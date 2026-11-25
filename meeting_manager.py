"""Meeting lifecycle orchestration around capture, processing, and persistence."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_capture import AudioCapture, AudioChunk
from audio_processor import AudioProcessor
from config import Settings, settings
from database import Database
from meeting_intelligence import MeetingIntelligence
from transcript_manager import TranscriptManager
from transcription_engine import TranscriptionEngine

logger = logging.getLogger(__name__)


class MeetingManager:
    def __init__(
        self,
        *,
        config: Settings = settings,
        database: Database | None = None,
        capture_factory: Any = AudioCapture,
        processor_factory: Any = AudioProcessor,
        transcriber: Any | None = None,
        intelligence: Any | None = None,
    ) -> None:
        self.config = config
        self.database = database or Database(config=config)
        self.capture_factory = capture_factory
        self.processor_factory = processor_factory
        self.transcriber = transcriber or TranscriptionEngine(config=config)
        self.intelligence = intelligence or MeetingIntelligence(config=config)
        self._lock = threading.RLock()
        self._capture: AudioCapture | None = None
        self._meeting_id: str | None = None
        self._transcript: TranscriptManager | None = None
        self._processors: dict[str, AudioProcessor] = {}
        self._workers: list[threading.Thread] = []
        self._worker_stop = threading.Event()
        self._active_started: float | None = None
        self._active_duration = 0.0
        self._subscribers: set[queue.Queue] = set()
        self._summary_jobs: dict[str, threading.Thread] = {}

    @property
    def meeting_id(self) -> str | None:
        return self._meeting_id

    def subscribe(self, event_queue: queue.Queue) -> None:
        with self._lock:
            self._subscribers.add(event_queue)

    def unsubscribe(self, event_queue: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(event_queue)

    def _publish(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(message)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(message)
                except queue.Empty:
                    pass

    def start_meeting(
        self,
        title: str,
        microphone_device: int | str | None = None,
        system_audio_device: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._meeting_id and self._status() in {"recording", "paused"}:
                raise RuntimeError("A meeting is already active")
            meeting_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            self.database.create_meeting(
                meeting_id, title.strip() or "Untitled meeting", now, now
            )
            self._meeting_id = meeting_id
            self._active_duration = 0.0
            self._active_started = time.monotonic()
            self._worker_stop.clear()
            self._transcript = TranscriptManager(
                self.database, meeting_id, on_update=self._publish
            )
            self._processors = {
                source: self.processor_factory(config=self.config)
                for source in ("microphone", "system")
            }
            self._capture = self.capture_factory(
                microphone_device=microphone_device,
                system_audio_device=system_audio_device,
                config=self.config,
            )
            try:
                paths = self._capture.start()
                self.database.update_meeting(
                    meeting_id,
                    microphone_recording=str(paths.get("microphone", "")) or None,
                    system_recording=str(paths.get("system", "")) or None,
                )
                sources = []
                if self.config.audio_capture_microphone:
                    sources.append("microphone")
                if self.config.audio_capture_system:
                    sources.append("system")
                self._workers = [
                    threading.Thread(
                        target=self._consume,
                        args=(source,),
                        daemon=True,
                        name=f"{source}-transcription",
                    )
                    for source in sources
                ]
                for worker in self._workers:
                    worker.start()
            except Exception:
                self._worker_stop.set()
                if self._capture:
                    self._capture.stop()
                self.database.update_meeting(
                    meeting_id,
                    status="error",
                    ended_at=datetime.now(timezone.utc).isoformat(),
                    duration=0,
                )
                self._meeting_id = None
                raise
            status = self.status()
            self._publish({"type": "status", "meeting": status})
            logger.info("Started meeting %s", meeting_id)
            return status

    def pause_meeting(self) -> dict[str, Any]:
        with self._lock:
            meeting_id = self._require_active()
            if self._status() == "paused":
                return self.status()
            if self._capture:
                self._capture.pause()
            self._active_duration += self._elapsed()
            self._active_started = None
            self.database.update_meeting(
                meeting_id, status="paused", updated_at=self._now()
            )
            status = self.status()
        self._publish({"type": "status", "meeting": status})
        return status

    def resume_meeting(self) -> dict[str, Any]:
        with self._lock:
            meeting_id = self._require_active()
            if self._status() == "recording":
                return self.status()
            if self._capture:
                self._capture.resume()
            self._active_started = time.monotonic()
            self.database.update_meeting(
                meeting_id, status="recording", updated_at=self._now()
            )
            status = self.status()
        self._publish({"type": "status", "meeting": status})
        return status

    def stop_meeting(self, *, summarize: bool = True) -> dict[str, Any]:
        with self._lock:
            meeting_id = self._require_active()
            self._active_duration += self._elapsed()
            self._active_started = None
            if self._capture:
                self._capture.stop()
            self._worker_stop.set()
            workers = list(self._workers)
        for worker in workers:
            worker.join()
        for source, processor in self._processors.items():
            for chunk in processor.flush(source):
                self._transcribe(chunk)
        now = self._now()
        self.database.update_meeting(
            meeting_id,
            status="completed",
            ended_at=now,
            duration=self._active_duration,
            updated_at=now,
        )
        if summarize and self._transcript and self._transcript.segments():
            self._schedule_summary(meeting_id)
        status = self.status()
        self._publish({"type": "status", "meeting": status})
        logger.info("Stopped meeting %s", meeting_id)
        return status

    def _schedule_summary(self, meeting_id: str) -> None:
        existing = self._summary_jobs.get(meeting_id)
        if existing and existing.is_alive():
            return

        def summarize_after_stop() -> None:
            try:
                self.generate_summary(meeting_id)
            except Exception:
                logger.exception("Automatic summary generation failed")

        worker = threading.Thread(
            target=summarize_after_stop,
            daemon=True,
            name=f"summary-{meeting_id[:8]}",
        )
        self._summary_jobs[meeting_id] = worker
        worker.start()

    def _consume(self, source: str) -> None:
        assert self._capture is not None
        audio_queue = (
            self._capture.microphone_queue
            if source == "microphone"
            else self._capture.system_audio_queue
        )
        processor = self._processors[source]
        while not self._worker_stop.is_set() or not audio_queue.empty():
            try:
                captured: AudioChunk = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                for chunk in processor.process(captured):
                    self._transcribe(chunk)
            except Exception:
                logger.exception("Audio processing failed for source %s", source)
            finally:
                audio_queue.task_done()

    def _transcribe(self, chunk) -> None:
        if not self._transcript:
            return
        try:
            result = self.transcriber.transcribe_chunk(chunk)
            for segment in result["segments"]:
                self._transcript.add_segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"],
                    source=chunk.source,
                    confidence=segment.get("confidence"),
                    language=result.get("language"),
                )
        except Exception:
            logger.exception("Transcription failed for source %s", chunk.source)

    def generate_summary(self, meeting_id: str | None = None) -> dict[str, Any]:
        meeting_id = meeting_id or self._meeting_id
        if not meeting_id:
            raise RuntimeError("No meeting selected")
        segments = self.database.get_segments(meeting_id)
        transcript = "\n".join(
            f"[{self._format_time(item['start'])}] {item['text']}" for item in segments
        )
        result = self.intelligence.summarize(transcript)
        self.database.save_summary(meeting_id, result)
        self._publish({"type": "summary", "meeting_id": meeting_id, "summary": result})
        return result

    def transcribe_recordings(self, meeting_id: str) -> int:
        meeting = self.database.get_meeting(meeting_id)
        if meeting is None:
            raise ValueError("Meeting not found")
        transcript = TranscriptManager(
            self.database, meeting_id, on_update=self._publish
        )
        count = 0
        sources = (
            ("system", meeting.get("system_recording")),
            ("microphone", meeting.get("microphone_recording")),
        )
        for source, path in sources:
            if not path or not Path(path).is_file():
                continue
            result = self.transcriber.transcribe_file(path)
            for segment in result["segments"]:
                saved = transcript.add_segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=segment["text"],
                    source=source,
                    confidence=segment.get("confidence"),
                    language=result.get("language"),
                )
                count += saved is not None
        return count

    def status(self) -> dict[str, Any]:
        if not self._meeting_id:
            return {"status": "idle", "meeting_id": None, "duration": 0.0}
        meeting = self.database.get_meeting(self._meeting_id) or {}
        duration = self._active_duration + self._elapsed()
        return {
            **meeting,
            "status": meeting.get("status", "idle"),
            "meeting_id": self._meeting_id,
            "duration": duration,
        }

    def list_meetings(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.database.list_meetings(limit)

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        return self.database.get_meeting(meeting_id)

    def get_transcript(self, meeting_id: str) -> list[dict[str, Any]]:
        return self.database.get_segments(meeting_id)

    def get_summary(self, meeting_id: str) -> dict[str, Any] | None:
        return self.database.get_summary(meeting_id)

    def rename_speaker(
        self, meeting_id: str, speaker_id: str, name: str
    ) -> dict[str, Any]:
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError("Speaker name must be between 1 and 100 characters")
        self.database.set_speaker(meeting_id, speaker_id, name)
        return {"id": speaker_id, "name": name}

    def configure(self, **changes: Any) -> dict[str, Any]:
        if self._meeting_id and self._status() in {"recording", "paused"}:
            raise RuntimeError("Configuration cannot change during an active meeting")
        allowed = {
            "whisper_model",
            "compute_device",
            "compute_type",
            "transcription_language",
            "ollama_model",
        }
        updates = {key: value for key, value in changes.items() if key in allowed}
        self.config = replace(self.config, **updates)
        self.transcriber = TranscriptionEngine(config=self.config)
        self.intelligence = MeetingIntelligence(config=self.config)
        return {
            "whisper_model": self.config.whisper_model,
            "compute_device": self.config.compute_device,
            "compute_type": self.config.compute_type,
            "transcription_language": self.config.transcription_language,
            "ollama_model": self.config.ollama_model,
        }

    def _require_active(self) -> str:
        if not self._meeting_id or self._status() not in {"recording", "paused"}:
            raise RuntimeError("No active meeting")
        return self._meeting_id

    def _status(self) -> str:
        meeting = self.database.get_meeting(self._meeting_id) if self._meeting_id else None
        return meeting["status"] if meeting else "idle"

    def _elapsed(self) -> float:
        if self._active_started is None:
            return 0.0
        return max(0.0, time.monotonic() - self._active_started)

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02}:{seconds:02}"

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
