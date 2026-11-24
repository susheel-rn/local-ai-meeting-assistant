"""FastAPI service exposing local meeting lifecycle and transcript APIs."""

from __future__ import annotations

import asyncio
import logging
import queue
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from audio_capture import AudioCapture
from config import configure_logging, settings
from export_manager import EXPORT_FORMATS, ExportManager
from meeting_manager import MeetingManager
from speaker_diarization import SpeakerDiarizer

configure_logging()
logger = logging.getLogger(__name__)
manager = MeetingManager()
exporter = ExportManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Local Meeting Assistant API starting")
    yield
    try:
        if manager.status()["status"] in {"recording", "paused"}:
            manager.stop_meeting(summarize=False)
    finally:
        manager.database.close()
        logger.info("Local Meeting Assistant API stopped")


app = FastAPI(title="Local AI Meeting Assistant", version="0.1.0", lifespan=lifespan)


class LocalOriginMiddleware:
    """Reject browser state-changing requests originating outside localhost."""

    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") in {
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
        }:
            headers = dict(scope.get("headers", []))
            origin = headers.get(b"origin") or headers.get(b"referer")
            try:
                host = (
                    urlsplit(origin.decode("latin-1")).hostname if origin else None
                )
            except ValueError:
                host = None
            if host not in {"localhost", "127.0.0.1", "::1"}:
                response = PlainTextResponse(
                    "A localhost Origin or Referer is required for state changes.",
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.application(scope, receive, send)


app.add_middleware(LocalOriginMiddleware)


class StartRequest(BaseModel):
    title: str = Field(default="Meeting", min_length=1, max_length=200)
    microphone_device: int | str | None = None
    system_audio_device: str | None = None


class SegmentUpdate(BaseModel):
    text: str | None = Field(default=None, max_length=10_000)
    speaker_id: str | None = Field(default=None, max_length=100)


class SpeakerUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class SummaryUpdate(BaseModel):
    summary: dict[str, Any]


class SettingsUpdate(BaseModel):
    whisper_model: str | None = None
    compute_device: str | None = None
    compute_type: str | None = None
    transcription_language: str | None = None
    ollama_model: str | None = Field(default=None, max_length=100)


def _not_found(meeting_id: str) -> None:
    if manager.get_meeting(meeting_id) is None:
        raise HTTPException(404, "Meeting not found")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/meeting/start")
def start_meeting(request: StartRequest):
    try:
        return manager.start_meeting(
            request.title, request.microphone_device, request.system_audio_device
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not start meeting")
        raise HTTPException(503, "Unable to start audio capture") from exc


@app.post("/api/meeting/stop")
def stop_meeting():
    try:
        return manager.stop_meeting()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/meeting/pause")
def pause_meeting():
    try:
        return manager.pause_meeting()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/meeting/resume")
def resume_meeting():
    try:
        return manager.resume_meeting()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/meeting/status")
def meeting_status():
    return manager.status()


@app.get("/api/meeting/devices")
def devices():
    return AudioCapture.list_audio_devices()


@app.get("/api/meeting/models")
def models():
    return {
        "whisper": ["tiny", "base", "small", "medium", "large-v3"],
        "selected_whisper": manager.config.whisper_model,
        "device": manager.config.compute_device,
        "language": manager.config.transcription_language,
        "ollama": manager.config.ollama_model,
    }


@app.put("/api/settings")
def update_settings(request: SettingsUpdate):
    try:
        changes = request.model_dump(exclude_unset=True, exclude_none=True)
        if not changes:
            raise HTTPException(422, "At least one setting is required")
        return manager.configure(**changes)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/meetings")
def meetings(limit: int = 100):
    return manager.list_meetings(limit)


@app.get("/api/meetings/{meeting_id}")
def meeting_detail(meeting_id: str):
    _not_found(meeting_id)
    return manager.get_meeting(meeting_id)


@app.get("/api/meetings/{meeting_id}/transcript")
def transcript(meeting_id: str):
    _not_found(meeting_id)
    return manager.get_transcript(meeting_id)


@app.put("/api/meetings/{meeting_id}/transcript/{segment_id}")
def edit_transcript(meeting_id: str, segment_id: str, request: SegmentUpdate):
    _not_found(meeting_id)
    fields = request.model_dump(exclude_unset=True, exclude_none=True)
    if not fields:
        raise HTTPException(422, "At least one transcript field is required")
    if "text" in fields and not fields["text"].strip():
        raise HTTPException(422, "Transcript text cannot be empty")
    if not manager.database.update_segment(meeting_id, segment_id, **fields):
        raise HTTPException(404, "Transcript segment not found")
    return next(
        item
        for item in manager.get_transcript(meeting_id)
        if item["id"] == segment_id
    )


@app.get("/api/meetings/{meeting_id}/summary")
def summary(meeting_id: str):
    _not_found(meeting_id)
    return manager.get_summary(meeting_id) or {}


@app.put("/api/meetings/{meeting_id}/summary")
def edit_summary(meeting_id: str, request: SummaryUpdate):
    _not_found(meeting_id)
    manager.database.save_summary(meeting_id, request.summary)
    return request.summary


@app.post("/api/meetings/{meeting_id}/summarize")
def summarize(meeting_id: str):
    _not_found(meeting_id)
    try:
        return manager.generate_summary(meeting_id)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/meetings/{meeting_id}/transcribe")
def transcribe_recordings(meeting_id: str):
    _not_found(meeting_id)
    try:
        return {"segments_added": manager.transcribe_recordings(meeting_id)}
    except Exception as exc:
        logger.exception("Saved recording transcription failed")
        raise HTTPException(503, "Could not transcribe saved audio") from exc


@app.put("/api/meetings/{meeting_id}/speakers/{speaker_id}")
def rename_speaker(meeting_id: str, speaker_id: str, request: SpeakerUpdate):
    _not_found(meeting_id)
    try:
        return manager.rename_speaker(meeting_id, speaker_id, request.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/meetings/{meeting_id}/diarize")
def diarize(meeting_id: str):
    _not_found(meeting_id)
    meeting = manager.get_meeting(meeting_id)
    audio_paths = [
        path
        for path in (
            meeting.get("system_recording"),
            meeting.get("microphone_recording"),
        )
        if path and Path(path).is_file()
    ]
    if not audio_paths:
        raise HTTPException(409, "No recording is available for diarization")
    try:
        segments = manager.get_transcript(meeting_id)
        diarizer = SpeakerDiarizer()
        all_turns = []
        for audio_path in audio_paths:
            turns = diarizer.diarize(audio_path)
            source = (
                "system"
                if audio_path == meeting.get("system_recording")
                else "microphone"
            )
            matching = [segment for segment in segments if segment["source"] == source]
            SpeakerDiarizer.assign_speakers(matching, turns)
            all_turns.extend({**turn, "source": source} for turn in turns)
        for segment in segments:
            manager.database.update_segment(
                meeting_id, segment["id"], speaker_id=segment["speaker_id"]
            )
            if segment["speaker_id"] not in {None, "Unknown"}:
                manager.rename_speaker(
                    meeting_id, segment["speaker_id"], segment["speaker_id"]
                )
        return {"segments": segments, "turns": all_turns}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("Diarization failed")
        raise HTTPException(503, "Diarization failed") from exc


@app.get("/api/meetings/{meeting_id}/export/{file_format}")
def export_meeting(meeting_id: str, file_format: str):
    _not_found(meeting_id)
    if file_format.lower() not in EXPORT_FORMATS:
        raise HTTPException(400, "Unsupported export format")
    meeting = manager.get_meeting(meeting_id)
    speakers = {item["id"]: item["name"] for item in meeting.get("speakers", [])}
    segments = manager.get_transcript(meeting_id)
    for segment in segments:
        segment["speaker_name"] = speakers.get(segment.get("speaker_id"))
    try:
        path = exporter.export(
            meeting, segments, manager.get_summary(meeting_id), file_format
        )
    except ImportError as exc:
        raise HTTPException(503, f"Export dependency unavailable: {exc.name}") from exc
    except Exception as exc:
        logger.exception("Export failed")
        raise HTTPException(500, "Could not export meeting") from exc
    return FileResponse(path, filename=path.name)


@app.websocket("/ws/transcription")
async def transcription_socket(websocket: WebSocket):
    await websocket.accept()
    events: queue.Queue = queue.Queue(maxsize=256)
    manager.subscribe(events)
    await websocket.send_json({"type": "status", "meeting": manager.status()})
    try:
        while True:
            try:
                event = await asyncio.to_thread(events.get, True, 1)
                await websocket.send_json(event)
            except queue.Empty:
                await websocket.send_json(
                    {"type": "heartbeat", "meeting": manager.status()}
                )
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(events)
