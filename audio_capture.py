"""Independent microphone and Windows desktop-audio capture."""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import Settings, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    """One timestamped block of audio from a single input source."""

    source: str
    timestamp: float
    sample_rate: int
    samples: np.ndarray


class AudioCapture:
    """Capture microphone and speaker loopback audio on independent queues.

    Microphone input is streamed by ``sounddevice``. Desktop output is read
    separately using SoundCard's Windows WASAPI loopback support. Audio remains
    float32 in memory; optional WAV files are saved as 16-bit PCM.
    """

    def __init__(
        self,
        microphone_device: int | str | None = None,
        system_audio_device: str | None = None,
        *,
        config: Settings = settings,
        recording_dir: str | Path | None = None,
        record_audio: bool | None = None,
        sounddevice_module: Any | None = None,
        soundcard_module: Any | None = None,
    ) -> None:
        self.microphone_device = microphone_device
        self.system_audio_device = system_audio_device
        self.config = config
        self.recording_dir = Path(recording_dir or config.recordings_dir)
        self.record_audio = (
            config.audio_recordings_enabled if record_audio is None else record_audio
        )
        self.microphone_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=config.audio_queue_size
        )
        self.system_audio_queue: queue.Queue[AudioChunk] = queue.Queue(
            maxsize=config.audio_queue_size
        )

        self._sounddevice_module = sounddevice_module
        self._soundcard_module = soundcard_module
        self._microphone_stream: Any | None = None
        self._system_microphone: Any | None = None
        self._system_thread: threading.Thread | None = None
        self._writer_threads: list[threading.Thread] = []
        self._writer_queues: dict[str, queue.Queue[AudioChunk]] = {}
        self._writer_stop = threading.Event()
        self._stop_event = threading.Event()
        self._system_stop_event = threading.Event()
        self._paused = threading.Event()
        self._system_ready = threading.Event()
        self._system_error: Exception | None = None
        self._system_active = False
        self._start_time: float | None = None
        self._paused_total = 0.0
        self._pause_started: float | None = None
        self._running = False
        self._lock = threading.RLock()
        self.recording_paths: dict[str, Path] = {}
        self.dropped_chunks = {"microphone": 0, "system": 0}
        self.dropped_recording_chunks = {"microphone": 0, "system": 0}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @staticmethod
    def list_audio_devices(
        *,
        sounddevice_module: Any | None = None,
        soundcard_module: Any | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return selectable input microphones and desktop output devices."""
        devices: dict[str, list[dict[str, Any]]] = {
            "microphones": [],
            "system_audio_outputs": [],
        }
        try:
            sd = sounddevice_module or importlib.import_module("sounddevice")
            for index, device in enumerate(sd.query_devices()):
                if device.get("max_input_channels", 0) > 0:
                    devices["microphones"].append(
                        {
                            "id": index,
                            "name": device["name"],
                            "channels": device["max_input_channels"],
                        }
                    )
        except Exception:
            logger.exception("Could not enumerate microphone devices")

        try:
            sc = soundcard_module or importlib.import_module("soundcard")
            for speaker in sc.all_speakers():
                devices["system_audio_outputs"].append(
                    {
                        "id": str(getattr(speaker, "id", speaker.name)),
                        "name": str(speaker.name),
                    }
                )
        except Exception:
            logger.exception("Could not enumerate system-audio output devices")
        return devices

    def start(self) -> dict[str, Path]:
        """Start available input streams and optional independent WAV writers."""
        with self._lock:
            if self._running:
                raise RuntimeError("Audio capture is already running")
            self._stop_event.clear()
            self._system_stop_event.clear()
            self._writer_stop.clear()
            self._paused.clear()
            self._system_ready.clear()
            self._system_error = None
            self._system_active = False
            self._paused_total = 0.0
            self._pause_started = None
            self._clear_queue(self.microphone_queue)
            self._clear_queue(self.system_audio_queue)
            self._start_time = time.monotonic()
            self._running = True
            self.recording_paths = {}
            self._writer_threads = []
            self._writer_queues = {}

        failures: list[str] = []
        try:
            if self.record_audio:
                self._start_writers()

            if self.config.audio_capture_microphone:
                try:
                    sd = self._get_sounddevice()
                    self._microphone_stream = sd.InputStream(
                        samplerate=self.config.audio_sample_rate,
                        channels=1,
                        dtype="float32",
                        device=self.microphone_device,
                        callback=self._microphone_callback,
                    )
                    self._microphone_stream.start()
                    logger.info("Microphone capture started")
                except Exception as exc:
                    failures.append(f"microphone: {exc}")
                    logger.exception("Microphone capture could not be started")
                    self._microphone_stream = None

            if self.config.audio_capture_system:
                try:
                    self._system_microphone = self._get_loopback_microphone()
                    self._system_thread = threading.Thread(
                        target=self._capture_system_audio,
                        name="system-audio-capture",
                        daemon=True,
                    )
                    self._system_thread.start()
                    if not self._system_ready.wait(timeout=5):
                        self._system_stop_event.set()
                        self._system_thread.join(timeout=5)
                        raise RuntimeError("Timed out while opening system-audio capture")
                    if self._system_error is not None:
                        raise RuntimeError(str(self._system_error)) from self._system_error
                    logger.info("System-audio loopback capture started")
                except Exception as exc:
                    failures.append(f"system audio: {exc}")
                    logger.exception("System-audio capture could not be started")

            if not self._has_active_source():
                raise RuntimeError(
                    "No audio source could be started. " + "; ".join(failures)
                )
            return self.recording_paths.copy()
        except Exception:
            self.stop()
            raise

    def pause(self) -> None:
        with self._lock:
            if not self._running:
                raise RuntimeError("Audio capture is not running")
            if not self._paused.is_set():
                self._pause_started = time.monotonic()
                self._paused.set()

    def resume(self) -> None:
        with self._lock:
            if not self._running:
                raise RuntimeError("Audio capture is not running")
            if self._paused.is_set():
                if self._pause_started is not None:
                    self._paused_total += time.monotonic() - self._pause_started
                self._pause_started = None
                self._paused.clear()

    def stop(self) -> dict[str, Path]:
        """Stop capture and drain queued audio to the optional WAV recordings."""
        with self._lock:
            if not self._running:
                return self.recording_paths.copy()
            self._running = False
            self._stop_event.set()
            self._system_stop_event.set()

        if self._microphone_stream is not None:
            try:
                self._microphone_stream.stop()
                self._microphone_stream.close()
            except Exception:
                logger.exception("Error while closing microphone capture")
            self._microphone_stream = None

        if self._system_thread is not None:
            self._system_thread.join(
                timeout=max(5.0, self.config.audio_chunk_duration * 3)
            )
            if self._system_thread.is_alive():
                logger.error("System-audio capture did not stop within the timeout")
            self._system_thread = None

        self._writer_stop.set()
        for thread in self._writer_threads:
            thread.join(timeout=10)
            if thread.is_alive():
                logger.error("WAV writer did not stop within the timeout")
        self._writer_threads.clear()
        return self.recording_paths.copy()

    def get_chunk(self, source: str, timeout: float | None = None) -> AudioChunk:
        """Get the next chunk from ``microphone`` or ``system``."""
        if source == "microphone":
            return self.microphone_queue.get(timeout=timeout)
        if source == "system":
            return self.system_audio_queue.get(timeout=timeout)
        raise ValueError("source must be 'microphone' or 'system'")

    def _has_active_source(self) -> bool:
        return self._microphone_stream is not None or (
            self._system_thread is not None
            and self._system_thread.is_alive()
            and self._system_active
        )

    @staticmethod
    def _clear_queue(target: queue.Queue) -> None:
        while True:
            try:
                target.get_nowait()
                target.task_done()
            except queue.Empty:
                return

    def _get_sounddevice(self) -> Any:
        if self._sounddevice_module is None:
            self._sounddevice_module = importlib.import_module("sounddevice")
        return self._sounddevice_module

    def _get_loopback_microphone(self) -> Any:
        if self._soundcard_module is None:
            self._soundcard_module = importlib.import_module("soundcard")
        sc = self._soundcard_module
        speakers = sc.all_speakers()
        if self.system_audio_device is None:
            speaker = sc.default_speaker()
        else:
            speaker = next(
                (
                    candidate
                    for candidate in speakers
                    if self.system_audio_device
                    in {
                        str(getattr(candidate, "id", candidate.name)),
                        str(candidate.name),
                    }
                ),
                None,
            )
            if speaker is None:
                raise ValueError(
                    f"System-audio output device not found: {self.system_audio_device}"
                )
        return sc.get_microphone(id=str(speaker.name), include_loopback=True)

    def _microphone_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        del frames, time_info
        if status:
            logger.warning("Microphone capture status: %s", status)
        if self._stop_event.is_set() or self._paused.is_set():
            return
        self._publish_chunk("microphone", indata.copy())

    def _capture_system_audio(self) -> None:
        microphone = self._system_microphone
        try:
            frames = max(
                1,
                int(
                    self.config.audio_sample_rate
                    * self.config.audio_chunk_duration
                ),
            )
            with microphone.recorder(
                samplerate=self.config.audio_sample_rate, channels=2
            ) as recorder:
                self._system_ready.set()
                self._system_active = True
                while not self._system_stop_event.is_set():
                    samples = recorder.record(numframes=frames)
                    if self._paused.is_set() or self._system_stop_event.is_set():
                        continue
                    if samples is not None and np.size(samples):
                        self._publish_chunk("system", samples)
        except Exception as exc:
            self._system_error = exc
            self._system_ready.set()
            if not self._stop_event.is_set():
                logger.exception("System-audio capture stopped unexpectedly")
        finally:
            self._system_active = False

    def _publish_chunk(self, source: str, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        chunk_duration = samples.shape[0] / self.config.audio_sample_rate
        with self._lock:
            if self._start_time is None or self._paused.is_set():
                return
            timestamp = max(
                0.0,
                time.monotonic()
                - self._start_time
                - self._paused_total
                - chunk_duration,
            )
        chunk = AudioChunk(
            source=source,
            timestamp=timestamp,
            sample_rate=self.config.audio_sample_rate,
            samples=samples,
        )
        target = (
            self.microphone_queue if source == "microphone" else self.system_audio_queue
        )
        try:
            target.put_nowait(chunk)
        except queue.Full:
            self.dropped_chunks[source] += 1
            logger.warning("%s audio queue is full; dropping a chunk", source)

        writer_queue = self._writer_queues.get(source)
        if writer_queue is not None:
            try:
                writer_queue.put_nowait(chunk)
            except queue.Full:
                self.dropped_recording_chunks[source] += 1
                logger.error("%s WAV queue is full; recording chunk was dropped", source)

    def _start_writers(self) -> None:
        meeting_dir = self.recording_dir / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
        meeting_dir.mkdir(parents=True, exist_ok=True)
        channels = {"microphone": 1, "system": 2}
        active_sources = []
        if self.config.audio_capture_microphone:
            active_sources.append("microphone")
        if self.config.audio_capture_system:
            active_sources.append("system")
        for source in active_sources:
            path = meeting_dir / f"{source}.wav"
            self.recording_paths[source] = path
            self._writer_queues[source] = queue.Queue(
                maxsize=self.config.audio_queue_size
            )
            thread = threading.Thread(
                target=self._write_wav,
                args=(source, path, channels[source]),
                name=f"{source}-wav-writer",
                daemon=True,
            )
            thread.start()
            self._writer_threads.append(thread)

    def _write_wav(self, source: str, path: Path, channels: int) -> None:
        chunks = self._writer_queues[source]
        try:
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.config.audio_sample_rate)
                while not self._writer_stop.is_set() or not chunks.empty():
                    try:
                        chunk = chunks.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    samples = chunk.samples
                    if samples.shape[1] != channels:
                        samples = samples[:, :channels]
                        if samples.shape[1] < channels:
                            samples = np.pad(
                                samples,
                                ((0, 0), (0, channels - samples.shape[1])),
                            )
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
                    wav_file.writeframesraw(pcm.tobytes())
                    chunks.task_done()
        except Exception:
            logger.exception("Could not write %s audio recording to %s", source, path)
