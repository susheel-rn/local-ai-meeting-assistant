"""Lazy local Faster-Whisper model loading with CPU fallback."""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any

import numpy as np

from audio_processor import ProcessedAudioChunk
from config import Settings, settings

logger = logging.getLogger(__name__)
SUPPORTED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}


class TranscriptionEngine:
    def __init__(
        self,
        *,
        config: Settings = settings,
        model_factory: Any | None = None,
    ) -> None:
        self.config = config
        if config.whisper_model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported Whisper model: {config.whisper_model}")
        self._model_factory = model_factory
        self._model: Any | None = None
        self.device = ""
        self.compute_type = ""

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._model_factory
        if factory is None:
            factory = importlib.import_module("faster_whisper").WhisperModel
        device = self.config.compute_device
        if device == "auto":
            device = "cuda" if self._cuda_available() else "cpu"
        compute_type = self.config.compute_type or (
            "float16" if device == "cuda" else "int8"
        )
        try:
            self._model = factory(
                self.config.whisper_model, device=device, compute_type=compute_type
            )
        except Exception:
            if device != "cuda":
                raise
            logger.exception("GPU Whisper initialization failed; falling back to CPU")
            device, compute_type = "cpu", "int8"
            self._model = factory(
                self.config.whisper_model,
                device=device,
                compute_type=compute_type,
            )
        self.device, self.compute_type = device, compute_type
        logger.info(
            "Loaded Faster-Whisper model=%s device=%s compute_type=%s",
            self.config.whisper_model,
            device,
            compute_type,
        )
        return self._model

    @staticmethod
    def _cuda_available() -> bool:
        try:
            ctranslate2 = importlib.import_module("ctranslate2")
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    def transcribe_chunk(self, chunk: ProcessedAudioChunk) -> dict[str, Any]:
        if chunk.samples.size == 0:
            return {"segments": [], "language": self.config.transcription_language}
        model = self._load_model()
        started = time.monotonic()
        segments, info = model.transcribe(
            chunk.samples,
            language=self.config.transcription_language,
            vad_filter=self.config.vad_enabled,
            beam_size=5,
        )
        output = []
        for segment in segments:
            start = max(chunk.start, chunk.start + float(segment.start))
            end = min(chunk.end, chunk.start + float(segment.end))
            if end <= start or not segment.text.strip():
                continue
            output.append(
                {
                    "start": start,
                    "end": end,
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, "avg_logprob", None),
                    "no_speech_probability": getattr(
                        segment, "no_speech_prob", None
                    ),
                }
            )
        logger.info(
            "Transcribed %s chunk in %.2fs; segments=%d",
            chunk.source,
            time.monotonic() - started,
            len(output),
        )
        return {
            "segments": output,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }

    def transcribe_file(self, path: str) -> dict[str, Any]:
        """Transcribe a saved WAV/other Faster-Whisper supported audio file."""
        model = self._load_model()
        started = time.monotonic()
        segments, info = model.transcribe(
            path,
            language=self.config.transcription_language,
            vad_filter=self.config.vad_enabled,
            beam_size=5,
        )
        output = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "confidence": getattr(segment, "avg_logprob", None),
                "no_speech_probability": getattr(segment, "no_speech_prob", None),
            }
            for segment in segments
            if float(segment.end) > float(segment.start) and segment.text.strip()
        ]
        logger.info("Transcribed file in %.2fs", time.monotonic() - started)
        return {
            "segments": output,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }
