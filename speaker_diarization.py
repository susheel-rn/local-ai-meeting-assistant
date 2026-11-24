"""Optional post-meeting pyannote speaker diarization."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    def __init__(
        self,
        *,
        model_name: str = "pyannote/speaker-diarization-3.1",
        token: str | None = None,
        pipeline: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.token = token or os.getenv("HF_TOKEN")
        self._pipeline = pipeline

    def _load(self) -> Any:
        if self._pipeline is None:
            if not self.token:
                raise RuntimeError(
                    "Set HF_TOKEN and accept the pyannote model terms on Hugging Face."
                )
            pipeline_class = importlib.import_module(
                "pyannote.audio"
            ).Pipeline
            try:
                self._pipeline = pipeline_class.from_pretrained(
                    self.model_name, token=self.token
                )
            except TypeError:
                self._pipeline = pipeline_class.from_pretrained(
                    self.model_name, use_auth_token=self.token
                )
        return self._pipeline

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        """Return anonymous speaker turns; never infer real-world identity."""
        try:
            annotation = self._load()(audio_path)
        except Exception:
            logger.exception("Post-meeting diarization failed")
            raise
        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker_id": str(speaker),
                }
            )
        return turns

    @staticmethod
    def assign_speakers(
        segments: list[dict[str, Any]], turns: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        labels: dict[str, str] = {}
        for segment in segments:
            scores: dict[str, float] = {}
            for turn in turns:
                overlap = min(segment["end"], turn["end"]) - max(
                    segment["start"], turn["start"]
                )
                if overlap > 0:
                    speaker = turn["speaker_id"]
                    scores[speaker] = scores.get(speaker, 0.0) + overlap
            if scores:
                original = max(scores, key=scores.get)
                labels.setdefault(original, f"Speaker {len(labels) + 1}")
                segment["speaker_id"] = labels[original]
            else:
                segment["speaker_id"] = "Unknown"
        return segments
