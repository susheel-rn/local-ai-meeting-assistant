"""Chronological transcript storage, edits, and conservative overlap deduplication."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Callable

from database import Database


class TranscriptManager:
    def __init__(
        self,
        database: Database,
        meeting_id: str,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.database = database
        self.meeting_id = meeting_id
        self.on_update = on_update

    def add_segment(
        self,
        *,
        start: float,
        end: float,
        text: str,
        source: str,
        speaker_id: str | None = None,
        confidence: float | None = None,
        language: str | None = None,
        status: str = "final",
    ) -> dict[str, Any] | None:
        text = text.strip()
        if not text or end <= start or source not in {"microphone", "system"}:
            return None
        normalized = self._normalize(text)
        for existing in self.database.find_overlapping_segments(
            self.meeting_id, source, start, end
        ):
            overlap = min(end, existing["end"]) - max(start, existing["start"])
            if overlap <= 0:
                continue
            similarity = self._similarity(normalized, self._normalize(existing["text"]))
            if similarity >= 0.82:
                return None
        segment = {
            "id": uuid.uuid4().hex,
            "meeting_id": self.meeting_id,
            "start": float(start),
            "end": float(end),
            "text": text,
            "source": source,
            "speaker_id": speaker_id,
            "status": status,
            "confidence": confidence,
            "language": language,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        self.database.save_segment(segment)
        if self.on_update:
            self.on_update({"type": "transcript", "segment": segment})
        return segment

    def segments(self) -> list[dict[str, Any]]:
        return self.database.get_segments(self.meeting_id)

    def edit(self, segment_id: str, text: str) -> dict[str, Any] | None:
        if not text.strip():
            raise ValueError("Transcript text cannot be empty")
        changed = self.database.update_segment(
            self.meeting_id, segment_id, text=text.strip()
        )
        if not changed:
            return None
        return next(
            (segment for segment in self.segments() if segment["id"] == segment_id),
            None,
        )

    def assign_speaker(self, segment_id: str, speaker_id: str) -> bool:
        return self.database.update_segment(
            self.meeting_id, segment_id, speaker_id=speaker_id
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        if left == right:
            return 1.0
        if not left or not right:
            return 0.0
        left_words, right_words = set(left.split()), set(right.split())
        return len(left_words & right_words) / max(len(left_words), len(right_words))
