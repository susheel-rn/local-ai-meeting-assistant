"""SQLite persistence for meetings, transcripts, speakers, and notes."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import Settings, settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(
        self, path: str | Path | None = None, *, config: Settings = settings
    ) -> None:
        self.path = Path(path or config.database_path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), check_same_thread=False, timeout=30
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        if str(self.path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self.initialize()

    def initialize(self) -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, status TEXT NOT NULL,
                started_at TEXT NOT NULL, ended_at TEXT, duration REAL NOT NULL DEFAULT 0,
                microphone_recording TEXT, system_recording TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS transcript_segments (
                id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id)
                    ON DELETE CASCADE,
                start REAL NOT NULL, end REAL NOT NULL, text TEXT NOT NULL,
                source TEXT NOT NULL, speaker_id TEXT, status TEXT NOT NULL DEFAULT 'final',
                confidence REAL, language TEXT, created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS speakers (
                id TEXT NOT NULL, meeting_id TEXT NOT NULL REFERENCES meetings(id)
                    ON DELETE CASCADE,
                name TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (meeting_id, id)
            )""",
            """CREATE TABLE IF NOT EXISTS meeting_summaries (
                meeting_id TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
                summary_json TEXT NOT NULL, updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id)
                    ON DELETE CASCADE,
                task TEXT NOT NULL, owner TEXT, deadline TEXT, status TEXT NOT NULL,
                timestamp REAL, created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL REFERENCES meetings(id)
                    ON DELETE CASCADE,
                text TEXT NOT NULL, timestamp REAL, created_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_segments_meeting_time "
            "ON transcript_segments(meeting_id, start, end)",
            "CREATE INDEX IF NOT EXISTS idx_meetings_started "
            "ON meetings(started_at DESC)",
        )
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                for statement in statements:
                    self._connection.execute(statement)
                self._connection.commit()
            except sqlite3.Error:
                self._connection.rollback()
                logger.exception("Could not initialize SQLite database")
                raise
        self.recover_incomplete_meetings()

    def recover_incomplete_meetings(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            rows = self._connection.execute(
                "SELECT id,started_at FROM meetings WHERE status IN ('recording','paused')"
            ).fetchall()
            for row in rows:
                try:
                    started = datetime.fromisoformat(row["started_at"])
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    duration = max(
                        0.0,
                        (now - started.astimezone(timezone.utc)).total_seconds(),
                    )
                except (TypeError, ValueError):
                    duration = 0.0
                self._connection.execute(
                    """UPDATE meetings SET status='interrupted',ended_at=?,duration=?,
                       updated_at=? WHERE id=?""",
                    (now.isoformat(), duration, now.isoformat(), row["id"]),
                )
            self._connection.commit()

    def execute(self, query: str, values: tuple[Any, ...] = ()) -> int:
        with self._lock:
            try:
                cursor = self._connection.execute(query, values)
                self._connection.commit()
                return cursor.rowcount
            except sqlite3.Error:
                self._connection.rollback()
                logger.exception("SQLite write failed")
                raise

    def fetch_all(
        self, query: str, values: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        with self._lock:
            try:
                return [dict(row) for row in self._connection.execute(query, values)]
            except sqlite3.Error:
                logger.exception("SQLite query failed")
                raise

    def fetch_one(
        self, query: str, values: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        rows = self.fetch_all(query, values)
        return rows[0] if rows else None

    def create_meeting(
        self,
        meeting_id: str,
        title: str,
        started_at: str,
        created_at: str,
    ) -> None:
        self.execute(
            """INSERT INTO meetings(id,title,status,started_at,created_at,updated_at)
               VALUES(?,?,'recording',?,?,?)""",
            (meeting_id, title, started_at, created_at, created_at),
        )

    def update_meeting(self, meeting_id: str, **fields: Any) -> None:
        allowed = {
            "title",
            "status",
            "ended_at",
            "duration",
            "microphone_recording",
            "system_recording",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = fields.get("updated_at") or datetime.now().astimezone().isoformat()
        columns = ", ".join(f"{key}=?" for key in updates)
        self.execute(
            f"UPDATE meetings SET {columns} WHERE id=?",
            tuple(updates.values()) + (meeting_id,),
        )

    def save_segment(self, segment: dict[str, Any]) -> None:
        self.execute(
            """INSERT OR REPLACE INTO transcript_segments
               (id,meeting_id,start,end,text,source,speaker_id,status,confidence,
                language,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                segment["id"],
                segment["meeting_id"],
                segment["start"],
                segment["end"],
                segment["text"],
                segment["source"],
                segment.get("speaker_id"),
                segment.get("status", "final"),
                segment.get("confidence"),
                segment.get("language"),
                segment["created_at"],
            ),
        )

    def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        meeting = self.fetch_one("SELECT * FROM meetings WHERE id=?", (meeting_id,))
        if meeting is not None:
            meeting["speakers"] = self.get_speakers(meeting_id)
        return meeting

    def list_meetings(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM meetings ORDER BY started_at DESC LIMIT ?",
            (max(1, min(int(limit), 500)),),
        )

    def get_segments(self, meeting_id: str) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM transcript_segments WHERE meeting_id=? "
            "ORDER BY start,end,id",
            (meeting_id,),
        )

    def find_overlapping_segments(
        self, meeting_id: str, source: str, start: float, end: float
    ) -> list[dict[str, Any]]:
        return self.fetch_all(
            """SELECT id,start,end,text,source FROM transcript_segments
               WHERE meeting_id=? AND source=? AND start < ? AND end > ?
               ORDER BY start DESC LIMIT 50""",
            (meeting_id, source, end, start),
        )

    def get_speakers(self, meeting_id: str) -> list[dict[str, Any]]:
        return self.fetch_all(
            "SELECT * FROM speakers WHERE meeting_id=? ORDER BY created_at,id",
            (meeting_id,),
        )

    def set_speaker(self, meeting_id: str, speaker_id: str, name: str) -> None:
        self.execute(
            """INSERT INTO speakers(id,meeting_id,name,created_at) VALUES(?,?,?,?)
               ON CONFLICT(meeting_id,id) DO UPDATE SET name=excluded.name""",
            (
                speaker_id,
                meeting_id,
                name,
                datetime.now().astimezone().isoformat(),
            ),
        )

    def update_segment(self, meeting_id: str, segment_id: str, **fields: Any) -> bool:
        allowed = {"text", "speaker_id", "status"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return False
        columns = ", ".join(f"{key}=?" for key in updates)
        return bool(
            self.execute(
                f"UPDATE transcript_segments SET {columns} WHERE meeting_id=? AND id=?",
                tuple(updates.values()) + (meeting_id, segment_id),
            )
        )

    def save_summary(self, meeting_id: str, summary: dict[str, Any]) -> None:
        summary_json = json.dumps(summary, ensure_ascii=False)
        updated_at = datetime.now().astimezone().isoformat()
        action_items = summary.get("action_items", [])
        if not isinstance(action_items, list):
            action_items = []
        decisions = summary.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                self._connection.execute(
                    """INSERT INTO meeting_summaries(meeting_id,summary_json,updated_at)
                       VALUES(?,?,?) ON CONFLICT(meeting_id) DO UPDATE SET
                       summary_json=excluded.summary_json,updated_at=excluded.updated_at""",
                    (meeting_id, summary_json, updated_at),
                )
                self._connection.execute(
                    "DELETE FROM action_items WHERE meeting_id=?", (meeting_id,)
                )
                self._connection.execute(
                    "DELETE FROM decisions WHERE meeting_id=?", (meeting_id,)
                )
                for item in action_items:
                    if not isinstance(item, dict) or not item.get("task"):
                        continue
                    self._connection.execute(
                        """INSERT INTO action_items
                           (id,meeting_id,task,owner,deadline,status,timestamp,created_at)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (
                            uuid.uuid4().hex,
                            meeting_id,
                            item["task"],
                            item.get("owner"),
                            item.get("deadline"),
                            item.get("status", "open"),
                            item.get("timestamp"),
                            updated_at,
                        ),
                    )
                for item in decisions:
                    if not isinstance(item, (dict, str)):
                        continue
                    text = item if isinstance(item, str) else item.get("text", "")
                    if not text:
                        continue
                    self._connection.execute(
                        "INSERT INTO decisions(id,meeting_id,text,timestamp,created_at) "
                        "VALUES(?,?,?,?,?)",
                        (
                            uuid.uuid4().hex,
                            meeting_id,
                            text,
                            None if isinstance(item, str) else item.get("timestamp"),
                            updated_at,
                        ),
                    )
                self._connection.commit()
            except (sqlite3.Error, TypeError, ValueError):
                self._connection.rollback()
                logger.exception("Could not save meeting summary")
                raise

    def get_summary(self, meeting_id: str) -> dict[str, Any] | None:
        row = self.fetch_one(
            "SELECT summary_json FROM meeting_summaries WHERE meeting_id=?",
            (meeting_id,),
        )
        return json.loads(row["summary_json"]) if row else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()
