"""Local transcript and meeting-note exports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import Settings, settings

EXPORT_FORMATS = {"txt", "json", "srt", "docx", "pdf"}


class ExportManager:
    def __init__(
        self, directory: str | Path | None = None, *, config: Settings = settings
    ) -> None:
        self.directory = Path(directory or config.exports_dir)

    def export(
        self,
        meeting: dict[str, Any],
        segments: list[dict[str, Any]],
        summary: dict[str, Any] | None,
        file_format: str,
    ) -> Path:
        file_format = file_format.lower()
        if file_format not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {file_format}")
        self.directory.mkdir(parents=True, exist_ok=True)
        meeting_id = re.sub(r"[^A-Za-z0-9_-]", "_", meeting["id"])
        path = self.directory / f"{meeting_id}.{file_format}"
        ordered = sorted(segments, key=lambda item: (item["start"], item["end"]))
        notes = summary or {}
        if file_format == "json":
            content = {"meeting": meeting, "transcript": ordered, "summary": notes}
            path.write_text(
                json.dumps(content, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        elif file_format == "srt":
            path.write_text(self._srt(ordered), encoding="utf-8")
        elif file_format == "txt":
            path.write_text(
                self._text(meeting, ordered, notes),
                encoding="utf-8",
            )
        elif file_format == "docx":
            self._docx(path, meeting, ordered, notes)
        else:
            self._pdf(path, meeting, ordered, notes)
        return path

    def _text(
        self,
        meeting: dict[str, Any],
        segments: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> str:
        lines = [
            meeting.get("title", "Meeting"),
            f"Date: {meeting.get('started_at', '')}",
            f"Duration: {meeting.get('duration', 0):.0f} seconds",
            "",
        ]
        if summary:
            lines.extend(
                [
                    "Executive summary",
                    str(summary.get("executive_summary", "")),
                    "",
                    "Key discussion points",
                    *[f"- {item}" for item in summary.get("key_points", [])],
                    "",
                    "Decisions",
                    *[f"- {self._label(item)}" for item in summary.get("decisions", [])],
                    "",
                    "Action items",
                    *[f"- {self._label(item)}" for item in summary.get("action_items", [])],
                    "",
                    "Open questions",
                    *[f"- {item}" for item in summary.get("open_questions", [])],
                    "",
                ]
            )
        lines.append("Transcript")
        for segment in segments:
            who = segment.get("speaker_name") or segment.get("speaker_id") or "Unknown"
            lines.append(
                f"[{self._timestamp(segment['start'])}] "
                f"[{who}] [{segment.get('source', '')}] {segment['text']}"
            )
        return "\n".join(lines) + "\n"

    def _srt(self, segments: list[dict[str, Any]]) -> str:
        blocks = []
        for index, segment in enumerate(segments, 1):
            speaker = segment.get("speaker_name") or segment.get("speaker_id")
            text = f"{speaker}: {segment['text']}" if speaker else segment["text"]
            blocks.append(
                f"{index}\n{self._srt_timestamp(segment['start'])} --> "
                f"{self._srt_timestamp(segment['end'])}\n{text}"
            )
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _docx(self, path, meeting, segments, summary) -> None:
        from docx import Document

        document = Document()
        document.add_heading(meeting.get("title", "Meeting"), 0)
        document.add_paragraph(f"Date: {meeting.get('started_at', '')}")
        document.add_paragraph(f"Duration: {meeting.get('duration', 0):.0f} seconds")
        if summary:
            document.add_heading("Executive summary", level=1)
            document.add_paragraph(str(summary.get("executive_summary", "")))
            for heading, key in (
                ("Key discussion points", "key_points"),
                ("Decisions", "decisions"),
                ("Action items", "action_items"),
                ("Open questions", "open_questions"),
            ):
                document.add_heading(heading, level=1)
                for item in summary.get(key, []):
                    document.add_paragraph(self._label(item), style="List Bullet")
        document.add_heading("Transcript", level=1)
        for segment in segments:
            speaker = segment.get("speaker_name") or segment.get("speaker_id") or "Unknown"
            document.add_paragraph(
                f"[{self._timestamp(segment['start'])}] {speaker} "
                f"({segment.get('source', '')}): {segment['text']}"
            )
        document.save(path)

    def _pdf(self, path, meeting, segments, summary) -> None:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        styles = getSampleStyleSheet()
        font_paths = (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        )
        font_path = next((candidate for candidate in font_paths if candidate.is_file()), None)
        if font_path:
            pdfmetrics.registerFont(TTFont("MeetingUnicode", str(font_path)))
            for style in styles.byName.values():
                style.fontName = "MeetingUnicode"
        story = [Paragraph(meeting.get("title", "Meeting"), styles["Title"])]
        story.extend(
            [
                Paragraph(f"Date: {meeting.get('started_at', '')}", styles["BodyText"]),
                Paragraph(
                    f"Duration: {meeting.get('duration', 0):.0f} seconds",
                    styles["BodyText"],
                ),
                Spacer(1, 12),
            ]
        )
        if summary:
            story.extend(
                [
                    Paragraph("Executive summary", styles["Heading2"]),
                    Paragraph(
                        self._escape(summary.get("executive_summary", "")),
                        styles["BodyText"],
                    ),
                    Spacer(1, 8),
                ]
            )
            for key, heading in (
                ("key_points", "Key discussion points"),
                ("decisions", "Decisions"),
                ("action_items", "Action items"),
                ("open_questions", "Open questions"),
            ):
                story.append(Paragraph(heading, styles["Heading2"]))
                story.extend(
                    Paragraph(self._escape(self._label(item)), styles["BodyText"])
                    for item in summary.get(key, [])
                )
        story.append(Paragraph("Transcript", styles["Heading2"]))
        for segment in segments:
            speaker = segment.get("speaker_name") or segment.get("speaker_id") or "Unknown"
            text = (
                f"[{self._timestamp(segment['start'])}] {speaker} "
                f"({segment.get('source', '')}): {segment['text']}"
            )
            story.append(Paragraph(self._escape(text), styles["BodyText"]))
        SimpleDocTemplate(str(path), pagesize=letter).build(story)

    @staticmethod
    def _label(item: Any) -> str:
        if isinstance(item, dict):
            parts = [str(item.get("task") or item.get("text") or "")]
            if item.get("owner"):
                parts.append(f"Owner: {item['owner']}")
            if item.get("deadline"):
                parts.append(f"Deadline: {item['deadline']}")
            return " — ".join(parts)
        return str(item)

    @staticmethod
    def _escape(text: Any) -> str:
        import xml.sax.saxutils

        return xml.sax.saxutils.escape(str(text))

    @staticmethod
    def _timestamp(seconds: float) -> str:
        minutes, remaining = divmod(max(0, int(seconds)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{remaining:02}"

    @staticmethod
    def _srt_timestamp(seconds: float) -> str:
        milliseconds = round(max(0, seconds) * 1000)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1_000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
