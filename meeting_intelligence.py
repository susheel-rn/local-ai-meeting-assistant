"""Local Ollama meeting summaries and transcript-aware action extraction."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from config import Settings, settings

logger = logging.getLogger(__name__)

SUMMARY_SCHEMA = {
    "executive_summary": "",
    "overview": "",
    "key_points": [],
    "topics": [],
    "observations": [],
    "decisions": [],
    "action_items": [],
    "open_questions": [],
    "risks_and_blockers": [],
    "follow_up_items": [],
}


class MeetingIntelligence:
    def __init__(
        self,
        *,
        config: Settings = settings,
        opener: Any | None = None,
        max_chars: int = 12_000,
    ) -> None:
        self.config = config
        self.opener = opener or urllib.request.urlopen
        self.max_chars = max_chars

    def _ask(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "stream": False,
                "format": "json",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Summarize only facts explicitly present in the transcript. "
                            "Never invent decisions, names, owners, or deadlines. "
                            "Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            logger.warning("Ollama summary request failed: %s", exc)
            raise RuntimeError(
                "Could not generate meeting notes. Check that Ollama is running "
                "locally and the selected model is available."
            ) from exc

    def summarize(self, transcript: str) -> dict[str, Any]:
        if not transcript.strip():
            return dict(SUMMARY_SCHEMA)
        sections = [
            transcript[index : index + self.max_chars]
            for index in range(0, len(transcript), self.max_chars)
        ]
        partials = []
        for index, section in enumerate(sections, 1):
            prompt = (
                "Extract a concise JSON object with fields executive_summary, "
                "key_points, decisions, action_items (task, owner, deadline, "
                "status, timestamp), open_questions, risks_and_blockers. "
                "Use null for owners or deadlines not explicitly stated. "
                f"Transcript section {index}/{len(sections)}:\n{section}"
            )
            partials.append(self._parse_json(self._ask(prompt)))
        if len(partials) == 1:
            result = dict(SUMMARY_SCHEMA)
            result.update(partials[0])
            return result
        combined = json.dumps(partials, ensure_ascii=False)
        result = self._parse_json(
            self._ask(
                "Combine these partial meeting notes into one final JSON summary. "
                "Preserve decisions and action items; remove duplicates; never "
                "invent facts. Include all fields executive_summary, overview, "
                "key_points, topics, observations, decisions, action_items, "
                "open_questions, risks_and_blockers, follow_up_items.\n"
                + combined
            )
        )
        final = dict(SUMMARY_SCHEMA)
        final.update(result)
        return final

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama returned a non-object summary")
        return parsed
