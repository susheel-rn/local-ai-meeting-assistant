"""Application configuration for the local meeting assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load simple KEY=VALUE settings without overriding process environment."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.is_file():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        os.environ.setdefault(key.strip(), value)


_load_env_file()


@dataclass(frozen=True)
class Settings:
    audio_sample_rate: int = 48_000
    audio_chunk_duration: float = 1.0
    audio_queue_size: int = 128
    audio_capture_microphone: bool = True
    audio_capture_system: bool = True
    audio_recordings_enabled: bool = True
    recordings_dir: Path = PROJECT_DIR / "recordings"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables, falling back to defaults."""
        defaults = cls()
        return cls(
            audio_sample_rate=int(
                os.getenv("AUDIO_SAMPLE_RATE", defaults.audio_sample_rate)
            ),
            audio_chunk_duration=float(
                os.getenv("AUDIO_CHUNK_DURATION", defaults.audio_chunk_duration)
            ),
            audio_queue_size=int(
                os.getenv("AUDIO_QUEUE_SIZE", defaults.audio_queue_size)
            ),
            audio_capture_microphone=os.getenv(
                "AUDIO_CAPTURE_MICROPHONE", str(defaults.audio_capture_microphone)
            ).lower()
            in {"1", "true", "yes", "on"},
            audio_capture_system=os.getenv(
                "AUDIO_CAPTURE_SYSTEM", str(defaults.audio_capture_system)
            ).lower()
            in {"1", "true", "yes", "on"},
            audio_recordings_enabled=os.getenv(
                "AUDIO_RECORDINGS_ENABLED",
                str(defaults.audio_recordings_enabled),
            ).lower()
            in {"1", "true", "yes", "on"},
            recordings_dir=Path(
                os.getenv("RECORDINGS_DIR", str(defaults.recordings_dir))
            ).expanduser(),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        )


settings = Settings.from_env()
