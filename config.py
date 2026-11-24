"""Application configuration for the local meeting assistant."""

from __future__ import annotations

import os
import logging
import math
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
    transcription_chunk_duration: float = 8.0
    transcription_overlap_duration: float = 1.0
    audio_capture_microphone: bool = True
    audio_capture_system: bool = True
    audio_recordings_enabled: bool = True
    whisper_model: str = "small"
    transcription_language: str | None = None
    compute_device: str = "auto"
    compute_type: str | None = None
    vad_enabled: bool = True
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    database_path: Path = PROJECT_DIR / "meeting_data.db"
    recordings_dir: Path = PROJECT_DIR / "recordings"
    exports_dir: Path = PROJECT_DIR / "exports"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        numeric_settings = (
            self.audio_sample_rate,
            self.audio_chunk_duration,
            self.transcription_chunk_duration,
            self.transcription_overlap_duration,
        )
        if not all(math.isfinite(value) for value in numeric_settings):
            raise ValueError("Audio durations and sample rate must be finite values")
        if self.audio_sample_rate <= 0 or self.audio_chunk_duration <= 0:
            raise ValueError("Audio sample rate and chunk duration must be positive")
        if self.audio_queue_size <= 0:
            raise ValueError("Audio queue size must be positive")
        if self.transcription_chunk_duration <= 0:
            raise ValueError("Transcription chunk duration must be positive")
        if not 0 <= self.transcription_overlap_duration < self.transcription_chunk_duration:
            raise ValueError(
                "Transcription overlap must be non-negative and shorter than its chunk"
            )
        if self.whisper_model not in {"tiny", "base", "small", "medium", "large-v3"}:
            raise ValueError(f"Unsupported Whisper model: {self.whisper_model}")
        if self.compute_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Compute device must be auto, cpu, or cuda")
        if self.compute_type and self.compute_type not in {
            "int8",
            "int8_float32",
            "int8_float16",
            "int8_bfloat16",
            "int16",
            "float16",
            "bfloat16",
            "float32",
        }:
            raise ValueError(f"Unsupported CTranslate2 compute type: {self.compute_type}")
        if self.log_level.upper() not in {
            "CRITICAL",
            "ERROR",
            "WARNING",
            "INFO",
            "DEBUG",
        }:
            raise ValueError(f"Invalid log level: {self.log_level}")

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables, falling back to defaults."""
        defaults = cls()

        def boolean(name: str, default: bool) -> bool:
            value = os.getenv(name)
            if value is None:
                return default
            normalized = value.lower()
            if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                raise ValueError(f"{name} must be a boolean value")
            return normalized in {"1", "true", "yes", "on"}

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
            transcription_chunk_duration=float(
                os.getenv(
                    "TRANSCRIPTION_CHUNK_DURATION",
                    defaults.transcription_chunk_duration,
                )
            ),
            transcription_overlap_duration=float(
                os.getenv(
                    "TRANSCRIPTION_OVERLAP_DURATION",
                    defaults.transcription_overlap_duration,
                )
            ),
            audio_capture_microphone=boolean(
                "AUDIO_CAPTURE_MICROPHONE", defaults.audio_capture_microphone
            ),
            audio_capture_system=boolean(
                "AUDIO_CAPTURE_SYSTEM", defaults.audio_capture_system
            ),
            audio_recordings_enabled=boolean(
                "AUDIO_RECORDINGS_ENABLED", defaults.audio_recordings_enabled
            ),
            whisper_model=os.getenv("WHISPER_MODEL", defaults.whisper_model),
            transcription_language=os.getenv("TRANSCRIPTION_LANGUAGE") or None,
            compute_device=os.getenv("COMPUTE_DEVICE", defaults.compute_device),
            compute_type=os.getenv("COMPUTE_TYPE") or None,
            vad_enabled=boolean("VAD_ENABLED", defaults.vad_enabled),
            ollama_url=os.getenv("OLLAMA_URL", defaults.ollama_url).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", defaults.ollama_model),
            database_path=Path(
                os.getenv("DATABASE_PATH", str(defaults.database_path))
            ).expanduser(),
            recordings_dir=Path(
                os.getenv("RECORDINGS_DIR", str(defaults.recordings_dir))
            ).expanduser(),
            exports_dir=Path(
                os.getenv("EXPORTS_DIR", str(defaults.exports_dir))
            ).expanduser(),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        )


settings = Settings.from_env()


def configure_logging(level: str = settings.log_level) -> None:
    """Configure concise application-wide logging without exposing credentials."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
