"""Convert independent capture streams into overlapped Whisper-ready windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_capture import AudioChunk
from config import Settings, settings


@dataclass(frozen=True)
class ProcessedAudioChunk:
    source: str
    start: float
    end: float
    sample_rate: int
    samples: np.ndarray


class AudioProcessor:
    """Buffer each source independently and emit mono 16 kHz audio windows."""

    def __init__(
        self,
        *,
        config: Settings = settings,
        target_sample_rate: int = 16_000,
        chunk_duration: float | None = None,
        overlap_duration: float | None = None,
        vad_enabled: bool | None = None,
    ) -> None:
        self.config = config
        if target_sample_rate <= 0:
            raise ValueError("Target sample rate must be positive")
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = (
            config.transcription_chunk_duration
            if chunk_duration is None
            else chunk_duration
        )
        self.overlap_duration = (
            config.transcription_overlap_duration
            if overlap_duration is None
            else overlap_duration
        )
        self.vad_enabled = config.vad_enabled if vad_enabled is None else vad_enabled
        if self.chunk_duration <= 0 or self.overlap_duration < 0:
            raise ValueError("Chunk duration must be positive and overlap non-negative")
        if self.overlap_duration >= self.chunk_duration:
            raise ValueError("Overlap duration must be shorter than chunk duration")
        self._buffers: dict[str, np.ndarray] = {}
        self._buffer_starts: dict[str, float] = {}
        self._source_rates: dict[str, int] = {}
        self._window_samples = round(self.chunk_duration * target_sample_rate)
        self._overlap_samples = round(self.overlap_duration * target_sample_rate)

    def process(self, chunk: AudioChunk) -> list[ProcessedAudioChunk]:
        if chunk.sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive")
        if chunk.source not in {"microphone", "system"}:
            raise ValueError(f"Unknown audio source: {chunk.source}")
        audio = self._convert(chunk.samples, chunk.sample_rate)
        if audio.size == 0:
            return []

        source = chunk.source
        previous_rate = self._source_rates.get(source)
        if previous_rate != self.target_sample_rate:
            self._buffers.pop(source, None)
            self._buffer_starts[source] = chunk.timestamp
            self._source_rates[source] = self.target_sample_rate
        if source not in self._buffers:
            self._buffers[source] = np.empty(0, dtype=np.float32)
            self._buffer_starts[source] = chunk.timestamp

        results: list[ProcessedAudioChunk] = []
        expected = self._buffer_starts[source] + (
            len(self._buffers[source]) / self.target_sample_rate
        )
        if len(self._buffers[source]) == 0:
            expected = chunk.timestamp
            self._buffer_starts[source] = chunk.timestamp
        # A discontinuity means samples were dropped or capture was paused.
        if abs(chunk.timestamp - expected) > 0.25:
            remaining = self._buffers[source]
            if remaining.size and not self._is_silence(remaining):
                results.append(
                    ProcessedAudioChunk(
                        source=source,
                        start=self._buffer_starts[source],
                        end=self._buffer_starts[source]
                        + len(remaining) / self.target_sample_rate,
                        sample_rate=self.target_sample_rate,
                        samples=remaining.copy(),
                    )
                )
            self._buffers[source] = np.empty(0, dtype=np.float32)
            self._buffer_starts[source] = chunk.timestamp
        self._buffers[source] = np.concatenate((self._buffers[source], audio))

        hop = self._window_samples - self._overlap_samples
        while len(self._buffers[source]) >= self._window_samples:
            start = self._buffer_starts[source]
            window = self._buffers[source][: self._window_samples].copy()
            if not self._is_silence(window):
                results.append(
                    ProcessedAudioChunk(
                        source=source,
                        start=start,
                        end=start + len(window) / self.target_sample_rate,
                        sample_rate=self.target_sample_rate,
                        samples=window,
                    )
                )
            self._buffers[source] = self._buffers[source][hop:]
            self._buffer_starts[source] += hop / self.target_sample_rate
        return results

    def flush(self, source: str | None = None) -> list[ProcessedAudioChunk]:
        """Return each remaining non-silent stream buffer at meeting shutdown."""
        sources = [source] if source else list(self._buffers)
        flushed: list[ProcessedAudioChunk] = []
        for name in sources:
            samples = self._buffers.get(name)
            if samples is None or samples.size == 0:
                continue
            if not self._is_silence(samples):
                flushed.append(
                    ProcessedAudioChunk(
                        source=name,
                        start=self._buffer_starts[name],
                        end=self._buffer_starts[name]
                        + len(samples) / self.target_sample_rate,
                        sample_rate=self.target_sample_rate,
                        samples=samples.copy(),
                    )
                )
            self._buffers[name] = np.empty(0, dtype=np.float32)
        return flushed

    def _convert(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1, dtype=np.float32)
        elif audio.ndim != 1:
            raise ValueError("Audio must be a one- or two-dimensional array")
        if not audio.size:
            return np.empty(0, dtype=np.float32)
        if sample_rate == self.target_sample_rate:
            return audio.copy()
        output_size = max(
            1, round(len(audio) * self.target_sample_rate / sample_rate)
        )
        old_positions = np.arange(len(audio), dtype=np.float64)
        new_positions = np.arange(output_size, dtype=np.float64) * (
            sample_rate / self.target_sample_rate
        )
        return np.interp(new_positions, old_positions, audio).astype(np.float32)

    def _is_silence(self, audio: np.ndarray) -> bool:
        if not self.vad_enabled or audio.size == 0:
            return False
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) < 0.003
