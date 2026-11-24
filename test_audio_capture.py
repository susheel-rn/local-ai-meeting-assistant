import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from audio_capture import AudioCapture
from config import Settings


class AudioCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = Settings(
            audio_sample_rate=16_000,
            audio_chunk_duration=0.01,
            audio_queue_size=8,
            recordings_dir=Path(self.temp_dir.name),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_device_listing_separates_microphones_and_outputs(self):
        sounddevice = SimpleNamespace(
            query_devices=lambda: [
                {"name": "Mic", "max_input_channels": 1},
                {"name": "Speakers", "max_input_channels": 0},
            ]
        )
        speaker = SimpleNamespace(id="out-1", name="Speakers")
        soundcard = SimpleNamespace(all_speakers=lambda: [speaker])

        devices = AudioCapture.list_audio_devices(
            sounddevice_module=sounddevice,
            soundcard_module=soundcard,
        )

        self.assertEqual(devices["microphones"][0]["name"], "Mic")
        self.assertEqual(devices["system_audio_outputs"][0]["id"], "out-1")

    def test_simultaneous_sources_are_queued_and_saved_separately(self):
        input_streams = []
        sounddevice = SimpleNamespace(
            InputStream=lambda **kwargs: self._make_input_stream(
                input_streams, kwargs
            )
        )

        system_samples = np.full((160, 2), 0.25, dtype=np.float32)

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def record(self, numframes):
                time.sleep(0.002)
                return system_samples[:numframes]

        loopback = SimpleNamespace(
            recorder=lambda **_kwargs: Recorder()
        )
        speaker = SimpleNamespace(name="Speakers", id="speaker-1")
        soundcard = SimpleNamespace(
            all_speakers=lambda: [speaker],
            default_speaker=lambda: speaker,
            get_microphone=lambda **_kwargs: loopback,
        )
        capture = AudioCapture(
            config=self.config,
            recording_dir=self.temp_dir.name,
            sounddevice_module=sounddevice,
            soundcard_module=soundcard,
        )

        paths = capture.start()
        input_streams[0].callback(
            np.full((160, 1), -0.25, dtype=np.float32), 160, None, None
        )

        microphone_chunk = capture.get_chunk("microphone", timeout=1)
        system_chunk = capture.get_chunk("system", timeout=1)
        self.assertEqual(microphone_chunk.source, "microphone")
        self.assertEqual(system_chunk.source, "system")
        self.assertEqual(microphone_chunk.samples.dtype, np.float32)
        self.assertGreaterEqual(system_chunk.timestamp, 0)
        self.assertTrue(capture.is_running)

        capture.pause()
        self.assertTrue(capture.is_paused)
        capture.resume()
        self.assertFalse(capture.is_paused)
        capture.stop()

        for source in ("microphone", "system"):
            self.assertTrue(paths[source].is_file())
            with wave.open(str(paths[source]), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16_000)
                self.assertGreater(wav_file.getnframes(), 0)
        self.assertNotEqual(paths["microphone"], paths["system"])

    def _make_input_stream(self, streams, kwargs):
        stream = MagicMock()
        stream.callback = kwargs["callback"]
        stream.start.side_effect = lambda: streams.append(stream)
        return stream


if __name__ == "__main__":
    unittest.main()
