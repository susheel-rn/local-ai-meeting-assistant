import json
import tempfile
import time
import unittest
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from audio_capture import AudioChunk
from audio_processor import AudioProcessor, ProcessedAudioChunk
from config import Settings
from database import Database
from export_manager import ExportManager
from meeting_intelligence import MeetingIntelligence
from meeting_manager import MeetingManager
from speaker_diarization import SpeakerDiarizer
from transcript_manager import TranscriptManager
from transcription_engine import TranscriptionEngine


class AudioProcessingTests(unittest.TestCase):
    def test_resample_overlap_and_flush(self):
        processor = AudioProcessor(
            target_sample_rate=16_000,
            chunk_duration=1,
            overlap_duration=0.25,
            vad_enabled=False,
        )
        audio = np.ones((64_000, 2), dtype=np.float32) * 0.1
        windows = processor.process(
            AudioChunk("microphone", 2.0, 32_000, audio)
        )
        tail = processor.flush("microphone")

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].samples.size, 16_000)
        self.assertEqual(windows[0].samples.dtype, np.float32)
        self.assertEqual(windows[0].start, 2.0)
        self.assertEqual(windows[1].start, 2.75)
        self.assertEqual(len(tail), 1)
        self.assertAlmostEqual(tail[0].end, 4.0)

    def test_capture_gap_flushes_previous_audio_instead_of_shifting_times(self):
        processor = AudioProcessor(
            target_sample_rate=16_000,
            chunk_duration=1,
            overlap_duration=0.1,
            vad_enabled=False,
        )
        processor.process(
            AudioChunk("system", 0.0, 16_000, np.ones(8_000, dtype=np.float32))
        )
        flushed = processor.process(
            AudioChunk("system", 1.5, 16_000, np.ones(3_200, dtype=np.float32))
        )
        self.assertEqual(flushed[0].start, 0)
        self.assertEqual(flushed[0].end, 0.5)
        self.assertEqual(processor.flush()[0].start, 1.5)

    def test_invalid_audio_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Settings(audio_queue_size=0)
        with self.assertRaises(ValueError):
            Settings(transcription_chunk_duration=1, transcription_overlap_duration=1)


class TranscriptionTests(unittest.TestCase):
    def test_fake_faster_whisper_model_reuses_model_and_offsets_timestamps(self):
        created = []

        class FakeModel:
            def transcribe(self, audio, **_kwargs):
                self.audio = audio
                return (
                    iter(
                        [
                            SimpleNamespace(
                                start=0.1,
                                end=0.4,
                                text=" hello ",
                                avg_logprob=-0.2,
                                no_speech_prob=0.01,
                            )
                        ]
                    ),
                    SimpleNamespace(language="en", language_probability=0.99),
                )

        def factory(name, **kwargs):
            created.append((name, kwargs))
            return FakeModel()

        engine = TranscriptionEngine(
            config=Settings(compute_device="cpu"),
            model_factory=factory,
        )
        chunk = ProcessedAudioChunk(
            "system", 3.0, 4.0, 16_000, np.ones(16_000, dtype=np.float32)
        )
        result = engine.transcribe_chunk(chunk)
        engine.transcribe_chunk(chunk)

        self.assertEqual(len(created), 1)
        self.assertEqual(result["segments"][0]["start"], 3.1)
        self.assertEqual(result["segments"][0]["text"], "hello")
        self.assertEqual(result["language"], "en")


class PersistenceAndTranscriptTests(unittest.TestCase):
    def test_history_transcript_edits_speakers_and_summary_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meetings.sqlite"
            db = Database(path)
            now = datetime.now(timezone.utc).isoformat()
            db.create_meeting("m1", "Planning", now, now)
            transcript = TranscriptManager(db, "m1")
            segment = transcript.add_segment(
                start=1,
                end=2,
                text="We will ship this.",
                source="microphone",
                speaker_id="speaker_0",
            )
            duplicate = transcript.add_segment(
                start=1.2,
                end=2.1,
                text="We will ship this.",
                source="microphone",
            )
            transcript.edit(segment["id"], "We will ship next week.")
            transcript.assign_speaker(segment["id"], "speaker_0")
            db.set_speaker("m1", "speaker_0", "Alex")
            summary = {
                "executive_summary": "Plan agreed.",
                "decisions": ["Ship the update."],
                "action_items": [
                    {
                        "task": "Prepare release",
                        "owner": "Alex",
                        "deadline": "Friday",
                    }
                ],
            }
            db.save_summary("m1", summary)
            db.close()

            reopened = Database(path)
            self.assertIsNone(duplicate)
            self.assertEqual(reopened.get_meeting("m1")["title"], "Planning")
            self.assertEqual(reopened.get_meeting("m1")["status"], "interrupted")
            self.assertEqual(
                reopened.get_segments("m1")[0]["text"], "We will ship next week."
            )
            self.assertEqual(reopened.get_speakers("m1")[0]["name"], "Alex")
            self.assertEqual(reopened.get_summary("m1")["executive_summary"], "Plan agreed.")
            self.assertEqual(
                reopened.fetch_one(
                    "SELECT owner,deadline FROM action_items WHERE meeting_id=?",
                    ("m1",),
                ),
                {"owner": "Alex", "deadline": "Friday"},
            )
            duplicate_ids = {
                **summary,
                "action_items": [
                    {"id": "shared-id", "task": "First task"},
                    {"id": "shared-id", "task": "Second task"},
                ],
            }
            reopened.save_summary("m1", duplicate_ids)
            self.assertEqual(
                reopened.fetch_one(
                    "SELECT COUNT(*) AS count FROM action_items WHERE meeting_id=?",
                    ("m1",),
                )["count"],
                2,
            )
            with self.assertRaises(TypeError):
                reopened.save_summary("m1", {"invalid": object()})
            self.assertEqual(
                reopened.fetch_one(
                    "SELECT COUNT(*) AS count FROM action_items WHERE meeting_id=?",
                    ("m1",),
                )["count"],
                2,
            )
            reopened.close()


class DiarizationTests(unittest.TestCase):
    def test_turn_overlap_gets_anonymous_stable_speaker(self):
        segments = [
            {"start": 0.2, "end": 1.2},
            {"start": 2.2, "end": 3.0},
            {"start": 5.0, "end": 6.0},
        ]
        turns = [
            {"start": 0, "end": 1.5, "speaker_id": "SPEAKER_00"},
            {"start": 2, "end": 3.5, "speaker_id": "SPEAKER_01"},
        ]

        SpeakerDiarizer.assign_speakers(segments, turns)
        self.assertEqual(
            [item["speaker_id"] for item in segments],
            ["Speaker 1", "Speaker 2", "Unknown"],
        )


class LocalOllamaTests(unittest.TestCase):
    def test_summary_request_is_local_and_parsed(self):
        content = json.dumps(
            {
                "executive_summary": "Discussed next steps.",
                "action_items": [],
                "decisions": [],
            }
        )
        requests = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps({"message": {"content": content}}).encode()

        def opener(request, timeout):
            requests.append((request, timeout))
            return Response()

        result = MeetingIntelligence(opener=opener).summarize(
            "[00:01] We should review the draft."
        )
        self.assertEqual(result["executive_summary"], "Discussed next steps.")
        self.assertTrue(requests[0][0].full_url.startswith("http://127.0.0.1"))
        prompt = json.loads(requests[0][0].data)["messages"][1]["content"]
        self.assertIn("review the draft", prompt)
        self.assertIn("never invent", json.loads(requests[0][0].data)["messages"][0]["content"].lower())


class ExportTests(unittest.TestCase):
    def test_all_export_formats_are_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter = ExportManager(directory)
            meeting = {
                "id": "test-meeting",
                "title": "Café planning",
                "started_at": "2026-01-01",
                "duration": 12,
            }
            transcript = [
                {
                    "start": 1.25,
                    "end": 2.5,
                    "text": "Résumé ready.",
                    "speaker_name": "Alex",
                    "source": "microphone",
                }
            ]
            summary = {
                "executive_summary": "Résumé reviewed.",
                "key_points": ["Next steps"],
                "decisions": ["Proceed"],
                "action_items": [{"task": "Send notes", "owner": "Alex"}],
            }
            for file_format in ("txt", "json", "srt", "docx", "pdf"):
                with self.subTest(file_format=file_format):
                    path = exporter.export(
                        meeting, transcript, summary, file_format
                    )
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)
            self.assertIn("00:00:01,250", (Path(directory) / "test-meeting.srt").read_text())
            exported = json.loads(
                (Path(directory) / "test-meeting.json").read_text()
            )
            self.assertEqual(exported["transcript"][0]["text"], "Résumé ready.")


class MeetingLifecycleTests(unittest.TestCase):
    def test_start_pause_resume_stop_drains_audio_and_persists(self):
        class FakeCapture:
            def __init__(self, **_kwargs):
                import queue

                self.microphone_queue = queue.Queue()
                self.system_audio_queue = queue.Queue()
                self.paused = False

            def start(self):
                self.microphone_queue.put(
                    AudioChunk(
                        "microphone", 0, 16_000, np.ones(160, dtype=np.float32)
                    )
                )
                return {}

            def pause(self):
                self.paused = True

            def resume(self):
                self.paused = False

            def stop(self):
                return {}

        class FakeTranscriber:
            def transcribe_chunk(self, chunk):
                return {
                    "language": "en",
                    "segments": [
                        {
                            "start": chunk.start,
                            "end": chunk.start + 0.01,
                            "text": "Test audio.",
                        }
                    ],
                }

        class FakeIntelligence:
            def summarize(self, transcript):
                return {"executive_summary": transcript, "action_items": []}

        config = Settings(
            audio_sample_rate=16_000,
            audio_chunk_duration=0.01,
            audio_queue_size=8,
            transcription_chunk_duration=0.01,
            transcription_overlap_duration=0,
            audio_recordings_enabled=False,
        )
        database = Database(":memory:", config=config)
        manager = MeetingManager(
            config=config,
            database=database,
            capture_factory=FakeCapture,
            transcriber=FakeTranscriber(),
            intelligence=FakeIntelligence(),
        )

        started = manager.start_meeting("Test")
        time.sleep(0.03)
        self.assertEqual(started["status"], "recording")
        self.assertEqual(manager.pause_meeting()["status"], "paused")
        self.assertEqual(manager.resume_meeting()["status"], "recording")
        stopped = manager.stop_meeting()

        self.assertEqual(stopped["status"], "completed")
        self.assertTrue(manager.get_transcript(started["meeting_id"]))
        deadline = time.monotonic() + 1
        while (
            manager.get_summary(started["meeting_id"]) is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertIsNotNone(manager.get_summary(started["meeting_id"]))
        database.close()


class CapturePauseTimestampTests(unittest.TestCase):
    def test_audio_chunk_timestamp_points_to_chunk_start(self):
        from audio_capture import AudioCapture

        capture = AudioCapture(
            config=Settings(
                audio_sample_rate=16_000,
                audio_capture_microphone=False,
                audio_capture_system=False,
                audio_recordings_enabled=False,
            )
        )
        capture._start_time = 10
        with patch("audio_capture.time.monotonic", return_value=12):
            capture._publish_chunk(
                "microphone", np.ones((1_600, 1), dtype=np.float32)
            )
        chunk = capture.get_chunk("microphone", timeout=1)
        self.assertAlmostEqual(chunk.timestamp, 1.9)

    def test_capture_timestamps_exclude_paused_time(self):
        import tempfile
        from types import SimpleNamespace

        from audio_capture import AudioCapture

        streams = []

        def input_stream(**kwargs):
            stream = SimpleNamespace(
                callback=kwargs["callback"],
                start=lambda: None,
                stop=lambda: None,
                close=lambda: None,
            )
            streams.append(stream)
            return stream

        capture = AudioCapture(
            config=Settings(
                audio_sample_rate=16_000,
                audio_capture_system=False,
                audio_recordings_enabled=False,
            ),
            sounddevice_module=SimpleNamespace(InputStream=input_stream),
        )
        with tempfile.TemporaryDirectory():
            capture.start()
            samples = np.ones((160, 1), dtype=np.float32)
            streams[0].callback(samples, 160, None, None)
            before_pause = capture.get_chunk("microphone", timeout=1)
            capture.pause()
            time.sleep(0.2)
            streams[0].callback(samples, 160, None, None)
            capture.resume()
            time.sleep(0.01)
            streams[0].callback(samples, 160, None, None)
            after_pause = capture.get_chunk("microphone", timeout=1)
            capture.stop()

        self.assertLess(after_pause.timestamp - before_pause.timestamp, 0.1)


class WebSocketTests(unittest.TestCase):
    def test_dashboard_websocket_receives_status(self):
        from fastapi.testclient import TestClient

        import main

        old_manager = main.manager
        test_manager = MeetingManager(database=Database(":memory:"))
        main.manager = test_manager
        try:
            with TestClient(main.app) as client:
                self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
                denied = client.post(
                    "/api/meeting/pause", headers={"Origin": "https://example.com"}
                )
                self.assertEqual(denied.status_code, 403)
                self.assertEqual(
                    client.post("/api/meeting/pause").status_code, 403
                )
                self.assertEqual(
                    client.post(
                        "/api/meeting/pause",
                        headers={"Origin": "http://localhost:8501"},
                    ).status_code,
                    409,
                )
                with client.websocket_connect("/ws/transcription") as websocket:
                    message = websocket.receive_json()
                    self.assertEqual(message["type"], "status")
                    self.assertEqual(message["meeting"]["status"], "idle")
                    test_manager._publish(
                        {"type": "transcript", "segment": {"text": "hello"}}
                    )
                    self.assertEqual(
                        websocket.receive_json()["segment"]["text"], "hello"
                    )
        finally:
            main.manager = old_manager


if __name__ == "__main__":
    unittest.main()
