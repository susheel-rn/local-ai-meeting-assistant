# Local AI Meeting Assistant

A locally running meeting assistant for capturing meeting audio and, in later
implementation stages, transcribing and organizing it. The initial
implementation provides centralized settings and independent microphone and
desktop-audio capture. No audio is sent to a cloud service.

## Stage 1: configuration and audio capture

`audio_capture.py` captures microphone input through `sounddevice` and desktop
output through SoundCard's Windows WASAPI loopback support. Each source has its
own bounded queue and timestamped float32 audio chunks. Optional recordings are
written asynchronously as separate 16-bit PCM WAV files under `recordings/`.
The Python audio-processing pipeline receives independent microphone and
system-audio streams; the system loopback stream includes audio played by
desktop applications such as Teams, Zoom, and Google Meet.

### Windows setup

Use 64-bit Python 3.10 or 3.11 and run PowerShell from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

In Windows **Settings > Privacy & security > Microphone**, allow desktop
applications to access the microphone. Connect the headset or speakers that
will play meeting audio before starting capture. The default Windows playback
device is selected unless a specific output is supplied to `AudioCapture`.
On Windows, SoundCard reads the selected output through WASAPI loopback; a
device or driver that does not expose loopback cannot provide system audio.
Capturing output audio does not capture an application's audio in isolation.

Settings can be changed in `.env` or as process environment variables.
Environment variables take precedence. Audio sample rate defaults to 48 kHz,
and recording files default to the project-local `recordings/` directory.
Choose `AUDIO_SAMPLE_RATE=16000` if a lower capture rate is preferable.

### Capture API

```python
from audio_capture import AudioCapture

capture = AudioCapture()
capture.start()
try:
    microphone_chunk = capture.get_chunk("microphone", timeout=2)
    system_chunk = capture.get_chunk("system", timeout=2)
finally:
    paths = capture.stop()
    print(paths)
```

`AudioCapture.list_audio_devices()` returns available microphone inputs and
speaker outputs. Pass a microphone device index or name and/or a system output
name when constructing `AudioCapture`. The system audio device is selected by
its output name (or reported ID). Call `pause()` and `resume()` to suspend and
resume publishing chunks; both capture workers continue to avoid accumulating
stale device audio. If one device fails, the other source remains usable. A
recording error or bounded-queue overflow is logged and does not stop capture.

Tests use mocked audio devices and do not require a live microphone, WASAPI
loopback device, or active meeting:

```powershell
python -m unittest discover -p "test_*.py"
```

## Planned implementation stages

This repository is being built incrementally. Faster-Whisper transcription,
audio chunk processing, diarization, meeting management, the FastAPI backend,
Streamlit dashboard, Ollama summaries, SQLite history, exports, and full
integration tests are not included yet.
