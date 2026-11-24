# Local AI Meeting Assistant

A Windows-first, locally running meeting recorder and notes application. It
captures microphone and desktop playback independently, transcribes audio
locally with Faster-Whisper, stores meeting history in SQLite, and offers a
Streamlit dashboard, FastAPI/WebSocket API, Ollama summaries, optional
pyannote diarization, and local exports.

No transcript or audio is sent to a hosted transcription service. Audio will
include whatever is played through the selected Windows output device; system
loopback does not isolate a meeting application from other desktop audio.

## Features

- Independent, timestamped microphone and Windows WASAPI loopback capture;
  separate float32 queues and optional WAV recordings.
- Bounded queues, selectable audio devices, pause/resume, separate capture
  workers, processing delay/drop logging, and graceful recording finalization.
- Mono 16 kHz conversion, overlapping 5–10 second windows, silence gating,
  timestamp offsets, and chunk-based (not true streaming) Faster-Whisper
  inference. CPU int8 is the fallback; CUDA/float16 is attempted when available.
- SQLite persistence for meetings, transcript segments, speakers, summaries,
  decisions, and action items.
- Local Ollama summarization with transcript sectioning and conservative
  instructions not to invent owners, deadlines, or decisions.
- Optional post-meeting pyannote diarization with anonymous speaker labels.
- FastAPI REST API and `/ws/transcription` WebSocket; Streamlit dashboard
  controls meetings, shows live transcript updates, edits notes/segments/speaker
  names, and downloads TXT, JSON, SRT, DOCX, and PDF files.

## Windows installation

Use 64-bit Python 3.11 for the best compatibility with the audio and ML wheels.
Python 3.10 is also supported by the project dependencies.

In PowerShell, from the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

In Windows **Settings > Privacy & security > Microphone**, permit desktop
applications to use the microphone. Connect the headset/speakers that will
play the meeting before opening the dashboard. SoundCard uses WASAPI loopback
on Windows; the output device or driver must expose loopback capture.

Settings can be changed in `.env` or process environment variables. Process
environment values take precedence. Common settings include:

| Variable | Default | Description |
|---|---:|---|
| `AUDIO_SAMPLE_RATE` | `48000` | Device capture sample rate |
| `AUDIO_CHUNK_DURATION` | `1.0` | System capture read size in seconds |
| `AUDIO_QUEUE_SIZE` | `128` | Bounded audio queue capacity |
| `TRANSCRIPTION_CHUNK_DURATION` | `8` | Whisper window size in seconds |
| `TRANSCRIPTION_OVERLAP_DURATION` | `1` | Overlap between windows in seconds |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `COMPUTE_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `TRANSCRIPTION_LANGUAGE` | unset | Language code, or unset for auto-detection |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Local Ollama server |
| `OLLAMA_MODEL` | `llama3.2` | Locally installed Ollama model |
| `DATABASE_PATH` | `meeting_data.db` | Local SQLite file |
| `RECORDINGS_DIR` | `recordings` | Separate microphone/system WAV directory |
| `EXPORTS_DIR` | `exports` | Export output directory |
| `HF_TOKEN` | unset | Optional Hugging Face token for diarization |

`AudioCapture.list_audio_devices()` lists microphone and output devices. The
dashboard lets you choose devices and configure model, execution device, and
language before starting a meeting.

## Faster-Whisper and hardware

The selected Faster-Whisper model is downloaded by CTranslate2 on first use and
then cached locally. Model size affects download size, memory use, and
transcription latency. Start with `small`; use `tiny` or `base` on CPU-only
machines with limited memory. The application does not require NVIDIA hardware.

For NVIDIA GPU inference, install a compatible NVIDIA driver and CUDA/cuDNN
runtime supported by the installed CTranslate2 build. Set `COMPUTE_DEVICE=cuda`
to require GPU initialization; if loading fails, the app logs the issue and
falls back to CPU int8. Check the Faster-Whisper/CTranslate2 installation
guidance for the exact driver/runtime combination for your machine.

## Ollama meeting notes

Install Ollama for Windows, then in PowerShell:

```powershell
ollama pull llama3.2
ollama serve
```

If Ollama is already running as a Windows service, the `ollama serve` command
may report that its port is in use; the service is sufficient. Choose another
locally installed model by setting `OLLAMA_MODEL`. Summary generation is
scheduled in a background worker after capture ends and can be requested again
from the dashboard or API. This prevents local LLM inference from blocking
audio capture or meeting finalization. If Ollama is
unavailable, capture and transcription continue; summary generation reports an
error and can be retried later.

## Optional speaker diarization

pyannote diarization runs after recording, independently of live transcription.
Create a Hugging Face access token with permission to download the model, visit
the `pyannote/speaker-diarization-3.1` model page, accept its access conditions,
then set `HF_TOKEN` in the ignored `.env` file. The model may require additional
manual access approval. The API endpoint
`POST /api/meetings/{meeting_id}/diarize` processes available recordings and
assigns anonymous labels. It does not identify real-world people. Diarization
is an optional, model-dependent feature; speakers can also be renamed manually.

## Run the application

Open two PowerShell windows, activate the same virtual environment in both,
and run from the project directory.

Backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Dashboard:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard.py
```

Open the Streamlit URL printed in the second window (normally
`http://localhost:8501`). Keep the FastAPI backend running at
`http://127.0.0.1:8000`. Both processes run locally; do not bind the backend to
a public interface unless you separately secure it.

Select microphone and output devices, select the Whisper model/device/language,
enter a meeting title, and start capture before joining Teams, Zoom, or Meet.
System audio comes from the chosen desktop playback output, while microphone
audio remains a separate stream and WAV. Pause/resume or stop from the
dashboard. Completed meetings appear under Meeting history, where transcript
segments and notes can be edited, speakers renamed, summaries regenerated, and
exports downloaded.

## API

The backend exposes `/docs` for interactive API documentation. Principal
endpoints:

- `POST /api/meeting/start`, `/pause`, `/resume`, `/stop`
- `GET /api/meeting/status`, `/devices`, `/models`
- `GET /api/meetings`, `/api/meetings/{id}`
- `GET /api/meetings/{id}/transcript`, `/summary`
- `POST /api/meetings/{id}/summarize`, `/transcribe`, `/diarize`
- `PUT /api/settings` to choose model, device, and transcription language
- `PUT /api/meetings/{id}/transcript/{segment_id}`
- `PUT /api/meetings/{id}/speakers/{speaker_id}`
- `GET /api/meetings/{id}/export/{txt|json|srt|docx|pdf}`
- WebSocket: `ws://127.0.0.1:8000/ws/transcription`

The dashboard uses the REST API and WebSocket. The API is intended for local
trusted use and currently does not implement authentication. Mutating HTTP
requests require a localhost `Origin` or `Referer` to prevent cross-site browser
forms from controlling a recording; non-browser API clients should send, for
example, `Origin: http://127.0.0.1`.

## Tests

The tests mock audio capture, Whisper inference, and Ollama so they do not
require an active meeting, a GPU, a Hugging Face token, or a running Ollama
server. Run:

```powershell
python -m unittest discover -p "test_*.py" -v
```

They cover device listing, simultaneous independent capture and WAV creation,
resampling/overlap/flush, transcription timestamps, transcript deduplication
and editing, SQLite persistence, anonymous speaker assignment, meeting
lifecycle, local summary requests, WebSocket status delivery, and all export
formats. A Windows machine with actual input/output hardware is still needed
to validate device drivers, permissions, and real WASAPI behavior.

## Troubleshooting

- **No devices / capture fails:** check Windows microphone permissions, select
  the active input/output device, and restart the backend after connecting it.
- **System audio is silent:** choose the output device actually used by Teams;
  check whether the device driver exposes WASAPI loopback.
- **Whisper is slow or out of memory:** select `tiny`/`base`, use CPU int8, or
  close other applications. Larger models need more memory.
- **Ollama summaries fail:** confirm the local service is running and that
  `ollama list` contains the configured model.
- **Diarization authorization error:** accept the model terms and set `HF_TOKEN`
  locally. Diarization is not required for recording or transcription.
- **Export dependency unavailable:** install the pinned packages from
  `requirements.txt` and restart the backend.

## Known limitations

Audio chunking provides near-real-time results, not word-level streaming.
Transcript accuracy, language support, diarization quality, and latency depend
on the selected local models and device performance. WASAPI captures the
selected output mix rather than a Teams-specific virtual channel. Model
downloads, NVIDIA driver/CUDA compatibility, Hugging Face model access, and
audio hardware must be configured on the target Windows computer.
