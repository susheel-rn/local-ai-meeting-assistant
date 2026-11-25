"""Streamlit browser dashboard for the local FastAPI meeting service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st
import websocket

API_URL = os.getenv("MEETING_ASSISTANT_API", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Local AI Meeting Assistant", layout="wide")
st.title("Local AI Meeting Assistant")


def api(path: str, method: str = "GET", data=None, timeout: float = 5):
    request = urllib.request.Request(
        API_URL + path,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers={
            "Content-Type": "application/json",
            "Origin": "http://127.0.0.1:8501",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        st.error(f"Cannot reach the local API at {API_URL}: {exc}")
        return None


def websocket_event():
    socket = st.session_state.get("transcription_socket")
    if socket is None:
        ws_url = API_URL.replace("https://", "wss://").replace(
            "http://", "ws://"
        ) + "/ws/transcription"
        try:
            socket = websocket.create_connection(ws_url, timeout=1)
            socket.settimeout(0.01)
            st.session_state.transcription_socket = socket
        except Exception:
            return None
    try:
        message = socket.recv()
        return json.loads(message) if message else None
    except websocket.WebSocketTimeoutException:
        return None
    except Exception:
        try:
            socket.close()
        except Exception:
            pass
        st.session_state.transcription_socket = None
        return None


status = api("/api/meeting/status")
if status is None:
    st.stop()

active = status.get("status") in {"recording", "paused"}
completed_meetings = sum(
    1
    for meeting in (api("/api/meetings?limit=500") or [])
    if meeting.get("status") == "completed"
)
column1, column2, column3, column4 = st.columns(4)
column1.metric("Application", "Ready")
column2.metric("Meeting", status.get("status", "idle").title())
column3.metric("Duration", f"{int(status.get('duration', 0) // 60):02}:"
               f"{int(status.get('duration', 0) % 60):02}")
column4.metric("Completed meetings", completed_meetings)

with st.sidebar:
    st.header("Meeting controls")
    title = st.text_input("Meeting title", "Meeting")
    device_info = api("/api/meeting/devices") or {}
    model_info = api("/api/meeting/models") or {}
    microphones = device_info.get("microphones", [])
    outputs = device_info.get("system_audio_outputs", [])
    mic_names = ["Default"] + [item["name"] for item in microphones]
    output_names = ["Default"] + [item["name"] for item in outputs]
    mic_choice = st.selectbox("Microphone", mic_names)
    output_choice = st.selectbox("System audio output", output_names)
    whisper_model = st.selectbox(
        "Whisper model",
        model_info.get("whisper", ["small"]),
        index=model_info.get("whisper", ["small"]).index(
            model_info.get("selected_whisper", "small")
        )
        if model_info.get("selected_whisper", "small")
        in model_info.get("whisper", ["small"])
        else 0,
        disabled=active,
    )
    compute_device = st.selectbox(
        "Inference device",
        ["auto", "cpu", "cuda"],
        index=["auto", "cpu", "cuda"].index(
            model_info.get("device", "auto")
            if model_info.get("device", "auto") in ["auto", "cpu", "cuda"]
            else "auto"
        ),
        disabled=active,
    )
    language_options = ["auto", "en", "es", "fr", "de", "it", "pt", "zh", "ja"]
    language = st.selectbox(
        "Transcription language",
        language_options,
        index=language_options.index(model_info.get("language") or "auto")
        if (model_info.get("language") or "auto") in language_options
        else 0,
        disabled=active,
    )
    mic_id = next(
        (item["id"] for item in microphones if item["name"] == mic_choice), None
    )
    output_id = next(
        (item["id"] for item in outputs if item["name"] == output_choice), None
    )
    if st.button("Save configuration", disabled=active, use_container_width=True):
        response = api(
            "/api/settings",
            "PUT",
            {
                "whisper_model": whisper_model,
                "compute_device": compute_device,
                "transcription_language": None if language == "auto" else language,
            },
        )
        if response:
            st.success("Configuration saved.")
    if st.button("Start meeting", disabled=active, use_container_width=True):
        result = api(
            "/api/meeting/start",
            "POST",
            {
                "title": title,
                "microphone_device": mic_id,
                "system_audio_device": output_id,
            },
        )
        if result:
            st.rerun()
    left, right = st.columns(2)
    if left.button("Pause", disabled=status.get("status") != "recording"):
        api("/api/meeting/pause", "POST", {})
        st.rerun()
    if right.button("Resume", disabled=status.get("status") != "paused"):
        api("/api/meeting/resume", "POST", {})
        st.rerun()
    if st.button("Stop meeting", disabled=not active, use_container_width=True):
        api("/api/meeting/stop", "POST", {}, timeout=300)
        st.rerun()

tab_live, tab_history = st.tabs(["Current meeting", "Meeting history"])
with tab_live:
    @st.fragment(run_every=2 if active else None)
    def render_live_transcript():
        latest_status = api("/api/meeting/status") or {}
        meeting_id = latest_status.get("meeting_id")
        event = websocket_event() if latest_status.get("status") in {
            "recording",
            "paused",
        } else None
        st.subheader("Live transcript")
        if event and event.get("type") == "transcript":
            st.caption("Live transcript update received")
        if not meeting_id:
            st.info("Start a meeting to display its transcript.")
            return
        segments = api(f"/api/meetings/{meeting_id}/transcript") or []
        if not segments:
            st.info("Transcript segments will appear here as audio is processed.")
        for segment in segments:
            speaker = segment.get("speaker_id") or segment.get("source", "Unknown")
            st.markdown(
                f"**{int(segment['start'] // 60):02}:"
                f"{int(segment['start'] % 60):02} · {speaker} · "
                f"{segment['source']}**  \n{segment['text']}"
            )
            edited = st.text_input(
                "Edit transcript",
                segment["text"],
                key=f"edit-{segment['id']}",
                label_visibility="collapsed",
            )
            if edited != segment["text"] and st.button(
                "Save edit", key=f"save-{segment['id']}"
            ):
                api(
                    f"/api/meetings/{meeting_id}/transcript/{segment['id']}",
                    "PUT",
                    {"text": edited},
                )
                st.rerun()

    render_live_transcript()
    if st.button("Refresh"):
        st.rerun()

with tab_history:
    history = api("/api/meetings") or []
    st.subheader("Previous meetings")
    if not history:
        st.info("Completed meetings will appear here.")
    else:
        meeting_labels = [
            f"{item['title']} · {item['started_at'][:10]} · "
            f"{int(item.get('duration', 0) // 60)} min"
            for item in history
        ]
        selection = st.selectbox(
            "Select a meeting", range(len(history)), format_func=meeting_labels.__getitem__
        )
        meeting = api(f"/api/meetings/{history[selection]['id']}") or history[selection]
        meeting_id = meeting["id"]
        transcript = api(f"/api/meetings/{meeting_id}/transcript") or []
        summary_data = api(f"/api/meetings/{meeting_id}/summary") or {}
        st.write(summary_data.get("executive_summary", "No summary yet."))
        for heading, key in (
            ("Key discussion points", "key_points"),
            ("Decisions", "decisions"),
            ("Action items", "action_items"),
            ("Open questions", "open_questions"),
            ("Risks and blockers", "risks_and_blockers"),
        ):
            items = summary_data.get(key, [])
            if items:
                st.markdown(f"**{heading}**")
                for item in items:
                    st.write(
                        f"- {item.get('task', item.get('text', ''))}"
                        if isinstance(item, dict)
                        else f"- {item}"
                    )
        if st.button("Generate / regenerate summary", key=f"summary-{meeting_id}"):
            summary_data = api(
                f"/api/meetings/{meeting_id}/summarize",
                "POST",
                {},
                timeout=240,
            ) or {}
            st.write(summary_data.get("executive_summary", ""))
        if meeting.get("system_recording") or meeting.get("microphone_recording"):
            if st.button("Run speaker diarization", key=f"diarize-{meeting_id}"):
                result = api(f"/api/meetings/{meeting_id}/diarize", "POST", {})
                if result:
                    st.success("Speaker assignments updated.")
        edited_summary = st.text_area(
            "Edit meeting notes (JSON)",
            json.dumps(summary_data, ensure_ascii=False, indent=2),
            key=f"notes-{meeting_id}",
        )
        if st.button("Save meeting notes", key=f"save-notes-{meeting_id}"):
            try:
                api(
                    f"/api/meetings/{meeting_id}/summary",
                    "PUT",
                    {"summary": json.loads(edited_summary)},
                )
            except json.JSONDecodeError:
                st.error("Notes must be valid JSON.")
        speakers = meeting.get("speakers", [])
        if speakers:
            st.markdown("**Speakers**")
            for speaker in speakers:
                renamed = st.text_input(
                    f"Rename {speaker['id']}",
                    speaker["name"],
                    key=f"speaker-{meeting_id}-{speaker['id']}",
                )
                if renamed != speaker["name"] and st.button(
                    "Save speaker name",
                    key=f"rename-{meeting_id}-{speaker['id']}",
                ):
                    api(
                        f"/api/meetings/{meeting_id}/speakers/{speaker['id']}",
                        "PUT",
                        {"name": renamed},
                    )
        st.markdown("**Transcript**")
        for segment in transcript:
            speaker_names = {item["id"]: item["name"] for item in speakers}
            speaker_label = speaker_names.get(
                segment.get("speaker_id"), segment.get("speaker_id") or "Unknown"
            )
            st.caption(
                f"{int(segment['start'] // 60):02}:"
                f"{int(segment['start'] % 60):02} · {speaker_label} · "
                f"{segment['source']}"
            )
            st.write(segment["text"])
            with st.expander(f"Edit segment {segment['id'][:8]}"):
                revised = st.text_area(
                    "Transcript text",
                    segment["text"],
                    key=f"history-text-{segment['id']}",
                )
                speaker_ids = ["Unknown"] + [item["id"] for item in speakers]
                current_speaker = segment.get("speaker_id") or "Unknown"
                if current_speaker not in speaker_ids:
                    speaker_ids.append(current_speaker)
                revised_speaker = st.selectbox(
                    "Speaker assignment",
                    speaker_ids,
                    index=speaker_ids.index(current_speaker),
                    key=f"history-speaker-{segment['id']}",
                )
                if st.button("Save segment", key=f"history-save-{segment['id']}"):
                    api(
                        f"/api/meetings/{meeting_id}/transcript/{segment['id']}",
                        "PUT",
                        {
                            "text": revised,
                            "speaker_id": None
                            if revised_speaker == "Unknown"
                            else revised_speaker,
                        },
                    )
        formats = {
            "TXT": ("txt", "text/plain"),
            "JSON": ("json", "application/json"),
            "SRT": ("srt", "application/x-subrip"),
            "DOCX": (
                "docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            "PDF": ("pdf", "application/pdf"),
        }
        selected_format = st.selectbox("Export format", list(formats))
        export_format, export_mime = formats[selected_format]
        if st.button("Prepare export", key=f"prepare-export-{meeting_id}"):
            try:
                with urllib.request.urlopen(
                    f"{API_URL}/api/meetings/{meeting_id}/export/{export_format}",
                    timeout=30,
                ) as response:
                    exported = response.read()
                st.session_state.prepared_export = {
                    "meeting_id": meeting_id,
                    "format": export_format,
                    "mime": export_mime,
                    "data": exported,
                }
            except urllib.error.HTTPError:
                st.error(f"{selected_format} export could not be generated.")
        prepared = st.session_state.get("prepared_export", {})
        if (
            prepared.get("meeting_id") == meeting_id
            and prepared.get("format") == export_format
        ):
            st.download_button(
                f"Download {selected_format}",
                data=prepared["data"],
                file_name=f"{meeting_id}.{export_format}",
                mime=export_mime,
                key=f"export-{export_format}-{meeting_id}",
            )
