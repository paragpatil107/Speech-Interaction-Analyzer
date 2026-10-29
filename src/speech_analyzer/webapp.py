from __future__ import annotations

import json
import os
import re
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.serving import make_server

from .analyzer import run_analysis

APP_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = APP_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
MAX_UPLOAD_BYTES = 180 * 1024 * 1024

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
SERVER = None

LOCK = threading.RLock()
STATE: dict[str, Any] = {
    "recording": False,
    "analyzing": False,
    "phase": "idle",
    "progress": 0,
    "message": "Your workspace is ready.",
    "error": None,
    "session_id": None,
    "started_at": None,
    "stream": None,
    "chunks": [],
    "chunk_lock": threading.Lock(),
    "settings": {},
    "token": None,
    "recording_duration": 0.0,
    "last_completed_session": None,
}

DEFAULT_SETTINGS = {
    "model": "base",
    "device": "cpu",
    "compute_type": "int8",
    "min_overlap": 0.15,
    "sample_rate": 16000,
}


def _new_session_id() -> str:
    numbers = []
    for path in OUTPUTS.glob("session-*"):
        match = re.fullmatch(r"session-(\d+)", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return f"session-{max(numbers, default=0) + 1:03d}"


def _clean_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    model = raw.get("model", DEFAULT_SETTINGS["model"])
    device = raw.get("device", DEFAULT_SETTINGS["device"])
    compute = raw.get("compute_type", DEFAULT_SETTINGS["compute_type"])
    try:
        min_overlap = float(raw.get("min_overlap", DEFAULT_SETTINGS["min_overlap"]))
    except (TypeError, ValueError):
        min_overlap = DEFAULT_SETTINGS["min_overlap"]
    try:
        sample_rate = int(raw.get("sample_rate", DEFAULT_SETTINGS["sample_rate"]))
    except (TypeError, ValueError):
        sample_rate = DEFAULT_SETTINGS["sample_rate"]

    if model not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise ValueError("Choose a supported transcription model.")
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose CPU or CUDA as the inference device.")
    if compute not in {"int8", "float16", "float32", "int8_float16"}:
        raise ValueError("Choose a supported compute type.")
    if min_overlap < 0 or min_overlap > 10:
        raise ValueError("Minimum overlap must be between 0 and 10 seconds.")
    if sample_rate not in {16000, 22050, 44100, 48000}:
        raise ValueError("Choose a supported microphone sample rate.")
    if device == "cpu" and compute in {"float16", "int8_float16"}:
        raise ValueError("For CPU inference, use int8 or float32 compute type.")
    return {"model": model, "device": device, "compute_type": compute,
            "min_overlap": min_overlap, "sample_rate": sample_rate}


def _get_token(raw_token: str | None) -> str | None:
    value = (raw_token or os.getenv("HF_TOKEN") or "").strip()
    return value or None


def _notify_progress(step: int, message: str) -> None:
    with LOCK:
        STATE["progress"] = min(94, int(round((step / 4) * 94)))
        STATE["phase"] = "analyzing"
        STATE["message"] = message
        STATE["error"] = None


def _analysis_worker(session_id: str, audio_path: Path, settings: dict[str, Any],
                     token: str, reference_path: Path | None = None) -> None:
    try:
        with LOCK:
            STATE["analyzing"] = True
            STATE["recording"] = False
            STATE["session_id"] = session_id
            STATE["phase"] = "analyzing"
            STATE["progress"] = 0
            STATE["message"] = "Preparing your recording for analysis…"
            STATE["error"] = None
        report = run_analysis(
            str(audio_path), str(audio_path.parent), settings["model"], settings["device"],
            settings["compute_type"], token, str(reference_path) if reference_path else None,
            settings["min_overlap"], progress_callback=_notify_progress,
        )
        with LOCK:
            STATE["progress"] = 100
            STATE["phase"] = "complete"
            STATE["message"] = "Analysis complete. Your results are ready."
            STATE["error"] = None
            STATE["last_completed_session"] = session_id
    except Exception as exc:
        # Keep a useful error for the local UI; credentials are never included in error strings.
        message = str(exc).replace(token, "[hidden token]") if token else str(exc)
        with LOCK:
            STATE["phase"] = "error"
            STATE["message"] = "Analysis could not be completed."
            STATE["error"] = message
            STATE["progress"] = 0
    finally:
        with LOCK:
            STATE["analyzing"] = False
            STATE["token"] = None
            STATE["started_at"] = None
            STATE["stream"] = None
            STATE["chunks"] = []
            STATE["recording_duration"] = 0.0


def _session_summary(path: Path) -> dict[str, Any]:
    report_path = path / "report.json"
    report: dict[str, Any] | None = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            report = None
    with LOCK:
        active = STATE.get("session_id") == path.name and (STATE.get("recording") or STATE.get("analyzing"))
        failed = STATE.get("session_id") == path.name and STATE.get("phase") == "error"
    duration = None
    try:
        if report:
            duration = report.get("audio", {}).get("duration_seconds")
        elif (path / "recording.wav").exists():
            info = sf.info(str(path / "recording.wav"))
            duration = info.frames / info.samplerate if info.samplerate else None
    except Exception:
        duration = None
    if active:
        status = "recording" if STATE.get("recording") else "analyzing"
    elif failed:
        status = "error"
    elif report:
        status = "complete"
    elif (path / "recording.wav").exists():
        status = "ready"
    else:
        status = "empty"
    created = datetime.fromtimestamp(path.stat().st_mtime).strftime("%d %b · %H:%M") if path.exists() else ""
    return {
        "id": path.name,
        "status": status,
        "created": created,
        "duration_seconds": duration,
        "speaker_count": report.get("speaker_count") if report else None,
        "overlap_seconds": report.get("overlap", {}).get("total_seconds") if report else None,
        "overlap_ratio": report.get("overlap", {}).get("recording_ratio") if report else None,
        "pause_count": report.get("pauses", {}).get("count") if report else None,
        "input_file": report.get("input_file") if report else "recording.wav",
        "has_timeline": (path / "timeline.png").exists(),
        "has_report": report_path.exists(),
    }


def _list_sessions() -> list[dict[str, Any]]:
    if not OUTPUTS.exists():
        return []
    paths = [p for p in OUTPUTS.iterdir() if p.is_dir() and p.name.startswith("session-")]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [_session_summary(p) for p in paths[:40]]


def _json_error(message: str, status: int = 400):
    return jsonify({"ok": False, "error": message}), status


def _busy() -> bool:
    return bool(STATE["recording"] or STATE["analyzing"])


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    with LOCK:
        started = STATE.get("started_at")
        elapsed = max(0.0, time.monotonic() - started) if started and STATE.get("recording") else 0.0
        snapshot = {
            "recording": STATE["recording"], "analyzing": STATE["analyzing"],
            "phase": STATE["phase"], "progress": STATE["progress"],
            "message": STATE["message"], "error": STATE["error"],
            "session_id": STATE["session_id"], "elapsed_seconds": round(elapsed, 1),
            "has_env_token": bool(os.getenv("HF_TOKEN")),
            "active_session": STATE.get("session_id"),
        }
    snapshot["sessions"] = _list_sessions()
    return jsonify(snapshot)


@app.get("/api/sessions/<session_id>")
def api_session(session_id: str):
    if not re.fullmatch(r"session-\d{3,}", session_id):
        return _json_error("Invalid session ID.", 404)
    folder = OUTPUTS / session_id
    if not folder.is_dir():
        return _json_error("Session not found.", 404)
    result = _session_summary(folder)
    report = None
    report_path = folder / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _json_error("The session report could not be read.", 500)
    result["report"] = report
    return jsonify(result)


@app.get("/api/sessions/<session_id>/files/<filename>")
def session_file(session_id: str, filename: str):
    if not re.fullmatch(r"session-\d{3,}", session_id):
        return _json_error("Invalid session ID.", 404)
    if filename not in {"timeline.png", "speaker_turns.csv", "summary.txt", "report.json", "recording.wav"}:
        return _json_error("File type is not available.", 404)
    folder = OUTPUTS / session_id
    file_path = folder / filename
    if not file_path.is_file():
        return _json_error("File not found.", 404)
    return send_from_directory(folder, filename, as_attachment=filename in {"speaker_turns.csv", "summary.txt", "report.json", "recording.wav"})


@app.post("/api/record/start")
def api_record_start():
    try:
        data = request.get_json(silent=True) or {}
        # Recording does not require model access. The token is requested only
        # when the user explicitly presses Analyze.
        settings = _clean_settings(data.get("settings"))
        with LOCK:
            if _busy():
                return _json_error("A recording or analysis is already in progress.", 409)
        try:
            import sounddevice as sd
        except ImportError as exc:
            return _json_error("Microphone recording needs sounddevice. Reinstall the project dependencies with: python -m pip install -e .", 500)

        session_id = _new_session_id()
        folder = OUTPUTS / session_id
        folder.mkdir(parents=True, exist_ok=False)
        chunks: list[np.ndarray] = []
        chunk_lock = threading.Lock()
        def callback(indata, frames, time_info, status):
            if status:
                # Device warnings are not credentials or user-provided content.
                with LOCK:
                    STATE["message"] = f"Recording · audio device warning: {status}"
            with chunk_lock:
                chunks.append(indata.copy())
        stream = sd.InputStream(samplerate=settings["sample_rate"], channels=1, dtype="float32", callback=callback, blocksize=0)
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        with LOCK:
            STATE.update({
                "recording": True, "analyzing": False, "phase": "recording", "progress": 0,
                "message": "Recording in progress. Speak naturally, or play another voice nearby.",
                "error": None, "session_id": session_id, "started_at": time.monotonic(),
                "stream": stream, "chunks": chunks, "chunk_lock": chunk_lock,
                "settings": settings, "token": None, "recording_duration": 0.0,
                "last_completed_session": None,
            })
        return jsonify({"ok": True, "session_id": session_id, "message": "Recording started."})
    except Exception as exc:
        return _json_error(f"Could not start microphone recording: {exc}. Check Windows microphone permissions and the default input device.", 500)


@app.post("/api/record/stop")
def api_record_stop():
    with LOCK:
        if not STATE["recording"]:
            return _json_error("There is no active recording to stop.", 409)
        session_id = STATE["session_id"]
        stream = STATE["stream"]
        chunks = STATE["chunks"]
        chunk_lock = STATE["chunk_lock"]
        settings = dict(STATE["settings"])
        STATE["recording"] = False
        STATE["phase"] = "saving"
        STATE["message"] = "Stopping the microphone and saving the recording…"
    try:
        stream.stop()
        stream.close()
        with chunk_lock:
            audio_chunks = list(chunks)
        if not audio_chunks:
            with LOCK:
                STATE["phase"] = "error"
                STATE["error"] = "No audio was captured. Check your microphone and try again."
                STATE["message"] = "Recording contained no audio."
                STATE["token"] = None
            return _json_error("No audio was captured. Check your microphone and try again.", 400)
        audio = np.concatenate(audio_chunks, axis=0)
        rate = settings["sample_rate"]
        duration = len(audio) / rate
        if duration < 1.0:
            with LOCK:
                STATE["phase"] = "error"
                STATE["error"] = "Recording was shorter than one second. Record for longer and try again."
                STATE["message"] = "Recording was too short."
                STATE["token"] = None
            return _json_error("Recording was shorter than one second. Record for longer and try again.", 400)
        audio_path = OUTPUTS / session_id / "recording.wav"
        sf.write(str(audio_path), audio, rate)
        with LOCK:
            STATE["recording_duration"] = round(duration, 2)
            STATE["recording"] = False
            STATE["analyzing"] = False
            STATE["phase"] = "ready"
            STATE["progress"] = 0
            STATE["message"] = "Recording saved. Click Analyze when you are ready."
            STATE["error"] = None
            STATE["token"] = None
            STATE["started_at"] = None
            STATE["stream"] = None
            STATE["chunks"] = []
        return jsonify({"ok": True, "session_id": session_id, "duration_seconds": round(duration, 2), "message": "Recording saved. Click Analyze when you are ready."})
    except Exception as exc:
        with LOCK:
            STATE["recording"] = False
            STATE["analyzing"] = False
            STATE["phase"] = "error"
            STATE["error"] = str(exc)
            STATE["message"] = "The recording could not be saved."
            STATE["token"] = None
        return _json_error(f"Could not save recording: {exc}", 500)


@app.post("/api/analyze/upload")
def api_analyze_upload():
    token = _get_token(request.form.get("hf_token"))
    if not token:
        return _json_error("Enter your Hugging Face read token to analyze audio. It is held in memory only and is not saved to disk.", 401)
    try:
        settings = _clean_settings(json.loads(request.form.get("settings", "{}")))
    except (ValueError, json.JSONDecodeError) as exc:
        return _json_error(str(exc), 400)
    file = request.files.get("audio")
    if not file or not file.filename:
        return _json_error("Choose an audio file first.", 400)
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        return _json_error("Supported formats are WAV, FLAC, OGG and AIFF. Convert other formats to WAV first.", 400)
    reference = request.files.get("reference")
    if reference and reference.filename and Path(reference.filename).suffix.lower() != ".txt":
        return _json_error("Reference transcript must be a .txt file.", 400)
    with LOCK:
        if _busy():
            return _json_error("A recording or analysis is already in progress.", 409)
    session_id = _new_session_id()
    folder = OUTPUTS / session_id
    folder.mkdir(parents=True, exist_ok=False)
    audio_path = folder / "recording.wav"
    try:
        # Normalize supported source formats to mono/stereo WAV at their original sample rate.
        samples, sample_rate = sf.read(file.stream, dtype="float32", always_2d=True)
        if not len(samples):
            raise ValueError("The selected audio file is empty.")
        sf.write(str(audio_path), samples, sample_rate)
        reference_path = None
        if reference and reference.filename:
            reference_path = folder / "reference.txt"
            reference_path.write_bytes(reference.read())
    except Exception as exc:
        return _json_error(f"Could not read this audio file: {exc}", 400)
    with LOCK:
        STATE["session_id"] = session_id
        STATE["settings"] = settings
        STATE["analyzing"] = True
        STATE["phase"] = "analyzing"
        STATE["progress"] = 0
        STATE["message"] = "Audio imported. Preparing analysis…"
        STATE["error"] = None
    threading.Thread(target=_analysis_worker, args=(session_id, audio_path, settings, token, reference_path), daemon=True).start()
    return jsonify({"ok": True, "session_id": session_id, "message": "Audio imported. Analysis has started."})


@app.post("/api/sessions/<session_id>/analyze")
def api_analyze_session(session_id: str):
    if not re.fullmatch(r"session-\d{3,}", session_id):
        return _json_error("Invalid session ID.", 404)
    folder = OUTPUTS / session_id
    audio_path = folder / "recording.wav"
    if not audio_path.is_file():
        return _json_error("This session has no recording.wav file to analyze.", 404)
    data = request.get_json(silent=True) or {}
    token = _get_token(data.get("hf_token"))
    if not token:
        return _json_error("Enter your Hugging Face read token to run analysis. It is held in memory only.", 401)
    try:
        settings = _clean_settings(data.get("settings"))
    except ValueError as exc:
        return _json_error(str(exc), 400)
    reference_path = folder / "reference.txt"
    with LOCK:
        if _busy():
            return _json_error("A recording or analysis is already in progress.", 409)
        STATE["session_id"] = session_id
        STATE["settings"] = settings
        STATE["analyzing"] = True
        STATE["phase"] = "analyzing"
        STATE["progress"] = 0
        STATE["message"] = "Preparing analysis…"
        STATE["error"] = None
    threading.Thread(target=_analysis_worker, args=(session_id, audio_path, settings, token,
                        reference_path if reference_path.exists() else None), daemon=True).start()
    return jsonify({"ok": True, "session_id": session_id, "message": "Analysis has started."})


@app.post("/api/shutdown")
def api_shutdown():
    """Stop the local server from the UI, but never interrupt active work."""
    with LOCK:
        if STATE["recording"] or STATE["analyzing"]:
            return _json_error("Stop the recording and wait for analysis to finish before closing the app.", 409)
    server = SERVER
    if server is None:
        return _json_error("The local server is not ready to shut down.", 503)

    def stop_server():
        # Let Flask finish sending the response before ending the server loop.
        time.sleep(0.35)
        server.shutdown()

    threading.Thread(target=stop_server, daemon=True).start()
    return jsonify({"ok": True, "message": "Speech Interaction Analyzer is closing."})


@app.errorhandler(413)
def too_large(_error):
    return _json_error("This file is too large. The maximum upload size is 180 MB.", 413)


def main() -> None:
    """Run the local UI. Closing it from the dashboard ends this process."""
    global SERVER
    port = int(os.getenv("SPEECH_ANALYZER_PORT", "8765"))
    host = "127.0.0.1"
    SERVER = make_server(host, port, app, threaded=True)
    url = f"http://{host}:{port}"
    print("\nSpeech Interaction Analyzer")
    print(f"Opening {url}")
    print("Use the Close app button in the page to stop the local server.\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        SERVER.serve_forever()
    finally:
        SERVER = None
        print("Speech Interaction Analyzer closed.")


if __name__ == "__main__":
    main()
