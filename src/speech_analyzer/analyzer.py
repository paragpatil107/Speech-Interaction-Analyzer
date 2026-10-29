from __future__ import annotations

import gc
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from .metrics import Interval, overlap_intervals, union_duration, word_error_rate


# Keep loaded models alive while the local app is running. Loading model weights
# repeatedly for each session adds avoidable startup time. Only one ASR model is
# cached at a time to avoid retaining several large models in RAM/VRAM.
_MODEL_CACHE_LOCK = threading.RLock()
_DIARIZATION_PIPELINE: Any | None = None
_DIARIZATION_DEVICE: str | None = None
_WHISPER_MODEL: Any | None = None
_WHISPER_MODEL_KEY: tuple[str, str, str] | None = None


def _get_diarization_pipeline(Pipeline: Any, device: str, token: str | None) -> Any:
    global _DIARIZATION_PIPELINE, _DIARIZATION_DEVICE
    device_key = device.lower()
    with _MODEL_CACHE_LOCK:
        if _DIARIZATION_PIPELINE is not None and _DIARIZATION_DEVICE == device_key:
            return _DIARIZATION_PIPELINE

        if not token:
            raise RuntimeError(
                "HF_TOKEN is required to load pyannote speaker diarization. "
                "Enter your Hugging Face read token in the app."
            )

        # Avoid keeping CPU and GPU copies of the same diarization model around.
        _DIARIZATION_PIPELINE = None
        _DIARIZATION_DEVICE = None
        gc.collect()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )
        if pipeline is None:
            raise RuntimeError(
                "Could not load pyannote model. Check Hugging Face access and model terms."
            )
        if device_key == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but PyTorch cannot access a CUDA GPU. Retry with --device cpu."
                )
            pipeline.to(torch.device("cuda"))
        _DIARIZATION_PIPELINE = pipeline
        _DIARIZATION_DEVICE = device_key
        return pipeline


def _get_whisper_model(WhisperModel: Any, model_size: str, device: str, compute_type: str) -> Any:
    global _WHISPER_MODEL, _WHISPER_MODEL_KEY
    key = (model_size, device.lower(), compute_type)
    with _MODEL_CACHE_LOCK:
        if _WHISPER_MODEL is not None and _WHISPER_MODEL_KEY == key:
            return _WHISPER_MODEL

        # Release the prior ASR model before loading a different size/precision.
        _WHISPER_MODEL = None
        _WHISPER_MODEL_KEY = None
        gc.collect()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WHISPER_MODEL = model
        _WHISPER_MODEL_KEY = key
        return model


def _load_audio_duration(path: Path) -> tuple[float, int, int]:
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("The audio file appears to be empty or unreadable.")
    return info.frames / info.samplerate, info.samplerate, info.channels


def _seconds(value: Any) -> float:
    return float(value)


def run_analysis(audio_path: str, output_dir: str, model_size: str = "base", device: str = "cpu",
                 compute_type: str = "int8", hf_token: str | None = None,
                 reference_path: str | None = None, min_overlap: float = 0.15,
                 progress_callback: Callable[[int, str], None] | None = None) -> dict:
    """Run diarization + ASR and save a JSON report, speaker-turn CSV, and timeline PNG.

    The optional callback receives (step_number, user-facing_message) for UI progress.
    """
    source = Path(audio_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    duration, sample_rate, channels = _load_audio_duration(source)

    token = hf_token or os.getenv("HF_TOKEN")

    try:
        from pyannote.audio import Pipeline
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Dependencies are missing. Install the project with: pip install -e .") from exc

    if progress_callback:
        progress_callback(1, "Loading or reusing speaker-diarization model…")
    print("[1/4] Loading or reusing speaker-diarization model…")
    pipeline = _get_diarization_pipeline(Pipeline, device, token)
    if progress_callback:
        progress_callback(2, "Detecting speaker turns and overlap…")
    print("[2/4] Detecting speaker turns and overlap…")
    diarization_result = pipeline(str(source))
    annotation = getattr(diarization_result, "speaker_diarization", diarization_result)
    turns: list[Interval] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(Interval(_seconds(segment.start), _seconds(segment.end), str(speaker)))
    turns.sort(key=lambda t: (t.start, t.end, t.speaker or ""))

    overlaps = overlap_intervals(turns, min_duration=min_overlap)
    overlap_seconds = union_duration((x["start"], x["end"]) for x in overlaps)
    overlap_ratio = overlap_seconds / duration if duration else 0.0
    unique_speakers = sorted({t.speaker for t in turns if t.speaker is not None})

    if progress_callback:
        progress_callback(3, "Transcribing audio…")
    print("[3/4] Transcribing audio…")
    whisper = _get_whisper_model(WhisperModel, model_size, device, compute_type)
    segments_iter, info = whisper.transcribe(str(source), word_timestamps=True, vad_filter=True)
    asr_segments: list[dict] = []
    words: list[dict] = []
    for seg in segments_iter:
        item = {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
        asr_segments.append(item)
        if seg.words:
            for word in seg.words:
                words.append({"start": float(word.start), "end": float(word.end), "word": word.word.strip()})
    transcript = " ".join(x["text"] for x in asr_segments).strip()

    pauses = []
    ordered_asr = sorted(asr_segments, key=lambda x: x["start"])
    for previous, current in zip(ordered_asr, ordered_asr[1:]):
        gap = current["start"] - previous["end"]
        if gap >= 0.4:
            pauses.append({"start": previous["end"], "end": current["start"], "duration": gap})

    # These are deliberately called candidate interruptions: overlap alone cannot prove intent.
    candidates = []
    for ov in overlaps:
        candidates.append({**ov, "classification": "candidate_interruption_or_overlap",
                           "interpretation": "Overlap detected; audio alone does not establish conversational intent."})

    reference_metrics = None
    if reference_path:
        ref_file = Path(reference_path).expanduser().resolve()
        if not ref_file.exists():
            raise FileNotFoundError(f"Reference transcript not found: {ref_file}")
        reference_text = ref_file.read_text(encoding="utf-8").strip()
        reference_metrics = word_error_rate(reference_text, transcript)
        reference_metrics["note"] = "Whole-recording WER. Segment-level overlap-vs-clean WER requires a time-aligned reference transcript; a plain text transcript cannot support that comparison reliably."

    report = {
        "project": "Speech Interaction Analyzer", "version": "0.2.0", "input_file": source.name,
        "audio": {"duration_seconds": round(duration, 3), "sample_rate_hz": sample_rate, "channels": channels},
        "models": {"diarization": "pyannote/speaker-diarization-3.1", "asr": f"faster-whisper/{model_size}",
                   "device": device, "compute_type": compute_type, "detected_language": getattr(info, "language", None),
                   "language_probability": getattr(info, "language_probability", None)},
        "speaker_count": len(unique_speakers), "speakers": unique_speakers,
        "speaker_turns": [{"start": round(t.start, 3), "end": round(t.end, 3), "duration": round(t.duration, 3), "speaker": t.speaker} for t in turns],
        "overlap": {"interval_count": len(overlaps), "total_seconds": round(overlap_seconds, 3),
                    "recording_ratio": round(overlap_ratio, 5), "intervals": overlaps},
        "candidate_interruption_events": candidates,
        "pauses": {"threshold_seconds": 0.4, "count": len(pauses), "intervals": pauses},
        "transcription": {"text": transcript, "segments": asr_segments, "words": words},
        "recognition_evaluation": reference_metrics,
        "limitations": [
            "An overlap is not automatically an intentional interruption; events are reported as candidates.",
            "Speaker labels are anonymous cluster IDs, not real identities.",
            "Voice similarity/uniqueness is not estimated in this first version; diarization labels are not voice-identity scores.",
            "Whole-recording WER is calculated only when a reference transcript is supplied.",
            "To compare WER during overlap versus non-overlap, use a word-level time-aligned reference annotation.",
            "A single recording cannot by itself establish general noise robustness; that requires controlled tests across recordings and conditions."
        ]
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_turns_csv(out / "speaker_turns.csv", report["speaker_turns"])
    if progress_callback:
        progress_callback(4, "Creating timeline plot and report…")
    print("[4/4] Creating timeline plot and report…")
    _plot_timeline(out / "timeline.png", duration, report["speaker_turns"], overlaps, asr_segments)
    _write_summary(out / "summary.txt", report)
    return report


def _write_turns_csv(path: Path, turns: list[dict]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start", "end", "duration", "speaker"])
        writer.writeheader(); writer.writerows(turns)


def _plot_timeline(path: Path, duration: float, turns: list[dict], overlaps: list[dict], segments: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    speakers = sorted({t["speaker"] for t in turns})
    y_for = {speaker: i for i, speaker in enumerate(speakers)}
    for turn in turns:
        y = y_for[turn["speaker"]]
        axes[0].barh(y, turn["end"] - turn["start"], left=turn["start"], height=0.55)
    axes[0].set_yticks(list(y_for.values()), list(y_for.keys()))
    axes[0].set_title("Speaker turns (overlapping bars indicate simultaneous speech)")
    axes[0].set_ylabel("Speaker")
    for ov in overlaps:
        axes[0].axvspan(ov["start"], ov["end"], alpha=0.18)
        axes[1].axvspan(ov["start"], ov["end"], alpha=0.22)
    for seg in segments:
        axes[1].barh(0, max(0.01, seg["end"] - seg["start"]), left=seg["start"], height=0.45)
    axes[1].set_yticks([0], ["ASR segments"])
    axes[1].set_title("Speech recognition segments; shaded regions are detected overlap")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_xlim(0, max(duration, 1))
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _write_summary(path: Path, report: dict) -> None:
    overlap = report["overlap"]
    lines = [
        "SPEECH INTERACTION ANALYZER — SUMMARY", "=" * 42,
        f"File: {report['input_file']}", f"Duration: {report['audio']['duration_seconds']:.2f} s",
        f"Detected speakers: {report['speaker_count']}", f"Overlap intervals: {overlap['interval_count']}",
        f"Total overlap: {overlap['total_seconds']:.2f} s ({overlap['recording_ratio'] * 100:.2f}% of recording)",
        f"Candidate interruption/overlap events: {len(report['candidate_interruption_events'])}",
        f"Pauses >= 0.4 s: {report['pauses']['count']}", "",
        "TRANSCRIPT", "-" * 10, report["transcription"]["text"] or "(No transcript produced)", "",
        "RECOGNITION EVALUATION", "-" * 22,
    ]
    metrics = report["recognition_evaluation"]
    if metrics and metrics.get("wer") is not None:
        lines.append(f"Whole-recording WER: {metrics['wer'] * 100:.2f}% ({metrics['errors']} edits / {metrics['reference_words']} reference words)")
        lines.append(metrics["note"])
    else:
        lines.append("Not calculated. Supply --reference path/to/reference.txt for whole-recording WER.")
    lines += ["", "LIMITATIONS", "-" * 11] + [f"- {x}" for x in report["limitations"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
