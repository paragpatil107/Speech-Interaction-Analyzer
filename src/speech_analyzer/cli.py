from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import soundfile as sf

from .analyzer import run_analysis


def _run_and_print(args, audio_path: str, output_dir: str) -> None:
    report = run_analysis(
        audio_path, output_dir, args.model, args.device, args.compute_type,
        args.hf_token, args.reference, args.min_overlap,
    )
    print("\nAnalysis complete.")
    print(f"Speakers: {report['speaker_count']}")
    print(f"Overlap: {report['overlap']['total_seconds']:.2f}s ({report['overlap']['recording_ratio'] * 100:.2f}%)")
    print(f"Results: {Path(output_dir).resolve()}")
    print("Files: report.json, summary.txt, speaker_turns.csv, timeline.png")


def _record_then_analyze(args) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Live recording dependency missing. Run: pip install -e .") from exc

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = output_dir / "recording.wav"
    chunks: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    callback_error: list[BaseException] = []

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio input warning: {status}", file=sys.stderr)
        if not stop_event.is_set():
            chunks.put(indata.copy())

    print("Live recording started. Speak normally; press Ctrl+C when finished.")
    print(f"Microphone: system default | Sample rate: {args.sample_rate} Hz | Channels: 1")
    print("Press Ctrl+C to stop recording and automatically analyze the session.\n")
    audio_chunks = []
    try:
        with sd.InputStream(samplerate=args.sample_rate, channels=1, dtype="float32", callback=callback, blocksize=0):
            while True:
                try:
                    audio_chunks.append(chunks.get(timeout=0.25))
                except queue.Empty:
                    continue
    except KeyboardInterrupt:
        stop_event.set()
        while not chunks.empty():
            try:
                audio_chunks.append(chunks.get_nowait())
            except queue.Empty:
                break
        print("\nRecording stopped. Saving audio...")
    except Exception as exc:
        stop_event.set()
        raise RuntimeError(f"Could not access the microphone: {exc}. Check Windows microphone permissions and the default input device.") from exc

    if not audio_chunks:
        raise RuntimeError("No audio was captured. Check the microphone and try again.")
    import numpy as np
    audio = np.concatenate(audio_chunks, axis=0)
    if len(audio) < args.sample_rate:
        raise RuntimeError("Recording was shorter than one second; record for longer and try again.")
    sf.write(str(recording_path), audio, args.sample_rate)
    print(f"Saved recording: {recording_path} ({len(audio) / args.sample_rate:.1f} seconds)")
    _run_and_print(args, str(recording_path), str(output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech-analyzer", description="Record or analyze speech interactions, overlap, speaker turns, pauses, and ASR output.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_analysis_options(command_parser):
        command_parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"], help="faster-whisper model size (base is a quicker CPU default)")
        command_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device; CPU is the safe default")
        command_parser.add_argument("--compute-type", default="int8", choices=["int8", "float16", "float32", "int8_float16"], help="Inference precision")
        command_parser.add_argument("--hf-token", default=None, help="Hugging Face token; preferably set HF_TOKEN environment variable instead")
        command_parser.add_argument("--reference", default=None, help="Optional plain-text reference transcript for whole-recording WER")
        command_parser.add_argument("--min-overlap", type=float, default=0.15, help="Minimum overlap duration in seconds to report")

    analyze = sub.add_parser("analyze", help="Analyze an existing audio file")
    analyze.add_argument("audio", help="Input audio file")
    analyze.add_argument("--output", default="outputs/run-001", help="Output directory")
    add_analysis_options(analyze)

    record = sub.add_parser("record", help="Record from your microphone; Ctrl+C stops recording and starts analysis")
    record.add_argument("--output", default="outputs/live-session-001", help="Output directory for recording and analysis")
    record.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate (default: 16000 Hz)")
    add_analysis_options(record)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "record":
            _record_then_analyze(args)
        else:
            _run_and_print(args, args.audio, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
