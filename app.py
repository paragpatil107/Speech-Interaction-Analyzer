
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


if os.name == "nt":
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    try:
        using_venv = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        using_venv = False
    if venv_python.is_file() and not using_venv:
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())])

# The current pyannote checkpoint used by this project requires the compatible
# torch.load behaviour for checkpoints from the trusted Hugging Face model repo.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
sys.path.insert(0, str(ROOT / "src"))

# Flask is the only new web-UI dependency; install it automatically if absent.
try:
    import flask  # noqa: F401
except ModuleNotFoundError:
    print("Flask is missing. Installing the web interface dependency…")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask>=3.0,<4.0"])
    except subprocess.CalledProcessError:
        print("\nCould not install Flask. Check your internet connection and Python environment.")
        input("Press Enter to close...")
        raise SystemExit(1)

try:
    from speech_analyzer.webapp import main
except ModuleNotFoundError as exc:
    missing = exc.name or "a required module"
    print(f"\nSpeech Interaction Analyzer cannot start: missing Python dependency '{missing}'.")
    print("Reinstall the project dependencies in its .venv, then double-click app.py again.")
    input("Press Enter to close...")
    raise SystemExit(1)

if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(f"\nCould not start the local web server: {exc}")
        print("Check whether http://127.0.0.1:8765 is already in use.")
        input("Press Enter to close...")
        raise SystemExit(1)
