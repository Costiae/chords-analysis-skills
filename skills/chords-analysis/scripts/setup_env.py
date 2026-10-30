#!/usr/bin/env python3
"""Cross-platform environment setup for the chords-analysis skill.

Creates an isolated venv at <skill-dir>/.venv and installs madmom + librosa.
Usage:  python setup_env.py
"""
import subprocess
import sys
import venv
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_DIR / ".venv"
PY = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main():
    if not VENV_DIR.exists():
        print(f"creating venv at {VENV_DIR} ...")
        venv.create(str(VENV_DIR), with_pip=True)
    else:
        print(f"venv already exists at {VENV_DIR}")

    pkgs = ["librosa", "madmom"]
    print(f"installing {', '.join(pkgs)} ...")
    r = subprocess.run([str(PY), "-m", "pip", "install", "--upgrade", "pip"])
    r = subprocess.run([str(PY), "-m", "pip", "install", *pkgs])
    if r.returncode != 0:
        print("\n[WARN] install failed. If madmom failed to build, try:")
        print(f'  {PY} -m pip install "numpy<2"')
        print(f"  {PY} -m pip install madmom")
        print("or use a Python 3.10 environment (conda create -n chords python=3.10).")
        sys.exit(2)

    r = subprocess.run([str(PY), "-c", "import madmom, librosa; print('OK: madmom', madmom.__version__)"])
    if r.returncode != 0:
        print("[WARN] madmom installed but import failed — see troubleshooting in SKILL.md")
        sys.exit(2)
    print("\nDone. Python for this skill:", PY)


if __name__ == "__main__":
    main()
