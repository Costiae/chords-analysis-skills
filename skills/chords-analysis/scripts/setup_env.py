#!/usr/bin/env python3
"""Cross-platform environment setup for the chords-analysis skill.

Strategy (madmom 0.16.1 has no wheels and only supports Python <= 3.9):
  1. If current Python is <= 3.9: create .venv in the skill dir, install deps.
  2. If current Python is >= 3.10: create/reuse a conda env (Python 3.9) instead —
     'collections.MutableSequence' was removed in 3.10, and numpy<1.24 has no
     wheels for 3.13+.
  3. Print the python executable to use for the scripts.

Verified recipe (Windows, Oct 2026): Python 3.9 + setuptools<81 + cython +
numpy 1.23.5 + madmom 0.16.1 (--no-build-isolation) + librosa 0.11.

Usage:  python setup_env.py
"""
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_DIR / ".venv"


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def pip_install(py, *pkgs, **kw):
    return run(py, "-m", "pip", "install", *pkgs, **kw)


def setup_venv():
    py = VENV_DIR / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not VENV_DIR.exists():
        print(f"creating venv at {VENV_DIR} ...")
        run(sys.executable, "-m", "venv", VENV_DIR)
    else:
        print(f"venv already exists at {VENV_DIR}")
    run(py, "-m", "pip", "install", "--upgrade", "pip")
    # madmom needs Cython + numpy<1.24 present *before* building (it fails to
    # declare them as build deps), and build isolation hides them -> disable it.
    pip_install(py, "--upgrade", "setuptools<81", "wheel")
    pip_install(py, "cython", "numpy<1.24")
    r = pip_install(py, "--no-build-isolation", "madmom")
    if r.returncode == 0:
        pip_install(py, "librosa")
    return py


def setup_conda(name):
    # create env if missing, then use its python directly
    r = run("conda", "env", "list")
    exists = r.returncode == 0 and name in (r.stdout or "")
    if not exists:
        r = run("conda", "create", "-n", name, "python=3.9", "-y")
        if r.returncode != 0:
            print(f"[ERROR] conda env creation failed")
            sys.exit(2)
    # locate env python
    import os
    envs = Path(os.environ.get("CONDA_EXE", "")).parent.parent / "envs" if os.environ.get("CONDA_EXE") else Path.home() / "miniconda3" / "envs"
    py = envs / name / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        print(f"[ERROR] cannot locate python for conda env '{name}' (tried {py})")
        sys.exit(2)
    run(py, "-m", "pip", "install", "--upgrade", "pip")
    # same madmom build pitfalls apply in the conda env; numpy must be <1.24
    # (np.float alias removed in 1.24)
    pip_install(py, "setuptools<81", "cython", "numpy<1.24")
    pip_install(py, "--no-build-isolation", "madmom")
    pip_install(py, "librosa")
    return py


def main():
    if sys.version_info >= (3, 10):
        print("Python >= 3.10 detected; madmom 0.16.1 needs Python <= 3.9")
        print("(collections.MutableSequence removed in 3.10; numpy<1.24 has no")
        print(" wheels for 3.13+). Falling back to a conda env with Python 3.9 ...")
        py = setup_conda("chords39")
    else:
        py = setup_venv()
    r = run(py, "-W", "ignore", "-c",
            "import madmom, librosa; print('OK: madmom', madmom.__version__)")
    if r.returncode != 0:
        print("[WARN] install finished but import failed — see troubleshooting in SKILL.md")
        sys.exit(2)
    print("\nDone. Use this python (with -W ignore) for the scripts:")
    print(py)


if __name__ == "__main__":
    main()
