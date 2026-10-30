#!/usr/bin/env python3
"""Cross-platform environment setup for the chords-analysis skill.

madmom 0.16.1 has no wheels and only supports Python <= 3.9
(collections.MutableSequence), with numpy<1.24 (np.float) and setuptools<81
(pkg_resources). So the analysis interpreter must be Python 3.9, whatever the
host's default Python is.

Strategy, in order of preference (all verified on Windows):
  1. If the current Python is 3.9 -> create `.venv` in the skill directory.
  2. Else if `uv` is available -> `uv python install 3.9` + `uv venv --python 3.9`
     (no admin rights, no conda required, self-contained).
  3. Else if `conda` is available -> create/reuse a conda env with Python 3.9.
  4. Else -> explain and exit non-zero.

Usage:  python setup_env.py [--target-dir DIR]
Prints the interpreter to use for every later step.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
VENV_DIR = SKILL_DIR / ".venv"

BUILD_DEPS = ["setuptools<81", "wheel", "cython", "numpy<1.24"]
RUNTIME = ["librosa", "transformers<5", "huggingface_hub", "mir_eval"]
OPTIONAL = ["basic-pitch", "all-in-one-infer @ git+https://github.com/openmirlab/all-in-one-infer.git"]


def run(cmd, **kw):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def pip(py, *pkgs, **kw):
    return run(py, "-m", "pip", "install", *pkgs, **kw)


def install_all(py: Path) -> Path:
    run(py, "-m", "pip", "install", "--upgrade", "pip")
    pip(py, *BUILD_DEPS)
    # madmom cannot declare its build deps, and build isolation hides them
    r = pip(py, "--no-build-isolation", "madmom")
    if r.returncode != 0:
        print("[WARN] madmom build failed — chord ensemble will run BTC-only")
    pip(py, *RUNTIME)
    for pkg in OPTIONAL:
        r = pip(py, pkg)
        if r.returncode != 0:
            print(f"[WARN] optional package failed: {pkg} "
                  f"(its layer will degrade, everything else still works)")
    return py


def venv_py(base: Path, dest: Path) -> Path:
    return dest / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def setup_local_venv() -> Path:
    if not VENV_DIR.exists():
        run(sys.executable, "-m", "venv", VENV_DIR)
    return install_all(venv_py(sys.executable, VENV_DIR))


def setup_uv(target: Path) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found")
    run(uv, "python", "install", "3.9")
    run(uv, "venv", "--python", "3.9", "--seed", target)
    return install_all(venv_py(uv, target))


def setup_conda(name="chords39") -> Path:
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("conda not found")
    envs = Path(os.environ.get("CONDA_PREFIX", Path.home() / "miniconda3")).parent / "envs"
    py = envs / name / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not py.exists():
        run(conda, "create", "-n", name, "python=3.9", "-y")
    if not py.exists():
        raise RuntimeError(f"cannot locate python for conda env '{name}'")
    return install_all(py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", default=None,
                    help="venv path for the uv route (default: <cwd>/.venv39)")
    args = ap.parse_args()

    print(f"host python: {sys.version.split()[0]}")
    py = None
    if sys.version_info[:2] == (3, 9):
        print("host python is 3.9 -> local venv route")
        py = setup_local_venv()
    else:
        target = Path(args.target_dir).resolve() if args.target_dir else Path.cwd() / ".venv39"
        print(f"host python is not 3.9 -> need an isolated 3.9 interpreter")
        try:
            py = setup_uv(target)
            print(f"uv route OK -> {py}")
        except Exception as e:
            print(f"[uv route failed] {type(e).__name__}: {e}")
            try:
                py = setup_conda()
                print(f"conda route OK -> {py}")
            except Exception as e2:
                print(f"[conda route failed] {type(e2).__name__}: {e2}")
                print("\nNo Python 3.9 available. Install one of:")
                print("  - uv (recommended):  pip install uv")
                print("  - Miniconda, then:   conda create -n chords39 python=3.9 -y")
                sys.exit(2)

    r = run(py, "-W", "ignore", "-c",
            "import madmom, librosa; print('OK: madmom', madmom.__version__)")
    if r.returncode != 0:
        print("[WARN] import check failed — see troubleshooting in SKILL.md")
    print("\nUse this interpreter (with -W ignore) for all scripts:")
    print(py)


if __name__ == "__main__":
    main()
