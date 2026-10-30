#!/usr/bin/env bash
# One-shot reproducible deployment for the chords-analysis skill (macOS / Linux).
#
# Creates a self-contained Python 3.9 environment next to this script:
#   .venv39/            Python 3.9 + all packages
#   .cache/{pip,torch,huggingface,models}
#
# No conda required. Safe to delete .cache/ — weights re-download on demand.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$ROOT/skills/chords-analysis"
VENV="$ROOT/.venv39"
PY="$VENV/bin/python"
CACHE="$ROOT/.cache"
MIRROR="${CHORDS_HF_MIRROR:-https://hf-mirror.com}"

echo "== chords-analysis deploy =="
echo "root   : $ROOT"
echo "python : $PY"
echo "cache  : $CACHE"

mkdir -p "$CACHE"
export PIP_CACHE_DIR="$CACHE/pip"
export TORCH_HOME="$CACHE/torch"
export HF_HOME="$CACHE/huggingface"
export CHORDS_MODEL_DIR="$CACHE/models"
export CHORDS_HF_MIRROR="$MIRROR"
[ "$MIRROR" != "none" ] && export HF_ENDPOINT="$MIRROR"

echo "[1/6] ensuring uv ..."
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --quiet --user uv || {
    echo "uv install failed; install it manually: https://docs.astral.sh/uv/"; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[2/6] ensuring Python 3.9 ..."
uv python install 3.9

echo "[3/6] venv ..."
[ -x "$PY" ] || uv venv --python 3.9 --seed "$VENV"

echo "[4/6] packages ..."
uv pip install --python "$PY" "setuptools<81" wheel cython "numpy<1.24"
uv pip install --python "$PY" --no-build-isolation "madmom==0.16.1" || \
  echo "[WARN] madmom failed; ensemble runs BTC-only"
uv pip install --python "$PY" -r "$SKILL_DIR/requirements.txt" || \
  echo "[WARN] some optional packages failed; layers degrade"

echo "[5/6] model weights (mirror: $MIRROR) ..."
"$PY" -W ignore "$SKILL_DIR/scripts/models.py" --fetch-all

echo "[6/6] smoke test ..."
"$PY" -W ignore -c "import madmom, librosa, transformers, mir_eval; print('core deps OK')"
"$PY" -W ignore -c "import basic_pitch; print('basic-pitch OK')" || true
"$PY" -W ignore -c "import allin1_infer; print('structure model OK')" || true

cat <<EOF

Done. Use this interpreter:
  $PY

One-shot analysis:
  $PY -W ignore "$SKILL_DIR/scripts/run_all.py" audio/<file> --outdir out

Add to your shell profile to reuse the caches:
  export TORCH_HOME="$CACHE/torch"
  export HF_HOME="$CACHE/huggingface"
  export CHORDS_MODEL_DIR="$CACHE/models"
EOF
