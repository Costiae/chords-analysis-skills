---
name: Chords Analysis
description: Audio harmony analysis workflow — extract chords from mp3/wav/flac/m4a into structured JSON, then analyze key, progressions, sections, and modulations, or answer free-form questions. Use when the user wants chord/chord-progression/harmony analysis of a song, or mentions "analyze this song's chords/harmony".
license: MIT
---

# Chords Analysis Workflow

A complete pipeline: audio → chord recognition → formatted JSON → LLM analysis / report / Q&A.
The standard data format is documented in `references/schema.md`.

## Directory conventions

- Audio files: `audio/`; all outputs: `out/` (created as needed relative to the
  agent's working directory — adjust to the user's project layout as appropriate)
- Authoritative data file: `out/<name>.chords.json` (chords-v1 schema)
- Scripts (paths relative to this skill's base directory): `scripts/01_extract.py`
  (recognition), `scripts/02_format.py` (normalization), `scripts/03_summarize.py` (summary)
- Engines are pluggable via `scripts/engines.py`; adding a better model in the future
  is one function + one registration line. Run `01_extract.py --list-engines` to
  see what's available.

## First-time environment setup

Run from this skill's base directory:

```
python scripts/setup_env.py
```

The script probes your Python and picks a route:
- Python ≤ 3.9 → creates `.venv` in the skill directory
- Python ≥ 3.10 → creates/reuses a conda env `chords39` (Python 3.9), because
  madmom 0.16.1 cannot run on 3.10+ (`collections.MutableSequence`) nor with
  numpy ≥ 1.24 (`np.float`)

The script prints the python executable to use for all later steps — call it
`PY` below. On a Python ≥3.10 machine it is typically
`~/miniconda3/envs/chords39/python.exe` (Windows) or `.../bin/python` (Unix),
NOT a `.venv` inside the skill dir.

madmom 0.16.1 install recipe (handled automatically, recorded for debugging):
`setuptools<81` (pkg_resources) + `cython` + `numpy<1.24` pre-installed, then
`pip install --no-build-isolation madmom`, then `librosa`.
Verified combo (Windows, Oct 2026): Python 3.9 / numpy 1.23.5 / madmom 0.16.1 / librosa 0.11.

**Always invoke the analysis python with `-W ignore`** — madmom emits a
pkg_resources deprecation warning on import that pollutes stderr.

Verify afterwards:

```
<PY> -W ignore -c "import madmom, librosa; print('OK')"
```

If setup fails, report the error to the user; do not silently switch engines.

## Workflow steps

1. **Locate input**: find the audio the user refers to (default scan `audio/` for
   mp3/wav/flac/m4a/ogg/aiff).
2. **Extract chords** (accuracy-first: ensemble cross-validation by default):
   ```
   <PY> -W ignore <skill-dir>/scripts/01_extract.py audio/<file> --outdir out
   ```
   Two independent madmom engines (DeepChroma and CNN+CRF) run by default; their
   agreement is printed and disagreeing intervals are marked `confidence: "conflict"`
   downstream. `--list-engines` shows available engines; `--engine a,b,c` runs any
   combination; if the primary engine fails, the next one is used automatically.
   Relay any failure output to the user in full.
2b. **Structure analysis** (segments / beats / BPM — do NOT guess sections from
   chords; run the dedicated model):
   ```
   <PY> -W ignore <skill-dir>/scripts/01b_structure.py audio/<file> --outdir out
   ```
   Uses all-in-one-infer (functional segments: intro/verse/chorus/bridge/outro +
   beats + BPM). If unavailable, falls back to madmom beat tracking (no segment
   labels — then say so instead of guessing). First run downloads model weights.
3. **Format**:
   ```
   <PY> -W ignore <skill-dir>/scripts/02_format.py out/<name>.raw.json --outdir out
   ```
   Produces `<name>.chords.json` + `<name>.lab`. Automatically merges
   `<name>.struct.json` if present: chords get `bar`/`beat` (from downbeats) and
   `section` labels; `structure` and `tempo_bpm` land in the JSON. `--bars` is the
   legacy manual-beats path; `--struct` overrides the auto-discovered file.
4. **Summarize** (for the user's overview and your own quick reading):
   ```
   <PY> -W ignore <skill-dir>/scripts/03_summarize.py out/<name>.chords.json
   ```
5. **Analyze / answer**: read `out/<name>.chords.json` (for long songs read `stats`
   first, then read `chords` in slices as needed), then answer or write a report to
   `out/<name>.analysis.md` following the "LLM analysis conventions" in
   `references/schema.md`.

## Analysis requirements

- Answers must be grounded in the actual JSON data; cite chords with timestamps
  (e.g. `1:23.5 (83.5s) G:maj`).
- Key: `song.key` is a heuristic inference. Give your own judgment with evidence
  (opening/closing chords, tonicization, characteristic pitches); if you disagree
  with the heuristic, say so.
- Report structure: overview → key & modulations → section progression patterns
  (annotated with Roman numerals) → cadences & highlights → summary.
- For long songs do not read all chords at once; use the merged timeline from
  03_summarize.py to grasp the whole, then zoom into sections the question targets.

## Troubleshooting

- madmom install/import failures are version-driven; the exact walls hit in
  real testing (Windows, Python 3.14 host):
  - Python ≥ 3.10: `ImportError: cannot import MutableSequence from collections`
    → must use Python 3.9 (`conda create -n chords39 python=3.9 -y`)
  - numpy ≥ 1.24: `np.float` removed → pin `numpy<1.24` (1.23.5 verified)
  - setuptools ≥ 81: `pkg_resources` removed → pin `setuptools<81`
  - default pip build isolation hides Cython → install madmom with
    `--no-build-isolation`, after `cython` and `numpy<1.24` are present
  - Python 3.13+: numpy<1.24 has no wheels at all — conda route is mandatory
- Audio decode failure: check ffmpeg is on PATH (librosa needs it for mp3/m4a;
  flac/wav decode natively).
- Obviously wrong results (all `N` or a single chord): warn the user that this engine
  performs poorly on pure percussion / dense electronic music. The `.lab` file can be
  hand-edited, then re-run 02_format.py.
