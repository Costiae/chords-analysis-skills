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

## First-time environment setup

Run from this skill's base directory:

```
python scripts/setup_env.py
```

This creates an isolated `.venv` inside the skill directory and installs madmom + librosa.
Verify afterwards:

```
<skill-dir>/.venv/Scripts/python -c "import madmom"      # Windows
<skill-dir>/.venv/bin/python -c "import madmom"          # macOS/Linux
```

If madmom fails to install or import, report the error to the user; do not silently
switch engines. Fallbacks are listed at the bottom of this file.

## Workflow steps

1. **Locate input**: find the audio the user refers to (default scan `audio/` for
   mp3/wav/flac/m4a/ogg/aiff).
2. **Extract chords** (accuracy-first: ensemble cross-validation by default):
   ```
   <skill-dir>/.venv/Scripts/python <skill-dir>/scripts/01_extract.py audio/<file> --outdir out
   ```
   (On macOS/Linux use `.venv/bin/python`.) Two independent madmom engines
   (DeepChroma and CNN+CRF) run by default; their agreement is computed and
   disagreeing intervals are marked. Relay any failure output to the user in full.
3. **Format**:
   ```
   <skill-dir>/.venv/Scripts/python <skill-dir>/scripts/02_format.py out/<name>.raw.json --outdir out
   ```
   Produces `<name>.chords.json` + `<name>.lab`. If the user has beat data (a JSON with
   `{"beats": [t1, t2, ...]}`), pass it via `--bars` to align chords to bars/beats.
4. **Summarize** (for the user's overview and your own quick reading):
   ```
   <skill-dir>/.venv/Scripts/python <skill-dir>/scripts/03_summarize.py out/<name>.chords.json
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

- madmom may fail on Python ≥ 3.12 due to numpy/compile issues: try
  `pip install "numpy<2"` first, then madmom; if it still fails, honestly tell the
  user and suggest a conda env with Python 3.10 (`conda create -n chords python=3.10`).
- Audio decode failure: check ffmpeg is on PATH (librosa needs it for mp3/m4a;
  flac/wav decode natively).
- Obviously wrong results (all `N` or a single chord): warn the user that this engine
  performs poorly on pure percussion / dense electronic music. The `.lab` file can be
  hand-edited, then re-run 02_format.py.
