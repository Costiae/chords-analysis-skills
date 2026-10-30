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
   beats + downbeats + BPM + meter). If unavailable, falls back to madmom beat
   tracking (no segment labels — then say so instead of guessing). Weights are
   fetched from HuggingFace; on blocked networks use `HF_ENDPOINT=https://hf-mirror.com`
   and pre-place the `harmonix-fold*.pth` files in `~/.cache/torch/hub/checkpoints`.
2c. **Voices: melody + bass** (mode/key evidence and inversions):
   ```
   <PY> -W ignore <skill-dir>/scripts/01c_voices.py audio/<file> --outdir out
   ```
   Runs demucs once (cached in `<outdir>/.stems/`), then pyin pitch tracking on the
   vocals stem (melody notes, pitch-class profile, phrase endings, key candidates)
   and on the bass stem (bass notes → slash chords). Vocals are the primary key
   evidence; if the track is instrumental the melody step fails gracefully and
   only bass is used.
3. **Format**:
   ```
   <PY> -W ignore <skill-dir>/scripts/02_format.py out/<name>.raw.json --outdir out
   ```
   Produces `<name>.chords.json` + `<name>.lab`. Automatically merges
   `<name>.struct.json` and `<name>.voices.json` if present: chords get
   `bar`/`beat`/`section`/`bass_pc`, key inference fuses chord histogram +
   melody pitch-class profile + phrase-ending anchors, section-local keys are
   estimated, chord labels are respelled for the local key (B♭ not A♯) and
   inversions become slash chords (C:maj/E). `--bars`/`--struct`/`--voices`
   override auto-discovery.
4. **Summarize** (for the user's overview and your own quick reading):
   ```
   <PY> -W ignore <skill-dir>/scripts/03_summarize.py out/<name>.chords.json
   ```
4b. **Harmonic map** (bar grid + Roman numerals + repeated progressions):
   ```
   <PY> -W ignore <skill-dir>/scripts/04_map.py out/<name>.chords.json --outdir out
   ```
   Writes `<name>.map.md`: per-bar chord grid with Roman numerals under the
   section's local key, per-section progression chains and harmonic rhythm, and
   repeated n-gram progressions ranked by frequency. This is the file to read
   for "和声地图"/"扒谱" style questions — don't re-derive it by hand.
5. **Analyze / answer**: read `out/<name>.chords.json` (for long songs read `stats`
   first, then read `chords` in slices as needed), then answer or write a report to
   `out/<name>.analysis.md` following the "LLM analysis conventions" in
   `references/schema.md`.

## Analysis requirements

- Answers must be grounded in the actual JSON data; cite chords with timestamps
  (e.g. `1:23.5 (83.5s) G:maj`).
- **Key**: `song.key` fuses three evidence sources (chord histogram, melody
  pitch-class profile, phrase-ending anchors). Report `song.key_candidates` and
  their `tonic_evidence` when the top two are close — that usually means a
  relative major/minor or parallel-mode ambiguity. Never present a
  near-related-key guess as certain.
- **Sections**: use `structure.segments[].key/key_confidence` for section-local
  keys; `song.modulations` lists every distinct key found. A low-confidence
  segment key (short segments) should be described as uncertain.
- **Progressions**: prefer `04_map.py` output (`<name>.map.md`) — it already has
  the bar grid, Roman numerals and mined repeated n-grams. Do not re-derive the
  map by hand from raw chords.
- **Inversions**: labels with `/bass` are inversions confirmed by the bass stem;
  the `bass_pc` field is the raw low note when not a chord tone.
- Report structure: overview → key & modulations → section progression patterns
  (Roman numerals) → harmonic rhythm → cadences & highlights → low-confidence list.
- For long songs do not read all chords at once; start from `03_summarize.py`
  and `<name>.map.md`, then zoom into sections the question targets.

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
