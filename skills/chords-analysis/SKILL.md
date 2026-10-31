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

The script probes your host and picks the first route that works:
- host Python is 3.9 → creates `.venv` in the skill directory
- otherwise, `uv` available → `uv python install 3.9` + `uv venv --python 3.9`
  (recommended; no admin rights, no conda required)
- otherwise, `conda` available → conda env `chords39` (Python 3.9)
- otherwise it stops and tells you what to install

It prints the interpreter (`PY`) to use for every later step. madmom 0.16.1 walls
hit in real testing: needs Python ≤3.9 (`collections.MutableSequence`),
`numpy<1.24` (`np.float`), `setuptools<81` (`pkg_resources`), and a
`--no-build-isolation` build with `cython` pre-installed.

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

0. **One-shot alternative**: `python run_all.py audio/<file> --outdir out` runs every
   step below in order and prints a per-layer OK/SKIP summary. Use it unless a
   step needs manual attention.
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
   <PY> -W ignore <skill-dir>/scripts/01c_voices.py audio/<file> --outdir out \
       [--bass-method auto|basic-pitch|pyin]
   ```
   Runs demucs once (cached in `<outdir>/.stems/`), then:
   - **melody**: pyin pitch tracking on the vocals stem → notes, pitch-class
     profile, phrase endings, key candidates. Primary key evidence.
   - **bass**: `scripts/bass.py`, which is event-based (Basic Pitch), **not**
     continuous F0 tracking. pyin on a plucked bass is systematically wrong:
     the 2nd harmonic usually outweighs the fundamental (octave flips D2/D3)
     and 100–180 ms notes are too short to track. The pipeline is
     Basic Pitch → octave folding into [C1, G3] → monophonization →
     sub-harmonic repair → chroma self-check. pyin is kept only as the
     degradation path (`--bass-method pyin`, or when basic-pitch is missing).
   Instrumental tracks simply skip the melody half; bass still runs.
   Read the printed diagnostics and report them if unusual:
   - `octave_folded` / `register_dropped` — notes moved into bass register / discarded
   - `subharmonic` — notes lowered an octave because f0/2 dominated in both
     the head and tail windows (harmonic misdetection, pitch class unchanged)
   - `chroma_agreement` (0..1) — note pitch-class distribution vs. audio chroma.
     It is a **self-check**, not an accuracy score. On a real plucked-bass stem,
     ~0.75–0.85 is normal for Basic Pitch; pyin scores ~0.43 and must not be
     used as the primary path.
   Bass notes are the sole evidence for slash chords; no bass stem means no
   inversions (never invent them).
2d. **Chord quality (large vocabulary)** — order matters:
   - If engine `btc-170` is available (weights cached), **skip this step**: BTC
     already outputs 170-class labels and 02_format ignores the template file
     when the engine is BTC.
   - Otherwise run the offline fallback:
   ```
   <PY> -W ignore <skill-dir>/scripts/01d_quality.py audio/<file> out/<name>.raw.json --outdir out
   ```
   Keeps madmom's root, refines quality by chroma template matching over
   13 qualities, with family restriction (never flips the third), energy
   validation and per-quality margins.
2e. **Transcription (扒谱, optional)**:
   ```
   <PY> -W ignore <skill-dir>/scripts/01e_transcribe.py audio/<file> --outdir out
   ```
   Basic Pitch per demucs stem (vocals/bass/other), monophonic filtering on
   vocals/bass, per-stem MIDI + combined MIDI + note JSON. Skips cleanly when
   basic-pitch is not installed. If `<name>.voices.json` already exists, the
   bass stem is **reused** from it rather than re-inferred, so the MIDI and the
   inversion analysis share one note list (GM program 33 = electric bass).

## Model fetching (shared, mirror-aware)

All remote weights go through `scripts/models.py` + `models/manifest.json`:

```
<PY> -W ignore <skill-dir>/scripts/models.py --status          # what's cached
<PY> -W ignore <skill-dir>/scripts/models.py --fetch-all       # download+verify
<PY> -W ignore <skill-dir>/scripts/models.py --check-updates   # HEAD-compare, no download
```

- Mirror: `CHORDS_HF_MIRROR` or `HF_ENDPOINT` (defaults to `https://hf-mirror.com`);
  set to `none` to force the official source.
- Every artifact is sha256-verified and atomically replaced; downloads resume.
- HF-repo models (BTC) are pinned to a **commit sha** in the manifest, never `main`.
- **Never let a download failure abort the pipeline** — each stage degrades:
  BTC-170 → madmom ensemble; structure → madmom beats; melody → bass only;
  large-vocab → template quality refinement; quality → raw maj/min;
  transcription → skipped. Always report which layer degraded and why.
3. **Format**:
   ```
   <PY> -W ignore <skill-dir>/scripts/02_format.py out/<name>.raw.json --outdir out
   ```
   Produces `<name>.chords.json` + `<name>.lab`. Automatically merges
   `<name>.struct.json`, `<name>.voices.json` and `<name>.quality.json` if
   present: chords get `bar`/`beat`/`section`/`bass_pc` and refined qualities,
   key inference fuses chord histogram + melody pitch-class profile +
   phrase-ending anchors, section-local keys are estimated, chord labels are
   respelled for the local key (B♭ not A♯) and inversions become slash chords
   (C:maj/E). `--bars`/`--struct`/`--voices`/`--quality` override auto-discovery.
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
4c. **Local key audit + tonicization** (section keys are averages — verify them):
   ```
   <PY> -W ignore <skill-dir>/scripts/06_local_keys.py out/<name>.chords.json --outdir out
   ```
   Two independent outputs:
   - **coverage gaps**: sliding windows where the section key stops explaining
     the chord tones (coverage < 0.8 for ≥2 consecutive windows). Each gap gets
     the best local alternative and a `departure` / `relative` label; `relative`
     means the alternative shares the same pitch-class set (relative major/minor,
     no real modulation). Boundaries are ±window; the `worst window` column is
     the closest thing to the true pivot.
   - **tonicizations**: `dominant → target` resolutions (root up a fourth),
     excluding the section's own V→I and I→IV. `strong` = dominant 7th or a
     major triad on a degree that should be minor/dim (secondary dominant);
     `weak` needs context. Events inside a gap are marked with the gap's hint —
     the section key has already been shown to fail there.
   Writes `<name>.localkeys.md`. **Read this before writing about key**: a
   section key with low confidence or `key_source: stable` is often an average
   of two keys, and the gap table shows exactly where it breaks.
4d. **Section materials from melody + chords** (which sections reuse the same material):
   ```
   <PY> -W ignore <skill-dir>/scripts/07_sections.py out/<name>.chords.json --outdir out
   ```
   Clusters sections on **transposition-invariant melodic contour** (0.45) +
   pitch-class profile (0.30) + chord-degree profile (0.25), then cross-tabulates
   the result against the model's labels. Writes `<name>.materials.md`:
   - per-section table: material ID, model label, melodic direction
     (regression slope + phrase up/down), cadence degree;
   - material groups and a **label × material** matrix;
   - cross-label reuse (e.g. an outro that reuses the chorus melody);
   - pre-chorus candidates: section immediately before a chorus that ends on a
     dominant function and whose material is unique. A full-length verse is
     reported as `weak`, not as a confirmed pre-chorus.
   Contour is what actually separates sections: in testing, pitch-class profiles
   sat at 0.85+ even between verse and chorus, while contour reached 0.99 for
   repeated choruses and went negative across different materials. Melody from
   pyin can carry octave errors, so a single low score is a hint, not proof.
4e. **Score export** (MusicXML + tempo-mapped MIDI):
   ```
   <PY> -W ignore <skill-dir>/scripts/08_score.py out/<name>.chords.json --outdir out
   ```
   Reads the real downbeat list and tempo curve from `<name>.struct.json`
   (`chords.json` only stores the downbeat count), quantizes to 16ths and writes:
   - `<name>.lead.musicxml` — melody + chord symbols + rehearsal marks + key changes;
   - `<name>.score.musicxml` — melody + bass + harmony **generated from chord
     symbols** (the part name says `generated`; it is not audio transcription);
   - `<name>.score.mid` — with tempo map and time signature, importable into a DAW.
   Measures are padded to exactly one bar; the score starts at the first downbeat
   (audio before it is not notated). Melody is folded into C3–C6 and octave-
   corrected. Requires `music21` (8.3.0 is the last release supporting Python
   3.9); the step skips cleanly without it.
4f. **Human review page** (the "AI got it wrong, a human can check" artifact):
   ```
   <PY> -W ignore <skill-dir>/scripts/09_review.py out/<name>.chords.json --outdir out
   ```
   Writes `<name>.review.html`: section bands (label · material · key), chord lane
   (engine conflicts in orange), melody piano roll (octave-corrected notes in red),
   bass roll, bar grid and a section table. Self-contained — open it in any
   browser. Audit against this page before asserting anything about a passage.
5. **Analyze / answer**: read `out/<name>.chords.json` (for long songs read `stats`
   first, then read `chords` in slices as needed), then answer or write a report to
   `out/<name>.analysis.md` following the "LLM analysis conventions" in
   `references/schema.md`.
6. **Regression check (optional)**: with a reference annotation,
   ```
   <PY> -W ignore <skill-dir>/scripts/05_eval.py out/<name>.chords.json ref.lab --vocab mirex
   ```
   prints root/majmin/triad/sevenths/mirex scores (mir_eval) and writes
   `<name>.chords.eval.json`. Run this before/after changing engines instead of
   arguing from impressions.

## Analysis requirements

- Answers must be grounded in the actual JSON data; cite chords with timestamps
  (e.g. `1:23.5 (83.5s) G:maj`).
- **Key is per-section, never per-song.** `song.key` is a duration-weighted
  summary, not an analysis unit — do not open a report with "this song is in X".
  Establish the tonic for each section first, then say how each section departs
  from it. A section can contain more than one tonic: `06_local_keys.py`'s
  coverage gaps expose those, and a section label like "B♭ (0.60)" is usually an
  average of two keys.
- **Tonicization ≠ modulation.** A brief `V/x → x` that returns to the section
  tonic is a tonicization; a sustained new tonic (and the section key that
  follows) is a modulation. Report the direction (toward vi, toward III, …) and
  how long it lasts. `06_local_keys.py` gives both, plus the return time.
- **Key evidence**: `song.key` fuses chord histogram, melody pitch-class profile
  and phrase-ending anchors. Report `song.key_candidates` and `tonic_evidence`
  when the top two are close — that usually means a relative major/minor or
  parallel-mode ambiguity. Never present a near-related-key guess as certain.
- **Sections**: use `structure.segments[].key/key_confidence` as a starting
  point, but treat `key_source: stable` as inherited, not measured — verify it
  against the coverage gaps before quoting it. `song.modulations` lists every
  distinct key found.
- **Structure labels are functional, not material.** A `chorus` label can cover
  several different melodies, and an `outro` can reuse the chorus material. Use
  `07_sections.py`'s label × material matrix before saying "the chorus returns";
  quote the material ID and the transposition if the repeat is shifted.
- **Pre-chorus** is only a pre-chorus if it is short, ends on a dominant function
  and has its own material. A full-length verse that happens to end on V is a
  verse; say so.
- **Do not claim transcription accuracy the pipeline cannot support.** The
  score is a review artifact: melody is single-voice and octave-corrected, the
  harmony staff is generated from chord symbols, drums and piano voicings are not
  transcribed. Say what the artifact is (lead sheet / review score), and point at
  `<name>.review.html` and the corrected notes in it.
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

- **The environment disappeared / `python.exe` not found**: this happened in
  testing when a conda install was being uninstalled under the running job.
  Just re-run `python scripts/setup_env.py` — model weights stay cached in
  `~/.cache/chords-analysis` and `~/.cache/torch`, so nothing large re-downloads.
  `run_all.py` runs a preflight check and says exactly which layer will degrade.
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
