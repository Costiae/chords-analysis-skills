# chords-analysis-skills

Portable agent skill for audio harmony analysis: **audio (mp3/wav/flac/m4a/ogg) →
chord recognition → structured JSON → LLM analysis / report / Q&A**.

Works with any agent harness that supports the [Agent Skills](https://agentskills.io)
convention (`SKILL.md` + supporting files), including OpenCode, Claude Code,
and other compatible runtimes.

## Install

### OpenCode

Clone into the project (or global `~/.config/opencode/skills`):

```sh
git clone https://github.com/Costiae/chords-analysis-skills.git
# per-project: copy/rename skills/chords-analysis into .opencode/skills/chords-analysis
# global:      copy into ~/.config/opencode/skills/chords-analysis
```

### Claude Code / other SKILL.md-compatible agents

Copy `skills/chords-analysis/` into the agent's skills directory
(e.g. `.claude/skills/chords-analysis` for Claude Code projects,
`~/.claude/skills/` globally).

### Any agent, manually

Point the agent at `skills/chords-analysis/SKILL.md` and let it follow the
workflow instructions. All steps are plain Python scripts — no harness-specific
features are required.

## Setup (once)

### One-command deployment (recommended)

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File deploy.ps1

# macOS / Linux
bash deploy.sh
```

This creates a self-contained `.venv39` (Python 3.9 via `uv`) and `.cache/` in the
repo root, installs every pinned dependency, fetches model weights (mirror-aware,
sha256-verified) and runs a smoke test. No conda, no admin rights, nothing large
written to the system drive. Delete `.cache/` freely — weights come back on demand.

### Manual setup

```sh
cd skills/chords-analysis
python scripts/setup_env.py   # prints the interpreter to use (PY below)
```

The script auto-routes to Python 3.9: local `.venv` if already 3.9, otherwise `uv`,
otherwise conda. madmom 0.16.1 constraints are handled automatically
(`numpy<1.24`, `setuptools<81`, `cython` + `--no-build-isolation`).

All later commands: `PY -W ignore scripts/01_extract.py ...` (the `-W ignore`
suppresses madmom's pkg_resources deprecation warning).

**Verified in the wild** (Windows, Oct 2026): a 4:31 96 kHz FLAC ran the whole
pipeline — 304 large-vocabulary chords, 12 functional segments, melody/bass
stems, 6.7k transcribed notes, harmonic map with Roman numerals.

## Pipeline

```
audio/xxx.flac
  │  scripts/run_all.py         ← one-shot orchestrator (per-layer OK/SKIP report)
  │
  ├─ 01_extract.py             chords: BTC-170 + madmom-cnn ensemble (falls back to madmom×2)
  ├─ 01b_structure.py          segments, beats, downbeats, BPM, meter, tempo curve
  ├─ 01c_voices.py             demucs stems → pyin melody (key evidence) + bass.py (event-based bass)
  ├─ 01d_quality.py            template quality refinement (skipped when BTC active)
  ├─ 01e_transcribe.py         Basic Pitch per stem → MIDI + note events
  ▼
out/xxx.struct.json / xxx.voices.json / xxx.quality.json / xxx.transcription.json
  │  scripts/02_format.py      fuse into chords-v1: bar/beat/section/bass per chord,
  │                            multi-evidence key, section-local keys, spelling, slash chords
  ▼
out/xxx.chords.json  ★ authoritative + xxx.lab (MIREX)
  │  scripts/03_summarize.py   timeline + harmonic rhythm
  │  scripts/04_map.py         harmonic map (bar grid, Roman numerals, n-grams)
  │  scripts/05_eval.py        mir_eval regression vs reference annotation
  │  scripts/06_local_keys.py  section-key audit: coverage gaps + tonicization directions
  │  scripts/07_sections.py   section materials from melody contour + chords, label × material
  │  scripts/08_score.py      MusicXML (lead sheet + 3-part) + tempo-mapped MIDI
  │  scripts/09_review.py     self-contained HTML review page (piano roll + chords + sections)
  ▼
out/xxx.map.md · xxx.lead.musicxml · xxx.score.midi · xxx.review.html · xxx.analysis.md
```

## What a complete analysis covers

| Layer | Source | Output |
|---|---|---|
| Chords | madmom ensemble (2 engines, cross-validated) | onset/offset + confidence |
| **Large-vocab chords** | **BTC-170 (BTC transformer, HF weights)** | **170 classes: maj7/min7/7/sus/dim/aug** |
| Bar grid | all-in-one downbeats + meter | `bar` / `beat` per chord |
| Sections | all-in-one functional labels | intro/verse/chorus/bridge/outro |
| Tempo | all-in-one + beat-interval curve | BPM + meter + tempo curve |
| Melody | demucs vocals + pyin | notes, pitch-class profile, phrase endings |
| Bass / inversions | demucs bass + Basic Pitch (event-based) → octave fold → monophonize → sub-harmonic repair | slash chords (`C:maj/E`), `bass_share` gate |
| Chord quality | BTC-170 → template fallback | maj7/min7/7/sus/dim/aug |
| **Transcription** | **Basic Pitch per demucs stem** | **note events + MIDI per stem** |
| Key / mode | chords + melody + phrase anchors + **section-scoped** key, coverage-checked | per-section tonic, gaps where it fails |
| Tonicization | `06_local_keys.py` (dominant → target resolutions) | direction (V/vi, V/III…), duration |
| Section material | `07_sections.py` (transposition-invariant melody contour + chords) | material IDs, label × material matrix, pre-chorus |
| **Score** | **`08_score.py` (downbeat grid + tempo curve + music21)** | **`.lead.musicxml`, `.score.musicxml`, tempo-mapped `.score.mid`** |
| **Human review** | **`09_review.py` (SVG, no deps)** | **`.review.html`: sections + chords + melody + bass on one timeline** |
| Modulations | per-section key estimation | `song.modulations`, per-segment keys |
| Progressions | n-gram mining | repeated progressions with Roman numerals |
| Evaluation | mir_eval (05_eval.py) | root/majmin/triad/sevenths/mirex |

## What the score is, and what it is not

The export is a **review score**: it is built from the analysis, not from a
dedicated score model, and it says so in the part names.

- Melody and bass are quantized to 16ths on the downbeat grid, with a real tempo
  map; measures are padded to one bar, so it opens correctly in MuseScore/DAW.
- The melody is single-voice, folded into C3–C6 and octave-corrected; corrected
  notes are red in `<name>.review.html`.
- The `Harmony (generated from symbols)` staff is **generated from chord symbols**,
  not transcribed from audio. It exists so the score is playable, not as evidence.
- Not transcribed: drums, piano/guitar voicing (which notes, which position),
  performance techniques, lyrics. Tab export is therefore not meaningful yet.

Honest gaps: instrument identification beyond the four demucs stems, dense
polyphonic transcription, and — most importantly — **no ground-truth
annotation**. `05_eval.py`'s 1.0000 is self-consistency; engine agreement
(75–77%) is not accuracy.

## Next: multi-instrument transcription (separate environment)

The current stack is pinned to Python 3.9 because of madmom. The strongest
open multi-instrument option, **YourMT3+** (MT3 family, 8-stem separation,
~536 MB weights, CPU-capable), requires **Python ≥ 3.10** — so it cannot live in
`.venv39`. `openmirlab/mt3-infer` wraps YourMT3/MR-MT3/MT3-PyTorch behind one
`transcribe()` API. The intended path is a second environment (`.venv311`) used
only for transcription, writing the same `transcription.json` shape, so the rest
of the pipeline (score, review page, analysis) is unchanged.

That is what would move the score from "review-grade" to "piano/drum parts
usable"; it does not remove the need for human annotation.

## Reproducibility

- Weights are pinned: BTC to a **commit sha**, harmonix artifacts to **sha256**;
  `models.py` verifies every download and never silently upgrades.
- `--check-updates` reports remote changes without applying them — upgrades are
  an explicit decision, then re-run `05_eval.py` against a reference `.lab` to
  confirm the change is actually an improvement.
- Degradation order is fixed and reported: BTC-170 → madmom ensemble → template
  quality → maj/min; structure → beat-only; melody → bass-only; bass →
  Basic Pitch → pyin; transcription → skipped.

## Model fetching

All remote weights are declared in `skills/chords-analysis/models/manifest.json`
and fetched by `scripts/models.py` (mirror-aware, sha256-verified, resumable,
`--check-updates` HEAD comparison). Mirror defaults to `https://hf-mirror.com`;
override with `CHORDS_HF_MIRROR`.

```sh
PY scripts/models.py --status        # cached artifacts
PY scripts/models.py --fetch-all     # download + verify
PY scripts/models.py --check-updates # detect remote updates without downloading
```

If a fetch fails nothing breaks: structure falls back to beat tracking, melody to
bass-only, large-vocab to template quality refinement, and finally to raw
maj/min chords — the pipeline always reports which layer degraded.

## The chords-v1 format

See [`skills/chords-analysis/references/schema.md`](skills/chords-analysis/references/schema.md).
Snapshot:

```json
{
  "schema": "chords-v1",
  "song": { "title": "...", "source_file": "audio/xxx.flac", "duration_sec": 213.4,
            "engine": "madmom-dcchord", "key": "C major (inferred, heuristic)" },
  "chords": [
    { "start": 1.6, "end": 2.5, "label": "F:maj", "bar": 1, "beat": 1 }
  ],
  "stats": { "total_chords": 87, "unique_labels": ["C:maj", "G:maj"],
             "chord_histogram": { "C:maj": 21.0, "G:maj": 18.0 } }
}
```

Labels follow the MIREX chord format (`root:quality` — maj, min, maj7, min7, 7,
dim, aug, sus2, sus4, hdim7, `N` for no-chord).

## Pluggable engines

Chord recognition is **accuracy-first and engine-agnostic**. No engine is 100%
accurate, so:

- Two independent engines run by default (ensemble) and cross-validate each other;
  disagreement intervals are flagged `confidence: "conflict"` in the JSON.
- All engines are registered in `skills/chords-analysis/scripts/engines.py`.
  Adding a future/better model (ChordFormer, BTC, a commercial API…) means writing
  one `detect(audio) -> [{"start","end","label"}]` function and one registration
  line — the rest of the pipeline never changes. `--engine a,b,c` ensembles any
  combination; `--list-engines` shows what's available.

Interpretation of `ensemble.agreement`: ≥0.85 trustworthy · 0.65–0.85 hedge your
wording · <0.65 suggest manual review (common for jazz / dense electronic /
percussion-only material).

## License

MIT
