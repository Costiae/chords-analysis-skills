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

```sh
cd skills/chords-analysis
python scripts/setup_env.py   # prints the python to use (PY below)
```

The script auto-routes: Python ≤3.9 → local `.venv`; Python ≥3.10 → conda env
with Python 3.9 (madmom 0.16.1 only supports ≤3.9, `numpy<1.24`, `setuptools<81`).
All later commands: `PY -W ignore scripts/01_extract.py ...` (the `-W ignore`
suppresses madmom's pkg_resources deprecation warning).

**Verified in the wild** (Windows, Python 3.14 host → conda py3.9, Oct 2026):
a 4:31 FLAC ran through the full pipeline — 155 chords, ensemble agreement 76.1%,
report generated from the JSON.

## Pipeline

```
audio/xxx.flac
  │  scripts/01_extract.py    chord recognition (madmom DeepChroma + CNN-CRF ensemble)
  ▼
out/xxx.raw.json               raw engine output + agreement/conflict intervals
out/xxx.struct.json            ← 01b_structure.py: segments, beats, BPM, meter
out/xxx.voices.json            ← 01c_voices.py: melody notes/key evidence, bass notes
  │  scripts/02_format.py     fuse into chords-v1: bar/beat/section per chord,
  │                           three-source key inference, section-local keys,
  │                           enharmonic spelling, slash chords
  ▼
out/xxx.chords.json    ★ standard chords-v1 (authoritative) + xxx.lab (MIREX)
  │  scripts/03_summarize.py  merged timeline + harmonic rhythm stats
  │  scripts/04_map.py        harmonic map: bar grid + Roman numerals + n-grams
  ▼
out/xxx.map.md + out/xxx.analysis.md
```

## What a complete analysis covers

| Layer | Source | Output |
|---|---|---|
| Chords | madmom ensemble (2 engines, cross-validated) | onset/offset + confidence |
| Bar grid | all-in-one downbeats + meter | `bar` / `beat` per chord |
| Sections | all-in-one functional labels | intro/verse/chorus/bridge/outro |
| Tempo | all-in-one | BPM + meter |
| Melody | demucs vocals + pyin | notes, pitch-class profile, phrase endings |
| Bass / inversions | demucs bass + pyin | slash chords (`C:maj/E`) |
| Chord quality | `01d_quality.py` chroma templates (offline) | maj7/min7/7/sus/dim/aug, 13 qualities |
| Key / mode | chords + melody + phrase anchors | candidates with evidence scores |
| Modulations | per-section key estimation | `song.modulations`, per-segment keys |
| Progressions | n-gram mining | repeated progressions with Roman numerals |

Honest gaps (not yet automated): instrument identification beyond the four
demucs stems, tempo curves for rubato passages, and full polyphonic
transcription (melody is monophonic pyin on the vocal stem). Chord qualities are
template-inferred offline — a neural large-vocabulary model can be registered in
`scripts/engines.py` (optional slot `chord-lv` in `models/manifest.json`) and
will take precedence, with the template path as automatic fallback.

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
