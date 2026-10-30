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
python scripts/setup_env.py        # creates .venv, installs madmom + librosa
```

## Pipeline

```
audio/xxx.flac
  │  scripts/01_extract.py    chord recognition (madmom DeepChroma → CNN+CRF, auto-fallback)
  ▼
out/xxx.raw.json               raw engine output
  │  scripts/02_format.py     normalize + merge fragments + duration-weighted stats
  │                           + heuristic key inference (optional bar alignment via --bars)
  ▼
out/xxx.chords.json    ★ standard chords-v1 format (authoritative) + xxx.lab (MIREX)
  │  scripts/03_summarize.py  merged timeline summary
  ▼
LLM analysis → conversational Q&A or out/xxx.analysis.md report
```

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

## Accuracy notes

The core engine is madmom (DeepChroma CNN + CRF decoding; ISMIR 2016/2018 line of
work). It performs well on pop/rock/folk music built from triads. Dense jazz
sevenths, pure percussion, and heavy electronic production are common weak spots —
the `.lab` output is hand-editable; re-run `02_format.py` after manual fixes.

## License

MIT
