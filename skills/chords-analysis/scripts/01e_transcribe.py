#!/usr/bin/env python3
"""Step 1e: 分轨转写(扒谱) —— Basic Pitch per-stem -> MIDI + 音符 JSON。

为什么分轨:整混合音的多音高转写精度差;demucs 已给出 vocals/bass/other/drums,
人声轨近似单旋律、bass 轨近似单低音、other 轨承载和声乐器。分轨转写再合并,
比一次转写整曲可靠得多,而且天然分声部(可直接当多声部扒谱的骨架)。

用法:
  python 01e_transcribe.py <audio> [--outdir out] [--stems vocals,bass,other]
                            [--onset-threshold 0.5] [--frame-threshold 0.3] [--min-note-ms 60]

依赖: basic-pitch(缺失时整个步骤跳过,不影响其它层)
输出: out/<name>.midi(合并 MIDI)、out/<name>.transcription.json(音符事件)
"""
import argparse, json, sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def stems_for(audio: Path, outdir: Path) -> dict:
    """复用 01c_voices.py 的 demucs 缓存;缺失则用原音频(单轨退化)。

    01c 里 demucs 跑在重采样后的文件上,缓存目录名可能是 `<stem>` 或 `<stem>.44k`。"""
    names = ["vocals", "bass", "drums", "other"]
    candidates = [audio.stem, f"{audio.stem}.44k"]
    for c in candidates:
        d = outdir / ".stems" / c
        if all((d / f"{n}.wav").exists() for n in names):
            return {n: d / f"{n}.wav" for n in names}
    return {"mix": audio}


def monophonize(notes, min_gap=0.02):
    """把重叠的音符压成单声部:重叠时保留 velocity 最高者(适用于人声/贝斯轨)。"""
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n["start"], -n["velocity"]))
    out = [notes[0]]
    for n in notes[1:]:
        prev = out[-1]
        if n["start"] < prev["end"] - min_gap:
            # 重叠:保留更强的,并把边界对齐
            if n["velocity"] > prev["velocity"]:
                prev["end"] = max(prev["end"], n["start"])
                out[-1] = n
                n["start"] = prev.get("start", n["start"])
            else:
                prev["end"] = min(prev["end"], max(n["start"], prev["start"] + 0.01))
        else:
            out.append(n)
    return [n for n in out if n["end"] - n["start"] >= 0.03]


def transcribe(path: Path, onset_thr=0.5, frame_thr=0.3, min_note_ms=60):
    from basic_pitch.inference import predict, Model
    from basic_pitch import ICASSP_2022_MODEL_PATH

    model = Model(ICASSP_2022_MODEL_PATH)
    model_output, midi_data, note_events = predict(
        str(path), model, onset_threshold=onset_thr,
        frame_threshold=frame_thr, minimum_note_length=min_note_ms)
    notes = []
    for ev in note_events:
        start, end, pitch, amplitude, pitch_bend = ev
        notes.append({"start": round(float(start), 3), "end": round(float(end), 3),
                      "midi": int(pitch), "velocity": round(float(amplitude), 3)})
    return notes, midi_data


def process(audio: Path, outdir: Path, stems_wanted, onset_thr, frame_thr, min_note_ms,
            mono_stems=("vocals", "bass")):
    try:
        import basic_pitch  # noqa: F401
    except Exception as e:
        print(f"[{audio.name}] basic-pitch unavailable ({type(e).__name__}); "
              f"skipping transcription layer", file=sys.stderr)
        return False

    stems = stems_for(audio, outdir)
    wanted = [s for s in stems_wanted if s in stems] or list(stems)
    all_notes, merged_midi = [], None
    per_stem = {}
    outdir.mkdir(parents=True, exist_ok=True)

    for stem in wanted:
        try:
            print(f"[{audio.name}] transcribing {stem} ...", flush=True)
            notes, midi = transcribe(stems[stem], onset_thr, frame_thr, min_note_ms)
            raw_count = len(notes)
            if stem in mono_stems:
                notes = monophonize(notes)
            for n in notes:
                n["stem"] = stem
            per_stem[stem] = {"notes": len(notes), "raw_notes": raw_count}
            all_notes.extend(notes)
            if midi is not None:
                midi_dest = outdir / f"{audio.stem}.{stem}.midi"
                try:
                    midi.write(str(midi_dest))
                    per_stem[stem]["midi"] = str(midi_dest)
                except Exception:
                    pass
            if merged_midi is None:
                merged_midi = midi
            elif midi is not None:
                try:
                    for inst in midi.instruments:
                        merged_midi.instruments.append(inst)
                except Exception:
                    pass
        except Exception as e:
            print(f"[{audio.name}] {stem} failed: {type(e).__name__}: {e}", file=sys.stderr)
            per_stem[stem] = {"error": f"{type(e).__name__}: {e}"}

    if not all_notes:
        return False

    outdir.mkdir(parents=True, exist_ok=True)
    midi_dest = outdir / f"{audio.stem}.midi"
    if merged_midi is not None:
        try:
            merged_midi.write(str(midi_dest))
        except Exception as e:
            print(f"  midi write failed: {e}", file=sys.stderr)

    doc = {
        "schema": "transcription-v1",
        "engine": "basic-pitch",
        "stems": per_stem,
        "notes": sorted(all_notes, key=lambda n: n["start"]),
        "midi": str(midi_dest) if midi_dest.exists() else None,
    }
    dest = outdir / f"{audio.stem}.transcription.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{audio.name}] transcription: {len(all_notes)} notes "
          f"({', '.join(k + '=' + str(v.get('notes', 'x')) for k, v in per_stem.items())})"
          f" -> {dest}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--stems", default="vocals,bass,other")
    ap.add_argument("--onset-threshold", type=float, default=0.5)
    ap.add_argument("--frame-threshold", type=float, default=0.3)
    ap.add_argument("--min-note-ms", type=int, default=60)
    args = ap.parse_args()
    src = Path(args.input)
    files = find_audio(src)
    if not files:
        print(f"no audio files under {src}", file=sys.stderr)
        sys.exit(1)
    outdir = Path(args.outdir)
    stems = [s.strip() for s in args.stems.split(",") if s.strip()]
    ok = sum(process(f, outdir, stems, args.onset_threshold,
                     args.frame_threshold, args.min_note_ms) for f in files)
    print(f"done: {ok}/{len(files)} succeeded")
    sys.exit(0 if ok == len(files) else 2)


if __name__ == "__main__":
    main()
