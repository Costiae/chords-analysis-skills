#!/usr/bin/env python3
"""Step 1: 音频 -> 原始和弦区间(.lab + raw json)。

准确性策略(可插拔引擎,见 engines.py):
  - 默认 ensemble:同时运行两个独立引擎交叉验证,一致区域可信度高,
    分歧区域标 conflict,由下游(LLM 分析)重点复核或提示用户人工确认。
  - 新增/更换引擎只改 engines.py 注册表,本脚本不动。

用法:
  python 01_extract.py <audio文件或目录> [--outdir out]
         [--engine ensemble|<引擎名>...]   # 逗号分隔多个引擎即 ensemble

输出: out/<name>.raw.json
  {
    "schema": "chords-raw-v1",
    "song": {...},
    "chords": [ {"start","end","label"(主引擎)}, ... ],
    "ensemble": {                            # 仅多引擎模式
       "engines": ["madmom-dcchord","madmom-cnn"],
       "agreement": 0.83,                    # 加权时长一致率
       "intervals": [ {"start","end","agree":bool,
                       "<engineA>": "C:maj", "<engineB>": "A:min"}, ... ]
    }
  }
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import engines as ENG

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def run_engine(audio: Path, name: str, detect):
    try:
        chords = detect(audio)
        return [{"start": round(c["start"], 3), "end": round(c["end"], 3),
                 "label": c["label"]} for c in chords]
    except Exception as e:
        print(f"[{audio.name}] engine {name} failed: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return None


def quality_compatible(a: str, b: str) -> bool:
    """实质一致:完全相同,或仅 maj/min 混淆(最常见且影响最小的分歧)。"""
    if a == b:
        return True
    ra, qa = (a.split(":", 1) + [""])[:2]
    rb, qb = (b.split(":", 1) + [""])[:2]
    return ra == rb and {qa, qb} <= {"maj", "min"}


def build_ensemble_intervals(results: dict):
    """合并多引擎区间到统一时间轴,标注每段是否全票一致。"""
    events = sorted({c["start"] for r in results.values() for c in r} |
                    {c["end"] for r in results.values() for c in r})

    def dominant(chords, s, e):
        weights = {}
        for c in chords:
            ov = min(c["end"], e) - max(c["start"], s)
            if ov > 0:
                weights[c["label"]] = weights.get(c["label"], 0.0) + ov
        return max(weights, key=weights.get) if weights else "N"

    names = list(results)
    intervals, agree_dur, total_dur = [], 0.0, 0.0
    for s, e in zip(events, events[1:]):
        if e - s < 0.05:
            continue
        votes = {n: dominant(results[n], s, e) for n in names}
        ok = all(quality_compatible(votes[names[0]], v) for v in votes.values())
        intervals.append({"start": round(s, 3), "end": round(e, 3),
                          "agree": ok, **votes})
        total_dur += e - s
        if ok:
            agree_dur += e - s
    return intervals, round(agree_dur / total_dur, 3) if total_dur else 0.0


def audio_duration(audio: Path):
    try:
        import librosa
        return round(float(librosa.get_duration(path=str(audio))), 2)
    except Exception:
        return None


def resample_44k(audio: Path, tmpdir: Path) -> Path:
    """madmom 频谱链按 44.1kHz 设计;hi-res 音频(48k/96k)会降低识别精度,
    统一重采样到 44.1k 单声道再分析(与 01b_structure.py 保持一致)。"""
    import librosa, soundfile as sf
    y, sr = librosa.load(str(audio), sr=None, mono=True)
    if abs(sr - 44100) < 1:
        return audio
    tmpdir.mkdir(parents=True, exist_ok=True)
    dest = tmpdir / f"{audio.stem}.44k.wav"
    if not dest.exists():
        y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sf.write(str(dest), y44, 44100)
        print(f"[{audio.name}] resampled {sr}Hz -> 44100Hz for analysis", flush=True)
    return dest


def process(audio: Path, outdir: Path, engine_arg: str):
    orig_stem = audio.stem
    audio = resample_44k(audio, outdir / ".resampled")
    # 解析引擎选择
    if engine_arg == "ensemble":
        chosen = ENG.default_ensemble()
    else:
        names = [n.strip() for n in engine_arg.split(",")]
        chosen = ENG.pick(names)
        if not chosen:
            print(f"no known engines in '{engine_arg}'. available: "
                  f"{', '.join(e.name for e in ENG.ENGINES)}", file=sys.stderr)
            return False
    outdir.mkdir(parents=True, exist_ok=True)

    results, primary_name = {}, None
    for name, detect in chosen:
        r = run_engine(audio, name, detect)
        if r:
            results[name] = r
            primary_name = primary_name or name
    if not results:
        print(f"[{orig_stem}] FAILED: all engines failed", file=sys.stderr)
        return False
    if primary_name != chosen[0][0]:
        print(f"[{orig_stem}] primary engine {chosen[0][0]} unavailable, "
              f"using {primary_name}", flush=True)

    result = results[primary_name]
    ensemble = None
    if len(results) >= 2:
        intervals, agreement = build_ensemble_intervals(results)
        ensemble = {"engines": list(results), "agreement": agreement,
                    "intervals": intervals}
        print(f"[{audio.name}] ensemble agreement: {agreement:.1%} "
              f"({sum(1 for i in intervals if not i['agree'])} conflict intervals)",
              flush=True)

    dur = audio_duration(audio)
    if dur and result and result[-1]["end"] < dur:
        result[-1]["end"] = dur

    doc = {
        "schema": "chords-raw-v1",
        "song": {"title": orig_stem, "source_file": str(audio),
                 "duration_sec": dur,
                 "engine": "+".join(results) if ensemble else primary_name},
        "chords": result,
    }
    if ensemble:
        doc["ensemble"] = ensemble
    dest = outdir / f"{orig_stem}.raw.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{orig_stem}] OK {len(result)} chords (engine={doc['song']['engine']}) "
          f"-> {dest}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--engine", default="ensemble",
                    help="ensemble(默认) 或逗号分隔的引擎名。可用: "
                         + ", ".join(e.name for e in ENG.ENGINES))
    ap.add_argument("--list-engines", action="store_true")
    args = ap.parse_args()

    if args.list_engines:
        for e in ENG.ENGINES:
            print(f"{e.name}  (tier {e.tier})  {e.note}")
        return

    if not args.input:
        ap.error("input is required (unless --list-engines)")
    src = Path(args.input)
    files = find_audio(src)
    if not files:
        print(f"no audio files found under {src}", file=sys.stderr)
        sys.exit(1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ok = sum(process(f, outdir, args.engine) for f in files)
    print(f"done: {ok}/{len(files)} succeeded")
    sys.exit(0 if ok == len(files) else 2)


if __name__ == "__main__":
    main()
