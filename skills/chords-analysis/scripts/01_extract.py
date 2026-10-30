#!/usr/bin/env python3
"""Step 1: 音频 -> 原始和弦区间(.lab + raw json)。

准确性优先策略:默认同时运行两个 madmom 引擎做交叉验证,
  - madmom-dcchord : DeepChroma + 深度色度和弦解码(整体精度更高,主引擎)
  - madmom-cnn     : CNN特征 + CRF(maj/min,独立模型)
两个引擎独立训练、架构不同,一致的区域可信度高;不一致的区域在输出中标记
"conflict",由下游(LLM 分析)重点复核或提示用户人工确认。

用法:
  python 01_extract.py <audio文件或目录> [--outdir out] [--engine auto|ensemble|madmom-dcchord|madmom-cnn]

输出: out/<name>.raw.json
  {
    "schema": "chords-raw-v1",
    "song": {...},
    "chords": [ {"start","end","label"(主引擎)}, ... ],
    "ensemble": {                            // 仅 ensemble 模式
       "engines": ["madmom-dcchord","madmom-cnn"],
       "agreement": 0.83,                    // 加权时长一致率
       "intervals": [ {"start","end","agree": true|false,
                       "dcchord": "C:maj", "cnn": "A:min"}, ... ]
    }
  }
"""
import argparse, json, sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def run_madmom(audio: Path, engine: str):
    if engine == "madmom-cnn":
        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
        feats = CNNChordFeatureProcessor()(str(audio))
        decoded = CRFChordRecognitionProcessor()(feats)
    else:
        from madmom.audio.chroma import DeepChromaProcessor
        from madmom.features.chords import DeepChromaChordRecognitionProcessor
        dcp = DeepChromaProcessor()
        decode = DeepChromaChordRecognitionProcessor()
        decoded = decode(dcp(str(audio)))
    return [{"start": round(float(s), 3), "end": round(float(e), 3), "label": str(l)}
            for s, e, l in decoded]


def quality_compatible(a: str, b: str) -> bool:
    """和弦是否"实质一致":完全相同,或仅 maj/min 混淆(最常见且影响最小的分歧)。"""
    if a == b:
        return True
    ra, qa = (a.split(":", 1) + [""])[:2]
    rb, qb = (b.split(":", 1) + [""])[:2]
    if ra != rb:
        return False
    return {qa, qb} <= {"maj", "min"}


def build_ensemble_intervals(a, b):
    """把两个引擎的区间合并到统一时间轴,标注每段是否一致。"""
    # 用主引擎(dcchord)区间为骨架;对每段统计 cnn 在该时段的多数标签
    events = sorted({c["start"] for c in a} | {c["end"] for c in a} |
                    {c["start"] for c in b} | {c["end"] for c in b})

    def label_at(chords, t):
        for c in chords:
            if c["start"] <= t < c["end"]:
                return c["label"]
        return "N"

    def dominant_cnn(s, e):
        # cnn 区间与 [s,e) 的重叠时长加权
        weights = {}
        for c in b:
            ov = min(c["end"], e) - max(c["start"], s)
            if ov > 0:
                weights[c["label"]] = weights.get(c["label"], 0.0) + ov
        return max(weights, key=weights.get) if weights else "N"

    intervals, agree_dur, total_dur = [], 0.0, 0.0
    for s, e in zip(events, events[1:]):
        if e - s < 0.05:
            continue
        la, lb = label_at(a, s), dominant_cnn(s, e)
        ok = quality_compatible(la, lb)
        intervals.append({"start": round(s, 3), "end": round(e, 3),
                          "agree": ok, "dcchord": la, "cnn": lb})
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


def process(audio: Path, outdir: Path, engine: str):
    errors = []
    result, ensemble = None, None

    if engine == "ensemble":
        try:
            print(f"[{audio.name}] ensemble: madmom-dcchord + madmom-cnn", flush=True)
            a = run_madmom(audio, "madmom-dcchord")
            b = run_madmom(audio, "madmom-cnn")
            intervals, agreement = build_ensemble_intervals(a, b)
            result = a
            ensemble = {"engines": ["madmom-dcchord", "madmom-cnn"],
                        "agreement": agreement, "intervals": intervals}
            print(f"[{audio.name}] ensemble agreement: {agreement:.1%} "
                  f"({sum(1 for i in intervals if not i['agree'])} conflict intervals)", flush=True)
        except Exception as e:
            errors.append(f"ensemble: {type(e).__name__}: {e}")
            print(f"[{audio.name}] ensemble failed, falling back to single engine", flush=True)

    if result is None:
        for eng in (["madmom-dcchord", "madmom-cnn"] if engine in ("auto", "ensemble")
                    else [engine]):
            try:
                print(f"[{audio.name}] trying engine: {eng}", flush=True)
                result = run_madmom(audio, eng)
                break
            except Exception as e:
                errors.append(f"{eng}: {type(e).__name__}: {e}")
    if result is None:
        print(f"[{audio.name}] FAILED\n  " + "\n  ".join(errors), file=sys.stderr)
        return False

    dur = audio_duration(audio)
    if dur and result and result[-1]["end"] < dur:
        result[-1]["end"] = dur

    doc = {
        "schema": "chords-raw-v1",
        "song": {"title": audio.stem, "source_file": str(audio),
                 "duration_sec": dur, "engine": "ensemble" if ensemble else engine},
        "chords": result,
    }
    if ensemble:
        doc["ensemble"] = ensemble
    dest = outdir / f"{audio.stem}.raw.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{audio.name}] OK {len(result)} chords -> {dest}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--engine", default="ensemble",
                    choices=["ensemble", "auto", "madmom-dcchord", "madmom-cnn"])
    args = ap.parse_args()

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
