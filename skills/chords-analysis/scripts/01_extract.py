#!/usr/bin/env python3
"""Step 1: 音频 -> 原始和弦区间(.lab + raw json)。

引擎可插拔,按顺序尝试:
  1. madmom  DeepChromaChordRecognitionProcessor (经典,精度高,纯 pip)
  2. madmom  CNNChordFeatureProcessor + CRF (maj/min,更快)

用法:
  python 01_extract.py <audio文件或目录> [--outdir out] [--engine auto|madmom-dcchord|madmom-cnn]

依赖(在 analysis/.venv 中): madmom, librosa(仅音频解码)
输出: out/<name>.raw.json  ( [{"start":s,"end":e,"label":"C:maj"}, ...] )
"""
import argparse, json, sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def run_madmom(audio: Path, engine: str):
    import numpy as np
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
    out = []
    for start, end, label in decoded:
        out.append({"start": round(float(start), 3), "end": round(float(end), 3),
                    "label": str(label)})
    return out


def audio_duration(audio: Path):
    try:
        import librosa
        return round(float(librosa.get_duration(path=str(audio))), 2)
    except Exception:
        return None


def process(audio: Path, outdir: Path, engine: str):
    result = None
    engines = ["madmom-dcchord", "madmom-cnn"] if engine == "auto" else [engine]
    errors = []
    for eng in engines:
        try:
            print(f"[{audio.name}] trying engine: {eng}", flush=True)
            result = run_madmom(audio, eng)
            used = eng
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
                 "duration_sec": dur, "engine": used},
        "chords": result,
    }
    dest = outdir / f"{audio.stem}.raw.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{audio.name}] OK {len(result)} chords -> {dest} (engine={used})", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "madmom-dcchord", "madmom-cnn"])
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
