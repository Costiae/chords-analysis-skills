#!/usr/bin/env python3
"""Step 1b: 音频 -> 结构分析(段落/节拍/下拍/BPM)。

引擎:all-in-one-infer (openmirlab fork of mir-aidj allin1, ISMIR23)
  - 纯 PyTorch 实现,无需 NATTEN 编译,Windows/macOS/Linux 皆可
  - 输出 functional segments(intro/verse/chorus/bridge/outro...)+ beats + BPM
  - 首次运行会从 HuggingFace 下载模型权重(数百 MB)

与和弦识别(01_extract)相互独立,产物在 02_format 中合入 chords-v1。
若 allin1 不可用,回退到 madmom RNNBeatDetector(仅节拍,无段落标签)。

用法:
  python 01b_structure.py <audio文件或目录> [--outdir out]

输出: out/<name>.struct.json
  {
    "schema": "struct-v1",
    "bpm": 100.0,
    "beats": [t1, t2, ...],
    "downbeats": [t1, t2, ...],
    "segments": [{"start","end","label"}],
    "engine": "allin1|madmom-beats|none"
  }
"""
import argparse, json, sys
from pathlib import Path

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def run_allin1(audio: Path):
    import allin1_infer as allin1
    r = allin1.analyze(str(audio), keep_byproducts=False)
    return {
        "schema": "struct-v1",
        "bpm": float(r.bpm) if r.bpm else None,
        "beats": [round(float(b), 3) for b in (r.beats or [])],
        "downbeats": [round(float(b), 3) for b in (r.downbeats or [])],
        "segments": [{"start": round(float(s.start), 3), "end": round(float(s.end), 3),
                      "label": str(s.label)} for s in (r.segments or [])],
        "engine": "allin1",
    }


def run_madmom_beats(audio: Path):
    """回退:madmom 节拍跟踪(无段落标签)。"""
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    act = RNNBeatProcessor()(str(audio))
    beats = DBNBeatTrackingProcessor(fps=100)(act)
    return {
        "schema": "struct-v1",
        "bpm": None,
        "beats": [round(float(b), 3) for b in beats],
        "downbeats": [],
        "segments": [],
        "engine": "madmom-beats",
    }


def resample_44k(audio: Path, tmpdir: Path) -> Path:
    """allin1/madmom 频谱链按 44.1kHz 设计(81 filterbank bins)。
    非 44.1k 输入(48k/96k hi-res 音频)会导致 bins 数不匹配而崩溃或精度下降,
    统一重采样到 44.1k 单声道再分析。"""
    import librosa, soundfile as sf
    y, sr = librosa.load(str(audio), sr=None, mono=True)
    if abs(sr - 44100) < 1:
        return audio
    dest = tmpdir / f"{audio.stem}.44k.wav"
    if not dest.exists():
        y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sf.write(str(dest), y44, 44100)
        print(f"[{audio.name}] resampled {sr}Hz -> 44100Hz for analysis", flush=True)
    return dest


def process(audio: Path, outdir: Path) -> bool:
    orig_stem = audio.stem
    tmpdir = outdir / ".resampled"
    tmpdir.mkdir(parents=True, exist_ok=True)
    audio = resample_44k(audio, tmpdir)
    try:
        print(f"[{orig_stem}] structure: all-in-one-infer ...", flush=True)
        doc = run_allin1(audio)
    except Exception as e:
        print(f"[{orig_stem}] allin1 failed ({type(e).__name__}: {e}), "
              f"falling back to madmom beats", file=sys.stderr, flush=True)
        try:
            doc = run_madmom_beats(audio)
        except Exception as e2:
            print(f"[{orig_stem}] structure FAILED entirely: {e2}", file=sys.stderr)
            return False
    dest = outdir / f"{orig_stem}.struct.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    segs = doc["segments"]
    print(f"[{orig_stem}] OK bpm={doc['bpm']} beats={len(doc['beats'])} "
          f"segments={len(segs)} engine={doc['engine']} -> {dest}", flush=True)
    if segs:
        preview = " ".join(f"{s['label']}[{s['start']:.0f}-{s['end']:.0f}s]" for s in segs[:10])
        print(f"  segments: {preview}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    src = Path(args.input)
    files = find_audio(src)
    if not files:
        print(f"no audio files found under {src}", file=sys.stderr)
        sys.exit(1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = sum(process(f, outdir) for f in files)
    print(f"done: {ok}/{len(files)} succeeded")
    sys.exit(0 if ok == len(files) else 2)


if __name__ == "__main__":
    main()
