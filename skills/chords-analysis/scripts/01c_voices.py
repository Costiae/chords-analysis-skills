#!/usr/bin/env python3
"""Step 1c: 音频 -> 旋律(调式证据) + 低音(转位证据)。

一次 demucs 分离,两份产物:
  - melody: 人声 stem 做 pyin F0 跟踪 -> 音符序列 + 音级时长分布 + 乐句终止音
            -> 调性候选(与和弦直方图交叉验证,解决近关系调误判)
  - bass  : 低音 stem -> 音符序列,供 02_format 推断转位(斜杠和弦)

demucs 产物缓存在 <outdir>/.stems/<name>/,重复运行不再分离。

用法:
  python 01c_voices.py <audio文件或目录> [--outdir out] [--skip-melody] [--skip-bass]

输出: out/<name>.voices.json
  {
    "schema": "voices-v1",
    "stem_model": "htdemucs",
    "melody": {
      "source": "vocals",
      "notes": [{"start","end","midi","name"}],
      "pitch_class_profile": {"A": 42.1, ...},      // 秒
      "phrase_endings": [{"time","name","midi"}],
      "key_candidates": [{"key": "A major", "score": 0.81}, ...]
    },
    "bass": {"source": "bass", "notes": [...]}
  }
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}
SR = 44100
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def resample_44k(audio: Path, tmpdir: Path) -> Path:
    import librosa, soundfile as sf
    y, sr = librosa.load(str(audio), sr=None, mono=True)
    if abs(sr - SR) < 1:
        return audio
    tmpdir.mkdir(parents=True, exist_ok=True)
    dest = tmpdir / f"{audio.stem}.44k.wav"
    if not dest.exists():
        sf.write(str(dest), librosa.resample(y, orig_sr=sr, target_sr=SR), SR)
        print(f"[{audio.name}] resampled {sr}Hz -> {SR}Hz", flush=True)
    return dest


def separate_stems(audio: Path, cache_dir: Path) -> dict:
    """返回 {'vocals': Path, 'bass': Path, ...};已存在则复用。

    demucs_infer.separate_file 返回 (mixture, {stem: tensor[channels, samples]}),
    需要自己落盘。"""
    import demucs_infer
    import soundfile as sf
    dest = cache_dir / audio.stem
    names = ["vocals", "bass", "drums", "other"]
    if all((dest / f"{n}.wav").exists() for n in names):
        return {n: dest / f"{n}.wav" for n in names}
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[{audio.name}] separating stems (demucs) ...", flush=True)
    result = demucs_infer.separate_file(str(audio))
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        stems = result[1]
    elif isinstance(result, dict):
        stems = result
    else:
        raise RuntimeError(f"unexpected demucs output type: {type(result)}")

    for n in names:
        if n not in stems or stems[n] is None:
            raise RuntimeError(f"demucs did not return stem '{n}'")
        arr = stems[n]
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if getattr(arr, "ndim", 1) == 2:
            arr = arr.T          # (channels, samples) -> (samples, channels)
        sf.write(str(dest / f"{n}.wav"), arr, SR)
    return {n: dest / f"{n}.wav" for n in names}


def f0_to_notes(f0, voiced, times, hop_sec, min_dur=0.08):
    """连续同音高的帧合并为音符。"""
    notes = []
    cur = None
    for i, (f, v) in enumerate(zip(f0, voiced)):
        if v and f > 0:
            midi = int(round(69 + 12 * (np.log2(f / 440.0))))
            if cur and cur["midi"] == midi:
                cur["end"] = times[i] + hop_sec
            else:
                if cur:
                    notes.append(cur)
                cur = {"start": times[i], "midi": midi,
                       "end": times[i] + hop_sec}
        else:
            if cur:
                notes.append(cur)
                cur = None
    if cur:
        notes.append(cur)
    out = []
    for n in notes:
        if n["end"] - n["start"] >= min_dur:
            out.append({"start": round(n["start"], 3), "end": round(n["end"], 3),
                        "midi": n["midi"], "name": pc_name(n["midi"])})
    return out


def pc_name(midi):
    octave = midi // 12 - 1
    return f"{PC_NAMES[midi % 12]}{octave}"


def pitch_class_profile(notes):
    prof = {p: 0.0 for p in PC_NAMES}
    for n in notes:
        prof[PC_NAMES[n["midi"] % 12]] += n["end"] - n["start"]
    return {k: round(v, 1) for k, v in sorted(prof.items(), key=lambda x: -x[1])}


# Krumhansl-Kessler 调性轮廓
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def key_candidates(prof, top=4):
    vec = np.array([prof.get(p, 0.0) for p in PC_NAMES], dtype=float)
    if vec.sum() <= 0:
        return []
    vec = vec / vec.sum()
    out = []

    def corr(a, b):
        a = np.asarray(a, float) - np.mean(a)
        b = np.asarray(b, float) - np.mean(b)
        d = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d else 0.0

    for i, root in enumerate(PC_NAMES):
        for mode, prof_ref in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            rotated = np.roll(prof_ref, i)
            out.append({"key": f"{root} {mode}", "score": round(corr(vec, rotated), 3)})
    out.sort(key=lambda x: -x["score"])
    return out[:top]


def phrase_endings(notes, min_gap=0.35):
    """音符间的静默>min_gap 视为乐句结束,取结束音。"""
    ends = []
    for a, b in zip(notes, notes[1:]):
        if b["start"] - a["end"] >= min_gap:
            ends.append({"time": round(a["end"], 2), "name": a["name"], "midi": a["midi"]})
    if notes:
        last = notes[-1]
        ends.append({"time": round(last["end"], 2), "name": last["name"], "midi": last["midi"]})
    return ends


def analyze_melody(audio: Path, stems: dict):
    import librosa
    src = stems["vocals"]
    y, sr = librosa.load(str(src), sr=SR, mono=True)
    if y.size == 0 or float(np.abs(y).max()) < 1e-4:
        raise RuntimeError("vocals stem is silent (instrumental track?)")
    f0, voiced, vprob = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, frame_length=2048)
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    hop_sec = 512 / sr
    notes = f0_to_notes(f0, voiced, times, hop_sec)
    if not notes:
        raise RuntimeError("no melody notes detected in vocals stem")
    prof = pitch_class_profile(notes)
    return {
        "source": "vocals",
        "notes": notes,
        "pitch_class_profile": prof,
        "phrase_endings": phrase_endings(notes),
        "key_candidates": key_candidates(prof),
    }


def analyze_bass(stems: dict):
    import librosa
    y, sr = librosa.load(str(stems["bass"]), sr=SR, mono=True)
    if y.size == 0 or float(np.abs(y).max()) < 1e-4:
        return {"source": "bass", "notes": []}
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C4"),
        sr=sr, frame_length=4096)
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    notes = f0_to_notes(f0, voiced, times, 512 / sr, min_dur=0.06)
    return {"source": "bass", "notes": notes}


def process(audio: Path, outdir: Path, do_melody=True, do_bass=True) -> bool:
    orig_stem = audio.stem
    work = resample_44k(audio, outdir / ".resampled")
    try:
        stems = separate_stems(work, outdir / ".stems")
    except Exception as e:
        print(f"[{orig_stem}] stem separation failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False

    doc = {"schema": "voices-v1", "stem_model": "htdemucs"}
    if do_melody:
        try:
            print(f"[{orig_stem}] melody: pyin on vocals ...", flush=True)
            doc["melody"] = analyze_melody(work, stems)
            kc = doc["melody"]["key_candidates"][:2]
            kc_txt = ", ".join("{} ({})".format(c["key"], c["score"]) for c in kc)
            print(f"  notes={len(doc['melody']['notes'])} key_candidates={kc_txt}",
                  flush=True)
        except Exception as e:
            print(f"[{orig_stem}] melody failed: {type(e).__name__}: {e}", file=sys.stderr)
    if do_bass:
        try:
            print(f"[{orig_stem}] bass: pyin on bass stem ...", flush=True)
            doc["bass"] = analyze_bass(stems)
            print(f"  bass notes={len(doc['bass']['notes'])}", flush=True)
        except Exception as e:
            print(f"[{orig_stem}] bass failed: {type(e).__name__}: {e}", file=sys.stderr)

    if "melody" not in doc and "bass" not in doc:
        return False
    dest = outdir / f"{orig_stem}.voices.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{orig_stem}] OK -> {dest}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--skip-melody", action="store_true")
    ap.add_argument("--skip-bass", action="store_true")
    args = ap.parse_args()
    src = Path(args.input)
    files = find_audio(src)
    if not files:
        print(f"no audio files found under {src}", file=sys.stderr)
        sys.exit(1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ok = sum(process(f, outdir, not args.skip_melody, not args.skip_bass) for f in files)
    print(f"done: {ok}/{len(files)} succeeded")
    sys.exit(0 if ok == len(files) else 2)


if __name__ == "__main__":
    main()
