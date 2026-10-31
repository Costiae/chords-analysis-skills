#!/usr/bin/env python3
"""Step 1c-b: 低音声部专用提取(电贝斯/低音提琴)。

为什么单独一个模块
------------------
用 pyin(单声部 F0 跟踪)直接跑低音 stem 在真实曲目上会系统性出错,
实测(电贝斯 + demucs bass stem,171 BPM 8 分音符段落):

  * **八度翻转**:拨弦的第 2 谐波常强于基频,跟踪器会在基频与 2 倍频之间跳
    (D2 73 Hz 与 D3 147 Hz 交替出现)。同一件乐器被读成"两件乐器齐奏"。
  * **漏音**:音符只有 100–180 ms,pyin 的滑窗来不及稳定,长音被切碎/吞掉
    (例如 44.0–45.5 s 的 Gb2 保持音整段丢失)。
  * **八度越界**:基频弱时跟踪器会锁到 3/4/5 谐波,输出 C4–B5 这种不可能出现在
    低音轨上的音。

因此低音走**事件式**流程而不是连续跟踪:

  1. 主路径 basic-pitch(onset 驱动,不依赖连续 F0;已在依赖里)
  2. 退化路径 pyin(未安装 basic-pitch 时,保持旧行为)
  3. 音域折返:超出 [C1, G3] 的音按八度折回;超过 2 个八度视为伪音丢弃
  4. 单声部化:重叠取权重(幅度×时值)最高者,弱音被截断而不是叠加
  5. 谐波复核:头尾都显示 f0/2 强于 f0 的音,降一个八度(2 次谐波误判)
  6. 质量自检:分块比较"音符音级分布"与"音频 chroma",给出 0..1 一致性

输出保持 voices-v1 的字段(start/end/midi/name),追加 amp/octave_fix 与
method/register/chroma_agreement 等诊断字段,02_format 无需改动。

CLI:
  python bass.py <bass.wav> [--method auto|basic-pitch|pyin] [--diagnose]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SR = 44100
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 低音轨的合理音域(C1..G3);超出即按八度折返
BASS_LO, BASS_HI = 24, 55
MAX_OCTAVE_SHIFT = 2
MIN_NOTE_DUR = 0.05

_MODEL = None


def pc_name(midi: int) -> str:
    return f"{PC_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


# ---------------------------------------------------------------- 音符来源

def _basic_pitch_model():
    """模型加载很贵,进程内只建一次。"""
    global _MODEL
    if _MODEL is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


def notes_from_basic_pitch(path, onset_thr=0.5, frame_thr=0.3, min_note_ms=60):
    """basic-pitch:onset 驱动的事件检测,对拨弦低音比连续 F0 跟踪可靠。"""
    from basic_pitch.inference import predict
    model = _basic_pitch_model()
    _, _, events = predict(str(path), model, onset_threshold=onset_thr,
                           frame_threshold=frame_thr,
                           minimum_note_length=min_note_ms)
    notes = []
    for ev in events:
        start, end, pitch, amp = ev[0], ev[1], ev[2], ev[3]
        notes.append({"start": round(float(start), 3),
                      "end": round(float(end), 3),
                      "midi": int(pitch),
                      "amp": round(float(amp), 3)})
    return notes


def notes_from_pyin(y, sr, min_dur=0.06):
    """退化路径:连续 F0 跟踪 + 同音高合并(旧行为)。"""
    import librosa
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("A0"), fmax=librosa.note_to_hz("C4"),
        sr=sr, frame_length=4096)
    times = librosa.times_like(f0, sr=sr, hop_length=512)
    hop = 512.0 / sr
    notes, cur = [], None
    for i, (f, v) in enumerate(zip(f0, voiced)):
        if v and f > 0:
            midi = int(round(69 + 12 * np.log2(f / 440.0)))
            if cur and cur["midi"] == midi:
                cur["end"] = times[i] + hop
            else:
                if cur:
                    notes.append(cur)
                cur = {"start": times[i], "midi": midi, "end": times[i] + hop}
        elif cur:
            notes.append(cur)
            cur = None
    if cur:
        notes.append(cur)
    return [{"start": round(n["start"], 3), "end": round(n["end"], 3),
             "midi": n["midi"]} for n in notes
            if n["end"] - n["start"] >= min_dur]


# ---------------------------------------------------------------- 清理

def _weight(n):
    dur = max(1e-3, n["end"] - n["start"])
    return dur * float(n.get("amp") or 1.0)


def fold_register(notes, lo=BASS_LO, hi=BASS_HI, max_octaves=MAX_OCTAVE_SHIFT):
    """把音域外的音按八度折回;偏差超过 max_octaves 的视为伪音丢弃。

    低音轨出现 C4 以上几乎总是跟踪到了高次谐波。保留音级、折回八度,
    比直接删掉更有用(音级信息仍然参与转位判定);折太远则是垃圾。
    """
    kept, folded, dropped = [], 0, 0
    for n in notes:
        m = n["midi"]
        steps = 0
        while m > hi and steps < max_octaves:
            m -= 12
            steps += 1
        while m < lo and steps < max_octaves:
            m += 12
            steps += 1
        if m > hi or m < lo:
            dropped += 1
            continue
        if steps:
            n["octave_fix"] = 12 * steps if m > n["midi"] else -12 * steps
            folded += 1
        n["midi"] = m
        kept.append(n)
    return kept, folded, dropped


def _amp_at(spec, freqs, f):
    return float(np.interp(f, freqs, spec))


def _sub_ratio(y, sr, start, dur, midi, nfft=16384):
    """半频(f0/2)与基频(f0)的谱幅比;窗口太短返回 None。"""
    a = int(start * sr)
    b = min(a + int(dur * sr), y.size)
    if b - a < 512:
        return None
    w = y[a:b] * np.hanning(b - a)
    m = np.abs(np.fft.rfft(w, n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    f0 = midi_to_hz(midi)
    lo = _amp_at(m, freqs, f0 / 2.0)
    hi = _amp_at(m, freqs, f0)
    return lo / max(hi, 1e-9)


def fix_subharmonics(y, sr, notes, lo=BASS_LO, threshold=0.75, min_midi=48):
    """基频弱于半频 -> 降一个八度。

    单根拨弦不可能让 f0/2 的能量高于 f0 本身;若音符的头尾两个窗口都如此,
    那它几乎一定是**低一个八度的音的 2 次谐波**(实测 D2 73 Hz / D3 147 Hz
    交替出现的就是这个)。音级不变,只修八度,所以对转位判定无影响。
    头尾都判定,是为了排除"前一个音的余响还在振"这种假阳性。

    只处理 C3 以上:低音区(八度 2)能量里混着底鼓与分轨伪影,把 A#2 折成
    A#1 会掉到电贝斯空弦 E1 以下,风险大于收益,故不自动处理。
    """
    fixed = 0
    for n in notes:
        if n["midi"] < min_midi:
            continue
        dur = n["end"] - n["start"]
        if dur < 0.06:
            continue
        head = _sub_ratio(y, sr, n["start"] + 0.02, min(0.06, dur * 0.5), n["midi"])
        tail = _sub_ratio(y, sr, max(n["start"] + 0.02, n["end"] - 0.07),
                          min(0.06, dur * 0.5), n["midi"])
        if head is None or tail is None:
            continue
        if head >= threshold and tail >= threshold and n["midi"] - 12 >= max(lo, 36):
            n["midi"] -= 12
            n["octave_fix"] = n.get("octave_fix", 0) - 12
            n["octave_fix_reason"] = "subharmonic"
            fixed += 1
    return notes, fixed


def monophonize(notes, min_dur=MIN_NOTE_DUR):
    """单声部化:重叠时保留权重高者,弱音被截断(不是叠加)。"""
    notes = sorted(notes, key=lambda n: (n["start"], -_weight(n)))
    out = []
    for n in notes:
        keep = True
        while out and n["start"] < out[-1]["end"] - 1e-3:
            prev = out[-1]
            if _weight(n) > _weight(prev):
                prev["end"] = n["start"]
                if prev["end"] - prev["start"] < min_dur:
                    out.pop()
                else:
                    break
            else:
                n["start"] = prev["end"]
                if n["end"] - n["start"] < min_dur:
                    keep = False
                break
        if keep and n["end"] - n["start"] >= min_dur:
            out.append(n)
    return out


# ---------------------------------------------------------------- 自检

def chroma_agreement(y, sr, notes, block=2.0, hop=2048, fmin="C1"):
    """分块比较"音符音级分布"与"音频 chroma"的余弦一致性(0..1)。

    这是低音层的**自检指标**:音符列表若漏音/错八度,音符分布与音频不符,
    分值会掉下来。它不是精度分数(音频 chroma 本身含谐波),但可用于比较
    不同提取方法在同一段上的可信度。
    """
    import librosa
    if not notes or y.size == 0:
        return 0.0
    try:
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop,
                                       fmin=librosa.note_to_hz(fmin),
                                       bins_per_octave=24)
    except Exception:
        return 0.0
    times = librosa.times_like(c[0], sr=sr, hop_length=hop)
    sims = []
    t_end = float(times[-1]) if len(times) else 0.0
    t = 0.0
    while t < t_end:
        sel = [n for n in notes if n["start"] < t + block and n["end"] > t]
        frames = (times >= t) & (times < t + block)
        if sel and frames.any():
            aud = c[:, frames].mean(axis=1)
            nov = np.zeros(12)
            for n in sel:
                ov = min(n["end"], t + block) - max(n["start"], t)
                nov[n["midi"] % 12] += max(0.0, ov)
            if aud.sum() > 0 and nov.sum() > 0:
                sims.append(float(aud @ nov / (np.linalg.norm(aud) * np.linalg.norm(nov))))
        t += block
    return round(float(np.mean(sims)), 3) if sims else 0.0


def register_stats(notes):
    oct_sec = {}
    for n in notes:
        oct_sec[n["midi"] // 12 - 1] = round(
            oct_sec.get(n["midi"] // 12 - 1, 0.0) + n["end"] - n["start"], 2)
    return dict(sorted(oct_sec.items()))


# ---------------------------------------------------------------- 入口

def extract(bass_path, sr=SR, method="auto", cached_notes=None, diagnose=True):
    """低音 stem -> 音符列表 + 诊断。

    method: auto(默认:basic-pitch -> pyin) / basic-pitch / pyin
    cached_notes: 已转录的 bass 音符(例如 01e 的产物),避免重复推理
    """
    used, notes, y = None, [], None
    if cached_notes:
        notes = [dict(n) for n in cached_notes if n.get("end", 0) > n.get("start", 0)]
        used = "basic-pitch (cached)"
    elif method in ("auto", "basic-pitch"):
        try:
            notes = notes_from_basic_pitch(bass_path)
            used = "basic-pitch"
        except Exception as e:
            if method == "basic-pitch":
                raise
            print(f"  [bass] basic-pitch unavailable ({type(e).__name__}: {e}), "
                  f"falling back to pyin", file=sys.stderr)
    if not notes and method in ("auto", "pyin"):
        import librosa
        y, _ = librosa.load(str(bass_path), sr=sr, mono=True)
        notes = notes_from_pyin(y, sr)
        used = "pyin"
    if not notes:
        return {"source": "bass", "method": "none", "notes": [],
                "reason": "no notes detected"}

    raw = len(notes)
    notes, folded, dropped = fold_register(notes)
    notes = monophonize(notes)
    if y is None:
        import librosa
        y, _ = librosa.load(str(bass_path), sr=sr, mono=True)
    notes, sub_fixed = fix_subharmonics(y, sr, notes)
    for n in notes:
        n["name"] = pc_name(n["midi"])

    doc = {
        "source": "bass",
        "method": used,
        "notes": notes,
        "raw_notes": raw,
        "octave_folded": folded,
        "register_dropped": dropped,
        "subharmonic_fixes": sub_fixed,
        "register": register_stats(notes),
    }
    if diagnose:
        import librosa
        doc["chroma_agreement"] = chroma_agreement(y, sr, notes)
        env = np.abs(y)
        if env.size:
            frames = librosa.util.frame(env, frame_length=2048, hop_length=1024)
            rms = np.sqrt((frames ** 2).mean(axis=0))
            doc["crest"] = round(float(rms.max() / max(np.median(rms), 1e-9)), 2)
    return doc


def main():
    ap = argparse.ArgumentParser(description="bass note extraction + diagnostics")
    ap.add_argument("bass", help="bass stem (wav/flac)")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "basic-pitch", "pyin"])
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--json", default=None, help="write the result json here")
    args = ap.parse_args()

    doc = extract(Path(args.bass), sr=args.sr, method=args.method,
                  diagnose=args.diagnose)
    notes = doc.get("notes") or []
    print(f"method      : {doc.get('method')}")
    print(f"notes       : {len(notes)} (raw {doc.get('raw_notes')})")
    print(f"octave fold : {doc.get('octave_folded')}  dropped {doc.get('register_dropped')}")
    print(f"subharmonic : {doc.get('subharmonic_fixes')}")
    print(f"register(s) : {doc.get('register')}")
    if "chroma_agreement" in doc:
        print(f"chroma agree: {doc['chroma_agreement']}   crest {doc.get('crest')}")
    if notes:
        pcs = {}
        for n in notes:
            pcs[PC_NAMES[n["midi"] % 12]] = pcs.get(PC_NAMES[n["midi"] % 12], 0) + 1
        print("pitch class : " + ", ".join(
            f"{k}={v}" for k, v in sorted(pcs.items(), key=lambda x: -x[1])))
    if args.json:
        Path(args.json).write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
