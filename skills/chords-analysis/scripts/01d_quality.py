#!/usr/bin/env python3
"""Step 1d: 和弦质量(大词汇)推断 —— 不依赖任何下载。

madmom 只给 maj/min 根音与三和弦性质。本步骤在**保留其根音**的前提下,
用区间内 chroma 与 14 类和弦模板做匹配,升级和弦性质:
  maj min dim aug maj7 min7 7 hdim7 dim7 sus2 sus4 6 min6

这是"大词汇模型的离线 fallback":若将来接入神经大词汇模型(manifest 的
chord-lv 槽位),直接覆盖本步骤结果;本步骤始终可用,保证无网也能给出
带七和弦/挂留的分析。

用法:
  python 01d_quality.py <audio> out/<name>.raw.json [--outdir out] [--min-dur 0.35]

输出: out/<name>.quality.json
  {"schema":"quality-v1","engine":"template-chroma",
   "chords":[{"start","end","root","quality","confidence","margin","template_scores"}]}
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import harmony as H

# 模板权重:根音/三音更重要,五音次之,七音/延伸音再次
QUALITY_WEIGHTS = {
    "maj":   {0: 1.0, 4: 0.95, 7: 0.8},
    "min":   {0: 1.0, 3: 0.95, 7: 0.8},
    "dim":   {0: 1.0, 3: 0.9, 6: 0.8},
    "aug":   {0: 1.0, 4: 0.9, 8: 0.8},
    "maj7":  {0: 1.0, 4: 0.9, 7: 0.75, 11: 0.7},
    "min7":  {0: 1.0, 3: 0.9, 7: 0.75, 10: 0.7},
    "7":     {0: 1.0, 4: 0.9, 7: 0.75, 10: 0.7},
    "hdim7": {0: 1.0, 3: 0.9, 6: 0.75, 10: 0.7},
    "dim7":  {0: 1.0, 3: 0.9, 6: 0.75, 9: 0.7},
    "sus2":  {0: 1.0, 2: 0.9, 7: 0.8},
    "sus4":  {0: 1.0, 5: 0.9, 7: 0.8},
    "6":     {0: 1.0, 4: 0.9, 7: 0.75, 9: 0.7},
    "min6":  {0: 1.0, 3: 0.9, 7: 0.75, 9: 0.7},
}

# 只允许"同族"升级:三音决定大小调,不轻易翻转(madmom 的三音判断更可靠)
FAMILY = {
    "maj": "maj", "maj7": "maj", "6": "maj", "sus2": "maj", "sus4": "maj", "aug": "maj",
    "min": "min", "min7": "min", "min6": "min", "hdim7": "min", "dim7": "min", "dim": "min",
    "7": "dom",
}
# 音数越多越可疑(chroma 谐波泄漏常伪造 9/11 音);复杂度惩罚
# 升级阈值按性质分级:越"花"的性质要求证据越强
UPGRADE_MARGIN = {
    "maj7": 0.035, "min7": 0.035, "7": 0.035,
    "dim": 0.05, "aug": 0.06,
    "6": 0.06, "min6": 0.07, "hdim7": 0.07, "dim7": 0.08,
    "sus2": 0.12, "sus4": 0.12,   # 挂留多为旋律经过,证据要求最高
}


def score_quality(chroma_vec, root_idx, quality):
    tpl = np.zeros(12)
    for interval, w in QUALITY_WEIGHTS[quality].items():
        tpl[(root_idx + interval) % 12] = w
    # 余弦相似度(模板稀疏,chroma 归一化)
    nv = np.linalg.norm(chroma_vec)
    nt = np.linalg.norm(tpl)
    if nv == 0 or nt == 0:
        return 0.0
    mag = float(chroma_vec @ tpl / (nv * nt))
    # 惩罚:模板外的音级若能量高则减分(抑制把七和弦误当三和弦/反之)
    inside = (tpl > 0)
    outside = ~inside
    if outside.any():
        leak = float(chroma_vec[outside].mean())
        mag -= 0.35 * leak
    return mag


def resample_path(audio: Path, tmpdir: Path) -> Path:
    import librosa, soundfile as sf
    y, sr = librosa.load(str(audio), sr=None, mono=True)
    if abs(sr - 44100) < 1:
        return audio
    tmpdir.mkdir(parents=True, exist_ok=True)
    dest = tmpdir / f"{audio.stem}.44k.wav"
    if not dest.exists():
        sf.write(str(dest), librosa.resample(y, orig_sr=sr, target_sr=44100), 44100)
    return dest


def validate_quality(chroma_vec, root_idx, quality):
    """物理校验:升级必须有实际音级能量支撑,否则退回原三和弦判断。"""
    def e(semi):
        return float(chroma_vec[(root_idx + semi) % 12])
    root_e = e(0) + 1e-9
    if quality in ("sus2", "sus4"):
        # 挂留:三音必须明显弱于根音(否则是小/大三和弦的旋律经过音)
        return e(4) < 0.55 * root_e and e(3) < 0.55 * root_e
    if quality in ("maj7", "min7", "7", "hdim7", "dim7"):
        seventh = {"maj7": 11, "min7": 10, "7": 10, "hdim7": 10, "dim7": 9}[quality]
        return e(seventh) > 0.42 * root_e
    if quality in ("6", "min6"):
        return e(9) > 0.45 * root_e
    if quality == "aug":
        return e(8) > 0.4 * root_e
    if quality == "dim":
        return e(6) > 0.35 * root_e
    return True


def analyze(audio: Path, raw_doc: dict, min_dur=0.35):
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024, bins_per_octave=36)
    times = librosa.times_like(chroma, sr=sr, hop_length=1024)

    out = []
    for c in raw_doc["chords"]:
        dur = c["end"] - c["start"]
        if dur < min_dur or c["label"] == "N":
            continue
        root, qual0 = H.parse_label(c["label"])
        if root is None:
            continue
        i0 = np.searchsorted(times, c["start"])
        i1 = np.searchsorted(times, c["end"])
        if i1 - i0 < 2:
            continue
        seg = chroma[:, i0:i1]
        vec = np.median(seg, axis=1)
        vec = vec / (vec.sum() + 1e-9)

        root_idx = H.pc_index(root)
        family = FAMILY.get(qual0)
        allowed = [q for q in QUALITY_WEIGHTS if FAMILY.get(q) == family]
        if not allowed:
            allowed = [qual0]
        scores = {q: score_quality(vec, root_idx, q) for q in allowed}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_q, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_s - second_s
        # 保守策略:margin 不足或物理校验不通过,就保留 madmom 的三和弦判断
        keep = qual0
        need = UPGRADE_MARGIN.get(best_q, 0.05)
        if margin >= need and validate_quality(vec, root_idx, best_q):
            keep = best_q
        out.append({
            "start": c["start"], "end": c["end"],
            "root": root,
            "quality": keep,
            "madmom_quality": qual0,
            "confidence": round(best_s, 3),
            "margin": round(margin, 3),
            "template_scores": {q: round(s, 3) for q, s in ranked[:4]},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("raw_json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--min-dur", type=float, default=0.35)
    args = ap.parse_args()

    audio = resample_path(Path(args.audio), Path(args.outdir) / ".resampled")
    raw = json.loads(Path(args.raw_json).read_text(encoding="utf-8-sig"))
    chords = analyze(audio, raw, args.min_dur)

    changed = [c for c in chords if c["quality"] != c["madmom_quality"]]
    doc = {
        "schema": "quality-v1",
        "engine": "template-chroma",
        "vocabulary": sorted(QUALITY_WEIGHTS),
        "changed": len(changed),
        "total": len(chords),
        "chords": chords,
    }
    stem = Path(args.raw_json).stem.removesuffix(".raw")
    dest = Path(args.outdir) / f"{stem}.quality.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"quality: {len(chords)} chords analyzed, {len(changed)} upgraded "
          f"({len(changed)/max(1,len(chords)):.0%}) -> {dest}")
    for c in changed[:8]:
        print(f"  {c['start']:6.1f}s {c['root']}:{c['madmom_quality']} -> "
              f"{c['root']}:{c['quality']}  (margin {c['margin']})")


if __name__ == "__main__":
    main()
