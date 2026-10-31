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

# 扩展语汇(9/11/13 与变化音)。延伸音权重低,因为 chroma 里它们常被谐波淹没。
EXTENDED_WEIGHTS = {
    "9":      {0: 1.0, 4: 0.9, 7: 0.6, 10: 0.75, 2: 0.55},
    "maj9":   {0: 1.0, 4: 0.9, 7: 0.6, 11: 0.75, 2: 0.55},
    "min9":   {0: 1.0, 3: 0.9, 7: 0.6, 10: 0.75, 2: 0.55},
    "11":     {0: 1.0, 4: 0.85, 7: 0.5, 10: 0.7, 2: 0.5, 5: 0.5},
    "min11":  {0: 1.0, 3: 0.85, 7: 0.5, 10: 0.7, 2: 0.5, 5: 0.5},
    "13":     {0: 1.0, 4: 0.85, 7: 0.5, 10: 0.7, 2: 0.45, 9: 0.55},
    "maj13":  {0: 1.0, 4: 0.85, 7: 0.5, 11: 0.7, 2: 0.45, 9: 0.55},
    "min13":  {0: 1.0, 3: 0.85, 7: 0.5, 10: 0.7, 2: 0.45, 9: 0.55},
    "7b9":    {0: 1.0, 4: 0.85, 7: 0.55, 10: 0.7, 1: 0.55},
    "7#9":    {0: 1.0, 4: 0.85, 7: 0.55, 10: 0.7, 3: 0.55},
    "7#11":   {0: 1.0, 4: 0.85, 7: 0.5, 10: 0.7, 6: 0.55},
    "7b13":   {0: 1.0, 4: 0.85, 7: 0.5, 10: 0.7, 8: 0.55},
    "7b5":    {0: 1.0, 4: 0.85, 6: 0.6, 10: 0.7},
    "7#5":    {0: 1.0, 4: 0.85, 8: 0.6, 10: 0.7},
    "aug7":   {0: 1.0, 4: 0.85, 8: 0.6, 10: 0.7},
    "maj7#11": {0: 1.0, 4: 0.85, 7: 0.5, 11: 0.7, 6: 0.55},
    "maj7#5": {0: 1.0, 4: 0.85, 8: 0.6, 11: 0.7},
    "add9":   {0: 1.0, 4: 0.9, 7: 0.7, 2: 0.6},
    "madd9":  {0: 1.0, 3: 0.9, 7: 0.7, 2: 0.6},
    "6/9":    {0: 1.0, 4: 0.85, 7: 0.6, 9: 0.55, 2: 0.55},
    "7sus4":  {0: 1.0, 5: 0.9, 7: 0.7, 10: 0.7},
}

# 扩展升级:只允许从"同一基础和弦"往上加音,避免把三和弦直接猜成 13 和弦
ALLOWED_FROM = {
    # 三和弦允许升到七和弦;9/11/13 要求七音与延伸音同时可测,避免"猜"出延伸音
    "maj": ["maj", "maj7", "6", "maj9", "maj13", "maj7#11", "maj7#5"],
    "maj7": ["maj7", "maj9", "maj13", "maj7#11", "maj7#5"],
    "min": ["min", "min7", "min6", "min9", "min11", "min13"],
    "min7": ["min7", "min9", "min11", "min13"],
    "7": ["7", "9", "11", "13", "7b9", "7#9", "7#11", "7b13",
          "7b5", "7#5", "aug7", "7sus4"],
    "aug": ["aug", "maj7#5", "7#5"],
    "6": ["6"],
}

# 扩展和弦不与"第二候选"比,而与**基础和弦**比:13 与 9 的模板彼此接近,
# 用 best-vs-second 会把两个都判成"证据不足"。
EXT_GAIN = {q: 0.03 for q in EXTENDED_WEIGHTS}
# 三和弦直接跳到 9/11/13 需要更强证据(七音 + 延伸音都要在)
TRIAD_EXT_GAIN = 0.10
# 延伸音之间(13/11/9)的最小分差;分不开就退回七和弦,不掷骰子
EXT_MARGIN = 0.02

# 音数越多越可疑(chroma 谐波泄漏常伪造 9/11 音);复杂度惩罚
# 升级阈值按性质分级:越"花"的性质要求证据越强
UPGRADE_MARGIN = {
    "maj7": 0.035, "min7": 0.035, "7": 0.035,
    "dim": 0.05, "aug": 0.06,
    "6": 0.06, "min6": 0.07, "hdim7": 0.07, "dim7": 0.08,
    "sus2": 0.12, "sus4": 0.12,   # 挂留多为旋律经过,证据要求最高
    # 扩展和弦:证据要求更高
    "9": 0.09, "maj9": 0.09, "min9": 0.09,
    "11": 0.12, "min11": 0.12, "13": 0.12, "maj13": 0.12, "min13": 0.12,
    "7b9": 0.11, "7#9": 0.11, "7#11": 0.11, "7b13": 0.11,
    "7b5": 0.10, "7#5": 0.10, "aug7": 0.10,
    "maj7#11": 0.11, "maj7#5": 0.10,
    "add9": 0.10, "madd9": 0.10, "6/9": 0.10, "7sus4": 0.12,
}


# 基础语汇的"同族"升级(三音决定大小调,不轻易翻转)
FAMILY = {
    "maj": "maj", "maj7": "maj", "6": "maj", "sus2": "maj", "sus4": "maj", "aug": "maj",
    "min": "min", "min7": "min", "min6": "min", "hdim7": "min", "dim7": "min", "dim": "min",
    "7": "dom",
}


def score_quality(chroma_vec, root_idx, quality, weights=None):
    table = weights or QUALITY_WEIGHTS
    tpl = np.zeros(12)
    for interval, w in table[quality].items():
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
    # --- 扩展和弦:特性音必须有实际能量;9/11/13 还要求七音同时存在 ---
    if quality in ("9", "min9"):
        return e(2) > 0.30 * root_e and e(10) > 0.38 * root_e
    if quality == "maj9":
        return e(2) > 0.30 * root_e and e(11) > 0.38 * root_e
    if quality in ("11", "min11"):
        return e(5) > 0.25 * root_e and e(10) > 0.35 * root_e
    if quality in ("13", "min13"):
        # 13 和弦常省略 9 音(爵士 voicing),只要求 13 音与七音
        return e(9) > 0.24 * root_e and e(10) > 0.35 * root_e
    if quality == "maj13":
        return e(9) > 0.24 * root_e and e(11) > 0.35 * root_e
    if quality == "7b9":
        return e(1) > 0.32 * root_e
    if quality == "7#9":
        return e(3) > 0.32 * root_e
    if quality == "7#11":
        return e(6) > 0.30 * root_e
    if quality == "7b13":
        return e(8) > 0.30 * root_e
    if quality == "7b5":
        return e(6) > 0.32 * root_e
    if quality in ("7#5", "aug7"):
        return e(8) > 0.32 * root_e
    if quality == "maj7#11":
        return e(11) > 0.40 * root_e and e(6) > 0.30 * root_e
    if quality == "maj7#5":
        return e(11) > 0.40 * root_e and e(8) > 0.32 * root_e
    if quality in ("add9", "madd9"):
        return e(2) > 0.35 * root_e
    if quality == "6/9":
        return e(9) > 0.35 * root_e and e(2) > 0.30 * root_e
    if quality == "7sus4":
        return e(5) > 0.35 * root_e and e(4) < 0.45 * root_e
    return True


def analyze(audio: Path, raw_doc: dict, min_dur=0.35, vocabulary="basic"):
    import librosa
    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024, bins_per_octave=36)
    times = librosa.times_like(chroma, sr=sr, hop_length=1024)
    extended = vocabulary == "extended"
    weights = dict(QUALITY_WEIGHTS)
    if extended:
        weights.update(EXTENDED_WEIGHTS)

    out = []
    for c in raw_doc["chords"]:
        dur = c["end"] - c["start"]
        if dur < min_dur or c["label"] == "N":
            continue
        root, qual0 = H.parse_label(c["label"])
        if root is None:
            continue
        qual0 = H.canonical_quality(qual0)
        i0 = np.searchsorted(times, c["start"])
        i1 = np.searchsorted(times, c["end"])
        if i1 - i0 < 2:
            continue
        seg = chroma[:, i0:i1]
        vec = np.median(seg, axis=1)
        vec = vec / (vec.sum() + 1e-9)

        root_idx = H.pc_index(root)
        if extended:
            # 扩展模式 = 基础候选 ∪ 扩展候选:不能因为"扩展"反而丢掉基本性质
            allowed = list(ALLOWED_FROM.get(qual0) or [])
            fam = FAMILY.get(qual0)
            if fam:
                allowed += [q for q in QUALITY_WEIGHTS if FAMILY.get(q) == fam]
            allowed = list(dict.fromkeys(allowed)) or [qual0]
        else:
            family = FAMILY.get(qual0)
            allowed = [q for q in QUALITY_WEIGHTS if FAMILY.get(q) == family]
            if not allowed:
                allowed = [qual0]
        scores = {q: score_quality(vec, root_idx, q, weights) for q in allowed}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_q, best_s = ranked[0]
        second_s = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_s - second_s
        base_s = scores.get(qual0, 0.0)
        gain = best_s - base_s
        # 保守策略:margin/增益不足或物理校验不通过,就保留输入的性质
        keep = qual0
        if extended and best_q in EXTENDED_WEIGHTS:
            # 已经带七音的基础:小幅增益即可;三和弦直接跳延伸音:要求强证据
            base7 = qual0 in ("maj7", "min7", "7", "6", "min6", "hdim7", "dim7")
            need = EXT_GAIN.get(best_q, 0.03) if base7 else TRIAD_EXT_GAIN
            # 延伸音之间(13 vs 11 vs 9)也要分得开,否则退回七和弦
            ok = gain >= need and margin >= EXT_MARGIN
            if ok and validate_quality(vec, root_idx, best_q):
                keep = best_q
            else:
                # 回退:在基础候选里选最好的(通常就是七和弦)
                basic_ranked = [(q, s) for q, s in ranked if q in QUALITY_WEIGHTS]
                if basic_ranked:
                    bq, bs = basic_ranked[0]
                    b_second = basic_ranked[1][1] if len(basic_ranked) > 1 else 0.0
                    if (bs - b_second >= UPGRADE_MARGIN.get(bq, 0.05)
                            and validate_quality(vec, root_idx, bq)):
                        keep = bq
        else:
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
            "gain": round(gain, 3),
            "template_scores": {q: round(s, 3) for q, s in ranked[:4]},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("raw_json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--min-dur", type=float, default=0.35)
    ap.add_argument("--vocabulary", default="basic", choices=["basic", "extended"],
                    help="basic=三/七和弦;extended=加 9/11/13 与变化音(证据要求更高)")
    args = ap.parse_args()

    audio = resample_path(Path(args.audio), Path(args.outdir) / ".resampled")
    raw = json.loads(Path(args.raw_json).read_text(encoding="utf-8-sig"))
    chords = analyze(audio, raw, args.min_dur, args.vocabulary)

    changed = [c for c in chords if c["quality"] != c["madmom_quality"]]
    vocab = sorted(QUALITY_WEIGHTS) if args.vocabulary == "basic" else sorted(
        {**QUALITY_WEIGHTS, **EXTENDED_WEIGHTS})
    doc = {
        "schema": "quality-v1",
        "engine": "template-chroma",
        "vocabulary_mode": args.vocabulary,
        "vocabulary": vocab,
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
              f"{c['root']}:{c['quality']}  (gain {c.get('gain')}, margin {c['margin']})")


if __name__ == "__main__":
    main()
