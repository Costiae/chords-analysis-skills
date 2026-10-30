#!/usr/bin/env python3
"""共享音乐理论工具:调性候选、等音拼写、和弦音构成、低音推断。

被 01c_voices / 02_format / 04_map 共用。
"""
import numpy as np

PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

FLAT_MAJOR_PCS = {5, 10, 3, 8, 1, 6, 11}   # F Bb Eb Ab Db Gb Cb
FLAT_MINOR_PCS = {2, 7, 0, 5, 10, 3, 8}    # d g c f bb eb ab

# 和弦音程(半音)
QUALITY_INTERVALS = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "7": [0, 4, 7, 10],
    "hdim7": [0, 3, 6, 10], "dim7": [0, 3, 6, 9],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "6": [0, 4, 7, 9], "min6": [0, 3, 7, 9],
    "maj6": [0, 4, 7, 9], "minmaj7": [0, 3, 7, 11],
}

# 不同引擎的标签别名(BTC 用 maj6,Chordino 用 6 …)
QUALITY_ALIASES = {"maj6": "6", "m6": "min6", "sus": "sus4", "dom7": "7",
                   "minmaj7": "minmaj7"}


def canonical_quality(q: str) -> str:
    return QUALITY_ALIASES.get(q, q)

KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def pc_index(name: str):
    """'A#' / 'Bb' / 'C' -> 0..11"""
    for i, n in enumerate(PC_SHARP):
        if name == n:
            return i
    for i, n in enumerate(PC_FLAT):
        if name == n:
            return i
    raise ValueError(f"unknown pitch class: {name}")


def pc_name(idx: int, use_flats=False):
    return (PC_FLAT if use_flats else PC_SHARP)[idx % 12]


def parse_label(label: str):
    """'C:maj' -> ('C', 'maj');'N' -> (None, None)"""
    if not label or label == "N":
        return None, None
    if ":" in label:
        root, qual = label.split(":", 1)
        return root, qual
    return label, "maj"


def label_pitch_classes(label: str, use_flats=False):
    root, qual = parse_label(label)
    if root is None:
        return []
    qual = canonical_quality(qual)
    intervals = QUALITY_INTERVALS.get(qual, [0, 4, 7])
    r = pc_index(root)
    return [pc_name((r + i) % 12, use_flats) for i in intervals]


def chord_profile(hist, use_flats=False):
    """和弦时长直方图 -> 音级权重(每个和弦音按 1/音数 分摊)。用于调性推断。"""
    prof = {pc_name(i, use_flats): 0.0 for i in range(12)}
    for label, dur in hist.items():
        pcs = label_pitch_classes(label, use_flats)
        if not pcs:
            continue
        w = float(dur) / len(pcs)
        for pc in pcs:
            prof[pc] += w
    return prof


def _corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 0.0


def key_candidates(profile, top=4, use_flats=False):
    """音级时长分布 -> Krumhansl 调性候选。profile 的键需与拼写一致。"""
    names = PC_FLAT if use_flats else PC_SHARP
    vec = np.array([profile.get(p, 0.0) for p in names], dtype=float)
    if vec.sum() <= 0:
        return []
    vec = vec / vec.sum()
    out = []
    for i, root in enumerate(names):
        for mode, ref in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            out.append({"key": f"{root} {mode}",
                        "score": round(_corr(vec, np.roll(ref, i)), 3),
                        "root": root, "mode": mode})
    out.sort(key=lambda x: -x["score"])
    return out[:top]


def merge_profiles(*profiles):
    """按音级名称合并多个分布(自动对齐 sharp/flat 拼写)。"""
    merged = {pc: 0.0 for pc in PC_SHARP}
    for prof in profiles:
        for name, v in prof.items():
            merged[pc_name(pc_index(name))] += float(v)
    return merged


def relative_key_candidates(key_str):
    """给定 'A minor',返回近关系调(关系大调/小调、属、下属)。用于歧义提示。"""
    root, mode = key_str.split(" ", 1)
    i = pc_index(root)
    rel_minor = pc_name((i + 9) % 12)
    rel_major = pc_name((i + 3) % 12)
    if mode == "minor":
        return {"relative_major": f"{rel_major} major",
                "dominant": f"{pc_name((i + 7) % 12)} minor",
                "subdominant": f"{pc_name((i + 5) % 12)} minor"}
    return {"relative_minor": f"{rel_minor} minor",
            "dominant": f"{pc_name((i + 7) % 12)} major",
            "subdominant": f"{pc_name((i + 5) % 12)} major"}


def is_flat_key(key_str: str) -> bool:
    """按音级判断调号是否用降号——'D# major' 与 'Eb major' 等价,都返回 True。"""
    root, mode = key_str.split(" ", 1)
    i = pc_index(root)
    return i in (FLAT_MAJOR_PCS if mode == "major" else FLAT_MINOR_PCS)


def canonical_key_name(key_str: str) -> str:
    """按调号的升降号习惯重写调名:'A# major' -> 'Bb major','D# minor' -> 'Eb minor'。"""
    root, mode = key_str.split(" ", 1)
    flats = is_flat_key(key_str)
    return f"{pc_name(pc_index(root), flats)} {mode}"


def respell_label(label: str, use_flats: bool, bass_pc=None):
    """等音规范化:'A#:maj' -> 'Bb:maj';bass_pc 与根音不同则输出斜杠和弦。"""
    root, qual = parse_label(label)
    if root is None:
        return label
    r = pc_index(root)
    out = f"{pc_name(r, use_flats)}:{qual}"
    if bass_pc is not None and bass_pc != r:
        out += f"/{pc_name(bass_pc, use_flats)}"
    return out


def dominant_bass_pc(bass_notes, start, end, min_overlap=0.15):
    """在给定区间内取出现时长最长的低音音级。"""
    weights = {}
    for n in bass_notes:
        ov = min(n["end"], end) - max(n["start"], start)
        if ov > min_overlap:
            pc = n["midi"] % 12
            weights[pc] = weights.get(pc, 0.0) + ov
    if not weights:
        return None, 0.0
    pc = max(weights, key=weights.get)
    return pc, weights[pc]
