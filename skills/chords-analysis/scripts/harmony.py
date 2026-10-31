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
    # --- 扩展语汇(9/11/13 与变化音) ---
    "9": [0, 4, 7, 10, 2], "maj9": [0, 4, 7, 11, 2], "min9": [0, 3, 7, 10, 2],
    "11": [0, 4, 7, 10, 2, 5], "min11": [0, 3, 7, 10, 2, 5],
    "13": [0, 4, 7, 10, 2, 9], "maj13": [0, 4, 7, 11, 2, 9],
    "min13": [0, 3, 7, 10, 2, 9],
    "7b9": [0, 4, 7, 10, 1], "7#9": [0, 4, 7, 10, 3],
    "7#11": [0, 4, 7, 10, 6], "7b13": [0, 4, 7, 10, 8],
    "7b5": [0, 4, 6, 10], "7#5": [0, 4, 8, 10], "aug7": [0, 4, 8, 10],
    "maj7#11": [0, 4, 7, 11, 6], "maj7#5": [0, 4, 8, 11],
    "add9": [0, 4, 7, 2], "madd9": [0, 3, 7, 2], "6/9": [0, 4, 7, 9, 2],
    "7sus4": [0, 5, 7, 10],
}

# 不同引擎的标签别名(BTC 用 maj6,Chordino 用 6 …)
QUALITY_ALIASES = {"maj6": "6", "m6": "min6", "sus": "sus4", "dom7": "7",
                   "minmaj7": "minmaj7",
                   # 扩展语汇的常见写法
                   "m7b5": "hdim7", "min7b5": "hdim7", "halfdim": "hdim7",
                   "dom9": "9", "dom13": "13", "dom7b9": "7b9",
                   "maj9#11": "maj7#11", "add2": "add9", "69": "6/9",
                   "7alt": "7#9", "min9b5": "hdim7"}


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


def split_slash(label: str):
    """把 'C:maj/E' 拆成 ('C:maj', 'E');'F:6/9' 不拆(9 不是音名)。"""
    if "/" not in label:
        return label, None
    body, tail = label.rsplit("/", 1)
    if len(tail) in (1, 2) and tail[0] in "ABCDEFG" and all(c in "#b♯♭" for c in tail[1:]):
        return body, tail
    return label, None


def parse_label(label: str):
    """'C:maj' -> ('C', 'maj');'N' -> (None, None)"""
    if not label or label == "N":
        return None, None
    label, _ = split_slash(label)
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


def key_score(profile, key_str, use_flats=False):
    """给定音级分布与调名,返回该调的 Krumhansl 相关分(0..1)。用于稳定性比较。"""
    names = PC_FLAT if use_flats else PC_SHARP
    vec = np.array([profile.get(p, 0.0) for p in names], dtype=float)
    if vec.sum() <= 0:
        return 0.0
    vec = vec / vec.sum()
    root, mode = key_str.split(" ", 1)
    ref = KK_MINOR if mode == "minor" else KK_MAJOR
    return round(_corr(vec, np.roll(ref, pc_index(root))), 3)


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


MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)


def scale_pcs(key_str):
    """调名 -> 调内音级集合。小调额外包含导音(和声小调)。"""
    root, mode = key_str.split(" ", 1)
    i = pc_index(root)
    steps = MINOR_STEPS if mode.startswith("min") else MAJOR_STEPS
    pcs = {(i + s) % 12 for s in steps}
    if mode.startswith("min"):
        pcs.add((i + 11) % 12)
    return pcs


def diatonic_coverage(chords, key_str):
    """和弦音落在调内的时长加权比例(0..1)。

    Krumhansl 只比较音级分布,相邻调(尤其是三度/五度远的关系)常常几乎同分:
    同一批和弦音可以是 Cb major(B major)的调内音,也可以属于 Db major。
    三和弦里的特征音(F 自然 vs F#)能把两者分开,所以用和弦本体的音做覆盖检查。
    """
    if not chords:
        return 0.0
    pcs = scale_pcs(key_str)
    num = den = 0.0
    for c in chords:
        label = c.get("label") if isinstance(c, dict) else c
        if isinstance(c, dict):
            dur = max(0.0, float(c.get("end", 0.0)) - float(c.get("start", 0.0)))
        else:
            dur = 1.0
        if not label or label == "N" or dur <= 0:
            continue
        tones = label_pitch_classes(label)
        if not tones:
            continue
        hit = sum(1 for t in tones if pc_index(t) in pcs) / len(tones)
        num += hit * dur
        den += dur
    return round(num / den, 3) if den else 0.0


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


LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
LETTER_PC = [0, 2, 4, 5, 7, 9, 11]
# 半音数 -> 音级序号(第几度,0 起);6 的减五/增四歧义按五度处理
DEGREE_FOR_SEMIS = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6}


def _split_accidental(name: str):
    letter = name[0].upper()
    acc = 0
    for ch in name[1:]:
        if ch in "#♯":
            acc += 1
        elif ch in "b♭":
            acc -= 1
    return letter, acc


def spell_interval(root_name: str, semis: int):
    """按根音的字母/升降号拼写 root+semis 的音名(支持 #/b/##/bb)。

    斜杠低音必须按**和弦音级**拼写,不能直接取音级名:
    C#:maj 的三音是 E# 而不是 F,即使两者音级相同。
    拼不出常规音名(需要三升/三降)时返回 None。
    """
    try:
        letter, acc = _split_accidental(root_name)
        li = LETTERS.index(letter)
    except (IndexError, ValueError):
        return None
    root_pc = (LETTER_PC[li] + acc) % 12
    deg = DEGREE_FOR_SEMIS[semis % 12]
    tgt_li = (li + deg) % 7
    natural = LETTER_PC[tgt_li]
    target_pc = (root_pc + semis) % 12
    diff = (target_pc - natural) % 12
    if diff > 6:
        diff -= 12
    if abs(diff) > 2:
        return None
    return LETTERS[tgt_li] + ("#" * diff if diff > 0 else "b" * (-diff))


def respell_label(label: str, use_flats: bool, bass_pc=None):
    """等音规范化:'A#:maj' -> 'Bb:maj';bass_pc 与根音不同则输出斜杠和弦。"""
    root, qual = parse_label(label)
    if root is None:
        return label
    r = pc_index(root)
    root_name = pc_name(r, use_flats)
    out = f"{root_name}:{qual}"
    if bass_pc is not None and bass_pc != r:
        name = spell_interval(root_name, (bass_pc - r) % 12) or pc_name(bass_pc, use_flats)
        out += f"/{name}"
    return out


def dominant_bass_pc(bass_notes, start, end, min_overlap=0.15):
    """在给定区间内取出现时长最长的低音音级。

    返回 (pc, 该音级时长, 全部低音时长)。第三个值用于判断"这个低音是否
    真的占主导":密集的低音线里经过音也会成为最长音级。
    """
    weights = {}
    for n in bass_notes:
        ov = min(n["end"], end) - max(n["start"], start)
        if ov > min_overlap:
            pc = n["midi"] % 12
            weights[pc] = weights.get(pc, 0.0) + ov
    if not weights:
        return None, 0.0, 0.0
    pc = max(weights, key=weights.get)
    return pc, weights[pc], sum(weights.values())
