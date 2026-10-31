#!/usr/bin/env python3
"""Step 3b: 段内调性缺口 + 离调检测。

为什么需要单独一步
------------------
调性不是"全曲属性",也不止"每段一个 key":
同一段内部会临时把某个和弦当主音(离调 / tonicization),
方向可以完全不同——向 vi、向 IV、向 ♭VI、向 II……
把全曲压成一个 key,或把一段压成一个 key,都会把这些抹平。

本脚本不问"全曲是什么调",只做两件具体的事:

1. **段级 key 的覆盖缺口**(`breaks`)
   滑窗计算"该段 key 能解释多少和弦音"(调内覆盖率)。
   覆盖率跌破阈值(默认 0.8)且连续多个窗口的区间,说明**段级 key 在这里失效**;
   给出该区间的最佳替代调,并区分:
     departure  音集不同——真的转调/离调到了另一个调
     relative   只是相对大小调(音集相同)——信息量不足以判定,单独标注
   这样不会在整首曲子上反复横跳,只在"说不通"的地方报警。

2. **属方向的离调事件**(`events`)
   找"属和弦 → 目标主音"的解决(根音上行四度),排除本调自己的 V→I
   和 I→IV(两者根音运动相同)。强度:
     strong  属七,或该级在本调里应是小三/减三却出现大三(= 副属和弦)
     weak    本调内的大三和弦(可能是 V/x,需要结合上下文看)
   并给出目标相对段级调性的级数、目标调名、以及多久回到段级主音。

输出:
  out/<name>.localkeys.json
  out/<name>.localkeys.md

用法:
  python 06_local_keys.py out/<name>.chords.json --outdir out
                            [--window 10] [--hop 4] [--threshold 0.8]
                            [--min-strength weak|strong]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import harmony as H

# ---------------------------------------------------------------- 常量

QUAL_FAMILY = {
    "maj": "maj", "6": "maj", "maj7": "maj",
    "min": "min", "min6": "min", "min7": "min", "minmaj7": "min",
    "7": "dom", "dim": "dim", "dim7": "dim", "hdim7": "dim",
    "aug": "aug", "sus2": "sus", "sus4": "sus",
}

# 大调各级应有的三和弦性质
EXPECT_MAJOR = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
# 小调(自然小调 + 和声小调导音)
EXPECT_MINOR = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj",
                10: "maj", 11: "dim"}

DEGREE_LABEL = {0: "I", 1: "♭II", 2: "II", 3: "♭III", 4: "III", 5: "IV",
                6: "♯IV", 7: "V", 8: "♭VI", 9: "VI", 10: "♭VII", 11: "VII"}
MINOR_DEGREE_LABEL = {0: "I", 2: "II", 3: "III", 5: "IV", 7: "V", 8: "VI",
                      10: "VII", 11: "♯VII"}

MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
MINOR_STEPS = (0, 2, 3, 5, 7, 8, 10)


# ---------------------------------------------------------------- 标签工具

def base_label(label):
    """去掉斜杠低音:'C#:maj/E#' -> 'C#:maj'。低音不影响调性判断。"""
    return label.split("/", 1)[0]


def root_pc(label):
    root, _ = H.parse_label(base_label(label))
    return None if root is None else H.pc_index(root)


def quality_family(label):
    _, q = H.parse_label(base_label(label))
    if q is None:
        return None
    return QUAL_FAMILY.get(H.canonical_quality(q), "other")


def chord_pcs(label):
    return [H.pc_index(p) for p in H.core_pitch_classes(base_label(label))]


def split_key(key_str):
    root, mode = key_str.split(" ", 1)
    return H.pc_index(root), ("minor" if mode.startswith("min") else "major")


def natural_pcs(root, mode):
    """自然音阶音集(不含和声小调导音)——用于判断"是不是同一个音集"。"""
    steps = MINOR_STEPS if mode == "minor" else MAJOR_STEPS
    return {(root + s) % 12 for s in steps}


def same_scale(k1, k2):
    """相对大小调的音集相同,和弦层面无法区分,不能算转调。"""
    if not k1 or not k2:
        return False
    return natural_pcs(*split_key(k1)) == natural_pcs(*split_key(k2))


def key_scale(root, mode):
    """用于覆盖率的音集——小调额外包含和声小调导音。"""
    pcs = natural_pcs(root, mode)
    if mode == "minor":
        pcs.add((root + 11) % 12)
    return pcs


def degree_label(pc, root, mode):
    d = (pc - root) % 12
    if mode == "minor":
        return MINOR_DEGREE_LABEL.get(d, DEGREE_LABEL[d])
    return DEGREE_LABEL[d]


# ---------------------------------------------------------------- 局部替代调

def window_slice(chords, t0, t1):
    """取与窗口重叠的和弦片段(按重叠长度裁剪 start/end)。"""
    seg = []
    for c in chords:
        if c["label"] == "N":
            continue
        a, b = max(c["start"], t0), min(c["end"], t1)
        if b - a > 1e-3:
            seg.append({"start": a, "end": b, "label": c["label"],
                        "section": c.get("section"), "key_context": c.get("key_context")})
    return seg


def score_keys(seg, flats):
    """窗口 -> 24 个调的评分(0.55 Krumhansl + 0.45 覆盖率)。"""
    if not seg:
        return []
    prof = np.zeros(12)
    tones = []
    for c in seg:
        dur = c["end"] - c["start"]
        pcs = chord_pcs(c["label"])
        if not pcs:
            continue
        for pc in pcs:
            prof[pc] += dur / len(pcs)
        tones.append((set(pcs), dur))
    total = prof.sum()
    if total <= 0:
        return []
    vec = prof / total

    out = []
    for root in range(12):
        for mode, ref in (("major", H.KK_MAJOR), ("minor", H.KK_MINOR)):
            corr = float(np.dot(vec, np.roll(np.array(ref, dtype=float), root)) /
                         (np.linalg.norm(vec) * np.linalg.norm(ref)))
            scale = key_scale(root, mode)
            num = sum(dur * (sum(1 for pc in pcs if pc in scale) / len(pcs))
                      for pcs, dur in tones)
            den = sum(dur for _, dur in tones)
            cov = num / den if den else 0.0
            out.append({"root": root, "mode": mode,
                        "key": f"{H.pc_name(root, flats)} {mode}",
                        "score": round(0.55 * corr + 0.45 * cov, 3),
                        "krumhansl": round(corr, 3),
                        "coverage": round(cov, 3)})
    out.sort(key=lambda x: -x["score"])
    return out


def coverage_breaks(chords, window=10.0, hop=4.0, threshold=0.8, min_windows=2):
    """段级 key 解释不了的连续区间 + 该区间的最佳替代调。"""
    seq = [c for c in chords if c["label"] != "N"]
    if not seq:
        return []
    t_end = max(c["end"] for c in seq)
    wins = []
    t = 0.0
    while t + window <= t_end:
        seg = window_slice(seq, t, t + window)
        dur = sum(c["end"] - c["start"] for c in seg)
        if seg and dur >= window * 0.5:
            mid = seg[len(seg) // 2]
            key = mid.get("key_context")
            if key:
                cov = H.diatonic_coverage(seg, key)
                if cov < threshold:
                    cands = score_keys(seg, H.is_flat_key(key))
                    wins.append({"start": round(t, 2), "end": round(t + window, 2),
                                 "section": mid.get("section"),
                                 "section_key": key, "coverage": cov,
                                 "alt": cands[0] if cands else None})
        t += hop

    regions = []
    for w in wins:
        r = regions[-1] if regions else None
        if (r and r["section_key"] == w["section_key"]
                and w["start"] - r["last_start"] <= hop + 1e-6):
            r["end"] = w["end"]
            r["last_start"] = w["start"]
            r["windows"] += 1
            r["coverage"] = min(r["coverage"], w["coverage"])
            if w["coverage"] <= r["worst_coverage"]:
                r["worst_coverage"] = w["coverage"]
                r["worst_start"] = w["start"]
                # 用"最差窗口"的替代调做区间代表,而不是分数最高的窗口
                if w["alt"]:
                    r["alt"] = w["alt"]
            elif w["alt"] and not r["alt"]:
                r["alt"] = w["alt"]
        else:
            regions.append({"start": w["start"], "end": w["end"],
                            "last_start": w["start"], "section": w["section"],
                            "section_key": w["section_key"], "coverage": w["coverage"],
                            "worst_coverage": w["coverage"], "worst_start": w["start"],
                            "alt": w["alt"], "windows": 1})

    out = []
    for r in regions:
        if r["windows"] < min_windows or not r["alt"]:
            continue
        alt_key = H.canonical_key_name(r["alt"]["key"])
        r.pop("last_start", None)
        r["alt_key"] = alt_key
        r["alt_coverage"] = r["alt"]["coverage"]
        r["alt_score"] = r["alt"]["score"]
        r["gain"] = round(r["alt"]["coverage"] - r["coverage"], 2)
        r["relation"] = "relative" if same_scale(alt_key, r["section_key"]) else "departure"
        r["alt_reliable"] = r["alt_coverage"] >= threshold
        r["duration"] = round(r["end"] - r["start"], 1)
        r.pop("alt", None)
        out.append(r)
    return out


# ---------------------------------------------------------------- 离调

def find_tonicizations(chords, sections, breaks=None):
    """找属方向解决(根音上行四度)、且目标不是段级主音的事件。

    级数一律相对**段级 key** 计算,不做局部替换——替换会让事件表失去可比性。
    如果事件落在 coverage_breaks 区间内,段级 key 在此处已失效,
    会标记 `in_break` 并给出该区间的局部替代调提示,由分析者判断。
    """
    seq = [c for c in chords if c["label"] != "N"]
    breaks = breaks or []

    def break_at(t):
        for b in breaks:
            if b["start"] <= t < b["end"]:
                return b
        return None

    events = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if b["start"] - a["end"] > 1.0:          # 中间有明显空白,不算解决
            continue
        ra, rb = root_pc(a["label"]), root_pc(b["label"])
        if ra is None or rb is None:
            continue
        if (rb - ra) % 12 != 5:                  # 根音上行四度 = 下行五度
            continue
        fa, fb = quality_family(a["label"]), quality_family(b["label"])
        if fa not in ("maj", "dom"):
            continue

        key_str = a.get("key_context")
        if not key_str:
            continue
        kroot, kmode = split_key(key_str)
        brk = break_at(a["start"])
        if rb == kroot:                          # 本调的 V -> I,不是离调
            continue
        if ra == kroot:                          # I -> IV 与 V -> I 根音运动相同,排除
            continue

        deg_a = (ra - kroot) % 12
        expect = (EXPECT_MINOR if kmode == "minor" else EXPECT_MAJOR).get(deg_a)
        if fa == "dom":
            strength = "strong"
        elif expect and expect != fa:
            strength = "strong"                  # 该级本该是小三/减三,却出现大三
        else:
            strength = "weak"

        target_mode = "minor" if fb == "min" else "major"
        target_key = H.canonical_key_name(
            f"{H.pc_name(rb, H.is_flat_key(key_str))} {target_mode}")

        back = None
        for c in seq[i + 1:]:
            if root_pc(c["label"]) == kroot:
                back = round(c["start"] - b["start"], 2)
                break
        nxt = next((s for s in sections if s.get("start", 0) > b["start"] - 1e-6), None)
        confirmed = bool(nxt and nxt.get("key") == target_key)

        if confirmed:
            kind = "modulation"
        elif back is not None and back <= 4.0:
            kind = "tonicization"
        else:
            kind = "prolonged"

        events.append({
            "time": round(a["start"], 2),
            "section": a.get("section"),
            "section_key": key_str,
            "in_break": bool(brk),
            "local_hint": brk["alt_key"] if brk else None,
            "from": a["label"], "to": b["label"],
            "from_root": ra, "to_root": rb,
            "degree": degree_label(rb, kroot, kmode),
            "target_key": target_key,
            "strength": strength,
            "back_to_tonic": back,
            "kind": kind,
        })
    return events


# ---------------------------------------------------------------- 输出

def build_md(song, breaks, events):
    out = [f"# {song.get('title', '')} — 段内调性与离调", ""]
    kbs = song.get("key_by_section") or []
    if kbs:
        out.append("- 段级调性(时长加权): " + ", ".join(
            f"{k['key']} {k['seconds']}s" for k in kbs))
    out.append(f"- 全局 key **只是摘要,不是分析单位**: {song.get('key')}"
               + ("(ambiguous)" if song.get("key_ambiguous") else ""))
    out.append("- 方法:窗口 10s / 步长 4s;段级 key 覆盖率跌破 0.8 且连续两个窗口 = 缺口")
    out.append("")
    out.append("## 1. 段级 key 解释不了的区间")
    out.append("")
    if breaks:
        out.append("| 起 | 止 | 段 | 段级 key | 段覆盖率 | 最差窗口 | 局部最佳 | 局部覆盖率 | 关系 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for b in breaks:
            if not b.get("alt_reliable"):
                rel = "过渡/半音段落"
            elif b["relation"] == "departure":
                rel = "**换调**"
            else:
                rel = "相对大小调(同音集)"
            out.append(f"| {b['start']:.1f}s | {b['end']:.1f}s | {b['section']} | "
                       f"{b['section_key']} | {b['coverage']:.2f} | "
                       f"{b['worst_start']:.1f}s ({b['worst_coverage']:.2f}) | "
                       f"{b['alt_key']} | {b['alt_coverage']:.2f} | {rel} |")
    else:
        out.append("(无:段级 key 在整个段落内都能解释和弦)")
    out.append("")
    out.append("> 区间边界来自 10s 窗口,所以起止有 ±10s 的模糊;`最差窗口` 是覆盖率最低的那个窗口的起点,")
    out.append("> 更接近真实转折点。")
    out.append("")
    out.append("## 2. 属方向的离调 / 转调事件")
    out.append("")
    out.append("| 时间 | 段 | 依据调 | 属 → 目标 | 目标级数 | 目标调 | 强度 | 回主音 | 判定 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for e in events:
        back = "-" if e["back_to_tonic"] is None else f"{e['back_to_tonic']:.1f}s"
        src = f"段级,缺口内→{e['local_hint']}" if e.get("in_break") else "段级"
        out.append(f"| {e['time']:.1f}s | {e['section']} | {e['section_key']} ({src}) | "
                   f"{e['from']} → {e['to']} | {e['degree']} | {e['target_key']} | "
                   f"{e['strength']} | {back} | {e['kind']} |")
    out.append("")
    out.append("## 3. 按段的离调方向")
    out.append("")
    out.append("| 段 | 段级 key | 方向(级数) | 事件数 |")
    out.append("|---|---|---|---|")
    by_sec = {}
    for e in events:
        by_sec.setdefault((e["section"], e["section_key"]), []).append(e)
    for (sec, key), evs in by_sec.items():
        dirs, seen = [], set()
        for e in evs:
            if e["degree"] in seen:
                continue
            seen.add(e["degree"])
            dirs.append(f"{e['degree']}({e['target_key']})")
        out.append(f"| {sec} | {key} | " + ", ".join(dirs) + f" | {len(evs)} |")
    return "\n".join(out) + "\n"


def process(path: Path, outdir: Path, window, hop, threshold, min_strength):
    doc = json.loads(path.read_text(encoding="utf-8"))
    song = doc.get("song") or {}
    song.setdefault("title", path.name.replace(".chords.json", ""))
    chords = doc.get("chords") or []
    sections = (doc.get("structure") or {}).get("segments") or []

    breaks = coverage_breaks(chords, window=window, hop=hop, threshold=threshold)
    events = find_tonicizations(chords, sections, breaks)
    if min_strength == "strong":
        events = [e for e in events if e["strength"] == "strong"]

    name = path.name.replace(".chords.json", "")
    md = build_md(song, breaks, events)
    (outdir / f"{name}.localkeys.md").write_text(md, encoding="utf-8")
    payload = {"schema": "localkeys-v1", "window": window, "hop": hop,
               "threshold": threshold, "breaks": breaks, "events": events}
    (outdir / f"{name}.localkeys.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{name}] breaks={len(breaks)} events={len(events)}")
    for b in breaks:
        flag = "" if b.get("alt_reliable") else "  (替代调不可靠:过渡/半音)"
        print(f"   gap {b['start']:6.1f}-{b['end']:6.1f}s  {b['section_key']} "
              f"cov={b['coverage']:.2f} -> {b['alt_key']} cov={b['alt_coverage']:.2f} "
              f"[{b['relation']}]{flag}")
    for e in events:
        print(f"   ev  {e['time']:6.1f}s  {e['from']:>12s} -> {e['to']:<12s} "
              f"{e['degree']:>4s} {e['target_key']:11s} {e['strength']:6s} {e['kind']}")
    return payload


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="section-key coverage gaps + tonicizations")
    ap.add_argument("chords", help="out/<name>.chords.json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=4.0)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-strength", default="weak", choices=["weak", "strong"])
    args = ap.parse_args()
    src = Path(args.chords)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        sys.exit(1)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    process(src, outdir, args.window, args.hop, args.threshold, args.min_strength)


if __name__ == "__main__":
    main()
