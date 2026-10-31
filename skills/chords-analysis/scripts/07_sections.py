#!/usr/bin/env python3
"""Step 4d: 用主旋律 + 和弦给乐段编"材料号",并对应到 Verse/Pre-Chorus/Chorus/Bridge。

为什么不能只靠结构模型
----------------------
结构模型给的是功能标签(intro/verse/chorus/bridge/outro),它不告诉你
"这两段是不是同一份材料"。而实际写作里:
  - 同一段材料会在不同调上重复(副歌升 key 再来一遍);
  - 同一个标签下可能藏着两种完全不同的材料(主歌 A / 主歌 B);
  - 前副歌常常被模型并进 verse,因为它没有独立标签。

本脚本不看标签,先看**材料本身**:
  1. 每个乐段算旋律音级分布(按时长加权)+ 旋律走向 + 乐句方向 + 和弦级数分布;
  2. 两段之间的旋律相似度用**移调不变**的余弦(在 12 个移调里取最大),
     所以 +1 半音的副歌会被认成同一材料,并报出移调量;
  3. 单链接聚类 -> 材料 A/B/C…;
  4. 最后才和模型标签做交叉表,看"标签是否真的对应同一材料"。

输出:
  out/<name>.materials.json
  out/<name>.materials.md

用法:
  python 07_sections.py out/<name>.chords.json [--voices out/<name>.voices.json]
                          [--outdir out] [--threshold 0.88]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import harmony as H

MIN_MELODY_SEC = 2.0      # 少于此秒数,旋律相似度不可用
PHRASE_GAP = 0.7          # 乐句切分间隔(秒)
SLOPE_BIN = 1.0           # 旋律走向的分箱宽度(秒)
CONTOUR_BINS = 16         # 旋律轮廓重采样点数(按时段归一化)

# 相似度权重:轮廓 > 音级分布 > 和弦级数
# 实测音级分布在同曲不同段之间普遍 0.85+,没有分辨力;轮廓能到 0.99 vs -0.2。
W_CONTOUR, W_PC, W_DEGREE = 0.45, 0.30, 0.25

DEGREE_NAMES = ["I", "♭II", "II", "♭III", "III", "IV", "♯IV", "V", "♭VI", "VI", "♭VII", "VII"]


# ---------------------------------------------------------------- 工具

def root_pc(label):
    root, _ = H.parse_label(label.split("/", 1)[0])
    return None if root is None else H.pc_index(root)


def split_key(key_str):
    root, mode = key_str.split(" ", 1)
    return H.pc_index(root), ("minor" if mode.startswith("min") else "major")


def degree_of(pc, key_str):
    if pc is None or not key_str:
        return None
    kroot, _ = split_key(key_str)
    return (pc - kroot) % 12


def degree_name(d):
    return DEGREE_NAMES[d] if d is not None else "-"


def pc_profile(notes, a, b):
    """旋律音级分布(时长加权,12 维,和为单位向量)。"""
    v = np.zeros(12)
    for n in notes:
        ov = min(n["end"], b) - max(n["start"], a)
        if ov > 0:
            v[n["midi"] % 12] += ov
    s = v.sum()
    return v / s if s > 0 else None


def degree_profile(chords, key_str):
    """和弦根音级数分布(时长加权)。"""
    v = np.zeros(12)
    for c in chords:
        d = degree_of(root_pc(c["label"]), key_str)
        if d is not None:
            v[d] += c["end"] - c["start"]
    s = v.sum()
    return v / s if s > 0 else None


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def best_rotation(a, b):
    """移调不变相似度:在 12 个移调里取最大余弦,返回 (相似度, 移调半音数)。"""
    best, best_r = -1.0, 0
    for r in range(12):
        c = cos(np.roll(a, r), b)
        if c > best:
            best, best_r = c, r
    if best_r > 6:                 # +11 与 -1 等价,取绝对值小的写法
        best_r -= 12
    # 约定:正数表示 a 比 b 高 r 个半音(便于读"这段升了 key")
    return round(best, 3), -best_r


def contour_vector(notes, a, b, n=CONTOUR_BINS):
    """把乐段旋律重采样成 n 个等时点,去均值并归一化。

    这是"旋律方向"的量化形式:只保留音的**形状**(升/降/拱),去掉调高、
    去掉绝对音域、去掉时长差异。同一旋律移调或换速度后仍然匹配。
    """
    vec = []
    for k in range(n):
        t0 = a + (b - a) * k / n
        t1 = a + (b - a) * (k + 1) / n
        vals = []
        for x in notes:
            ov = min(x["end"], t1) - max(x["start"], t0)
            if ov > 0:
                vals.append((x["midi"], ov))
        if vals:
            tot = sum(w for _, w in vals)
            vec.append(sum(m * w for m, w in vals) / tot)
        else:
            vec.append(np.nan)
    v = np.array(vec, dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < 4:
        return None
    v = np.interp(np.arange(n), np.arange(n)[ok], v[ok])
    v = v - v.mean()
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 0 else None


def melody_stats(notes, a, b):
    """乐段内旋律:时长、音域、走向、乐句方向。"""
    sel = []
    for n in notes:
        s, e = max(n["start"], a), min(n["end"], b)
        if e - s > 1e-3:
            sel.append({"start": s, "end": e, "midi": n["midi"]})
    if not sel:
        return {"notes": 0, "sung": 0.0}

    sung = sum(n["end"] - n["start"] for n in sel)
    midis = np.array([n["midi"] for n in sel], dtype=float)
    weights = np.array([n["end"] - n["start"] for n in sel], dtype=float)
    mean_pitch = float((midis * weights).sum() / weights.sum())

    # 走向:1s 分箱取时长加权中位音高,再做最小二乘(单位:半音/10s)
    bins = defaultdict(list)
    for n in sel:
        t = n["start"]
        while t < n["end"]:
            bins[int((t - a) // SLOPE_BIN)].append((n["midi"], min(t + SLOPE_BIN, n["end"]) - t))
            t += SLOPE_BIN
    xs, ys = [], []
    for k in sorted(bins):
        ws = bins[k]
        tot = sum(w for _, w in ws)
        if tot > 0:
            xs.append(k * SLOPE_BIN + SLOPE_BIN / 2)
            ys.append(sum(m * w for m, w in ws) / tot)
    slope = None
    if len(xs) >= 4:
        slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0] * 10.0)

    # 乐句:按间隔切分
    sel.sort(key=lambda n: n["start"])
    phrases, cur = [], [sel[0]]
    for n in sel[1:]:
        if n["start"] - cur[-1]["end"] > PHRASE_GAP:
            phrases.append(cur)
            cur = [n]
        else:
            cur.append(n)
    phrases.append(cur)
    deltas = [p[-1]["midi"] - p[0]["midi"] for p in phrases]
    up = sum(1 for d in deltas if d >= 2)
    down = sum(1 for d in deltas if d <= -2)

    cv = contour_vector(notes, a, b)
    return {
        "notes": len(sel),
        "sung": round(sung, 1),
        "mean_midi": round(mean_pitch, 1),
        "mean_note": H.pc_name(int(round(mean_pitch)) % 12, True) + str(int(round(mean_pitch)) // 12 - 1),
        "range": [int(np.percentile(midis, 10)), int(np.percentile(midis, 90))],
        "slope_per_10s": None if slope is None else round(slope, 1),
        "phrases": len(phrases),
        "phrase_up": up,
        "phrase_down": down,
        "phrase_deltas": [int(d) for d in deltas],
        "contour": None if cv is None else [round(float(x), 4) for x in cv],
    }


def direction_text(m):
    if not m.get("notes"):
        return "无旋律"
    s = m.get("slope_per_10s")
    if s is None:
        return "样本不足"
    tag = "上行" if s > 1.5 else ("下行" if s < -1.5 else "平稳/环绕")
    txt = f"{tag} {s:+.1f} st/10s"
    if m.get("sung", 0) < 6.0:
        txt += " (样本少)"
    return txt


# ---------------------------------------------------------------- 聚类

def build_segments(doc, voices):
    segs = ((doc.get("structure") or {}).get("segments") or [])
    chords = doc.get("chords") or []
    mel = ((voices.get("melody") or {}).get("notes") or [])
    out = []
    for i, s in enumerate(segs):
        a, b = float(s["start"]), float(s["end"])
        if b - a < 1.0:
            continue
        cs = [c for c in chords if c["start"] < b and c["end"] > a]
        if not cs:
            continue
        labels = Counter(c.get("section") for c in cs if c.get("section"))
        key = s.get("key") or next((c.get("key_context") for c in cs if c.get("key_context")), None)
        m = melody_stats(mel, a, b)
        prof = pc_profile(mel, a, b) if m.get("sung", 0) >= 1.0 else None
        chord_dur = sum(c["end"] - c["start"] for c in cs)
        stat = {
            "index": len(out),
            "start": round(a, 2), "end": round(b, 2), "duration": round(b - a, 1),
            "label": labels.most_common(1)[0][0] if labels else "-",
            "key": key,
            "melody": m,
            "melody_profile": None if prof is None else [round(float(x), 4) for x in prof],
            "degree_profile": None,
            "chords_per_min": round(len(cs) / max(b - a, 1e-6) * 60, 1),
            "chord_duration": round(chord_dur, 1),
            "end_chord": cs[-1]["label"],
            "end_degree": degree_name(degree_of(root_pc(cs[-1]["label"]), key)),
            "top_degrees": [],
        }
        dp = degree_profile(cs, key)
        if dp is not None:
            stat["degree_profile"] = [round(float(x), 4) for x in dp]
            stat["top_degrees"] = [
                {"degree": DEGREE_NAMES[d], "share": round(float(dp[d]), 2)}
                for d in np.argsort(-dp)[:3] if dp[d] > 0.02]
        out.append(stat)
    return out


def segment_vectors(seg):
    """取一个乐段的三种特征向量(缺失的返回 None)。"""
    mel = seg.get("melody") or {}
    contour = np.array(mel["contour"]) if mel.get("contour") else None
    pc = np.array(seg["melody_profile"]) if seg.get("melody_profile") else None
    deg = np.array(seg["degree_profile"]) if seg.get("degree_profile") else None
    return contour, pc, deg


def similarity(a, b):
    """返回 (综合, 轮廓, 音级, 移调, 和弦) —— 缺失的项为 None。"""
    ca, pa, da = segment_vectors(a)
    cb, pb, db = segment_vectors(b)
    contour = float(np.dot(ca, cb)) if (ca is not None and cb is not None) else None
    rot = None
    if pa is not None and pb is not None:
        pc, rot = best_rotation(pa, pb)
    else:
        pc = None
    degree = cos(da, db) if (da is not None and db is not None) else None

    parts, weights = [], []
    if contour is not None:
        parts.append(contour); weights.append(W_CONTOUR)
    if pc is not None:
        parts.append(pc); weights.append(W_PC)
    if degree is not None:
        parts.append(degree); weights.append(W_DEGREE)
    if not parts:
        return 0.0, contour, pc, rot, degree
    combined = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    return round(float(combined), 3), contour, pc, rot, degree


def cluster(segs, threshold=0.8):
    """单链接聚类:与任一成员综合相似度 >= threshold 即同材料。

    没有旋律的乐段(纯器乐/静音尾)不参与聚类,材料记 "-"：
    否则它只能靠和弦相似度入组,会变成桥接点,把整首歌并成一块。
    """
    groups = []
    for s in segs:
        ca, pa, _ = segment_vectors(s)
        if ca is None and pa is None:
            s["material"] = "-"
            s["similar_to"] = None
            s["similarity"] = None
            continue
        best = None
        for g in groups:
            for other in g["members"]:
                sim, contour, pc, rot, degree = similarity(s, segs[other])
                if best is None or sim > best[0]:
                    best = (sim, other, contour, pc, rot, degree)
        if best and best[0] >= threshold:
            gi = groups.index(next(g for g in groups if best[1] in g["members"]))
            groups[gi]["members"].append(s["index"])
            s["similar_to"] = best[1]
            s["similarity"] = round(best[0], 3)
            s["contour_similarity"] = None if best[2] is None else round(best[2], 3)
            s["melodic_similarity"] = best[3]
            s["harmonic_similarity"] = None if best[5] is None else round(best[5], 3)
            s["transposition"] = best[4]
        else:
            groups.append({"members": [s["index"]]})
            s["similar_to"] = None
            s["similarity"] = None
    # 材料编号按首次出现顺序
    for gi, g in enumerate(groups):
        g["id"] = chr(ord("A") + gi)
        for idx in g["members"]:
            segs[idx]["material"] = g["id"]
    return groups


# ---------------------------------------------------------------- 前副歌

def pre_chorus_candidates(segs):
    """前副歌候选:紧接副歌、收在属功能、且材料不属于已有的 verse/chorus。

    前副歌常被结构模型并进 verse,所以这里不按标签判定,而看三件事:
      1. 下一段是 chorus;
      2. 本段收束在属功能(V / II / ♯IV / ♭II)——半终止;
      3. 材料只出现这一次(如果同一材料在别处当 verse,那它就是 verse)。
    长度只用来分强弱:短于副歌 75% 记为 strong,否则 weak。
    """
    choruses = [s for s in segs if s["label"] == "chorus"]
    if not choruses:
        return []
    med = float(np.median([c["duration"] for c in choruses]))
    freq = Counter(s.get("material") for s in segs)
    out = []
    for i, s in enumerate(segs[:-1]):
        nxt = segs[i + 1]
        if nxt["label"] != "chorus" or s["label"] not in ("verse", "inst"):
            continue
        if s["end_degree"] not in ("V", "II", "♯IV", "♭II"):
            continue
        if freq.get(s.get("material"), 0) > 1:
            continue
        short = s["duration"] <= 0.75 * med
        out.append({
            "index": s["index"], "start": s["start"], "end": s["end"],
            "duration": s["duration"], "label": s["label"],
            "end_chord": s["end_chord"], "end_degree": s["end_degree"],
            "next": nxt["start"], "material": s["material"],
            "strength": "strong" if short else "weak",
            "chorus_median": round(med, 1),
        })
    return out


# ---------------------------------------------------------------- 输出

def build_md(song, segs, groups, pcs):
    out = [f"# {song.get('title', '')} — 乐段材料与旋律方向", ""]
    out.append("> 材料号由**旋律(移调不变)+ 和弦级数**聚类得到,标签只用于最后对照。")
    out.append("")
    out.append("## 1. 乐段材料表")
    out.append("")
    out.append("| # | 时间 | 模型标签 | 材料 | 调 | 旋律 | 均值音高 | 走向 | 乐句↑/↓ | 收束 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in segs:
        m = s["melody"]
        up, dn = m.get("phrase_up", 0), m.get("phrase_down", 0)
        phr = f"{up}/{dn}" if m.get("phrases") else "-"
        out.append(f"| {s['index']} | {s['start']:.1f}–{s['end']:.1f}s | {s['label']} | "
                   f"**{s['material']}** | {s['key']} | {m.get('sung', 0):.1f}s | "
                   f"{m.get('mean_note', '-')} | {direction_text(m)} | {phr} | "
                   f"{s['end_chord']}（{s['end_degree']}） |")
    out.append("")
    out.append("## 2. 材料分组")
    out.append("")
    for g in groups:
        members = [segs[i] for i in g["members"]]
        labels = Counter(m["label"] for m in members)
        keys = Counter(m["key"] for m in members)
        out.append(f"### 材料 {g['id']}（{len(members)} 段，{sum(m['duration'] for m in members):.1f}s）")
        out.append("")
        out.append("- 乐段: " + ", ".join(
            f"#{m['index']} {m['start']:.0f}–{m['end']:.0f}s({m['label']})" for m in members))
        out.append("- 模型标签: " + ", ".join(f"{k}×{v}" for k, v in labels.items()))
        out.append("- 调: " + ", ".join(f"{k}×{v}" for k, v in keys.items()))
        out.append("- 旋律走向: " + "; ".join(
            f"#{m['index']} {direction_text(m['melody'])}" for m in members))
        out.append("- 收束级数: " + ", ".join(
            f"#{m['index']} {m['end_degree']}" for m in members))
        out.append("")
    out.append("## 3. 标签 × 材料（同一标签用了几个材料）")
    out.append("")
    labels = sorted({s["label"] for s in segs})
    mats = [g["id"] for g in groups]
    out.append("| 标签 | " + " | ".join(mats) + " | 合计 |")
    out.append("|---|" + "---|" * len(mats) + "---|")
    for lab in labels:
        row = [sum(1 for s in segs if s["label"] == lab and s["material"] == m) for m in mats]
        out.append(f"| {lab} | " + " | ".join(str(x) if x else "" for x in row) + f" | {sum(row)} |")
    out.append("")
    out.append("## 4. 跨标签复用（同一材料出现在不同标签下）")
    out.append("")
    reused = []
    for g in groups:
        labs = Counter(segs[i]["label"] for i in g["members"])
        if len(labs) > 1:
            reused.append((g, labs))
    if reused:
        for g, labs in reused:
            out.append(f"- **材料 {g['id']}** 同时用于 " + ", ".join(
                f"`{k}`×{v}" for k, v in labs.items()) + "：" + ", ".join(
                f"#{i} {segs[i]['start']:.0f}s" for i in g["members"]))
    else:
        out.append("(无:每个材料只出现在一个标签下)")
    out.append("")
    if pcs:
        out.append("## 5. 前副歌候选（模型可能并进 verse）")
        out.append("")
        out.append("| # | 时间 | 模型标签 | 时长 | 收束 | 下一段 | 材料 | 强度 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for p in pcs:
            out.append(f"| {p['index']} | {p['start']:.1f}–{p['end']:.1f}s | {p['label']} | "
                       f"{p['duration']:.1f}s | {p['end_chord']}（{p['end_degree']}） | "
                       f"chorus @{p['next']:.1f}s | {p['material']} | {p['strength']} |")
        out.append("")
        out.append(f"> 判定:紧接副歌 + 收束在属功能 + 材料唯一。`strong` = 短于副歌中位数"
                   f"({pcs[0]['chorus_median']}s)的 75%;`weak` = 长度与副歌相当,"
                   f"更像 verse 的尾段而不是独立前副歌。")
    else:
        out.append("## 5. 前副歌候选")
        out.append("")
        out.append("(无:没有「紧接副歌 + 收在属功能 + 材料唯一」的段落)")
    out.append("")
    out.append("## 6. 自动判读")
    out.append("")
    for lab in labels:
        used = sorted({s["material"] for s in segs if s["label"] == lab and s["material"] not in (None, "-")})
        if not used:
            out.append(f"- `{lab}` 没有可用的旋律材料（器乐/静音段）,不做材料判定。")
        elif len(used) == 1:
            out.append(f"- `{lab}` 只使用材料 **{used[0]}** —— 标签与材料一一对应。")
        else:
            out.append(f"- `{lab}` 使用了 {len(used)} 个材料（{', '.join(used)}）—— 同一标签下有不同写法,"
                       f"不能当成同一段材料。")
    return "\n".join(out) + "\n"


def process(chords_path: Path, voices_path: Path, outdir: Path, threshold):
    doc = json.loads(chords_path.read_text(encoding="utf-8"))
    voices = json.loads(voices_path.read_text(encoding="utf-8")) if voices_path.exists() else {}
    song = doc.get("song") or {}
    song.setdefault("title", chords_path.name.replace(".chords.json", ""))

    segs = build_segments(doc, voices)
    groups = cluster(segs, threshold=threshold)
    pcs = pre_chorus_candidates(segs)

    name = chords_path.name.replace(".chords.json", "")
    (outdir / f"{name}.materials.md").write_text(
        build_md(song, segs, groups, pcs), encoding="utf-8")
    payload = {"schema": "materials-v1", "threshold": threshold,
               "segments": segs,
               "materials": [{"id": g["id"], "members": g["members"]} for g in groups],
               "pre_chorus_candidates": pcs}
    (outdir / f"{name}.materials.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"[{name}] segments={len(segs)} materials={len(groups)}")
    for g in groups:
        mem = [segs[i] for i in g["members"]]
        print(f"   {g['id']}: " + ", ".join(
            f"#{m['index']} {m['label']} {m['start']:.0f}s" for m in mem))
    for s in segs:
        if s.get("similar_to") is not None:
            ct = s.get("contour_similarity")
            mel = s.get("melodic_similarity")
            harm = s.get("harmonic_similarity")
            tr = s.get("transposition")
            print(f"   #{s['index']} ~ #{s['similar_to']}  sim={s['similarity']} "
                  f"(contour={ct}, pc={mel}, harm={harm}, "
                  f"transpose={0 if tr is None else tr:+d})")
    for p in pcs:
        print(f"   pre-chorus candidate #{p['index']} {p['start']:.0f}s "
              f"({p['duration']:.0f}s, ends {p['end_chord']}={p['end_degree']})")
    return payload


def main():
    ap = argparse.ArgumentParser(description="section materials from melody + chords")
    ap.add_argument("chords", help="out/<name>.chords.json")
    ap.add_argument("--voices", default=None, help="out/<name>.voices.json")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--threshold", type=float, default=0.80)
    args = ap.parse_args()
    src = Path(args.chords)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        sys.exit(1)
    voices = Path(args.voices) if args.voices else src.with_name(
        src.name.replace(".chords.json", ".voices.json"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    process(src, voices, outdir, args.threshold)


if __name__ == "__main__":
    main()
