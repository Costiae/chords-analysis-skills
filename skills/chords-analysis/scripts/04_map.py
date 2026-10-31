#!/usr/bin/env python3
"""Step 4: 生成小节级和声地图 + 进行模式挖掘。

把 chords-v1 变成人和 LLM 都能直接读的结构化地图:
  - 小节网格:每小节的拍位和弦(合并同小节重复)
  - 段落摘要:每段的进行链(去重)、终止式、局部调性
  - 进行挖掘:统计重复出现的和弦 n-gram(2-6 和弦),标注出现位置
  - 罗马数字:按段落调性把和弦转成级数

用法:
  python 04_map.py out/<name>.chords.json [--outdir out] [--top-ngrams 8]
输出: out/<name>.map.md  +  stdout
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harmony as H

ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "bV", "V", "bVI", "VI", "bVII", "VII"]
ROMAN_MINOR = ["I", "bII", "II", "III", "#IV", "IV", "V", "bVI", "VI", "bVII", "VII", "bVII"]


def roman(numeral_key: str, label: str) -> str:
    """和弦 -> 级数标记(含转位与和弦性质)。"""
    if not label or label == "N":
        return "N"
    root_part = label.split("/")[0]
    root, qual = H.parse_label(root_part)
    if root is None:
        return "N"
    kroot, kmode = numeral_key.split(" ", 1)
    deg = (H.pc_index(root) - H.pc_index(kroot)) % 12
    table = ROMAN_MINOR if kmode == "minor" else ROMAN
    r = table[deg]
    # 小调惯用:自然小调的 v/III 等保留原级数,只标记性质
    minor_q = {"min", "min6", "min7", "min9", "min11", "min13", "minmaj7", "madd9"}
    if qual in minor_q:
        r = r.lower()
    suffix = ""
    if qual in ("maj7", "min7", "7", "hdim7", "dim7", "6", "min6"):
        suffix = {"maj7": "maj7", "min7": "7", "7": "7", "hdim7": "ø7",
                  "dim7": "°7", "6": "6", "min6": "6"}[qual]
    elif qual in ("9", "maj9", "min9", "11", "min11", "13", "maj13", "min13",
                  "7b9", "7#9", "7#11", "7b13", "7b5", "7#5", "aug7",
                  "maj7#11", "maj7#5", "add9", "madd9", "6/9", "7sus4"):
        suffix = {"min9": "9", "min11": "11", "min13": "13", "madd9": "add9",
                  "aug7": "+7"}.get(qual, qual)
    elif qual == "dim":
        suffix = "°"
    elif qual == "aug":
        suffix = "+"
    elif qual.startswith("sus"):
        suffix = qual.replace("sus", "sus")
    inv = label.split("/", 1)[1] if "/" in label else ""
    return f"{r}{suffix}" + (f"/{inv}" if inv else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--top-ngrams", type=int, default=8)
    ap.add_argument("--min-ngram-len", type=int, default=2)
    ap.add_argument("--max-ngram-len", type=int, default=6)
    args = ap.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    doc = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    song = doc["song"]
    chords = doc["chords"]
    struct = doc.get("structure") or {}
    segments = struct.get("segments") or []
    global_key = song.get("key") or "C major"

    def key_at(t):
        for s in segments:
            if s["start"] <= t < s["end"] and s.get("key"):
                return s["key"]
        return global_key

    def key_over_span(t0, t1):
        """按时间跨度取重叠最长的段落调性(避免边界前 0.1s 取错段)。"""
        best, best_ov = None, 0.0
        for s in segments:
            ov = min(s["end"], t1) - max(s["start"], t0)
            if ov > best_ov and s.get("key"):
                best, best_ov = s["key"], ov
        return best or key_at(t0)

    out = []
    out.append(f"# {song.get('title', Path(args.input).stem)} — 和声地图")
    out.append("")
    meta = [f"调性 **{global_key}**"]
    if song.get("tempo_bpm"):
        meta.append(f"BPM {song['tempo_bpm']}")
    if song.get("meter"):
        meta.append(f"{song['meter']}/4")
    if struct.get("segments"):
        meta.append(f"{len(segments)} 段")
    out.append("- " + " · ".join(meta))
    if doc.get("ensemble"):
        e = doc["ensemble"]
        out.append(f"- 识别一致率 {e['agreement']:.1%}(冲突 {e['conflict_seconds']}s)")
    if doc.get("voices"):
        v = doc["voices"]
        out.append(f"- 旋律 {v.get('melody_notes')} 音 / 低音 {v.get('bass_notes')} 音 / "
                   f"转位 {v.get('bass_inversions')}")
    out.append("")

    # ---- 小节网格 ----
    bars = [c for c in chords if c.get("bar")]
    if bars:
        out.append("## 小节网格(每小节和弦 · 级数)")
        out.append("")
        by_bar = defaultdict(list)
        for c in chords:
            if c.get("bar"):
                by_bar[c["bar"]].append(c)
        cur_seg = None
        for b in sorted(by_bar):
            cs = by_bar[b]
            seg = cs[0].get("section")
            if seg != cur_seg:
                k = key_at(cs[0]["start"])
                out.append("")
                out.append(f"**{seg}** · {k}")
                out.append("")
                cur_seg = seg
            # 小节内和弦(去重相邻重复)
            labels = []
            for c in cs:
                if not labels or labels[-1] != c["label"]:
                    labels.append(c["label"])
            k = key_at(cs[0]["start"])
            rome = " → ".join(roman(k, l) for l in labels)
            conf = " ⚠️" if any(c.get("confidence") == "conflict" for c in cs) else ""
            out.append(f"- m{b:>3}: {' | '.join(labels)}   [{rome}]{conf}")
        out.append("")

    # ---- 段落摘要 ----
    if segments:
        out.append("## 段落和声摘要")
        out.append("")
        for s in segments:
            seg_chords = [c for c in chords if s["start"] <= c["start"] < s["end"]]
            if not seg_chords:
                continue
            k = s.get("key") or key_at(s["start"])
            # 去重进行链(连续相同合并)
            chain = []
            for c in seg_chords:
                if not chain or chain[-1] != c["label"]:
                    chain.append(c["label"])
            rome = " → ".join(roman(k, l) for l in chain)
            dur = s["end"] - s["start"]
            rate = len(chain) / (dur / 60) if dur else 0
            kc = ""
            if s.get("key_candidates") and len(s["key_candidates"]) > 1:
                alt = s["key_candidates"][1]
                kc = f"(备选 {alt['key']} {alt['score']})"
            out.append(f"### {s['label']} {int(s['start']//60)}:{s['start']%60:04.1f}–"
                       f"{int(s['end']//60)}:{s['end']%60:04.1f} · {k}{kc}")
            out.append(f"- 和声: {rome}")
            out.append(f"- 和声节奏: {rate:.1f} 和弦/分钟,平均 {dur/len(chain):.1f}s/和弦")
            out.append("")

    # ---- 进行挖掘(n-gram)----
    # 注意:去重后的序号不能拿去索引原始和弦列表(会错位,连带调性上下文出错)
    seq_items = []
    for c in chords:
        if not seq_items or seq_items[-1][0] != c["label"]:
            seq_items.append((c["label"], c["start"]))
    seq = [s[0] for s in seq_items]
    counts = defaultdict(list)
    for n in range(args.min_ngram_len, args.max_ngram_len + 1):
        for i in range(len(seq) - n + 1):
            gram = tuple(seq[i:i + n])
            if gram[0] == "N":
                continue
            counts[gram].append(i)
    ranked = sorted(((g, idx) for g, idx in counts.items() if len(idx) >= 2),
                    key=lambda x: (-len(x[1]), -len(x[0])))
    if ranked:
        out.append("## 重复进行(自动挖掘)")
        out.append("")
        out.append("| 次数 | 长度 | 进行 | 罗马数字 | 首次出现 |")
        out.append("|---|---|---|---|---|")
        for gram, idxs in ranked[:args.top_ngrams]:
            i0 = idxs[0]
            t = seq_items[i0][1]
            t_end = seq_items[i0 + len(gram)][1] if i0 + len(gram) < len(seq_items) \
                else chords[-1]["end"]
            k = key_over_span(t, t_end)
            out.append(f"| {len(idxs)} | {len(gram)} | {' → '.join(gram)} | "
                       f"{' → '.join(roman(k, g) for g in gram)} | "
                       f"{int(t//60)}:{t%60:04.1f} |")
        out.append("")

    text = "\n".join(out)
    dest = Path(args.outdir) / f"{Path(args.input).stem.removesuffix('.chords')}.map.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
