#!/usr/bin/env python3
"""Step 2: raw json -> 标准 chords-v1 格式(归一化、合并碎片、统计、可选节拍/调性)。

用法:
  python 02_format.py <raw.json或目录> [--outdir out] [--bars beats.json]

节拍对齐(可选): 传入 allin1 或 madmom 产生的 beats json:
  {"beats": [t1, t2, ...]}   每拍一个时间点,4 拍一小节
输出: out/<name>.chords.json + out/<name>.lab
"""
import argparse, json, bisect, sys
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUAL_WEIGHT = {"maj": 0, "min": 1}  # 调性推断只看大小三和弦


def merge_short(chords, min_dur=0.15):
    """合并过短的碎片(<0.15s)到相邻和弦。"""
    out = [c for c in chords if (c["end"] - c["start"]) >= min_dur or c["label"] == "N"]
    if not out:
        return chords
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            out[i]["start"] = out[i - 1]["end"]
    return out


def infer_key(hist_dur):
    """简易 Krumhansl 风格调性推断:统计根音时长加权,优先含大/小三和弦多的根音。"""
    root_maj, root_min = defaultdict(float), defaultdict(float)
    for c, d in hist_dur.items():
        if ":" not in c:
            continue
        root, q = c.split(":", 1)
        w = QUAL_WEIGHT.get(q)
        if w == 0:
            root_maj[root] += d
        elif w == 1:
            root_min[root] += d
    if not root_maj and not root_min:
        return None
    # 简单规则:取时长最大的大三和弦根音为 I,否则小三
    cand = max(root_maj, key=root_maj.get) if root_maj else None
    m = max(root_min, key=root_min.get) if root_min else None
    if cand and (not m or root_maj[cand] >= root_min[m] * 0.8):
        return f"{cand} major (inferred, heuristic)"
    if m:
        return f"{m} minor (inferred, heuristic)"
    return None


def fmt_sec(s):
    return f"{int(s // 60)}:{s % 60:04.1f}"


def process(path: Path, outdir: Path, beats_file: Path | None):
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    chords = merge_short(doc["chords"])

    # 时长加权直方图
    hist = defaultdict(float)
    for c in chords:
        hist[c["label"]] += c["end"] - c["start"]
    hist = {k: round(v, 1) for k, v in sorted(hist.items(), key=lambda x: -x[1])}

    # 可选节拍对齐
    beats = None
    if beats_file and beats_file.exists():
        beats = json.loads(beats_file.read_text(encoding="utf-8"))["beats"]

    out_chords = []
    for c in chords:
        bar = beat = None
        if beats:
            i = bisect.bisect_right(beats, c["start"]) - 1
            if i >= 0:
                bar, beat = divmod(i, 4)
                bar, beat = bar + 1, beat + 1
        out_chords.append({"start": c["start"], "end": c["end"], "label": c["label"],
                           "bar": bar, "beat": beat})

    song = dict(doc["song"])
    key = infer_key(hist)
    if key:
        song["key"] = key

    result = {
        "schema": "chords-v1",
        "song": song,
        "chords": out_chords,
        "stats": {
            "total_chords": len(out_chords),
            "unique_labels": list(hist.keys()),
            "chord_histogram": hist,
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.stem.removesuffix(".raw")
    dest = outdir / f"{stem}.chords.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # MIREX .lab
    lab = outdir / f"{stem}.lab"
    lab.write_text("\n".join(f"{c['start']:.3f}\t{c['end']:.3f}\t{c['label']}" for c in chords),
                   encoding="utf-8")

    # 控制台摘要
    top = ", ".join(f"{k}({fmt_sec(v)})" for k, v in list(hist.items())[:8])
    print(f"{path.stem}: {len(out_chords)} chords | key={key or '?'}")
    print(f"  top: {top}")
    print(f"  -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--bars", default=None, help="beats json, {'beats':[...]}")
    args = ap.parse_args()
    src = Path(args.input)
    files = [src] if src.is_file() else sorted(src.glob("*.raw.json"))
    beats = Path(args.bars) if args.bars else None
    for f in files:
        process(f, Path(args.outdir), beats)


if __name__ == "__main__":
    main()
