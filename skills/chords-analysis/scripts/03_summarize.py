#!/usr/bin/env python3
"""Step 3: chords-v1 json -> 分析用文本摘要(供 LLM 直接读,或人类快速浏览)。

用法: python 03_summarize.py <file.chords.json> [--section 30]
输出到 stdout:调性/统计概览 + 压缩后的和弦时间线(相同和弦连续段合并,
按 --section 秒分块展示)。
"""
import argparse, json, sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--section", type=float, default=30.0, help="时间线分块秒数")
    args = ap.parse_args()
    doc = json.loads(Path(args.input).read_text(encoding="utf-8"))
    s, chords, hist = doc["song"], doc["chords"], doc["stats"]["chord_histogram"]

    print(f"# {s.get('title', Path(args.input).stem)}")
    print(f"- 时长: {fmt(s.get('duration_sec') or 0)} | 引擎: {s.get('engine')} | 调性: {s.get('key', '?')}")
    if s.get("tempo_bpm"):
        print(f"- BPM: {s['tempo_bpm']}")
    print(f"- 和弦数: {doc['stats']['total_chords']} | 种类: {len(hist)}")
    print(f"- 和弦分布(按时长): {', '.join(f'{k} {v}s' for k, v in hist.items())}")

    # 和声节奏:每小节/每段和弦更换次数
    struct = doc.get("structure") or {}
    segments = struct.get("segments") or []
    bars = [c.get("bar") for c in chords if c.get("bar")]
    if bars:
        from collections import Counter
        per_bar = Counter()
        for c in chords:
            if c.get("bar"):
                per_bar[c["bar"]] += 1
        n_bars = max(bars)
        avg = sum(per_bar.values()) / max(1, n_bars)
        print(f"\n## 和声节奏(harmonic rhythm)")
        print(f"- 覆盖小节: 1-{n_bars}(共 {n_bars}) | 平均 {avg:.2f} 个和弦/小节")
        dist = Counter(per_bar.values())
        dist_line = ", ".join(f"{k}和弦/小节×{v}小节" for k, v in sorted(dist.items()))
        print(f"- 分布: {dist_line}")
    if segments:
        print(f"\n## 段落和声节奏(模型分段)")
        for seg in segments:
            seg_chords = [c for c in chords if seg["start"] <= c["start"] < seg["end"]]
            if not seg_chords:
                continue
            dur = seg["end"] - seg["start"]
            # 段内和弦更换次数(连续相同合并后)
            changes = 1
            for a, b in zip(seg_chords, seg_chords[1:]):
                if a["label"] != b["label"]:
                    changes += 1
            print(f"- {seg['label']:<7} {fmt(seg['start'])}-{fmt(seg['end'])} "
                  f"({dur:.0f}s): {changes} 次更换, {changes / (dur / 60):.1f} 个/分钟, "
                  f"平均 {(dur / changes):.1f}s/和弦")

    # 连续相同和弦合并
    merged = []
    for c in chords:
        if merged and merged[-1]["label"] == c["label"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))

    print(f"\n## 和弦时间线(合并重复,共 {len(merged)} 段)")
    cur_block, block_start = None, None
    for c in merged:
        blk = int(c["start"] // args.section)
        if blk != cur_block:
            if cur_block is not None:
                print()
            print(f"[{fmt(blk * args.section)}]")
            cur_block, block_start = blk, True
        print(f"  {fmt(c['start'])}-{fmt(c['end'])}  {c['label']}")
    print()


if __name__ == "__main__":
    main()
