#!/usr/bin/env python3
"""Step 5: 把一首/多首曲子的全部产物汇总成一个入口页(out/index.html)。

为什么需要
----------
管线会产出十几份文件,人和 AI 都容易找不到该看哪个。这个页面只做索引:

  先看什么   -> .review.html(核对页)
  要演奏     -> .lead.musicxml / .score.musicxml / .score.mid
  要分析     -> .analysis.md / .map.md / .localkeys.md / .materials.md
  要数据     -> .chords.json / .voices.json / .transcription.json

不重新计算任何东西,只读已有 JSON 做摘要,所以不会影响任何准确度指标。

用法:
  python 10_bundle.py --outdir out
  python 10_bundle.py --outdir out --title "我的分析集"
"""
import argparse
import html
import json
import sys
from pathlib import Path
from urllib.parse import quote

# 产物 -> (中文名, 说明, 优先级)
ARTIFACTS = [
    ("review.html", "核对页", "段落/和弦/旋律/低音/鼓点并排", 1),
    ("lead.musicxml", "功能谱", "旋律 + 和弦符号 + 段落标记", 1),
    ("score.musicxml", "三声部总谱", "旋律 + 低音 + 生成和声", 1),
    ("score.mid", "带速度表的 MIDI", "可导入 DAW,小节对齐", 1),
    ("analysis.md", "分析报告", "调性/离调/材料/低音", 2),
    ("map.md", "和声地图", "小节网格 + 级数 + n-gram", 2),
    ("localkeys.md", "段内调性核验", "覆盖率缺口 + 离调方向", 2),
    ("materials.md", "乐段材料", "旋律轮廓聚类 + 标签对照", 2),
    ("chords.json", "标准数据", "chords-v1(权威)", 3),
    ("voices.json", "声部数据", "旋律/低音音符与诊断", 3),
    ("transcription.json", "转写事件", "分轨音符 + MIDI", 3),
    ("midi", "原始 MIDI", "未量化,仅调试用", 3),
    ("lab", "MIREX .lab", "和弦时间标签", 3),
    ("score.json", "谱面自检", "小节/量化/丢弃统计", 3),
]


def esc(s):
    return html.escape(str(s), quote=True)


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect(outdir: Path):
    tracks = []
    for chords_path in sorted(outdir.glob("*.chords.json")):
        name = chords_path.name.replace(".chords.json", "")
        doc = load(chords_path)
        song = doc.get("song") or {}
        ch = doc.get("chords") or []
        st = doc.get("structure") or {}
        score = load(outdir / f"{name}.score.json")
        mats = load(outdir / f"{name}.materials.json")
        loc = load(outdir / f"{name}.localkeys.json")

        inv = len([c for c in ch if "/" in c.get("label", "")])
        non_n = len([c for c in ch if c.get("label") != "N"])
        segs = st.get("segments") or []
        kbs = song.get("key_by_section") or []
        files = []
        for suffix, label, desc, prio in ARTIFACTS:
            p = outdir / f"{name}.{suffix}"
            if p.exists():
                files.append({"suffix": suffix, "label": label, "desc": desc,
                              "prio": prio, "name": p.name,
                              "kb": round(p.stat().st_size / 1024, 1)})
        tracks.append({
            "name": name,
            "title": song.get("title") or name,
            "duration": song.get("duration_sec"),
            "engine": song.get("engine"),
            "bpm": song.get("tempo_bpm") or st.get("bpm"),
            "meter": song.get("meter") or st.get("meter"),
            "key": song.get("key"),
            "ambiguous": song.get("key_ambiguous"),
            "key_by_section": kbs,
            "chords": non_n,
            "inversions": inv,
            "sections": len(segs),
            "bars": (score.get("checks") or {}).get("bars") or score.get("bars"),
            "checks": score.get("checks") or {},
            "materials": len(mats.get("materials") or []),
            "tonicizations": len(loc.get("events") or []),
            "gaps": len(loc.get("breaks") or []),
            "files": files,
        })
    return tracks


def render(tracks, title, outdir):
    cards = []
    for t in tracks:
        dur = t.get("duration")
        dur_txt = f"{int(dur)//60}:{int(dur)%60:02d}" if dur else "-"
        keys = " · ".join(f"{k['key']} {k['seconds']}s" for k in (t.get("key_by_section") or [])[:4])
        amb = " <span class='warn'>ambiguous</span>" if t.get("ambiguous") else ""
        checks = t.get("checks") or {}
        check_bits = []
        if checks:
            check_bits.append(f"{checks.get('bars', '-')} 小节")
            if checks.get("measures_uniform"):
                check_bits.append("小节等长 ✓")
            else:
                check_bits.append("小节不等长 ⚠")
            if checks.get("melody_octave_fixed") is not None:
                check_bits.append(f"八度修正 {checks['melody_octave_fixed']}")
        stats = [
            ("时长", dur_txt),
            ("速度", f"{t.get('bpm')} BPM · {t.get('meter')}/4" if t.get("bpm") else "-"),
            ("和弦", f"{t.get('chords')} 个 · 转位 {t.get('inversions')}"),
            ("结构", f"{t.get('sections')} 段 · 材料 {t.get('materials')}"),
            ("离调", f"{t.get('tonicizations')} 事件 · 缺口 {t.get('gaps')}"),
            ("谱面", " · ".join(check_bits) if check_bits else "-"),
        ]
        stat_html = "".join(
            f"<div class='stat'><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in stats)
        prio = {}
        for f in t["files"]:
            prio.setdefault(f["prio"], []).append(f)
        blocks = []
        for p, label in ((1, "直接看 / 直接演奏"), (2, "分析"), (3, "数据与调试")):
            if p not in prio:
                continue
            items = "".join(
                f"<li><a href=\"{esc(quote(f['name']))}\">{esc(f['label'])}</a>"
                f"<span class='fn'>{esc(f['name'])}</span>"
                f"<em>{esc(f['desc'])}</em><i>{f['kb']} KB</i></li>"
                for f in sorted(prio[p], key=lambda x: x["suffix"]))
            blocks.append(f"<div class='group'><h4>{label}</h4><ul>{items}</ul></div>")
        cards.append(f"""
<section class="card">
  <h2>{esc(t['title'])}</h2>
  <p class="sub">{esc(t['engine'] or '?')} · 调性摘要 {esc(t['key'] or '-')}{amb}<br>
     {esc(keys)}</p>
  <div class="stats">{stat_html}</div>
  {''.join(blocks)}
</section>""")

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:24px;color:#263238;background:#fafafa}}
 h1{{font-size:20px;margin:0 0 6px}}
 .lead{{color:#546e7a;font-size:13px;line-height:1.7;margin-bottom:18px;max-width:900px}}
 .lead code{{background:#eceff1;padding:1px 5px;border-radius:3px}}
 .card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:16px 18px;margin-bottom:16px}}
 .card h2{{font-size:16px;margin:0 0 4px}}
 .sub{{color:#607d8b;font-size:12px;margin:0 0 10px}}
 .warn{{color:#e65100;font-size:11px}}
 .stats{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}}
 .stat{{background:#f5f7f8;border-radius:6px;padding:6px 10px;font-size:12px;min-width:120px}}
 .stat span{{display:block;color:#78909c;font-size:11px}}
 .stat b{{font-weight:600}}
 .group h4{{font-size:12px;color:#546e7a;margin:10px 0 4px;font-weight:600}}
 ul{{list-style:none;margin:0;padding:0}}
 li{{font-size:13px;padding:3px 0;border-bottom:1px solid #f2f2f2}}
 li a{{color:#1565c0;text-decoration:none;font-weight:600}}
 li a:hover{{text-decoration:underline}}
 .fn{{color:#90a4ae;font-size:11px;margin-left:8px}}
 li em{{color:#607d8b;font-style:normal;font-size:12px;margin-left:8px}}
 li i{{float:right;color:#b0bec5;font-style:normal;font-size:11px}}
 .foot{{color:#90a4ae;font-size:11px;margin-top:8px}}
</style></head><body>
<h1>{esc(title)}</h1>
<div class="lead">
 <b>先看核对页</b>：<code>.review.html</code> 把段落、和弦、旋律、低音、鼓点画在同一时间轴上，
 橙色=两引擎冲突，红色=被八度修正的音。<br>
 <b>要演奏</b>：<code>.lead.musicxml</code>（旋律+和弦符号）或 <code>.score.musicxml</code>
 （加低音与生成和声），<code>.score.mid</code> 带速度表，可直接进 DAW。<br>
 <b>要分析</b>：<code>.analysis.md</code> 是结论，<code>.map.md</code> 是小节级证据，
 <code>.localkeys.md</code> 核验段级调性，<code>.materials.md</code> 给出材料号。<br>
 <b>边界</b>：旋律是单声部且经八度修正；和声声部由和弦符号生成，不是转写；
 鼓点只显示 onset，不是鼓谱。
</div>
{''.join(cards)}
<p class="foot">由 10_bundle.py 生成 · 只读已有 JSON，不改变任何分析结果</p>
</body></html>"""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="bundle all artifacts into out/index.html")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--title", default="音乐分析 · 产物索引")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    tracks = collect(outdir)
    if not tracks:
        print(f"no *.chords.json under {outdir}", file=sys.stderr)
        sys.exit(1)
    page = render(tracks, args.title, outdir)
    out = outdir / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"[bundle] {len(tracks)} tracks -> {out}")


if __name__ == "__main__":
    main()
