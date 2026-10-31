#!/usr/bin/env python3
"""Step 4f: 生成人工核对页(自包含 HTML/SVG,直接用浏览器打开)。

为什么需要
----------
AI 会错,人得能一眼看出来错在哪。这份页面把四层证据并排画在同一时间轴上:

  段落带   : 结构模型的分段 + 材料号 + 段级调性
  和弦轨   : 每个和弦的区间与标签;两引擎冲突的标橙色
  旋律轨   : 音符 piano-roll;被八度折返修正的音标红
  低音轨   : 音符 piano-roll;低音是转位判定的依据

再附一张乐段表(材料/旋律走向/收束级数),方便人工对着音频抽查。

不依赖任何库,输出单文件:
  out/<name>.review.html

用法:
  python 09_review.py out/<name>.chords.json [--outdir out] [--px 8]
"""
import argparse
import html
import json
import sys
from pathlib import Path

PX_PER_SEC = 8.0
MARGIN_L = 130
MARGIN_R = 20
PAD = 18

MELODY_LO, MELODY_HI = 45, 88      # A2..E6
BASS_LO, BASS_HI = 24, 60          # C1..C4

SECTION_COLORS = {
    "intro": "#e8eaf6", "inst": "#ede7f6", "verse": "#e3f2fd",
    "chorus": "#fff3e0", "bridge": "#f3e5f5", "solo": "#e8f5e9",
    "outro": "#fce4ec", "start": "#f5f5f5",
}


def esc(s):
    return html.escape(str(s), quote=True)


def pitch_y(midi, lo, hi, top, height):
    hi = max(hi, lo + 1)
    frac = (midi - lo) / (hi - lo)
    return top + height - frac * height


def fold_into(midi, lo, hi):
    while midi > hi:
        midi -= 12
    while midi < lo:
        midi += 12
    return midi


def build_html(chords_doc, voices, materials, struct, px):
    song = chords_doc.get("song") or {}
    chords = chords_doc.get("chords") or []
    segs = (chords_doc.get("structure") or {}).get("segments") or []
    st = struct or chords_doc.get("structure") or {}
    downbeats = st.get("downbeats") or []
    if isinstance(downbeats, int):
        downbeats = []
    duration = float(song.get("duration_sec") or max(c["end"] for c in chords))
    width = MARGIN_L + duration * px + MARGIN_R

    melody = ((voices.get("melody") or {}).get("notes") or [])
    bass = ((voices.get("bass") or {}).get("notes") or [])

    mat = {}
    for s in (materials or {}).get("segments") or []:
        mat[round(s["start"], 2)] = s

    # 布局
    y_sec = PAD
    h_sec = 34
    y_chord = y_sec + h_sec + 10
    h_chord = 44
    y_mel = y_chord + h_chord + 12
    h_mel = 150
    y_bass = y_mel + h_mel + 12
    h_bass = 90
    y_axis = y_bass + h_bass + 8
    height = y_axis + 44

    x0 = MARGIN_L

    def x(t):
        return x0 + t * px

    parts = []
    add = parts.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'font-family="Segoe UI, Microsoft YaHei, sans-serif" font-size="11">')
    add(f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>')

    # 段落带
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        label = s.get("label") or s.get("name") or ""
        color = SECTION_COLORS.get(label, "#f5f5f5")
        add(f'<rect x="{x(a):.1f}" y="{y_sec}" width="{(b-a)*px:.1f}" height="{h_sec}" '
            f'fill="{color}" stroke="#cfd8dc"/>')
        m = mat.get(round(a, 2), {}).get("material")
        txt = label + (f" · {m}" if m else "")
        if (b - a) * px > 34:
            add(f'<text x="{x(a)+4:.1f}" y="{y_sec+14}" fill="#37474f" font-weight="600">{esc(txt)}</text>')
        k = s.get("key") or ""
        if (b - a) * px > 60 and k:
            add(f'<text x="{x(a)+4:.1f}" y="{y_sec+28}" fill="#607d8b">{esc(k)}</text>')
        add(f'<title>{esc(label)} {a:.1f}-{b:.1f}s {esc(k)}</title>')

    # 和弦轨
    add(f'<text x="8" y="{y_chord+16}" fill="#546e7a">和弦</text>')
    for c in chords:
        a, b = float(c["start"]), float(c["end"])
        if b - a <= 0:
            continue
        conflict = c.get("confidence") == "conflict"
        fill = "#ffe0b2" if conflict else "#e1f5fe"
        stroke = "#f57c00" if conflict else "#b3e5fc"
        if c["label"] == "N":
            fill, stroke = "#eeeeee", "#e0e0e0"
        add(f'<rect x="{x(a):.1f}" y="{y_chord}" width="{(b-a)*px:.1f}" height="{h_chord}" '
            f'fill="{fill}" stroke="{stroke}"/>')
        w = (b - a) * px
        if w > 30 and c["label"] != "N":
            add(f'<text x="{x(a)+2:.1f}" y="{y_chord+16}" fill="#01579b">{esc(c["label"])}</text>')
        tip = f'{c["label"]} {a:.1f}-{b:.1f}s bar {c.get("bar") or "-"}'
        if conflict:
            tip += " (conflict)"
        add(f'<title>{esc(tip)}</title>')

    # 旋律轨
    add(f'<text x="8" y="{y_mel+16}" fill="#546e7a">旋律</text>')
    add(f'<text x="8" y="{y_mel+32}" fill="#90a4ae">自动 · 红色=八度修正</text>')
    for n in melody:
        m = int(n["midi"])
        fixed = not (MELODY_LO <= m <= MELODY_HI)
        mf = fold_into(m, MELODY_LO, MELODY_HI)
        yy = pitch_y(mf, MELODY_LO, MELODY_HI, y_mel, h_mel)
        add(f'<rect x="{x(float(n["start"])):.1f}" y="{yy:.1f}" '
            f'width="{max(1.5, (float(n["end"])-float(n["start"]))*px):.1f}" height="4" '
            f'rx="1" fill="{"#c62828" if fixed else "#1565c0"}"/>')
        add(f'<title>{esc(n.get("name", m))} {float(n["start"]):.2f}s{"" if not fixed else " (octave-corrected)"}</title>')

    # 低音轨
    add(f'<text x="8" y="{y_bass+16}" fill="#546e7a">低音</text>')
    add(f'<text x="8" y="{y_bass+32}" fill="#90a4ae">自动 · 转位依据</text>')
    for n in bass:
        m = int(n["midi"])
        mf = fold_into(m, BASS_LO, BASS_HI)
        yy = pitch_y(mf, BASS_LO, BASS_HI, y_bass, h_bass)
        add(f'<rect x="{x(float(n["start"])):.1f}" y="{yy:.1f}" '
            f'width="{max(1.5, (float(n["end"])-float(n["start"]))*px):.1f}" height="4" '
            f'rx="1" fill="#2e7d32"/>')
        add(f'<title>{esc(n.get("name", m))} {float(n["start"]):.2f}s</title>')

    # 网格与时间轴
    for t in downbeats:
        add(f'<line x1="{x(t):.1f}" y1="{y_sec}" x2="{x(t):.1f}" y2="{y_axis}" '
            f'stroke="#eceff1" stroke-width="1"/>')
    t = 0
    while t <= duration:
        add(f'<line x1="{x(t):.1f}" y1="{y_axis}" x2="{x(t):.1f}" y2="{y_axis+5}" stroke="#90a4ae"/>')
        mm, ss = divmod(int(round(t)), 60)
        add(f'<text x="{x(t):.1f}" y="{y_axis+18}" fill="#607d8b" text-anchor="middle">'
            f'{mm}:{ss:02d}</text>')
        t += 30
    add(f'<line x1="{x0}" y1="{y_axis}" x2="{x(duration):.1f}" y2="{y_axis}" stroke="#b0bec5"/>')

    add("</svg>")
    svg = "\n".join(parts)

    # 乐段表
    rows = []
    for s in segs:
        a = round(float(s["start"]), 2)
        m = mat.get(a, {})
        mel = m.get("melody") or {}
        slope = mel.get("slope_per_10s")
        direction = "-"
        if slope is not None:
            tag = "↑" if slope > 1.5 else ("↓" if slope < -1.5 else "→")
            direction = f"{tag} {slope:+.1f}"
        rows.append(
            f"<tr><td>{esc(s.get('label') or s.get('name'))}</td>"
            f"<td>{float(s['start']):.1f}–{float(s['end']):.1f}s</td>"
            f"<td><b>{esc(m.get('material', '-'))}</b></td>"
            f"<td>{esc(s.get('key') or '-')}</td>"
            f"<td>{esc(direction)}</td>"
            f"<td>{esc(m.get('end_chord', '-'))} ({esc(m.get('end_degree', '-'))})</td></tr>")

    body = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{esc(song.get('title',''))} · 人工核对</title>
<style>
 body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:16px;color:#263238}}
 h1{{font-size:18px;margin:0 0 4px}}
 .meta{{color:#607d8b;font-size:12px;margin-bottom:10px}}
 .legend span{{display:inline-block;margin-right:14px;font-size:12px}}
 .sw{{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:4px}}
 .wrap{{overflow-x:auto;border:1px solid #eceff1;border-radius:6px;margin-bottom:14px}}
 table{{border-collapse:collapse;font-size:12px}}
 th,td{{border-bottom:1px solid #eceff1;padding:5px 10px;text-align:left}}
 th{{background:#fafafa;color:#546e7a;font-weight:600}}
 .note{{color:#78909c;font-size:12px;line-height:1.6}}
 code{{background:#f5f5f5;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>{esc(song.get('title',''))} · 人工核对页</h1>
<div class="meta">
 识别层：和弦 {len([c for c in chords if c['label']!='N'])} 个（冲突标橙）·
 旋律 {len(melody)} 音 · 低音 {len(bass)} 音 · 横轴 {px:.0f}px/秒
</div>
<div class="legend">
 <span><i class="sw" style="background:#e1f5fe;border:1px solid #b3e5fc"></i>和弦</span>
 <span><i class="sw" style="background:#ffe0b2;border:1px solid #f57c00"></i>两引擎冲突</span>
 <span><i class="sw" style="background:#1565c0"></i>旋律</span>
 <span><i class="sw" style="background:#c62828"></i>八度修正</span>
 <span><i class="sw" style="background:#2e7d32"></i>低音</span>
</div>
<div class="wrap">{svg}</div>
<table>
<thead><tr><th>段</th><th>时间</th><th>材料</th><th>段级调</th><th>旋律走向</th><th>收束</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">
 读法：和弦冲突（橙）优先人工复核；旋律红条是被八度折返修正的音，若听觉上不对，说明该处
 八度判定错了；低音是转位（斜杠和弦）的唯一依据。<br>
 本页只画识别结果，不画由和弦符号生成的和声声部（那部分在 <code>.score.musicxml</code> 里，
 已在声部名标注 generated）。
</p>
</body></html>"""
    return body


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="human review page")
    ap.add_argument("chords", help="out/<name>.chords.json")
    ap.add_argument("--voices", default=None)
    ap.add_argument("--materials", default=None)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--px", type=float, default=PX_PER_SEC, help="pixels per second")
    args = ap.parse_args()

    src = Path(args.chords)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        sys.exit(1)
    def side(suffix, given):
        if given:
            return Path(given)
        return src.with_name(src.name.replace(".chords.json", suffix))
    vp = side(".voices.json", args.voices)
    mp = side(".materials.json", args.materials)
    sp = side(".struct.json", args.struct)

    doc = json.loads(src.read_text(encoding="utf-8"))
    voices = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    mats = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    struct = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = src.name.replace(".chords.json", "")

    page = build_html(doc, voices, mats, struct, args.px)
    out = outdir / f"{name}.review.html"
    out.write_text(page, encoding="utf-8")
    print(f"[{name}] -> {out}")


if __name__ == "__main__":
    main()
