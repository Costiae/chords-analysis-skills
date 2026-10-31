#!/usr/bin/env python3
"""Step 4h: 播放器式谱面 —— 播放原曲,谱面跟着走。

和 09_review 的区别
-------------------
review 页是"横向总览 + 人工核对";这一页是**跟随播放的谱面**:
音频在放,红色播放头在谱上移动,当前音高亮,当前和弦高亮,视口自动滚动。

实现选择
--------
没有用 MusicXML 渲染库(需要联网/CDN,且排版引擎对量化错误很敏感),
而是**直接用 chords.json / voices.json 里的音符画简化五线谱**:
  - 五线谱线、小节线、拍点、变音记号、符干、加线
  - **按段落调号**:每个音用"它所在乐段"的调号拼写;段内换调(由 06_local_keys
    的可靠缺口给出)也会切开,调号符号画在每次换调处;只有调外音才单独标升降号
  - **段落带第二行是调式**,段内换调显示为 "Bb major → A major",左侧色条按调区分
  - **只画散点**:音高 + 出现时刻;不画符干/符尾/时值线——转写出的音符长度
    不足以判定时值,画出来等于假装知道节奏。横向间距本身就是节奏。
  - **白色描边**,把点从谱线里分离出来
  - 上方和弦符号 + 段落带(标签·材料·调)
  - 音符位置来自 seconds,所以和音频严格同步(不依赖量化)
这是"可读的谱面",不是排版成品。

音频:用 ffmpeg 生成 96k 单声道 AAC 预览(out/audio/<name>.m4a);
没有 ffmpeg 就退化为引用原文件。

用法:
  python 11_player.py out/<name>.chords.json --outdir out [--px 70]
"""
import argparse
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PX_PER_SEC = 70.0
LEFT = 96
TOP = 0
SEC_H = 38
CHORD_Y = 66
TREBLE_TOP, TREBLE_BOT = 102, 138          # 五线,线距 9
BASS_TOP, BASS_BOT = 196, 232
SPACE = 9.0                                # 线距
HALF = SPACE / 2.0
HEIGHT = 314
PLAYHEAD_COLOR = "#e53935"

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LETTER_STEP = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
NATURAL_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FLAT_MAJOR_PCS = {5, 10, 3, 8, 1, 6, 11}
FLAT_MINOR_PCS = {2, 7, 0, 5, 10, 3, 8}


def esc(s):
    return html.escape(str(s), quote=True)


def parse_key(ks):
    try:
        root, mode = ks.split(" ", 1)
    except ValueError:
        return 0, "major"
    root = root.replace("♭", "b").replace("♯", "#")
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[root[0]]
    for ch in root[1:]:
        if ch == "#":
            base += 1
        elif ch == "b":
            base -= 1
    return base % 12, ("minor" if mode.startswith("min") else "major")


def key_scale(root, mode):
    steps = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    pcs = {(root + s) % 12 for s in steps}
    if mode == "minor":
        pcs.add((root + 11) % 12)
    return pcs


def natural_scale(root, mode):
    """自然音阶(小调不含和声导音)——调号用这个。"""
    steps = (0, 2, 3, 5, 7, 8, 10) if mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return {(root + s) % 12 for s in steps}


def key_signature(ks):
    """调号 -> {letter: accidental}。

    不能用"音级是否在音阶里"推,因为半音相邻时会有歧义(比如 D♭ 大调里
    D 同时挨着 D♭ 和 E♭)。调号本质是五度圈:先求调在五度圈上的位置,
    再按 ♯F C G D A E B / ♭B E A D G C F 的顺序落升降号。
    """
    root, mode = parse_key(ks)
    if mode == "minor":
        root = (root + 3) % 12          # 关系大调
    fifths = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6,
              1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
    n = fifths[root]
    sig = {letter: 0 for letter in NATURAL_PC}
    if n > 0:
        for letter in ("F", "C", "G", "D", "A", "E", "B")[:n]:
            sig[letter] = 1
    elif n < 0:
        for letter in ("B", "E", "A", "D", "G", "C", "F")[:-n]:
            sig[letter] = -1
    return sig


def is_flat_key(ks):
    root, mode = parse_key(ks)
    return root in (FLAT_MINOR_PCS if mode == "minor" else FLAT_MAJOR_PCS)


def spell(midi, flats, sig):
    """midi -> (letter, accidental, step, y_offset_from_letter)。"""
    name = (FLAT if flats else SHARP)[midi % 12]
    letter = name[0]
    acc = name.count("#") - name.count("b")
    octave = midi // 12 - 1
    step = octave * 7 + LETTER_STEP[letter]
    # 调号里该字母的默认升降
    show = None
    if acc != 0 and sig.get(letter, 0) != acc:
        show = "♯" if acc > 0 else "♭"
    elif acc == 0 and sig.get(letter, 0) != 0:
        show = "♮"
    return letter, acc, step, show


def staff_y(step, staff):
    if staff == "treble":
        return TREBLE_BOT - (step - (4 * 7 + 2)) * HALF   # 底线 E4
    return BASS_BOT - (step - (2 * 7 + 4)) * HALF         # 底线 G2


def staff_bounds(staff):
    if staff == "treble":
        return 4 * 7 + 2, 4 * 7 + 2 + 8   # E4..F5
    return 2 * 7 + 4, 2 * 7 + 4 + 8       # G2..A3


def ledger_steps(step, staff):
    lo, hi = staff_bounds(staff)
    out = []
    if step < lo:
        s = lo - 2
        while s >= step:
            out.append(s)
            s -= 2
    elif step > hi:
        s = hi + 2
        while s <= step:
            out.append(s)
            s += 2
    return out


def find_audio(outdir: Path, name: str):
    for cand in (outdir.parent / "audio" / f"{name}.flac",
                 outdir.parent / "audio" / f"{name}.wav",
                 outdir.parent / "audio" / f"{name}.mp3",
                 outdir.parent / "audio" / f"{name}.m4a",
                 outdir / ".resampled" / f"{name}.44k.wav"):
        if cand.exists():
            return cand
    return None


def ensure_preview(src: Path, outdir: Path, name: str):
    """生成 96k 单声道 AAC 预览,返回相对 out/ 的路径。"""
    adir = outdir / "audio"
    adir.mkdir(parents=True, exist_ok=True)
    dst = adir / f"{name}.m4a"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return f"audio/{dst.name}"
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src),
                        "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "96k", str(dst)],
                       check=True)
        return f"audio/{dst.name}"
    except Exception:
        return None


def build_page(doc, voices, materials, struct, audio_rel, px, localkeys=None):
    song = doc.get("song") or {}
    chords = [c for c in (doc.get("chords") or []) if c["label"] != "N"]
    segs = (doc.get("structure") or {}).get("segments") or []
    st = struct or {}
    downbeats = st.get("downbeats") or []
    beats = st.get("beats") or []
    if isinstance(downbeats, int):
        downbeats = []
    if isinstance(beats, int):
        beats = []
    duration = float(song.get("duration_sec") or max(c["end"] for c in chords))
    width = LEFT + duration * px + 80

    melody = ((voices.get("melody") or {}).get("notes") or [])
    bass = ((voices.get("bass") or {}).get("notes") or [])
    mat = {round(s["start"], 2): s for s in (materials or {}).get("segments") or []}

    # 段落调性 -> 每个音符用"它所在段落"的调号,而不是全曲调号。
    # 段内换调(06_local_keys 检出的可靠缺口)也插进来,否则整段都会按段级 key 拼写。
    key_regions = []
    for s in segs:
        if s.get("key"):
            key_regions.append((float(s["start"]), s.get("key"), "section"))

    def section_at(t):
        for s in segs:
            if float(s["start"]) <= t < float(s["end"]):
                return s
        return None

    intra = {}          # 段起点 -> [(pivot, alt_key)]
    for br in (localkeys or {}).get("breaks") or []:
        if br.get("relation") != "departure" or not br.get("alt_reliable"):
            continue
        pivot = float(br.get("worst_start") or br["start"])
        alt = br.get("alt_key")
        sec = section_at(pivot)
        if not sec or not alt:
            continue
        key_regions.append((pivot, alt, "intra"))
        intra.setdefault(round(float(sec["start"]), 2), []).append((pivot, alt))
        # 缺口在段结束前就结束时,恢复段级 key
        if float(br["end"]) < float(sec["end"]) - 0.5 and sec.get("key"):
            key_regions.append((float(br["end"]), sec["key"], "restore"))
    key_regions.sort(key=lambda kv: kv[0])

    key_order = []
    for _, k, _ in key_regions:
        if k and k not in key_order:
            key_order.append(k)
    sig_cache = {}

    def sig_for(ks):
        if ks not in sig_cache:
            sig_cache[ks] = key_signature(ks)
        return sig_cache[ks]

    def key_at(t):
        cur = song.get("key") or "C major"
        for start, ks, _ in key_regions:
            if start <= t + 1e-6:
                cur = ks
            else:
                break
        return cur

    # 调性 -> 色条颜色(按出现顺序分配)
    KEY_COLORS = ["#1565c0", "#6a1b9a", "#00695c", "#ad1457", "#ef6c00", "#37474f"]
    key_color = {k: KEY_COLORS[i % len(KEY_COLORS)] for i, k in enumerate(key_order)}

    def x(t):
        return LEFT + t * px

    parts = []
    add = parts.append
    add(f'<svg id="score" xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{HEIGHT}" '
        f'font-family="Segoe UI, Microsoft YaHei, sans-serif" font-size="11">')
    add(f'<rect x="0" y="0" width="{width:.0f}" height="{HEIGHT}" fill="#ffffff"/>')

    # 段落带(标签 · 材料 / 调式),左侧色条 = 调性
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        label = s.get("label") or s.get("name") or ""
        m = mat.get(round(a, 2), {}).get("material")
        ks = s.get("key")
        ink = {"intro": "#e8eaf6", "inst": "#ede7f6", "verse": "#e3f2fd",
               "chorus": "#fff3e0", "bridge": "#f3e5f5", "solo": "#e8f5e9",
               "outro": "#fce4ec"}.get(label, "#f5f5f5")
        add(f'<rect x="{x(a):.1f}" y="0" width="{(b-a)*px:.1f}" height="{SEC_H}" fill="{ink}" '
            f'stroke="#cfd8dc"/>')
        if ks:
            add(f'<rect x="{x(a):.1f}" y="0" width="4" height="{SEC_H}" '
                f'fill="{key_color.get(ks, "#90a4ae")}"/>')
        if (b - a) * px > 40:
            add(f'<text x="{x(a)+8:.1f}" y="15" fill="#37474f" font-weight="600">'
                f'{esc(label)}{" · " + esc(m) if m else ""}</text>')
            if ks:
                alt = intra.get(round(a, 2), [])
                key_txt = esc(ks) + (" → " + esc(alt[0][1]) if alt else "")
                add(f'<text x="{x(a)+8:.1f}" y="30" fill="#546e7a">{key_txt}</text>')
        add(f'<title>{esc(label)} {a:.1f}-{b:.1f}s {esc(ks or "")}</title>')

    # 小节线 / 拍点
    for t in downbeats:
        if 0 <= t <= duration:
            add(f'<line x1="{x(t):.1f}" y1="{SEC_H}" x2="{x(t):.1f}" y2="{BASS_BOT+18}" '
                f'stroke="#cfd8dc" stroke-width="1"/>')
    for t in beats:
        if 0 <= t <= duration:
            add(f'<line x1="{x(t):.1f}" y1="{BASS_BOT+6}" x2="{x(t):.1f}" y2="{BASS_BOT+12}" '
                f'stroke="#eceff1"/>')

    # 谱线
    for staff, top in (("treble", TREBLE_TOP), ("bass", BASS_TOP)):
        for i in range(5):
            y = top + i * SPACE
            add(f'<line x1="{LEFT}" y1="{y}" x2="{x(duration):.1f}" y2="{y}" '
                f'stroke="#b0bec5" stroke-width="1"/>')
    add(f'<text x="8" y="{TREBLE_TOP+4}" fill="#607d8b">旋律</text>')
    add(f'<text x="8" y="{BASS_TOP+4}" fill="#607d8b">低音</text>')

    # 调号:每个调性变化的段落起头画调号符号 + 调名
    SHARP_POS = {"treble": [77, 72, 79, 74, 69, 76, 71],
                 "bass": [53, 48, 55, 50, 45, 52, 47]}
    FLAT_POS = {"treble": [71, 76, 69, 74, 67, 72, 65],
                "bass": [47, 52, 45, 50, 43, 48, 41]}
    key_changes = []
    prev = None
    for start, k, _ in key_regions:
        if k != prev:
            key_changes.append((start, k))
            prev = k
    if not key_changes:
        key_changes = [(0.0, song.get("key") or "C major")]

    for start, ks in key_changes:
        gx = x(start) + 8
        flats = is_flat_key(ks)
        sig = sig_for(ks)
        n_sh = sum(1 for v in sig.values() if v > 0)
        n_fl = sum(1 for v in sig.values() if v < 0)
        add(f'<text x="{gx:.1f}" y="{TREBLE_TOP-18}" fill="#37474f" font-weight="600">'
            f'{esc(ks)}</text>')
        for staff, pos in (("treble", SHARP_POS if n_sh else FLAT_POS),
                           ("bass", SHARP_POS if n_sh else FLAT_POS)):
            count = n_sh if n_sh else n_fl
            for i in range(min(count, 7)):
                midi = pos[staff][i]
                step = spell(midi, flats, sig)[2]
                gy = staff_y(step, staff) + 4
                add(f'<text x="{gx + i * 7:.1f}" y="{gy:.1f}" fill="#455a64" '
                    f'font-size="12">{("♯" if n_sh else "♭")}</text>')

    # 和弦符号
    for c in chords:
        a, b = float(c["start"]), float(c["end"])
        w = (b - a) * px
        if w < 30:
            continue
        cur = " cur" if c.get("confidence") == "conflict" else ""
        add(f'<g class="ch{cur}" data-s="{a:.3f}"><text x="{x(a)+2:.1f}" y="{CHORD_Y}" '
            f'fill="#01579b" font-weight="600">{esc(c["label"])}</text>'
            f'<title>{esc(c["label"])} {a:.1f}s bar {c.get("bar") or "-"}</title></g>')

    # 音符:只画"音高 + 出现时刻"的散点。
    # 时值(符头/符尾)依赖转写出的音符长度,不可靠,所以不做任何节奏编码;
    # 横向间距本身就是节奏,悬停里给出测得的秒数(不是音符时值)。
    def draw_notes(notes, staff, color):
        for n in notes:
            m = int(n["midi"])
            a, b = float(n["start"]), float(n["end"])
            ks = key_at(a)
            flats = is_flat_key(ks)
            letter, acc, step, show = spell(m, flats, sig_for(ks))
            y = staff_y(step, staff)
            add(f'<g class="nt" data-s="{a:.3f}" data-e="{b:.3f}">')
            for ls in ledger_steps(step, staff):
                ly = staff_y(ls, staff)
                add(f'<line x1="{x(a)-8:.1f}" y1="{ly:.1f}" x2="{x(a)+8:.1f}" y2="{ly:.1f}" '
                    f'stroke="#b0bec5" stroke-width="1"/>')
            # 白色描边光晕:让点从谱线里"浮"出来
            add(f'<ellipse class="halo" cx="{x(a):.1f}" cy="{y:.1f}" rx="6.2" ry="5.0" '
                f'fill="#fff"/>')
            add(f'<ellipse class="head" cx="{x(a):.1f}" cy="{y:.1f}" rx="4.8" ry="3.8" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.6"/>')
            if show:
                add(f'<text x="{x(a)-12:.1f}" y="{y+4:.1f}" fill="{color}" font-size="11">'
                    f'{show}</text>')
            add(f'<title>{esc(n.get("name") or m)} {a:.2f}s · 测得 {b-a:.2f}s · {esc(ks)}</title></g>')

    draw_notes(melody, "treble", "#1a237e")
    draw_notes(bass, "bass", "#1b5e20")

    # 播放头
    add(f'<line id="ph" x1="{LEFT}" y1="0" x2="{LEFT}" y2="{HEIGHT}" stroke="{PLAYHEAD_COLOR}" '
        f'stroke-width="2" opacity="0.9"/>')
    add(f'<circle id="phdot" cx="{LEFT}" cy="{SEC_H}" r="3.5" fill="{PLAYHEAD_COLOR}"/>')
    add("</svg>")
    svg = "\n".join(parts)

    audio_tag = (f'<audio id="au" src="{esc(audio_rel)}" preload="auto"></audio>'
                 if audio_rel else '<audio id="au" preload="auto"></audio>')
    audio_note = "" if audio_rel else (
        '<p class="warn">未找到音频文件或 ffmpeg,播放器将静音;谱面仍可拖动查看。</p>')

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{esc(song.get('title',''))} · 跟随播放</title>
<style>
 html,body{{margin:0;height:100%;background:#fafafa;color:#263238;
   font-family:"Segoe UI","Microsoft YaHei",sans-serif}}
 header{{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #e0e0e0;
   padding:8px 14px}}
 h1{{font-size:15px;margin:0 0 6px}}
 .bar{{display:flex;align-items:center;gap:10px;font-size:12px;color:#546e7a}}
 button{{width:34px;height:30px;border:1px solid #cfd8dc;background:#fff;border-radius:6px;
   cursor:pointer;font-size:14px}}
 button:hover{{background:#f5f5f5}}
 input[type=range]{{flex:1;accent-color:#e53935}}
 .time{{font-variant-numeric:tabular-nums;min-width:96px;text-align:right}}
 .wrap{{overflow-x:auto;overflow-y:hidden;border-bottom:1px solid #e0e0e0;background:#fff;
   cursor:pointer}}
 .hint{{font-size:11px;color:#90a4ae;padding:6px 14px}}
 #score .nt{{opacity:.62}}
 #score .nt.on{{opacity:1}}
 #score .nt.on .head{{fill:#e53935;stroke:#fff}}
 #score .ch{{opacity:.45}}
 #score .ch.cur{{opacity:1}}
 #score .ch.cur text{{fill:#e53935}}
 .warn{{color:#e65100;font-size:12px;padding:6px 14px}}
</style></head><body>
<header>
 <h1>{esc(song.get('title',''))} · 跟随播放</h1>
 <div class="bar">
  <button id="pp" title="播放/暂停 (空格)">▶</button>
  <input id="seek" type="range" min="0" max="{duration:.2f}" step="0.01" value="0">
  <span class="time"><span id="cur">0:00</span> / <span id="dur">0:00</span></span>
  <label><input id="follow" type="checkbox" checked> 跟随</label>
 </div>
</header>
{audio_tag}
{audio_note}
<div class="wrap" id="wrap">{svg}</div>
<div class="hint">点击谱面任意位置跳转 · 空格播放/暂停 · 红色=当前音/和弦 · 段落带第二行=调式,左侧色条按调区分 · 谱上 ♯/♭ 是调号,只有调外音才单独标升降号 · 横轴间距即节奏,悬停显示测得秒数</div>
<script>
const au=document.getElementById('au'), wrap=document.getElementById('wrap');
const ph=document.getElementById('ph'), dot=document.getElementById('phdot');
const seek=document.getElementById('seek'), cur=document.getElementById('cur');
const durEl=document.getElementById('dur'), pp=document.getElementById('pp');
const follow=document.getElementById('follow');
const LEFT={LEFT}, PX={px};
const notes=[...document.querySelectorAll('#score .nt')];
const chs=[...document.querySelectorAll('#score .ch')];
let playing=false, raf=null;

function fmt(t){{ if(!isFinite(t)) return '0:00'; const m=Math.floor(t/60), s=Math.floor(t%60);
  return m+':'+String(s).padStart(2,'0'); }}
function total(){{ return isFinite(au.duration)?au.duration:{duration:.3f}; }}
function setDur(){{ durEl.textContent=fmt(total()); seek.max=total(); }}
au.addEventListener('loadedmetadata',setDur);
au.addEventListener('durationchange',setDur);
setDur();

function activeNotes(t){{
  for(const n of notes){{
    const s=+n.dataset.s, e=+n.dataset.e;
    n.classList.toggle('on', t>=s && t<e+0.02);
  }}
  let curCh=null;
  for(const c of chs){{ if(+c.dataset.s<=t+0.001) curCh=c; }}
  for(const c of chs) c.classList.toggle('cur', c===curCh);
}}
function frame(){{
  const t=au.currentTime;
  const x=LEFT+t*PX;
  ph.setAttribute('x1',x); ph.setAttribute('x2',x);
  dot.setAttribute('cx',x);
  activeNotes(t);
  seek.value=t;
  cur.textContent=fmt(t);
  if(follow.checked){{
    const target=x-wrap.clientWidth*0.35;
    if(target>0) wrap.scrollLeft=target; else wrap.scrollLeft=0;
  }}
  if(playing) raf=requestAnimationFrame(frame);
}}
function toggle(){{
  if(playing){{ au.pause(); playing=false; pp.textContent='▶'; cancelAnimationFrame(raf); }}
  else {{ au.play(); playing=true; pp.textContent='❚❚'; raf=requestAnimationFrame(frame); }}
}}
pp.onclick=toggle;
au.addEventListener('ended',()=>{{playing=false;pp.textContent='▶';}});
seek.oninput=()=>{{ au.currentTime=+seek.value; if(!playing) frame(); }};
wrap.onclick=(ev)=>{{
  const r=wrap.querySelector('svg').getBoundingClientRect();
  const x=ev.clientX-r.left;
  const t=Math.max(0,Math.min(total(),(x-LEFT)/PX));
  au.currentTime=t; if(!playing) frame();
}};
document.addEventListener('keydown',(e)=>{{
  if(e.code==='Space'){{ e.preventDefault(); toggle(); }}
}});
frame();
</script>
</body></html>"""


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="audio-following score player (HTML)")
    ap.add_argument("chords", help="out/<name>.chords.json")
    ap.add_argument("--voices", default=None)
    ap.add_argument("--materials", default=None)
    ap.add_argument("--struct", default=None)
    ap.add_argument("--localkeys", default=None, help="out/<name>.localkeys.json (intra-section key changes)")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--px", type=float, default=PX_PER_SEC)
    args = ap.parse_args()

    src = Path(args.chords)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        sys.exit(1)

    def side(suffix, given):
        if given:
            return Path(given)
        return src.with_name(src.name.replace(".chords.json", suffix))

    doc = json.loads(src.read_text(encoding="utf-8"))
    vp, mp, sp = side(".voices.json", args.voices), side(".materials.json", args.materials), side(".struct.json", args.struct)
    lp = side(".localkeys.json", getattr(args, "localkeys", None))
    voices = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    mats = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    struct = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    loc = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    outdir = Path(args.outdir)
    name = src.name.replace(".chords.json", "")

    audio_src = Path(args.audio) if args.audio else find_audio(outdir, name)
    audio_rel = ensure_preview(audio_src, outdir, name) if audio_src else None
    if audio_rel is None and audio_src:
        audio_rel = Path(os.path.relpath(audio_src.resolve(), outdir.resolve())).as_posix()

    page = build_page(doc, voices, mats, struct, audio_rel, args.px, localkeys=loc)
    out = outdir / f"{name}.player.html"
    out.write_text(page, encoding="utf-8")
    print(f"[{name}] -> {out}  audio={audio_rel}")


if __name__ == "__main__":
    main()
