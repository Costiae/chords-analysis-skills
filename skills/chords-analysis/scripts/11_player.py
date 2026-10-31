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
  - **时值可读**:空心符头=二分/全音符,实心=四分及更短,符尾=八分/十六分,
    符头右侧一条淡横线表示时值长度
  - **符头白色描边**,把音符从谱线里分离出来
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
SEC_H = 26
CHORD_Y = 52
TREBLE_TOP, TREBLE_BOT = 92, 128          # 五线,线距 9
BASS_TOP, BASS_BOT = 186, 222
SPACE = 9.0                                # 线距
HALF = SPACE / 2.0
HEIGHT = 300
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


def key_signature(ks):
    """调号 -> {letter: accidental}(该调里这个字母应奏的音)。"""
    root, mode = parse_key(ks)
    scale = key_scale(root, mode)
    sig = {}
    for letter, nat in NATURAL_PC.items():
        if nat in scale:
            sig[letter] = 0
        elif (nat + 1) % 12 in scale:
            sig[letter] = 1
        elif (nat - 1) % 12 in scale:
            sig[letter] = -1
        else:
            sig[letter] = 0
    return sig


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


def build_page(doc, voices, materials, struct, audio_rel, px):
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

    global_flats = False
    try:
        root, mode = parse_key(song.get("key") or "C major")
        global_flats = root in (FLAT_MINOR_PCS if mode == "minor" else FLAT_MAJOR_PCS)
    except Exception:
        global_flats = False

    def x(t):
        return LEFT + t * px

    parts = []
    add = parts.append
    add(f'<svg id="score" xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{HEIGHT}" '
        f'font-family="Segoe UI, Microsoft YaHei, sans-serif" font-size="11">')
    add(f'<rect x="0" y="0" width="{width:.0f}" height="{HEIGHT}" fill="#ffffff"/>')

    # 段落带
    for s in segs:
        a, b = float(s["start"]), float(s["end"])
        label = s.get("label") or s.get("name") or ""
        m = mat.get(round(a, 2), {}).get("material")
        ink = {"intro": "#e8eaf6", "inst": "#ede7f6", "verse": "#e3f2fd",
               "chorus": "#fff3e0", "bridge": "#f3e5f5", "solo": "#e8f5e9",
               "outro": "#fce4ec"}.get(label, "#f5f5f5")
        add(f'<rect x="{x(a):.1f}" y="0" width="{(b-a)*px:.1f}" height="{SEC_H}" fill="{ink}" '
            f'stroke="#cfd8dc"/>')
        if (b - a) * px > 40:
            add(f'<text x="{x(a)+4:.1f}" y="17" fill="#37474f" font-weight="600">'
                f'{esc(label)}{" · " + esc(m) if m else ""}</text>')
        add(f'<title>{esc(label)} {a:.1f}-{b:.1f}s {esc(s.get("key") or "")}</title>')

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

    # 调号文字 + 段落调性变化
    add(f'<text x="{LEFT+4}" y="{TREBLE_TOP-16}" fill="#37474f" font-weight="600">'
        f'{esc(song.get("key") or "")}</text>')
    prev_key = song.get("key")
    for s in segs:
        k = s.get("key")
        if k and k != prev_key:
            add(f'<text x="{x(float(s["start"]))+4:.1f}" y="{TREBLE_TOP-16}" fill="#455a64">'
                f'{esc(k)}</text>')
            prev_key = k

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

    # 音符
    curve = st.get("tempo_curve") or []
    bpm0 = float(st.get("bpm") or 120.0)

    def beat_sec_at(t):
        if curve:
            near = min(curve, key=lambda x: abs(float(x["time"]) - t))
            return 60.0 / max(float(near["bpm"]), 1e-6)
        return 60.0 / max(bpm0, 1e-6)

    def note_value(beats):
        if beats >= 3.5:
            return "whole", "全音符"
        if beats >= 1.75:
            return "half", "二分音符"
        if beats >= 0.875:
            return "quarter", "四分音符"
        if beats >= 0.4375:
            return "eighth", "八分音符"
        return "16th", "十六分音符"

    def draw_notes(notes, staff, color, flats, base_sig):
        sig = base_sig
        for n in notes:
            m = int(n["midi"])
            letter, acc, step, show = spell(m, flats, sig)
            y = staff_y(step, staff)
            a, b = float(n["start"]), float(n["end"])
            beats = (b - a) / max(beat_sec_at(a), 1e-6)
            val, val_cn = note_value(beats)
            hollow = val in ("whole", "half")
            stem_up = y > (TREBLE_TOP + TREBLE_BOT) / 2 if staff == "treble" else y > (BASS_TOP + BASS_BOT) / 2
            add(f'<g class="nt {"hollow" if hollow else ""}" data-s="{a:.3f}" data-e="{b:.3f}">')
            for ls in ledger_steps(step, staff):
                ly = staff_y(ls, staff)
                add(f'<line x1="{x(a)-8:.1f}" y1="{ly:.1f}" x2="{x(a)+8:.1f}" y2="{ly:.1f}" '
                    f'stroke="#b0bec5" stroke-width="1"/>')
            # 时值横线:把"这个音持续多久"直接画出来
            dur_px = (b - a) * px
            if dur_px > 8:
                add(f'<line x1="{x(a)+5.5:.1f}" y1="{y:.1f}" x2="{x(b):.1f}" y2="{y:.1f}" '
                    f'stroke="{color}" stroke-width="2" opacity="0.22"/>')
            # 白色描边光晕:让符头从谱线里"浮"出来
            add(f'<ellipse class="halo" cx="{x(a):.1f}" cy="{y:.1f}" rx="6.4" ry="5.0" '
                f'fill="#fff" transform="rotate(-18 {x(a):.1f} {y:.1f})"/>')
            fill = "#fff" if hollow else color
            add(f'<ellipse class="head" cx="{x(a):.1f}" cy="{y:.1f}" rx="5.2" ry="3.8" '
                f'fill="{fill}" stroke="{color}" stroke-width="1.5" '
                f'transform="rotate(-18 {x(a):.1f} {y:.1f})"/>')
            if val != "whole":
                sx = x(a) + 4.9 if stem_up else x(a) - 4.9
                sy = y - 24 if stem_up else y + 24
                add(f'<line x1="{x(a)+(4.9 if stem_up else -4.9):.1f}" y1="{y:.1f}" '
                    f'x2="{sx:.1f}" y2="{sy:.1f}" stroke="{color}" stroke-width="1.4"/>')
                # 符尾:八分 1 条,十六分 2 条
                if val in ("eighth", "16th"):
                    for k in range(1 if val == "eighth" else 2):
                        oy = sy + (0 if stem_up else 0) + (6 * k if stem_up else -6 * k)
                        add(f'<path d="M {sx:.1f} {oy:.1f} q 7 {-4 if stem_up else 4} '
                            f'{7 if stem_up else -7} {12 if stem_up else -12}" fill="none" '
                            f'stroke="{color}" stroke-width="1.6"/>')
            if show:
                add(f'<text x="{x(a)-12:.1f}" y="{y+4:.1f}" fill="{color}" font-size="11">'
                    f'{show}</text>')
            add(f'<title>{esc(n.get("name") or m)} {a:.2f}s · {val_cn} ({b-a:.2f}s)</title></g>')

    mel_sig = key_signature(song.get("key") or "C major")
    draw_notes(melody, "treble", "#1a237e", global_flats, mel_sig)
    draw_notes(bass, "bass", "#1b5e20", global_flats, mel_sig)

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
 #score .nt.on .head{{fill:#e53935;stroke:#e53935}}
 #score .nt.hollow.on .head{{fill:#fff;stroke:#e53935}}
 #score .nt.on line,#score .nt.on path{{stroke:#e53935}}
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
<div class="hint">点击谱面任意位置跳转 · 空格播放/暂停 · 红色=当前音/和弦 · 空心符头=二分/全音符,实心=四分及更短,符尾=八分/十六分,符头右侧横线=时值长度 · 这是由识别结果直接绘制的简化谱,不是排版引擎输出</div>
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
    voices = json.loads(vp.read_text(encoding="utf-8")) if vp.exists() else {}
    mats = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    struct = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    outdir = Path(args.outdir)
    name = src.name.replace(".chords.json", "")

    audio_src = Path(args.audio) if args.audio else find_audio(outdir, name)
    audio_rel = ensure_preview(audio_src, outdir, name) if audio_src else None
    if audio_rel is None and audio_src:
        audio_rel = Path(os.path.relpath(audio_src.resolve(), outdir.resolve())).as_posix()

    page = build_page(doc, voices, mats, struct, audio_rel, args.px)
    out = outdir / f"{name}.player.html"
    out.write_text(page, encoding="utf-8")
    print(f"[{name}] -> {out}  audio={audio_rel}")


if __name__ == "__main__":
    main()
