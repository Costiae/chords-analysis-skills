#!/usr/bin/env python3
"""Step 4e: 把分析结果写成可演奏的谱(MusicXML + 带速度表的 MIDI)。

为什么需要这一步
----------------
01e 产出的 MIDI 只是"音符事件":固定 120 BPM、没有拍号、没有小节、没有量化。
它不能当谱用——导入 DAW 或 MuseScore 会整体错位。

本脚本用已有的结构信息(tempo curve / downbeats / meter / key / sections)
把音符落到小节网格上,输出:

  <name>.lead.musicxml   主旋律 + 和弦符号 + 段落标记(功能谱)
  <name>.score.musicxml  主旋律 + 低音 + 由和弦符号生成的和声(三声部总谱骨架)
  <name>.score.mid       带速度表/拍号的 MIDI(可导入 DAW)
  <name>.score.json      量化与清理的统计(可核对)

三声部里,旋律与低音来自音频转写,和声声部是**由和弦符号生成的**
(part 名明确标注 generated),不是从音频里"听"出来的——这样人一眼能看出
哪些是识别结果、哪些是推导结果。

依赖: music21(缺失时只写 MIDI,不写 XML)
用法:
  python 08_score.py out/<name>.chords.json [--voices ...] [--materials ...]
                     [--outdir out] [--no-harmony-part]
"""
import argparse
import json
import sys
from bisect import bisect_right
from pathlib import Path

QUANT = 4                 # 每个四分音符的格数:4 = 十六分音符
MELODY_RANGE = (48, 84)   # C3..C6,人声/主旋律的合理音域
BASS_RANGE = (28, 55)     # E1..G3
HARMONY_RANGE = (48, 72)  # 生成和声的排列音域(C3..C5)

QUALITY_TO_M21 = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "maj7": "maj7", "min7": "m7", "7": "7", "hdim7": "m7b5", "dim7": "dim7",
    "sus2": "sus2", "sus4": "sus4", "6": "6", "min6": "m6", "maj6": "6",
    "minmaj7": "mMaj7",
}


def pc_of(name):
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    v = base[name[0].upper()]
    for ch in name[1:]:
        if ch in "#♯":
            v += 1
        elif ch in "b♭":
            v -= 1
    return v % 12


def split_label(label):
    """'Bb:maj7/F' -> ('Bb', 'maj7', 'F')。"""
    body = label.split("/", 1)
    root, qual = body[0].split(":", 1) if ":" in body[0] else (body[0], "maj")
    bass = body[1] if len(body) > 1 else None
    return root, qual, bass


def to_m21_symbol(label):
    """'Bb:maj7/F' -> 'B-maj7/F'。music21 用 '-' 表示降号根音。"""
    root, qual, bass = split_label(label)
    def m21_root(name):
        return name.replace("♭", "b").replace("♯", "#").replace("b", "-")
    suffix = QUALITY_TO_M21.get(qual, "")
    fig = f"{m21_root(root)}{suffix}"
    if bass:
        fig += "/" + m21_root(bass)
    return fig


# ---------------------------------------------------------------- 时间网格

class Grid:
    """downbeats(秒) -> 小节/拍网格;beat 位置以四分音符为单位。

    谱从**第一个下拍**开始:第一个下拍之前的音频(前奏噪声)不进谱,
    否则第一小节会变成非 4/4 的残小节,导出后节拍全错。
    """

    def __init__(self, downbeats, meter, end_time, bpm=None):
        self.db = [float(x) for x in downbeats if x is not None]
        self.meter = int(meter or 4)
        if not self.db:
            bar = 4 * 60.0 / max(float(bpm or 120.0), 1e-6)
            n = int(end_time // bar) + 2
            self.db = [i * bar for i in range(n)]
        if len(self.db) >= 2:
            tail = sorted(b - a for a, b in zip(self.db, self.db[1:]))
            bar = tail[len(tail) // 2]
        else:
            bar = 4 * 60.0 / max(float(bpm or 120.0), 1e-6)
        while self.db[-1] < end_time:
            self.db.append(self.db[-1] + bar)

    @property
    def start(self):
        return self.db[0]

    @property
    def bars(self):
        return len(self.db) - 1

    def bar_len(self, k):
        return self.db[k + 1] - self.db[k]

    def time_to_pos(self, t):
        if t < self.db[0] - 1e-6:
            return None
        k = bisect_right(self.db, t) - 1
        k = max(0, min(k, self.bars - 1))
        frac = (t - self.db[k]) / max(self.bar_len(k), 1e-6)
        return k * self.meter + frac * self.meter

    def quant(self, t):
        pos = self.time_to_pos(t)
        if pos is None:
            return None
        return round(pos * QUANT) / QUANT

    def bar_of(self, pos):
        return max(0, min(int(pos // self.meter + 1e-9), self.bars - 1))


# ---------------------------------------------------------------- 音高清理

def fold_into(midi, lo, hi):
    while midi > hi:
        midi -= 12
    while midi < lo:
        midi += 12
    return midi


def clean_notes(notes, lo, hi, grid, mono=True):
    """音域折返 + 八度连续性修正 + 单声部化 + 量化。"""
    items = []
    for n in notes:
        m = fold_into(int(n["midi"]), lo, hi)
        items.append({"start": float(n["start"]), "end": float(n["end"]),
                      "midi": m, "orig": int(n["midi"])})
    items.sort(key=lambda n: (n["start"], -n["midi"]))

    # 八度连续性:相邻音跳超过一个八度时,尝试把当前音移 ±12 靠近前一个
    prev = None
    for n in items:
        if prev is not None and abs(n["midi"] - prev) > 12:
            for cand in (n["midi"] - 12, n["midi"] + 12):
                if lo <= cand <= hi and abs(cand - prev) < abs(n["midi"] - prev):
                    n["midi"] = cand
                    break
        prev = n["midi"]

    # 单声部化(旋律/低音都是单声部):重叠处截断前一个音
    if mono:
        out = []
        for n in items:
            if out and n["start"] < out[-1]["end"] - 1e-4:
                out[-1]["end"] = max(out[-1]["start"], n["start"])
                if out[-1]["end"] - out[-1]["start"] < 0.03:
                    out.pop()
            out.append(n)
        items = out

    # 量化
    q = []
    for n in items:
        s = grid.quant(n["start"])
        e = grid.quant(n["end"])
        if s is None or e is None:
            continue
        if e - s < 1.0 / QUANT:
            e = s + 1.0 / QUANT
        if grid.bar_of(s) >= grid.bars:
            continue
        q.append({"pos": s, "end": e, "midi": n["midi"], "orig": n["orig"]})
    q.sort(key=lambda n: n["pos"])
    # 量化后可能有两音落到同一格(0.24s 与 0.26s 都变成 0.25),重叠会让
    # MusicXML 的小节长度超界(同一 voice 里时值相加)。量化后再单声部化一次。
    if mono:
        fixed = []
        for n in q:
            if fixed and n["pos"] < fixed[-1]["end"] - 1e-9:
                fixed[-1]["end"] = n["pos"]
                if fixed[-1]["end"] - fixed[-1]["pos"] < 1.0 / QUANT - 1e-9:
                    fixed.pop()
            fixed.append(n)
        q = fixed
    return q


# ---------------------------------------------------------------- 和声

def chord_pitches(label, lo, hi):
    """由和弦符号生成一个可读的排列(根音在低音区,其余音挤在 lo..hi)。"""
    root, qual, bass = split_label(label)
    r = pc_of(root)
    quality = QUALITY_TO_M21.get(qual, "")
    intervals = {
        "": [0, 4, 7], "m": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
        "maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "7": [0, 4, 7, 10],
        "m7b5": [0, 3, 6, 10], "dim7": [0, 3, 6, 9],
        "sus2": [0, 2, 7], "sus4": [0, 5, 7], "6": [0, 4, 7, 9],
        "m6": [0, 3, 7, 9], "mMaj7": [0, 3, 7, 11],
    }.get(quality, [0, 4, 7])
    pcs = [(r + i) % 12 for i in intervals]

    # 把音级排到 lo..hi 区间,根音尽量低
    pitches = []
    for pc in pcs:
        p = lo + ((pc - lo) % 12)
        while p < lo:
            p += 12
        pitches.append(p)
    pitches = sorted(set(pitches))
    if bass:
        bp = pc_of(bass)
        b = 36 + ((bp - 36) % 12)
        pitches = [b] + [p for p in pitches if p % 12 != bp]
    return pitches


# ---------------------------------------------------------------- 写谱

def build_score(doc, voices, materials, struct=None, with_harmony=True):
    from music21 import (stream, note, chord, meter, key, tempo, harmony,
                         clef, tie, pitch, expressions)
    RehearsalMark = expressions.RehearsalMark

    st = struct or doc.get("structure") or {}
    segs = (doc.get("structure") or {}).get("segments") or []
    song = doc.get("song") or {}
    chords = [c for c in (doc.get("chords") or []) if c["label"] != "N"]
    if not chords:
        return None
    meter_n = int(st.get("meter") or 4)
    end_time = max(c["end"] for c in doc["chords"])
    grid = Grid(st.get("downbeats") or [], meter_n, end_time, bpm=st.get("bpm"))

    # 调号(全曲)与段落调性
    key_str = song.get("key") or "C major"
    def m21_key(ks):
        root, mode = ks.split(" ", 1)
        return key.Key(root.replace("♭", "-").replace("b", "-"), mode)

    melody_notes = ((voices.get("melody") or {}).get("notes") or [])
    bass_notes = ((voices.get("bass") or {}).get("notes") or [])
    melody = clean_notes(melody_notes, *MELODY_RANGE, grid)
    bass = clean_notes(bass_notes, *BASS_RANGE, grid)

    score = stream.Score()
    score.metadata = None
    parts = []

    def new_part(name, clef_obj, program=None):
        p = stream.Part()
        p.partName = name
        p.insert(0, clef_obj)
        if program is not None:
            from music21 import instrument
            p.insert(0, instrument.instrumentFromMidiProgram(program))
        return p

    mel_part = new_part("Melody (auto)", clef.TrebleClef())
    bass_part = new_part("Bass (auto)", clef.BassClef(), 33)
    parts = [mel_part, bass_part]

    # --- 每小节建容器
    def make_measures(part):
        ms = []
        for k in range(grid.bars):
            m = stream.Measure(number=k + 1)
            ms.append(m)
        return ms

    mel_measures = make_measures(mel_part)
    bass_measures = make_measures(bass_part)

    # 调号/拍号
    for part, ms in ((mel_part, mel_measures), (bass_part, bass_measures)):
        ms[0].insert(0, meter.TimeSignature(f"{meter_n}/4"))
        try:
            ms[0].insert(0, m21_key(key_str))
        except Exception:
            pass

    # 段落调性变化 + 排练号
    mat_by_start = {}
    if materials:
        for s in materials.get("segments") or []:
            mat_by_start[round(s["start"], 2)] = s

    for seg in segs:
        pos = grid.quant(seg["start"])
        if pos is None:
            continue
        bar = grid.bar_of(pos)
        off = max(0.0, pos - bar * meter_n)
        label = seg.get("label") or seg.get("name") or ""
        mat = mat_by_start.get(round(seg["start"], 2), {}).get("material")
        text = f"{label}{f' ({mat})' if mat else ''}"
        for ms in (mel_measures, bass_measures):
            ms[bar].insert(off, RehearsalMark(text))
        # 调性变化
        if seg.get("key"):
            prev = None
            for s2 in segs:
                if s2["start"] < seg["start"] and s2.get("key"):
                    prev = s2["key"]
            if prev and prev != seg["key"]:
                try:
                    ms0 = mel_measures[bar]
                    ms0.insert(off, m21_key(seg["key"]))
                except Exception:
                    pass

    # --- 音符落谱(跨小节拆分 + 连音线)
    def add_pitched(measures, part, items, is_chord=None):
        for it in items:
            s, e = it["pos"], it["end"]
            first = True
            while s < e - 1e-9:
                bar = grid.bar_of(s)
                bar_end = (bar + 1) * meter_n
                seg_end = min(e, bar_end)
                off = s - bar * meter_n
                ql = seg_end - s
                if ql <= 1e-9:
                    break
                n = (is_chord(it) if is_chord else note.Note(it["midi"]))
                n.quarterLength = ql
                if is_chord is None and it["midi"] != it["orig"]:
                    n.style.color = "#c0392b"     # 红=被八度修正
                if seg_end < e - 1e-9:
                    n.tie = tie.Tie("start" if first else "continue")
                elif not first:
                    n.tie = tie.Tie("stop")
                measures[bar].insert(off, n)
                s = seg_end
                first = False

    add_pitched(mel_measures, mel_part, melody)
    add_pitched(bass_measures, bass_part, bass)

    # --- 和弦符号(挂到旋律谱上)
    seen = set()
    for c in chords:
        pos = grid.quant(c["start"])
        if pos is None:
            continue
        key_pt = (round(pos, 4), c["label"])
        if key_pt in seen:
            continue
        seen.add(key_pt)
        bar = grid.bar_of(pos)
        off = max(0.0, pos - bar * meter_n)
        try:
            cs = harmony.ChordSymbol(to_m21_symbol(c["label"]))
            mel_measures[bar].insert(off, cs)
        except Exception:
            continue

    # --- 速度表
    curve = st.get("tempo_curve") or []
    last_bpm = None
    for k in range(grid.bars):
        t = grid.db[k]
        bpm = None
        if curve:
            near = min(curve, key=lambda x: abs(x["time"] - t))
            if abs(near["time"] - t) <= 2.0:
                bpm = float(near["bpm"])
        if bpm is None and st.get("bpm"):
            bpm = float(st["bpm"])
        if bpm and (last_bpm is None or abs(bpm - last_bpm) >= 1.0):
            mel_measures[k].insert(0, tempo.MetronomeMark(number=round(bpm, 1)))
            last_bpm = bpm

    # --- 生成的和声声部
    if with_harmony:
        harm_part = new_part("Harmony (generated from symbols)", clef.TrebleClef())
        parts.append(harm_part)
        h_measures = make_measures(harm_part)
        h_measures[0].insert(0, meter.TimeSignature(f"{meter_n}/4"))
        try:
            h_measures[0].insert(0, m21_key(key_str))
        except Exception:
            pass
        items = []
        for c in chords:
            s = grid.quant(c["start"])
            if s is None:
                continue
            e = max(s + 1.0 / QUANT, grid.quant(c["end"]))
            if items and s < items[-1]["end"] - 1e-9:
                items[-1]["end"] = s
                if items[-1]["end"] - items[-1]["pos"] < 1.0 / QUANT - 1e-9:
                    items.pop()
            items.append({"pos": s, "end": e, "label": c["label"]})
        add_pitched(h_measures, harm_part, items,
                    is_chord=lambda it: chord.Chord(chord_pitches(it["label"], *HARMONY_RANGE)))

    for part, ms in zip(parts, [mel_measures, bass_measures] + ([h_measures] if with_harmony else [])):
        for m in ms:
            part.insert((m.number - 1) * meter_n, m)

    # 收敛到总长度:空小节补全休止(否则 MusicXML 会出现 0 长度小节)
    for ms in ([mel_measures, bass_measures] + ([h_measures] if with_harmony else [])):
        for m in ms:
            m.makeRests(fillGaps=True, inPlace=True)
            if m.highestTime < meter_n - 1e-6:
                m.insert(max(0.0, m.highestTime),
                         note.Rest(quarterLength=meter_n - m.highestTime))
    score.insert(0, parts[0])
    for p in parts[1:]:
        score.insert(0, p)
    return score


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="write MusicXML + tempo-mapped MIDI")
    ap.add_argument("chords", help="out/<name>.chords.json")
    ap.add_argument("--voices", default=None)
    ap.add_argument("--materials", default=None)
    ap.add_argument("--struct", default=None, help="out/<name>.struct.json (downbeats list)")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-harmony-part", action="store_true")
    args = ap.parse_args()

    src = Path(args.chords)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        sys.exit(1)
    voices_p = Path(args.voices) if args.voices else src.with_name(
        src.name.replace(".chords.json", ".voices.json"))
    mat_p = Path(args.materials) if args.materials else src.with_name(
        src.name.replace(".chords.json", ".materials.json"))
    struct_p = Path(args.struct) if args.struct else src.with_name(
        src.name.replace(".chords.json", ".struct.json"))
    doc = json.loads(src.read_text(encoding="utf-8"))
    voices = json.loads(voices_p.read_text(encoding="utf-8")) if voices_p.exists() else {}
    mats = json.loads(mat_p.read_text(encoding="utf-8")) if mat_p.exists() else {}
    struct = json.loads(struct_p.read_text(encoding="utf-8")) if struct_p.exists() else {}
    # chords.json 里的 downbeats 只是数量;真正的下拍时间序列在 struct.json
    merged = dict(doc.get("structure") or {})
    for k in ("downbeats", "beats", "tempo_curve", "tempo_range", "bpm", "meter"):
        if k in struct:
            merged[k] = struct[k]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = src.name.replace(".chords.json", "")

    try:
        import music21  # noqa: F401
    except Exception as e:
        print(f"music21 unavailable ({type(e).__name__}); skip MusicXML", file=sys.stderr)
        sys.exit(2)

    try:
        score = build_score(doc, voices, mats, struct=merged,
                            with_harmony=not args.no_harmony_part)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"score build failed: {e}", file=sys.stderr)
        sys.exit(1)

    lead = score
    lead_path = outdir / f"{name}.lead.musicxml"
    full_path = outdir / f"{name}.score.musicxml"
    mid_path = outdir / f"{name}.score.mid"

    # 功能谱:只保留旋律声部(带和弦符号)
    try:
        from music21 import stream
        lead = stream.Score()
        lead.insert(0, score.parts[0])
        lead.write("musicxml", fp=str(lead_path))
    except Exception as e:
        print(f"lead export failed: {e}", file=sys.stderr)

    try:
        score.write("musicxml", fp=str(full_path))
        score.write("midi", fp=str(mid_path))
    except Exception as e:
        print(f"score export failed: {e}", file=sys.stderr)
        sys.exit(1)

    st = merged
    grid = Grid(st.get("downbeats") or [], st.get("meter") or 4,
                max(c["end"] for c in doc["chords"]), bpm=st.get("bpm"))
    summary = {
        "schema": "score-v1",
        "bars": grid.bars,
        "meter": st.get("meter"),
        "quantization": f"1/{QUANT * 4} note ({QUANT}/quarter)",
        "parts": [p.partName for p in score.parts],
        "files": [str(lead_path), str(full_path), str(mid_path)],
    }
    (outdir / f"{name}.score.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] {grid.bars} bars, parts={summary['parts']}")
    print(f"  -> {lead_path}")
    print(f"  -> {full_path}")
    print(f"  -> {mid_path}")


if __name__ == "__main__":
    main()
