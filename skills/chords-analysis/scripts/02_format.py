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
from typing import Optional

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


def process(path: Path, outdir: Path, beats_file: Optional[Path], struct_file: Optional[Path],
            voices_file: Optional[Path] = None, quality_file: Optional[Path] = None,
            transcription_file: Optional[Path] = None, force_quality: bool = False):
    sys.path.insert(0, str(Path(__file__).parent))
    import harmony as H

    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    chords = merge_short(doc["chords"])

    # 时长加权直方图
    hist = defaultdict(float)
    for c in chords:
        hist[c["label"]] += c["end"] - c["start"]
    hist = {k: round(v, 1) for k, v in sorted(hist.items(), key=lambda x: -x[1])}

    # 结构层(优先 01b_structure.py 产物;否则 --bars 显式文件)
    struct = None
    if struct_file and struct_file.exists():
        struct = json.loads(struct_file.read_text(encoding="utf-8-sig"))
    beats, downbeats, segments = None, None, None
    if struct:
        beats = struct.get("beats")
        downbeats = struct.get("downbeats") or []
        segments = struct.get("segments") or []
    elif beats_file and beats_file.exists():
        beats = json.loads(beats_file.read_text(encoding="utf-8"))["beats"]

    # 旋律/低音层(01c_voices.py)
    voices = None
    if voices_file and voices_file.exists():
        voices = json.loads(voices_file.read_text(encoding="utf-8-sig"))
    melody_notes = ((voices or {}).get("melody") or {}).get("notes") or []
    bass_notes = ((voices or {}).get("bass") or {}).get("notes") or []

    # 大词汇和弦质量(01d_quality.py;神经大词汇模型可用时优先于它)
    quality_doc = None
    engine = (doc.get("song") or {}).get("engine", "")
    if quality_file and quality_file.exists() and (force_quality or "btc" not in engine.lower()):
        quality_doc = json.loads(quality_file.read_text(encoding="utf-8-sig"))
    quality_by_start = {}
    for q in ((quality_doc or {}).get("chords") or []):
        quality_by_start[round(q["start"], 3)] = q

    def section_at(t):
        if not segments:
            return None
        for s in segments:
            if s["start"] <= t < s["end"]:
                return s["label"]
        return None

    out_chords = []
    for c in chords:
        bar = beat = None
        if downbeats:
            # 下拍已知:小节号 = 下拍索引,拍内位置按 beats 推
            i = bisect.bisect_right(downbeats, c["start"] + 1e-6) - 1
            if i >= 0:
                j = bisect.bisect_right(beats or downbeats, c["start"] + 1e-6) - 1
                beat = max(1, j - i + 1)
                bar = i + 1
        elif beats:
            i = bisect.bisect_right(beats, c["start"]) - 1
            if i >= 0:
                bar, beat = divmod(i, 4)
                bar, beat = bar + 1, beat + 1
        if bass_notes:
            bass_pc, bass_w, bass_tot = H.dominant_bass_pc(bass_notes, c["start"], c["end"])
            bass_share = round(bass_w / bass_tot, 2) if bass_tot else None
        else:
            bass_pc, bass_share = None, None
        label = c["label"]
        qsource = None
        qmargin = None
        q = quality_by_start.get(round(c["start"], 3))
        if q and label != "N":
            root, qual0 = H.parse_label(label)
            if q["quality"] != qual0:
                label = f"{root}:{q['quality']}"
                qsource = "template-chroma"
                qmargin = q["margin"]
        entry = {"start": c["start"], "end": c["end"], "label": label,
                 "bar": bar, "beat": beat,
                 "section": section_at(c["start"]),
                 "bass_pc": bass_pc, "bass_share": bass_share}
        if qsource:
            entry["quality_source"] = qsource
            entry["quality_margin"] = qmargin
        out_chords.append(entry)

    # 集成验证(ensemble)结果透传 + 冲突时长统计
    ens = doc.get("ensemble")
    conflicts = None
    if ens:
        out_chords_by = {(c["start"], c["end"]): c for c in out_chords}
        for it in ens["intervals"]:
            if it["agree"]:
                continue
            # 找覆盖该时段的主引擎和弦,标记冲突
            for c in out_chords:
                if c["start"] <= it["start"] and c["end"] >= it["end"]:
                    c.setdefault("confidence", "conflict")
                    break
        conf_secs = sum(c["end"] - c["start"] for c in out_chords
                        if c.get("confidence") == "conflict")
        conflicts = {"engines": ens["engines"], "agreement": ens["agreement"],
                     "conflict_seconds": round(conf_secs, 1)}

    song = dict(doc["song"])
    song_dur = float(song.get("duration_sec") or 1.0)
    # ---- 调性:和弦 + 旋律双证据 + 乐句终止音,sharp/flat 两套拼写取更优 ----
    melody_prof = None
    if melody_notes:
        melody_prof = {pc: 0.0 for pc in H.PC_SHARP}
        for n in melody_notes:
            melody_prof[H.pc_name(n["midi"] % 12)] += n["end"] - n["start"]
    phrase_ends = ((voices or {}).get("melody") or {}).get("phrase_endings") or []
    end_weight = defaultdict(float)
    for e in phrase_ends:
        end_weight[e["midi"] % 12] += 1.0

    def combined_profile(use_flats):
        prof = H.chord_profile(hist, use_flats=use_flats)
        if melody_prof:
            mel = {H.pc_name(H.pc_index(k), use_flats): v for k, v in melody_prof.items()}
            cs, ms = sum(prof.values()) or 1, sum(mel.values()) or 1
            # 旋律稀疏时(纯器乐/人声很轻)降低其权重:两三个音不足以改写调性
            mel_w = 1.2 * min(1.0, sum(melody_prof.values()) / max(1e-6, song_dur))
            for k in prof:
                prof[k] += mel.get(k, 0.0) * (cs / ms) * mel_w
        return prof

    def rank_keys(use_flats, top=4):
        """相对大/小调同音集无法区分时,用乐句终止音的属主音权重打破平局。"""
        cands = H.key_candidates(combined_profile(use_flats), top=24, use_flats=use_flats)
        total_ends = sum(end_weight.values()) or 1
        # 终止音证据至少 3 个样本才满权重
        end_scale = min(1.0, len(phrase_ends) / 3.0)
        for c in cands:
            tonic_share = end_weight.get(H.pc_index(c["root"]), 0.0) / total_ends
            # 相对关系调(大/小)同主音,终止音证据能直接分开
            c["tonic_evidence"] = round(tonic_share, 3)
            c["diatonic"] = H.diatonic_coverage(out_chords, c["key"])
            # Krumhansl 相关 + 调内覆盖率 + 旋律终止音证据
            c["score"] = round(0.55 * c["score"] + 0.45 * c["diatonic"]
                               + 0.6 * tonic_share * end_scale, 3)
        cands.sort(key=lambda x: -x["score"])
        return cands[:top]

    cands_sharp = rank_keys(False)
    cands_flat = rank_keys(True)
    best = max(cands_sharp[:1] + cands_flat[:1], key=lambda x: x["score"]) if (cands_sharp or cands_flat) else None
    key = H.canonical_key_name(best["key"]) if best else None
    use_flats = H.is_flat_key(key) if key else False
    key_cands = rank_keys(use_flats)

    # ---- 分段局部调性(先算,便于按段落拼写和弦) ----
    if segments:
        for seg in segments:
            seg_chords = [c for c in out_chords if seg["start"] <= c["start"] < seg["end"]]
            if not seg_chords:
                continue
            seg_hist = defaultdict(float)
            for c in seg_chords:
                seg_hist[c["label"]] += c["end"] - c["start"]
            seg_use_flats = H.is_flat_key(key) if key else use_flats
            seg_prof = H.chord_profile(seg_hist, use_flats=seg_use_flats)
            mel_notes = [n for n in melody_notes if seg["start"] <= n["start"] < seg["end"]]
            if mel_notes:
                mel = {}
                for n in mel_notes:
                    pc = H.pc_name(n["midi"] % 12, seg_use_flats)
                    mel[pc] = mel.get(pc, 0.0) + (n["end"] - n["start"])
                cs, ms = sum(seg_prof.values()) or 1, sum(mel.values()) or 1
                # 同上:旋律时长占本段比例越低,权重越小(器乐段几乎不参与)
                mel_w = 1.2 * min(1.0, sum(mel.values()) / max(1e-6, seg["end"] - seg["start"]))
                for k in seg_prof:
                    seg_prof[k] += mel.get(k, 0.0) * (cs / ms) * mel_w
            seg_ends = [n for n in phrase_ends
                        if seg["start"] <= n["time"] < seg["end"]]
            # 终止音证据至少 3 个样本才满权重,避免单个音决定调性
            ev_scale = min(1.0, len(seg_ends) / 3.0)
            seg_keys = H.key_candidates(seg_prof, top=24, use_flats=seg_use_flats)
            tot = len(seg_ends) or 1
            for c in seg_keys:
                raw_ev = sum(1 for e in seg_ends
                             if e["midi"] % 12 == H.pc_index(c["root"])) / tot
                ev = raw_ev * ev_scale
                c["corr"] = c["score"]
                c["tonic_evidence"] = round(raw_ev, 3)
                c["diatonic"] = H.diatonic_coverage(seg_chords, c["key"])
                c["score"] = round(0.55 * c["corr"] + 0.45 * c["diatonic"] + 0.25 * ev, 3)
            seg_keys.sort(key=lambda x: -x["score"])
            if not seg_keys:
                continue
            dur = seg["end"] - seg["start"]
            strong = (dur >= 12 and len(seg_chords) >= 4
                      and seg_keys[0]["score"] >= 0.5)
            # 稳定性先验:上一段调性在本段得分接近最优时保持不变
            # (解决 A major ↔ F# minor / Bb major ↔ G minor 这类关系调摇摆)
            prev_key = None
            for prev in segments:
                if prev is seg:
                    break
                if prev.get("key"):
                    prev_key = prev["key"]
            stable = None
            if strong and prev_key:
                prev_e = sum(1 for e in seg_ends
                             if e["midi"] % 12 == H.pc_index(prev_key.split()[0])) / tot
                alt = (0.55 * H.key_score(seg_prof, prev_key, seg_use_flats)
                       + 0.45 * H.diatonic_coverage(seg_chords, prev_key)
                       + 0.25 * prev_e * ev_scale)
                if seg_keys[0]["score"] - alt < 0.12:
                    stable = H.canonical_key_name(prev_key)
            if strong:
                seg["key"] = stable or H.canonical_key_name(seg_keys[0]["key"])
                seg["key_confidence"] = seg_keys[0]["score"]
                seg["key_source"] = "stable" if stable else "estimated"
                seg["key_candidates"] = [{"key": H.canonical_key_name(k["key"]),
                                          "score": k["score"],
                                          "diatonic": k.get("diatonic")}
                                         for k in seg_keys[:2]]
            else:
                # 短段/弱证据:继承上一个可信段落的调性(避免半音拼写漂移)
                inherited = key if key else H.canonical_key_name(seg_keys[0]["key"])
                for prev in segments:
                    if prev is seg:
                        break
                    if prev.get("key") and prev.get("key_source") == "estimated":
                        inherited = prev["key"]
                seg["key"] = inherited
                seg["key_confidence"] = seg_keys[0]["score"]
                seg["key_source"] = "inherited"
                seg["key_candidates"] = [{"key": H.canonical_key_name(k["key"]),
                                          "score": k["score"],
                                          "diatonic": k.get("diatonic")}
                                         for k in seg_keys[:2]]

    # ---- 等音规范化(按段落调性)+ 转位(斜杠和弦) ----
    def key_for_time(t):
        for s in (segments or []):
            if s["start"] <= t < s["end"] and s.get("key"):
                return s["key"]
        return key

    # 全局调性:段落调性按时长加权(比整曲 KK 更稳,也能暴露双调性)
    if segments:
        dur_by_key = defaultdict(float)
        for s in segments:
            if s.get("key"):
                dur_by_key[s["key"]] += s["end"] - s["start"]
        if dur_by_key:
            ranked = sorted(dur_by_key.items(), key=lambda x: -x[1])
            key = ranked[0][0]
            song["key_by_section"] = [{"key": k, "seconds": round(v, 1)}
                                      for k, v in ranked]
            total = sum(dur_by_key.values())
            if len(ranked) > 1 and ranked[1][1] / max(1e-9, ranked[0][1]) > 0.6:
                song["key_ambiguous"] = True

    for c in out_chords:
        k = key_for_time(c["start"])
        f = H.is_flat_key(k) if k else use_flats
        pcs = {H.pc_index(p) for p in H.label_pitch_classes(c["label"], f)}
        # 只有低音在**和弦音**上、且占到该和弦低音时长的 50% 才写成转位;
        # 密集低音线里的经过音不应该制造斜杠和弦
        slash = (c["bass_pc"] if (c["bass_pc"] is not None and c["bass_pc"] in pcs
                                  and (c.get("bass_share") or 0) >= 0.5) else None)
        c["label"] = H.respell_label(c["label"], f, slash)
        c["key_context"] = k

    hist = {}
    for c in out_chords:
        hist[c["label"]] = round(hist.get(c["label"], 0.0) + (c["end"] - c["start"]), 1)
    hist = dict(sorted(hist.items(), key=lambda x: -x[1]))

    if key:
        song["key"] = key
        song["key_candidates"] = [{"key": H.canonical_key_name(k["key"]),
                                   "score": k["score"],
                                   "diatonic": k.get("diatonic"),
                                   "tonic_evidence": k.get("tonic_evidence")}
                                  for k in key_cands[:3]]
        song["key_evidence"] = {
            "chords": len(out_chords),
            "melody_notes": len(melody_notes),
            "phrase_endings": len(phrase_ends),
            "relative_keys": H.relative_key_candidates(key),
        }
        # 转调提示:分段调性与全曲不一致时列出
        seg_keys = {s["key"] for s in (segments or []) if s.get("key")}
        if len(seg_keys) > 1:
            song["modulations"] = sorted(seg_keys)

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
    if conflicts:
        result["ensemble"] = conflicts
    if struct:
        result["structure"] = {
            "engine": struct.get("engine"),
            "bpm": struct.get("bpm"),
            "meter": struct.get("meter"),
            "tempo_range": struct.get("tempo_range"),
            "tempo_curve": struct.get("tempo_curve"),
            "downbeats": len(downbeats),
            "segments": segments,
        }
        if struct.get("bpm"):
            result["song"]["tempo_bpm"] = struct["bpm"]
        if struct.get("meter"):
            result["song"]["meter"] = struct["meter"]
    if voices:
        result["voices"] = {
            "melody_source": (voices.get("melody") or {}).get("source"),
            "melody_notes": len(melody_notes),
            "melody_pitch_class_profile": (voices.get("melody") or {}).get("pitch_class_profile"),
            "melody_phrase_endings": ((voices.get("melody") or {}).get("phrase_endings") or [])[:40],
            "bass_notes": len(bass_notes),
            "bass_inversions": sum(1 for c in out_chords if "/" in c["label"]),
        }
    if quality_doc:
        upgraded = sum(1 for c in out_chords if c.get("quality_source") == "template-chroma")
        result["quality"] = {
            "engine": quality_doc.get("engine"),
            "vocabulary": quality_doc.get("vocabulary"),
            "upgraded": upgraded,
            "total": len(quality_doc.get("chords") or []),
        }
    if transcription_file and transcription_file.exists():
        tr = json.loads(transcription_file.read_text(encoding="utf-8-sig"))
        result["transcription"] = {
            "engine": tr.get("engine"),
            "notes_total": len(tr.get("notes") or []),
            "stems": {k: v.get("notes") for k, v in (tr.get("stems") or {}).items()},
            "midi": tr.get("midi"),
        }
    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.stem.removesuffix(".raw")
    dest = outdir / f"{stem}.chords.json"
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # MIREX .lab(等音规范化后的标签)
    lab = outdir / f"{stem}.lab"
    lab.write_text("\n".join(f"{c['start']:.3f}\t{c['end']:.3f}\t{H.to_harte(c['label'])}"
                             for c in out_chords), encoding="utf-8")

    # 控制台摘要
    top = ", ".join(f"{k}({fmt_sec(v)})" for k, v in list(hist.items())[:8])
    print(f"{path.stem}: {len(out_chords)} chords | key={key or '?'}")
    print(f"  top: {top}")
    if conflicts:
        print(f"  ensemble: agreement={conflicts['agreement']:.1%}, "
              f"conflict={conflicts['conflict_seconds']}s — 见 chords[].confidence")
    if segments:
        seg_line = " ".join(f"{s['label']}[{fmt_sec(s['start'])}]" for s in segments)
        print(f"  structure: {seg_line}")
    if voices:
        print(f"  voices: melody={len(melody_notes)} notes, bass={len(bass_notes)} notes, "
              f"inversions={sum(1 for c in out_chords if '/' in c['label'])}")
    print(f"  -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--bars", default=None, help="beats json, {'beats':[...]}")
    ap.add_argument("--struct", default=None,
                    help="01b_structure.py 输出 (.struct.json);缺省时自动在 outdir 找同名文件")
    ap.add_argument("--voices", default=None,
                    help="01c_voices.py 输出 (.voices.json);缺省时自动在 outdir 找同名文件")
    ap.add_argument("--quality", default=None,
                    help="01d_quality.py 输出 (.quality.json);缺省时自动在 outdir 找同名文件")
    ap.add_argument("--transcription", default=None,
                    help="01e_transcribe.py 输出 (.transcription.json);缺省时自动发现")
    args = ap.parse_args()
    src = Path(args.input)
    files = [src] if src.is_file() else sorted(src.glob("*.raw.json"))
    beats = Path(args.bars) if args.bars else None
    for f in files:
        stem = f.stem.removesuffix(".raw")
        sf = Path(args.struct) if args.struct else Path(args.outdir) / f"{stem}.struct.json"
        vf = Path(args.voices) if args.voices else Path(args.outdir) / f"{stem}.voices.json"
        qf = Path(args.quality) if args.quality else Path(args.outdir) / f"{stem}.quality.json"
        tf = Path(args.transcription) if args.transcription else \
             Path(args.outdir) / f"{stem}.transcription.json"
        process(f, Path(args.outdir), beats,
                sf if sf.exists() else None,
                vf if vf.exists() else None,
                qf if qf.exists() else None,
                tf if tf.exists() else None,
                force_quality=bool(args.quality))


if __name__ == "__main__":
    main()
