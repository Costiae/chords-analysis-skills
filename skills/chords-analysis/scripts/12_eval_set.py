#!/usr/bin/env python3
"""Step 6: 用公开标注集做评测(不是自洽校验)。

为什么必须做
------------
05_eval.py 是拿管线自己生成的 .lab 比,永远 1.0。要判断一个改动是不是真的
更好,必须有**外部标注**。本脚本接数据集,跑真实音频->和弦的完整链路,再用
mir_eval 算 root / majmin / mirex / sevenths。

第一个适配器:POP909(流行曲,CC/MIT,带和弦、拍、MIDI)。
  - 用 MIDI 合成音频(简单的加法合成),时间轴与标注严格对齐;
  - 和弦标注取 chord_midi.txt(与 MIDI 同时间轴);
  - 跑 01_extract.py,读 raw.json,和标注比。
注意:合成音色和真实录音不同,所以这里的数字是**回归基线**(改一处,看有没有
变差),不是"真实歌曲准确率"的最终结论。

用法:
  python 12_eval_set.py pop909 --dir .cache/eval/pop909 --songs 001,002,003 \
         --limit 90 --outdir out/eval [--engine madmom-dcchord,madmom-cnn] [--fetch]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
SR = 44100
POP909_RAW = "https://raw.githubusercontent.com/music-x-lab/POP909-Dataset/master/POP909/{sid}/{name}"


def fetch(url, dst: Path, tries=5):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "chords-analysis-eval"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                f.write(r.read())
            if dst.stat().st_size > 0:
                return True
        except Exception:
            time.sleep(2 + i)
    return False


def ensure_song(base: Path, sid: str, do_fetch: bool):
    d = base / sid
    need = {"mid": f"{sid}.mid", "chords": "chord_midi.txt"}
    ok = True
    for key, name in need.items():
        p = d / name
        if p.exists() and p.stat().st_size > 0:
            continue
        if not do_fetch:
            ok = False
            continue
        ok = fetch(POP909_RAW.format(sid=sid, name=name), p) and ok
    return d if ok else None


def synth_midi(midi_path: Path, wav_path: Path, limit=None):
    import pretty_midi
    import soundfile as sf
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    dur = pm.get_end_time()
    if limit:
        dur = min(dur, float(limit))
    y = np.zeros(int(dur * SR) + SR, dtype=np.float32)

    def tone(f, n):
        t = np.arange(n) / SR
        env = np.minimum(1.0, t / 0.005) * np.exp(-t * 3.2)
        w = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(4 * np.pi * f * t)
             + 0.22 * np.sin(6 * np.pi * f * t) + 0.10 * np.sin(8 * np.pi * f * t))
        return (w * env).astype(np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            if note.start >= dur:
                continue
            end = min(note.end, dur)
            n = int((end - note.start) * SR)
            if n <= 0:
                continue
            a = int(note.start * SR)
            y[a:a + n] += tone(440.0 * 2 ** ((note.pitch - 69) / 12), n) * (note.velocity / 127.0)
    peak = float(np.abs(y).max())
    if peak > 0:
        y /= peak
    sf.write(str(wav_path), y, SR)
    return dur


def load_ref(path: Path, limit):
    iv, lab = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        s, e, l = float(p[0]), float(p[1]), p[2]
        if limit:
            if s >= limit:
                break
            e = min(e, limit)
        if e > s:
            iv.append([s, e])
            lab.append(l)
    return np.array(iv, dtype=float), lab


def load_est(path: Path, limit):
    d = json.loads(path.read_text(encoding="utf-8"))
    iv, lab = [], []
    for c in d.get("chords") or []:
        s, e, l = float(c["start"]), float(c["end"]), c["label"]
        if limit:
            if s >= limit:
                break
            e = min(e, limit)
        if e > s:
            iv.append([s, e])
            lab.append(l)
    return np.array(iv, dtype=float), lab


def evaluate_song(outdir: Path, work: Path, sid: str, limit, est_name=None):
    import mir_eval
    ref_iv, ref_lab = load_ref(work / sid / "chord_midi.txt", limit)
    est_path = outdir / (est_name or f"{sid}.raw.json")
    if not est_path.exists():
        return None
    est_iv, est_lab = load_est(est_path, limit)
    if len(ref_lab) == 0 or len(est_lab) == 0:
        return None
    m = mir_eval.chord.evaluate(ref_iv, ref_lab, est_iv, est_lab)
    keep = ("root", "majmin", "mirex", "sevenths", "triads", "bass")
    return {k: float(m[k]) for k in keep if k in m}


def run_pop909(args):
    base = Path(args.dir)
    ids = [s.strip() for s in args.songs.split(",") if s.strip()]
    work_out = Path(args.outdir) / "raw"
    work_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        song_dir = ensure_song(base, sid, args.fetch)
        if song_dir is None:
            print(f"[{sid}] missing files (use --fetch)", file=sys.stderr)
            continue
        wav = song_dir / f"{sid}.wav"
        if not wav.exists() or wav.stat().st_mtime < (song_dir / f"{sid}.mid").stat().st_mtime:
            print(f"[{sid}] synthesizing audio ...", flush=True)
            synth_midi(song_dir / f"{sid}.mid", wav, limit=args.limit)
        print(f"[{sid}] extracting chords ...", flush=True)
        cmd = [sys.executable, "-W", "ignore", str(SCRIPTS / "01_extract.py"),
               str(wav), "--outdir", str(work_out)]
        if args.engine:
            cmd += ["--engine", args.engine]
        r = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", capture_output=True)
        if r.returncode != 0:
            print(f"[{sid}] extract failed:\n{r.stderr[-800:]}", file=sys.stderr)
            continue
        est_name = f"{sid}.raw.json"
        if args.refine != "none":
            print(f"[{sid}] quality refinement ({args.refine}) ...", flush=True)
            r = subprocess.run([sys.executable, "-W", "ignore",
                                str(SCRIPTS / "01d_quality.py"), str(wav),
                                str(work_out / f"{sid}.raw.json"), "--outdir", str(work_out),
                                "--vocabulary", args.refine], text=True, encoding="utf-8", errors="replace", capture_output=True)
            if r.returncode != 0:
                print(f"[{sid}] quality failed:\n{r.stderr[-600:]}", file=sys.stderr)
            else:
                r = subprocess.run([sys.executable, "-W", "ignore",
                                    str(SCRIPTS / "02_format.py"),
                                    str(work_out / f"{sid}.raw.json"),
                                    "--outdir", str(work_out)], text=True, encoding="utf-8", errors="replace", capture_output=True)
                if r.returncode != 0:
                    print(f"[{sid}] format failed:\n{r.stderr[-600:]}", file=sys.stderr)
                else:
                    est_name = f"{sid}.chords.json"
        sc = evaluate_song(work_out, base, sid, args.limit, est_name)
        if sc is None:
            print(f"[{sid}] no usable chords", file=sys.stderr)
            continue
        sc["song"] = sid
        rows.append(sc)
        print(f"[{sid}] " + " ".join(f"{k}={v:.3f}" for k, v in sc.items() if k != "song"))

    if not rows:
        print("no songs evaluated", file=sys.stderr)
        return 1
    keys = [k for k in ("root", "majmin", "mirex", "sevenths", "triads", "bass")
            if k in rows[0]]
    agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}
    payload = {
        "schema": "eval-set-v1",
        "dataset": "pop909",
        "engine": args.engine or "ensemble",
        "refine": args.refine,
        "limit_seconds": args.limit,
        "synthesized_audio": True,
        "note": ("POP909 audio is synthesized from MIDI for licensing-free reproduction; "
                 "these numbers are a regression baseline, not a real-recording accuracy claim."),
        "songs": rows,
        "aggregate": agg,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = args.refine if args.refine != "none" else "raw"
    (outdir / f"pop909.{tag}.eval.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# POP909 evaluation ({len(rows)} songs, {args.limit}s each)", "",
          f"- engine: `{args.engine or 'ensemble'}`",
          f"- quality refinement: `{args.refine}`",
          "- audio: synthesized from MIDI (scale/level normalized)",
          "- reference: `chord_midi.txt` (same timeline as the MIDI)",
          "", "| song | root | majmin | mirex | sevenths | triads |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['song']} | " + " | ".join(
            f"{r.get(k, float('nan')):.3f}" for k in ("root", "majmin", "mirex", "sevenths", "triads")) + " |")
    md.append("| **mean** | " + " | ".join(
        f"**{agg.get(k, float('nan')):.3f}**" for k in ("root", "majmin", "mirex", "sevenths", "triads")) + " |")
    md += ["", "> 合成音色与真实录音不同,这是回归基线;改动前后比较同一张表才有意义。"]
    (outdir / f"pop909.{tag}.eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\naggregate:", agg)
    print(f"-> {outdir / f'pop909.{tag}.eval.json'}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="evaluate against an annotated dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pop909", help="POP909 pop-song dataset")
    p.add_argument("--dir", default=".cache/eval/pop909")
    p.add_argument("--songs", default="001,002,003")
    p.add_argument("--limit", type=float, default=90.0, help="seconds per song, 0 = full")
    p.add_argument("--outdir", default="out/eval")
    p.add_argument("--engine", default="madmom-dcchord,madmom-cnn",
                   help="engine list passed to 01_extract (default: madmom only, faster)")
    p.add_argument("--refine", default="none", choices=["none", "basic", "extended"],
                   help="run 01d_quality+02_format with this vocabulary after extraction")
    p.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.cmd == "pop909":
        if args.limit <= 0:
            args.limit = None
        sys.exit(run_pop909(args))


if __name__ == "__main__":
    main()
