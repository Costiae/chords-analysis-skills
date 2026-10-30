#!/usr/bin/env python3
"""一键跑完整分析管线(每一步独立降级,最后汇总哪些层成功/降级)。

用法:
  python run_all.py <audio文件或目录> [--outdir out] [--skip stems,transcribe,quality]

步骤顺序:
  01_extract      和弦 ensemble(BTC-170 + madmom / madmom 双引擎)
  01b_structure   结构(段落/拍/BPM/拍号/速度曲线)
  01c_voices      旋律 + 低音(demucs + pyin)
  01d_quality     质量推断(模板;BTC 成功时自动跳过,避免覆盖大词汇结果)
  01e_transcribe  扒谱(Basic Pitch,per-stem)
  02_format       融合为 chords-v1
  03_summarize    摘要
  04_map          和声地图
"""
import argparse, json, subprocess, sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aiff", ".aif", ".wma"}


def find_audio(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def py() -> str:
    return sys.executable


def preflight():
    """环境预检:解释器可用性 + 关键依赖。损坏的环境要立刻暴露,而不是跑到一半崩。"""
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    ok = True
    if not Path(sys.executable).exists():
        print("[FATAL] interpreter missing — conda env was removed/damaged. "
              "Re-run scripts/setup_env.py", file=sys.stderr)
        ok = False
    checks = [("madmom", "chords"), ("librosa", "audio io"),
              ("allin1_infer", "structure"), ("transformers", "BTC-170"),
              ("basic_pitch", "transcription"), ("mir_eval", "evaluation")]
    for mod, purpose in checks:
        try:
            __import__(mod)
            print(f"  ok   {mod:14s} ({purpose})")
        except Exception:
            print(f"  MISS {mod:14s} ({purpose}) -> that layer will degrade")
    if not ok:
        sys.exit(2)
    print()


def run(label, args):
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([py(), "-W", "ignore", *args], text=True)
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'DEGRADED'}] {label}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--skip", default="",
                    help="逗号分隔,可含 stems/transcribe/quality/map")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    files = find_audio(Path(args.input))
    if not files:
        print("no audio found", file=sys.stderr)
        sys.exit(1)

    summary = []
    for audio in files:
        stem = audio.stem
        print(f"\n########## {audio.name} ##########")
        layers = {}

        layers["chords"] = run("01_extract (chords)", [
            str(SCRIPTS / "01_extract.py"), str(audio), "--outdir", args.outdir])
        layers["structure"] = run("01b_structure", [
            str(SCRIPTS / "01b_structure.py"), str(audio), "--outdir", args.outdir])

        if "stems" not in skip:
            layers["voices"] = run("01c_voices (melody+bass)", [
                str(SCRIPTS / "01c_voices.py"), str(audio), "--outdir", args.outdir])

        # BTC 大词汇已生效时,模板质量推断会被 02_format 优先覆盖,跳过更省时
        raw = Path(args.outdir) / f"{stem}.raw.json"
        engine = ""
        if raw.exists():
            engine = json.loads(raw.read_text(encoding="utf-8"))["song"].get("engine", "")
        if "quality" not in skip and "btc" not in engine:
            layers["quality"] = run("01d_quality", [
                str(SCRIPTS / "01d_quality.py"), str(audio), str(raw),
                "--outdir", args.outdir])

        if "transcribe" not in skip:
            layers["transcription"] = run("01e_transcribe (basic-pitch)", [
                str(SCRIPTS / "01e_transcribe.py"), str(audio), "--outdir", args.outdir])

        layers["format"] = run("02_format", [
            str(SCRIPTS / "02_format.py"), str(raw), "--outdir", args.outdir])
        layers["summary"] = run("03_summarize", [
            str(SCRIPTS / "03_summarize.py"),
            str(Path(args.outdir) / f"{stem}.chords.json")])
        if "map" not in skip:
            layers["map"] = run("04_map", [
                str(SCRIPTS / "04_map.py"),
                str(Path(args.outdir) / f"{stem}.chords.json"),
                "--outdir", args.outdir])

        summary.append((audio.name, engine, layers))

    print("\n================ PIPELINE SUMMARY ================")
    for name, engine, layers in summary:
        bad = [k for k, v in layers.items() if not v]
        print(f"{name}\n  engine: {engine or '?'}")
        layer_txt = ", ".join("{}={}".format(k, "OK" if v else "SKIP")
                              for k, v in layers.items())
        print("  layers: " + layer_txt)
        if bad:
            print(f"  DEGRADED: {', '.join(bad)} (下游已按降级链继续)")


if __name__ == "__main__":
    main()
