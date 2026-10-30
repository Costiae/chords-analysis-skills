#!/usr/bin/env python3
"""Step 5: 和弦识别精度评估(mir_eval)——换模型前后必须可量化。

用法:
  python 05_eval.py <pred.chords.json|pred.lab> <reference.lab> [--vocab majmin|triad|mirex]

参考标注格式:MIREX .lab,每行 `起始秒 结束秒 标签`(标签可为 Harte 记法)。
输出: 控制台指标 + <pred>.eval.json

指标口径:
  root       = 根音正确率(WCSR root)
  majmin     = 根音+大小三和弦正确率
  triad      = 三和弦(含 sus/dim/aug)
  sevenths   = 七和弦
  mirex      = MIREX 综合分
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_lab(path: Path):
    starts, ends, labels = [], [], []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split("\t") if "\t" in line else line.strip().split()
        if len(parts) < 3:
            continue
        s, e, lab = parts[0], parts[1], parts[2]
        try:
            s, e = float(s), float(e)
        except ValueError:
            continue
        starts.append(s)
        ends.append(e)
        labels.append(lab)
    return starts, ends, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("reference")
    ap.add_argument("--vocab", default="majmin",
                    choices=["majmin", "triad", "mirex"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mir_eval
    import numpy as np

    pred_path = Path(args.pred)
    if pred_path.suffix == ".json":
        doc = json.loads(pred_path.read_text(encoding="utf-8-sig"))
        chords = doc["chords"]
        rs = [c["start"] for c in chords]
        re_ = [c["end"] for c in chords]
        rl = [c["label"] for c in chords]
    else:
        rs, re_, rl = load_lab(pred_path)

    es, ee, el = load_lab(Path(args.reference))

    def norm(labels):
        """规范化到 mir_eval 可接受的 Harte 记法。

        MIREX 和弦评分不计转位,因此去掉 /bass;并把引擎别名(6→maj6)统一。"""
        alias = {"6": "maj6", "m6": "min6", "sus": "sus4", "hdim": "hdim7"}
        out = []
        for l in labels:
            if l in ("N", "X", ""):
                out.append("N")
                continue
            core = l.split("/")[0]
            if ":" in core:
                root, qual = core.split(":", 1)
            else:
                root, qual = core, "maj"
            qual = alias.get(qual, qual)
            if qual in ("minmaj7", "minmaj"):
                qual = "minmaj7"
            out.append(f"{root}:{qual}")
        return out

    rs = np.array(rs, dtype=float)
    re_ = np.array(re_, dtype=float)
    es = np.array(es, dtype=float)
    ee = np.array(ee, dtype=float)
    rl, el = norm(rl), norm(el)

    ref_intervals = np.array([es, ee]).T
    est_intervals = np.array([rs, re_]).T
    results = {}
    scores = mir_eval.chord.evaluate(ref_intervals, el, est_intervals, rl,
                                     reduce_to_triadic=(args.vocab != "majmin"))
    for k, v in scores.items():
        results[k] = round(float(v), 4)

    print(f"pred={pred_path.name} ref={Path(args.reference).name} vocab={args.vocab}")
    for k in ("root", "majmin", "triad", "sevenths", "mirex"):
        if k in results:
            print(f"  {k:9s}: {results[k]:.4f}")
    for k, v in results.items():
        if k not in ("root", "majmin", "triad", "sevenths", "mirex"):
            print(f"  {k:9s}: {v:.4f}")

    dest = Path(args.out) if args.out else pred_path.with_suffix(".eval.json")
    dest.write_text(json.dumps({"pred": str(pred_path), "reference": str(args.reference),
                                "vocab": args.vocab, "scores": results},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
