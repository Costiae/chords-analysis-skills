#!/usr/bin/env python3
"""可插拔和弦识别引擎注册表。

每个引擎实现 `detect(audio_path) -> [{"start", "end", "label"}, ...]`(MIREX/Harte 标签),
并声明元数据。01_extract.py 通过 ENGINES 注册表调用,新增引擎只需:
  1. 写一个 detect 函数(可懒加载重型依赖)
  2. 在 ENGINES 里 register 一行

质量分层(用于 ensemble 主引擎选择与降级顺序):
  tier 1: 验证充分、精度最高,优先作为主引擎
  tier 2: 可用,主要作为交叉验证的第二意见
"""
import os
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = Path(os.environ.get("CHORDS_MODEL_DIR",
                                        Path.home() / ".cache" / "chords-analysis" / "models"))

# BTC 权重仓库与固定版本(可复现:固定 commit,不用 main)
BTC_REPO = "puar-playground/btc-chord"
BTC_REVISION = "d436f2f664f5107cd987774279b8ce171846e376"   # pinned commit (reproducible)
BTC_FILES = ["config.json", "modeling_btc.py", "btc_model_large_voca.pt", "btc_model.pt"]


class Engine:
    def __init__(self, name, detect, tier, note="", vocabulary=None):
        self.name, self.detect, self.tier, self.note = name, detect, tier, note
        self.vocabulary = vocabulary or []


def _hf_endpoint():
    v = os.environ.get("CHORDS_HF_MIRROR") or os.environ.get("HF_ENDPOINT")
    if v and v.lower() == "none":
        return None
    return v or "https://hf-mirror.com"


def btc_local_dir() -> Path:
    return DEFAULT_MODEL_DIR / "btc-chord"


def btc_available() -> bool:
    """权重是否已就绪(不触发下载)。"""
    d = btc_local_dir()
    return all((d / f).exists() for f in ["config.json", "modeling_btc.py",
                                          "btc_model_large_voca.pt"])


def fetch_btc(revision=None) -> bool:
    """下载 BTC 权重与代码(镜像 + 固定 revision)。失败返回 False,绝不抛异常。"""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return False
    d = btc_local_dir()
    d.mkdir(parents=True, exist_ok=True)
    ep = _hf_endpoint()
    try:
        snapshot_download(
            repo_id=BTC_REPO,
            revision=revision or os.environ.get("CHORDS_BTC_REVISION"),
            local_dir=str(d),
            allow_patterns=["*.py", "*.json", "*.pt"],
            endpoint=ep,
            max_workers=4,
        )
        return btc_available()
    except Exception as e:
        print(f"  [btc] weight fetch failed: {type(e).__name__}: {e}", flush=True)
        return False


def _btc_detect(audio):
    """BTC 大词汇(170 和弦)推理。权重缺失时尝试自动拉取一次,仍失败则抛出。"""
    import torch
    from transformers import AutoModel

    if not btc_available():
        if not fetch_btc():
            raise RuntimeError("BTC weights unavailable")

    model = AutoModel.from_pretrained(
        str(btc_local_dir()), trust_remote_code=True,
        large_voca=True, device="cpu")
    preds = model.predict(str(audio))
    out = []
    for p in preds:
        label = p.get("chord") or p.get("label")
        if not label:
            continue
        # Harte 记法 'C' (无性质) 视为 maj
        if ":" not in label and label != "N":
            label = f"{label}:maj"
        out.append({"start": float(p["start"]), "end": float(p["end"]),
                    "label": str(label)})
    return out


# ---------------------------------------------------------------- tier 1

def _madmom_dcchord(audio):
    from madmom.audio.chroma import DeepChromaProcessor
    from madmom.features.chords import DeepChromaChordRecognitionProcessor
    decoded = DeepChromaChordRecognitionProcessor()(DeepChromaProcessor()(str(audio)))
    return [{"start": float(s), "end": float(e), "label": str(l)} for s, e, l in decoded]


def _madmom_cnn(audio):
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    decoded = CRFChordRecognitionProcessor()(CNNChordFeatureProcessor()(str(audio)))
    return [{"start": float(s), "end": float(e), "label": str(l)} for s, e, l in decoded]


# ---------------------------------------------------------------- registry
# 顺序即降级顺序;ensemble 取前两个不同 tier 或不同模型的引擎。

ENGINES = [
    Engine("btc-170", _btc_detect, tier=1,
           note="BTC bi-directional transformer, 170-chord vocabulary (ISMIR19). "
                "Large vocabulary: 7ths/sus/dim/aug. Requires transformers.",
           vocabulary=["maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7",
                       "dim7", "sus2", "sus4", "6", "min6"]),
    Engine("madmom-dcchord", _madmom_dcchord, tier=1,
           note="DeepChroma CNN + chord decoding (ISMIR16). Fallback primary."),
    Engine("madmom-cnn", _madmom_cnn, tier=2,
           note="CNN features + CRF (ISMIR18). Independent second opinion, maj/min only."),
]

BY_NAME = {}
def _rebuild_index():
    BY_NAME.clear()
    BY_NAME.update({e.name: e for e in ENGINES})
_rebuild_index()


def register(name, detect, tier, note="", vocabulary=None):
    """运行时注册引擎(测试或插件用)。"""
    e = Engine(name, detect, tier, note, vocabulary)
    ENGINES.append(e)
    BY_NAME[e.name] = e
    return e


def pick(names):
    """按名称列表取引擎,返回 [(name, detect), ...];忽略未知名称。"""
    return [(e.name, e.detect) for n in names if (e := BY_NAME.get(n))]


def default_ensemble():
    """默认 ensemble:精度最高的两个独立引擎。

    优先 BTC(大词汇)+ madmom-cnn(独立架构,用于交叉验证);
    BTC 权重不可用时退化为 madmom 双引擎。返回 [(name, detect), ...]。"""
    btc = BY_NAME.get("btc-170")
    cnn = BY_NAME.get("madmom-cnn")
    dc = BY_NAME.get("madmom-dcchord")
    if btc and btc_local_dir().joinpath("btc_model_large_voca.pt").exists():
        sel = [btc, cnn] if cnn else [btc]
    else:
        sel = [dc, cnn] if (dc and cnn) else [dc or cnn]
    return [(e.name, e.detect) for e in sel if e]

