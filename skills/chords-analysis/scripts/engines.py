#!/usr/bin/env python3
"""可插拔和弦识别引擎注册表。

每个引擎实现 `detect(audio_path) -> [{"start", "end", "label"}, ...]`(MIREX 标签),
并声明元数据。01_extract.py 通过 ENGINES 注册表调用,新增引擎只需:
  1. 写一个 detect 函数(可懒加载重型依赖)
  2. 在 ENGINES 里 register 一行
未来接入更好模型(autochord on Linux、ChordFormer、BTC-SLIM、商业 API 等)时
不动管线其它部分。

质量分层(用于 ensemble 主引擎选择与降级顺序):
  tier 1: 验证充分、精度最高,优先作为主引擎
  tier 2: 可用,主要作为交叉验证的第二意见
"""
import importlib


class Engine:
    def __init__(self, name, detect, tier, note=""):
        self.name, self.detect, self.tier, self.note = name, detect, tier, note
        # detect 内部 import,保证单个引擎缺依赖不影响其它引擎


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
    Engine("madmom-dcchord", _madmom_dcchord, tier=1,
           note="DeepChroma CNN + chord decoding (ISMIR16). Primary engine."),
    Engine("madmom-cnn", _madmom_cnn, tier=2,
           note="CNN features + CRF (ISMIR18). Independent second opinion, maj/min only."),
    # 未来引擎示例(取消注释并实现 detect 即可):
    # Engine("chordformer", _chordformer, tier=1, note="Conformer, large vocabulary"),
    # Engine("btc-slim", _btc_slim, tier=2, note="Bi-directional transformer"),
]

BY_NAME = {}
def _rebuild_index():
    BY_NAME.clear()
    BY_NAME.update({e.name: e for e in ENGINES})
_rebuild_index()


def register(name, detect, tier, note=""):
    """运行时注册引擎(测试或插件用)。"""
    e = Engine(name, detect, tier, note)
    ENGINES.append(e)
    BY_NAME[e.name] = e
    return e


def pick(names):
    """按名称列表取引擎,返回 [(name, detect), ...];忽略未知名称。"""
    return [(e.name, e.detect) for n in names if (e := BY_NAME.get(n))]


def default_ensemble():
    """默认 ensemble 组合:两个模型最独立的高 tier 引擎。返回 [(name, detect), ...]。"""
    primary = ENGINES[0]
    others = [e for e in ENGINES[1:] if e.tier <= 2]
    sel = [primary] + (others[:1] if others else [])
    return [(e.name, e.detect) for e in sel]
