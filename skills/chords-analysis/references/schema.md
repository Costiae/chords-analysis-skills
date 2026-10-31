# 和弦数据 JSON 格式 (chords-v1)

本工作流的标准交换格式。所有引擎输出统一归一化到该 schema,LLM 分析环节读写此格式。

## 顶层结构

```json
{
  "schema": "chords-v1",
  "song": {
    "title": "歌名(来自文件名,可人工改)",
    "source_file": "audio/xxx.mp3",
    "duration_sec": 213.4,
    "engine": "madmom-dcchord|madmom-cnn|chordino|manual",
    "analyzed_at": "2026-10-30T12:00:00",
    "key": "C major",          // 可选,由 03_analyze.py 推断
    "tempo_bpm": 120.0         // 可选
  },
  "chords": [
    { "start": 0.0, "end": 1.6, "label": "N",        "bar": null, "beat": null },
    { "start": 1.6, "end": 2.5, "label": "F:maj",    "bar": 1,    "beat": 1 }
  ],
  "sections": [                // 可选,歌曲段落结构
    { "start": 0.0, "end": 15.2, "name": "intro" },
    { "start": 15.2, "end": 45.0, "name": "verse" }
  ],
  "stats": {                   // 由 02_format.py 生成
    "total_chords": 87,
    "unique_labels": ["C:maj", "G:maj", "A:min"],
    "chord_histogram": { "C:maj": 21, "G:maj": 18 }
  }
}
```

## 字段说明

- `chords[].label`:MIREX 格式 `根音:音质`,根音为 `A..G` 可带 `#`/`b`,音质常见值:
  `maj` `min` `maj7` `min7` `7` `dim` `aug` `sus2` `sus4` `hdim7`,无和弦为 `N`。
- `chords[].confidence`:可选。`"conflict"` 表示 ensemble 交叉验证中两个独立引擎
  对该时段分歧(仅 maj/min 混淆不算冲突),LLM 分析时必须降低确信度并明示用户。
- `ensemble`(可选,`02_format.py` 透传):`{engines, agreement, conflict_seconds}`。
  `agreement` 为两个引擎按时长加权的一致率。**解读基准**:
  - ≥ 0.85:整体可信,conflict 段落单独提示即可
  - 0.65–0.85:大体可信,分析结论避免"确定/必然"措辞,conflict 段落必须标注
  - < 0.65:识别质量差(常见于爵士、电子、纯打击乐),建议人工核对或更换音频源
- `chords[].section`:可选。结构分析模型(01b)给出的段落标签
  (intro/verse/chorus/bridge/outro 等),按和弦起始时间归属。
- `chords[].bass_pc`:可选。低音 stem 在该和弦区间的主导音级(0-11),
  用于转位判定;`label` 里出现 `/X` 表示已确认的转位。
- `chords[].bass_share`:可选。该主导低音音级占整段低音时长的比例(0..1)。
  **`/X` 只有在 `bass_share ≥ 0.5` 时才写入 label**——密集低音线里的经过音
  也会成为"最长音级",不加这个闸门会凭空造出大量假转位。
- **斜杠低音的拼写规则**:按和弦音级拼写,不是直接取等音名。例如 `C#:maj/F`
  应写作 `C#:maj/E#`(C# 大三和弦的三音是 E#)。`harmony.spell_interval()`
  负责这一步;拼不出常规音名(需三升/三降)时才退回等音名。
- `chords[].key_context`:可选。该和弦所处的段落局部调性(拼写依据)。
- `bar` / `beat`:小节号与拍位(来自 01b 的下拍对齐);无结构数据时为 null。
- `structure`(可选):`{engine, bpm, meter, downbeats, segments}`。
  `segments[].key` / `key_confidence` / `key_source` / `key_candidates` 为段落局部调性。
  `key_source` 取值:`estimated`(本段独立判定)、`stable`(与相邻段同调,
  防止关系调摇摆)、`inherited`(段太短/证据不足,沿用上一可信段)。
  **乐段划分以此为准,engine 非 allin1 时无段落标签。**
- `song.key`: **时长加权的摘要,不是分析单位**。分析必须以
  `structure.segments[].key`(乐段级)为起点;`key_by_section` 给出各调的
  累计时长。`key_ambiguous: true` 表示前两名分差过小。
  段级 key 也可能是平均值——用 `06_local_keys.py` 的覆盖率缺口核验。
- `song.key_candidates` / `segments[].key_candidates`:调性候选。`score` 为
  `0.55×Krumhansl 相关 + 0.45×调内覆盖率 + 终止音证据`,同时给出
  `diatonic`(和弦音落在该调内的时长加权比例)与 `tonic_evidence`
  (旋律乐句终止音落在主音的比例)。
  `song.key_ambiguous: true` 表示前两名分差过小,应当视为不确定并引用候选。
  `song.modulations`:全曲出现的所有调性。
- `voices`(可选):`{melody_source, melody_notes, melody_pitch_class_profile,
  melody_phrase_endings, bass_notes, bass_inversions}`。
  完整结构见下节的 `voices-v1`。
- `quality`(可选):`{engine, vocabulary, upgraded, total}`。engine 为
  `template-chroma` 表示离线模板推断;若接入神经大词汇模型则为对应引擎名。
  `chords[].quality_source` / `quality_margin` 标记被升级的和弦及其证据强度。
- `stats.chord_histogram`:按出现时长(秒)加权的和弦统计,四舍五入到 0.1s。

## voices-v1 (`<name>.voices.json`)

```json
{
  "schema": "voices-v1",
  "stem_model": "htdemucs",
  "melody": {
    "source": "vocals",
    "notes": [{"start": 1.2, "end": 1.7, "midi": 69, "name": "A4"}],
    "pitch_class_profile": {"A": 42.1},
    "phrase_endings": [{"time": 12.4, "name": "A", "midi": 69}],
    "key_candidates": [{"key": "A major", "score": 0.81}]
  },
  "bass": {
    "source": "bass",
    "method": "basic-pitch",
    "notes": [{"start": 0.5, "end": 0.68, "midi": 38, "name": "D2",
               "amp": 0.71, "octave_fix": -12,
               "octave_fix_reason": "subharmonic"}],
    "raw_notes": 1544,
    "octave_folded": 38,
    "register_dropped": 0,
    "subharmonic_fixes": 16,
    "register": {"1": 63.0, "2": 133.9, "3": 32.5},
    "chroma_agreement": 0.799,
    "crest": 1.69
  }
}
```

低音层字段(bass.py):

| Field | Meaning |
|---|---|
| `method` | `basic-pitch`(主路径)/ `pyin`(退化路径)/ `basic-pitch (cached)` |
| `raw_notes` | 清理前的原始音符数(用于衡量清理强度) |
| `octave_folded` | 被折回 [C1, G3] 音域的音数 |
| `register_dropped` | 折回后仍越界、被丢弃的音数(通常是高次谐波伪音) |
| `subharmonic_fixes` | 因 f0/2 强于 f0 而降低一个八度的音数(音级不变) |
| `register` | 各八度累计时长(秒),用于确认低音真的在低音区 |
| `chroma_agreement` | 音符音级分布 vs 音频 chroma 的余弦一致性(0..1),自检指标,不是精度 |
| `crest` | 包络峰值/中位数,判断是否为拨弦型乐器 |

约定:

- `notes` 已单声部化(不重叠)并按 `start` 排序;这是转位判定的唯一依据。
- `octave_fix` 记录被移动的半音数,`octave_fix_reason` 说明原因
  (`subharmonic` 或音域折返)。音级(音高类)在 `subharmonic` 修复中不变,
  因此不影响转位结论。
- `chroma_agreement` 只用于**比较同一段音频上不同提取方法**的可信度,
  不能当成 mir_eval 精度分数。低于 0.6 时应在分析报告中提示低音线索可能不可靠。

## localkeys-v1 (`<name>.localkeys.md` / `.json`)

`06_local_keys.py` 的输出,用于回答"段级 key 在这里还成立吗、段内有没有离调"。

```json
{
  "schema": "localkeys-v1",
  "window": 10.0, "hop": 4.0, "threshold": 0.8,
  "breaks": [
    {"start": 128.0, "end": 154.0, "section": "chorus",
     "section_key": "Bb major", "coverage": 0.28,
     "worst_start": 136.0, "worst_coverage": 0.28,
     "alt_key": "A major", "alt_coverage": 0.99,
     "gain": 0.71, "relation": "departure", "alt_reliable": true}
  ],
  "events": [
    {"time": 64.3, "section": "chorus", "section_key": "A major",
     "degree": "VI", "target_key": "F# minor", "strength": "strong",
     "back_to_tonic": 1.6, "kind": "tonicization", "in_break": false}
  ]
}
```

字段约定:

| Field | Meaning |
|---|---|
| `breaks[].coverage` | 段级 key 在该区间的最差窗口覆盖率(和弦音落在调内的时长比例) |
| `breaks[].worst_start` | 覆盖率最低的窗口起点,比区间端点更接近真实转折 |
| `breaks[].relation` | `departure`(音集不同,真的换调)/ `relative`(相对大小调,同音集,不算换调) |
| `breaks[].alt_reliable` | 替代调的覆盖率是否也 ≥ threshold;为 false 时表示"过渡/半音段落,无稳定主音" |
| `events[].degree` | 目标相对**段级** key 的级数(V/vi 的目标记作 VI) |
| `events[].strength` | `strong` = 属七或该级本应小三/减三却出现大三(副属);`weak` 需结合上下文 |
| `events[].back_to_tonic` | 目标之后第一次回到段级主音根音的秒数,用于区分离调与转调 |
| `events[].kind` | `tonicization`(≤4s 回主音)/ `prolonged` / `modulation`(与下一段 key 一致) |
| `events[].in_break` | 该事件落在覆盖率缺口内,段级 key 已失效,需对照 `local_hint` 读 |

**解读规则**:`breaks` 只说明"段级 key 在这里解释不了和弦",不直接等于转调;
`relation: relative` 表示音集相同(相对大小调),合唱/听感层面可能切换了重心,
但和弦数据无法判定,不要写成"转调"。`events` 的级数一律相对段级 key,
不要把 `in_break` 事件的级数当成局部调的级数。

## 衍生文件

| File | Content |
|---|---|
| `<name>.chords.json` | standard chords-v1 (authoritative) |
| `<name>.struct.json` | structure model raw output (segments/beats/BPM/meter/tempo curve) |
| `<name>.voices.json` | melody notes + pitch-class profile + phrase endings + bass notes |
| `<name>.quality.json` | offline quality refinement (template; skipped when BTC is active) |
| `<name>.transcription.json` | Basic Pitch note events per stem |
| `<name>.midi`, `<name>.<stem>.midi` | transcription MIDI (combined + per stem) |
| `<name>.lab` | MIREX .lab time tags (Sonic Visualiser, mir_eval) |
| `<name>.map.md` | **harmonic map**: bar grid + Roman numerals + section progressions + n-grams |
| `<name>.localkeys.md` | **section-key audit**: coverage gaps (where the section key fails) + tonicization events with direction |
| `<name>.chords.eval.json` | evaluation scores vs a reference annotation (05_eval.py) |
| `<name>.analysis.md` | LLM analysis report |

## LLM 分析约定

- 分析/回答一律基于 `chords` 数组,按 `start` 升序。
- **调性以乐段为单位,全曲 `song.key` 只是时长摘要**。报告不要用"这首歌是 X 调"
  开头;先逐段定主音,再说每段如何偏离。段级 key 也可能是两个调的平均值——
  用 `<name>.localkeys.md` 的覆盖率缺口核验,`key_source: stable` 是继承值,
  不是本段独立判定。
- **离调(tonicization)与转调(modulation)要分开说**:短时把某级当主音、随后
  回到段级主音是离调;主音持续并进入下一段才是转调。给出方向(向 vi、向 III…)
  和持续时间。`localkeys` 的 `degree` 一律相对段级 key,不要混用局部调的级数。
- **乐段划分必须来自 `structure.segments`(结构分析模型输出),禁止靠和弦走势猜乐段**;
  若 `structure` 缺失或 `engine` 为 `madmom-beats`,明确说明"无段落标签,以下分段为
  依据和声变化的推测"并在报告开头标注。段内和声归纳才属于 LLM 的工作。
- **准确性优先于完整性**:识别层给出的和弦是"证据"而非"真相"。与音乐常识冲突时
  (如某和弦与前后文毫无功能联系),明确说"此处识别可能有误",不要强行解释。
- `confidence: "conflict"` 的和弦,引用时必须附带"两引擎分歧"提示。
- 报告应包含:逐段调性、段内离调方向、进行模式(如 4536251、卡农进行)、
  段落和声对比、转调点、终止式、节奏与和声密度;末尾附"低置信度段落清单"。
- 引用具体和弦时附带时间戳,如 `2:35 (155.0s) A:min → D:7`。
