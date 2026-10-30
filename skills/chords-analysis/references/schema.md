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
- `chords[].key_context`:可选。该和弦所处的段落局部调性(拼写依据)。
- `bar` / `beat`:小节号与拍位(来自 01b 的下拍对齐);无结构数据时为 null。
- `structure`(可选):`{engine, bpm, meter, downbeats, segments}`。
  `segments[].key` / `key_confidence` / `key_candidates` 为段落局部调性。
  **乐段划分以此为准,engine 非 allin1 时无段落标签。**
- `song.key_candidates`:调性候选(含 `tonic_evidence` 乐句终止音证据权重)。
  `song.modulations`:全曲出现的所有调性。
- `voices`(可选):`{melody_source, melody_notes, melody_pitch_class_profile,
  melody_phrase_endings, bass_notes, bass_inversions}`。
- `stats.chord_histogram`:按出现时长(秒)加权的和弦统计,四舍五入到 0.1s。

## 衍生文件

| 文件 | 内容 |
|---|---|
| `<name>.chords.json` | 上面的标准格式(权威数据) |
| `<name>.struct.json` | 结构模型原始输出(段落/拍/BPM/拍号) |
| `<name>.voices.json` | 旋律音符 + 音级分布 + 乐句终止音 + 低音音符 |
| `<name>.lab` | MIREX .lab 时间标签,便于外部工具(Sonic Visualiser 等) |
| `<name>.map.md` | **和声地图**:小节网格 + 级数 + 段落进行 + 重复进行挖掘 |
| `<name>.analysis.md` | LLM 生成的分析报告 |

## LLM 分析约定

- 分析/回答一律基于 `chords` 数组,按 `start` 升序。
- **乐段划分必须来自 `structure.segments`(结构分析模型输出),禁止靠和弦走势猜乐段**;
  若 `structure` 缺失或 `engine` 为 `madmom-beats`,明确说明"无段落标签,以下分段为
  依据和声变化的推测"并在报告开头标注。段内和声归纳才属于 LLM 的工作。
- **准确性优先于完整性**:识别层给出的和弦是"证据"而非"真相"。与音乐常识冲突时
  (如某和弦与前后文毫无功能联系),明确说"此处识别可能有误",不要强行解释。
- `confidence: "conflict"` 的和弦,引用时必须附带"两引擎分歧"提示。
- 报告应包含:调性判断与依据、进行模式(如 4536251、卡农进行)、段落和声对比、
  转调/离调点、终止式、节奏与和声密度;末尾附"低置信度段落清单"。
- 引用具体和弦时附带时间戳,如 `2:35 (155.0s) A:min → D:7`。
