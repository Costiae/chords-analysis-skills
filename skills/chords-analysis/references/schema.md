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
- `bar` / `beat`:若提供了节拍对齐(可选步骤),标注该和弦起始落在第几小节第几拍;否则为 null。
- `stats.chord_histogram`:按出现时长(秒)加权的和弦统计,四舍五入到 0.1s。

## 衍生文件

| 文件 | 内容 |
|---|---|
| `<name>.chords.json` | 上面的标准格式(权威数据) |
| `<name>.lab` | MIREX .lab 时间标签,便于外部工具(Sonic Visualiser 等) |
| `<name>.analysis.md` | LLM 生成的分析报告 |

## LLM 分析约定

- 分析/回答一律基于 `chords` 数组,按 `start` 升序。
- 报告应包含:调性判断与依据、进行模式(如 4536251、卡农进行)、段落和声对比、
  转调/离调点、终止式、节奏与和声密度。
- 引用具体和弦时附带时间戳,如 `2:35 (155.0s) A:min → D:7`。
