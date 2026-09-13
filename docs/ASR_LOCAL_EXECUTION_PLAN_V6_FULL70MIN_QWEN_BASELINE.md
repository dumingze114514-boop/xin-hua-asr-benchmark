# 心花 ASR 本地执行方案 V6：70 分钟全量 + Qwen3 原版对照

本版本替代 V5。当前 Demo 约 70 分钟，**所有上游候选都必须完整处理 70 分钟**，不得用 30/60 分钟抽样替代完整转录。

## 1. 唯一目标

最终输出完整：`speaker + start_ms + end_ms + text`。

当前首要瓶颈是：模糊、远场、轻声、混响条件下的中文文字识别准确率。说话人和时间戳是固定下游，不做模型海选。

## 2. 四个上游 ASR：并行，不串联

同一份 RAW 16 kHz mono PCM16 音频分别完整跑：

1. `Qwen/Qwen3-ASR-1.7B`：原版千问基线。
2. `Mega-ASR with router`：基于 Qwen3-ASR 的远场/噪声/混响鲁棒化版本。
3. `FireRedTeam/FireRedASR2-LLM`：普通话/meeting 强候选；显存不够时只允许回退 `FireRedASR2-AED`。
4. `zai-org/GLM-ASR-Nano-2512`：低音量/quiet speech 专长候选。

它们是四份独立逐字稿：

```text
RAW 70min -> Qwen3      -> qwen3_full
RAW 70min -> Mega-ASR   -> mega_asr_full
RAW 70min -> FireRed2   -> firered2_full
RAW 70min -> GLM-ASR    -> glm_asr_full
```

禁止 `Qwen -> Mega -> FireRed -> GLM` 串联。

## 3. 为什么新增原版 Qwen3

Mega-ASR 的底座是 Qwen3-ASR-1.7B。V6 必须直接回答：

```text
Qwen3 original vs Mega-ASR
```

即鲁棒适配是否真的改善心花远场模糊音频。

最终报告必须单独给出：

- `CER_AB`
- `CER_A`
- `CER_B`（模糊但人工可辨）
- `Deletion Rate`
- `Backchannel Recall`

并计算：

```text
Mega improvement = Qwen3 error - Mega error
```

如果 Mega 不如原版 Qwen，生产直接用原版 Qwen。

## 4. 四个候选全部跑满 70 分钟

必须产生：

```text
outputs/qwen3_full/transcript_full.jsonl
outputs/mega_asr_full/transcript_full.jsonl
outputs/firered2_full/transcript_full.jsonl
outputs/glm_asr_full/transcript_full.jsonl
```

每套还必须有：

```text
coverage_report.json
run_config.json
```

`coverage_report` 必须证明源音频 0 ms 到结尾全部经过该模型处理；静音无文字不算缺失。

不得：

- 跑前 30 分钟就淘汰；
- 跑前 60 分钟就停止；
- 只跑困难片段；
- 因 chunk 报错跳过时间段。

## 5. 推理方式

四个模型都吃同一个 RAW 工作音频，不做预先降噪、normalize、EQ、AGC、compressor。

外层可以用不超过约 5 分钟的 block 做断点恢复，但模型内部优先采用各自官方 offline/long-form 推理路径。不要为了形式公平强制所有模型使用同一个 30 秒切块或同一个 VAD。

## 6. Gold 与评价

机器转录必须全量。人工 Gold 建议最终覆盖整 70 分钟；若暂时不足，也不能阻塞四套全量机器转录。

Gold 标：

```text
A = 清晰可辨
B = 模糊但人工最终可确认
C = 人工也无法可靠确认
```

主指标只在 A+B 上计算；C 报告时长占比。

核心指标：

- `CER_AB`
- `CER_B`
- Substitution / Deletion / Insertion
- Backchannel Recall

特别重视 `CER_B` 与 `Deletion Rate`。

中文 CER：Unicode NFKC、英文小写、去标点/空格；normalized 额外统一简繁和常规数字；必须保留“嗯、啊、呃、哦”、重复与自我修正。

## 7. 选择唯一 PRIMARY_ASR

有全量 Gold 时：

1. 先比较 `CER_AB`；
2. 第一与第二差 >= 0.5 个绝对百分点，直接选第一；
3. 若 <0.5，依次比较 `CER_B`、Deletion Rate、Backchannel Recall、运行稳定性。

最终只选一个 `PRIMARY_ASR`，第一阶段不做 ensemble。

没有全量 Gold 时，仍完成四套 70 分钟结果，并生成 `model_disagreement_queue.jsonl`；只能给 provisional best，不得宣称最终最优。

## 8. 唯一条件增强

只有 PRIMARY_ASR 的 `CER_B > 12%` 或 `Deletion Rate > 6%` 才测试 `MossFormerGAN_SE_16K`。

只有同时满足：

```text
CER_B 改善 >= 1.5 个绝对百分点
Deletion Rate 不增加
Backchannel Recall 下降 <= 5%
```

才保留增强；若保留，必须把完整 70 分钟重新生成增强版 PRIMARY_ASR。否则永久弃用，不搜索第二个增强模型。

## 9. 固定下游：时间戳

最终文字统一：

```text
PRIMARY_ASR full transcript
-> Qwen/Qwen3-ForcedAligner-0.6B
```

ForcedAligner 只对齐时间，不改写文字。必须处理全部最终文本并恢复绝对 `start_ms/end_ms`。

随机 200 token 只用于质量检查：MAE <=100 ms 通过；100-250 ms 可用但报告限制；>250 ms 触发 timestamp failure。

## 10. 固定下游：双人说话人

整段 RAW 70 分钟：

```text
-> DiariZen-Large-s80
num_speakers = 2
```

人工抽查 10 分钟只用于评价，不是只处理 10 分钟。speaker confusion <=5% 接受；>5% 停止并报告，不自动换模型。

## 11. 最终融合

```text
PRIMARY_ASR 文本
-> Qwen3 ForcedAligner 时间戳
+
RAW 音频
-> DiariZen speaker timeline
=
最终 speaker + time + text
```

每个 token 按时间 overlap 分配 speaker；无 overlap 时前后搜索 300 ms；仍无法确定标 `uncertain`。同 speaker 且 gap <800 ms 合并 utterance。

每场只需人工一次映射：`speaker_0/speaker_1 -> therapist/client`。

## 12. 最终文件

必须生成：

```text
outputs/final/demo_full_transcript.jsonl
outputs/final/demo_full_transcript.md
outputs/final/demo_full_transcript.txt
outputs/review/model_disagreement_queue.jsonl
outputs/review/missed_speech_intervals.jsonl
```

`missed_speech_intervals` 用 DiariZen speech timeline 对照最终 ASR/alignment，找“检测到有人讲话但没有文字”的区间。

## 13. Codex 精确执行树

```text
0 环境检查
1 统一生成 RAW 16k mono PCM16
2 创建完整 70min processing manifest
3 Qwen3 完整 70min
4 Mega-ASR 完整 70min
5 FireRedASR2 完整 70min
6 GLM-ASR 完整 70min
7 四份 transcript + coverage reports
8 计算/补充 Gold 评价与 Qwen3-vs-Mega 对照
9 选唯一 PRIMARY_ASR
10 条件满足时才跑 MossFormerGAN；若通过则完整 70min 重跑
11 最终文字全量 Qwen3-ForcedAligner
12 RAW 70min 全量 DiariZen 两人分离
13 全量 token-speaker 融合
14 人工一次确认 therapist/client
15 输出完整 JSONL/MD/TXT + disagreement/missed-speech queues
16 完整性检查
```

## 14. 必须阅读

- Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Qwen3-ASR paper: https://arxiv.org/abs/2601.21337
- Mega-ASR: https://github.com/xzf-thu/Mega-ASR
- Mega-ASR paper: https://arxiv.org/abs/2605.19833
- Voices-in-the-Wild: https://github.com/xzf-thu/Voices-in-the-Wild-Bench
- FireRedASR2S: https://github.com/FireRedTeam/FireRedASR2S
- FireRedASR2 paper: https://arxiv.org/abs/2603.10420
- GLM-ASR: https://github.com/zai-org/GLM-ASR
- GLM-ASR model: https://huggingface.co/zai-org/GLM-ASR-Nano-2512
- DiariZen: https://github.com/ntuspeechlab/Diarizen
- CER edit distance: https://github.com/jitsi/jiwer

条件触发才看 ClearerVoice：
https://github.com/modelscope/ClearerVoice-Studio

## 15. 不要自行扩展

第一阶段不要增加 Paraformer、SenseVoice、Whisper、MOSS、Step-Audio2、pyannote、3D-Speaker、云 API、LLM 纠错、ROVER、视频 active speaker。

只有四个 ASR 都明显失败、DiariZen >5% speaker confusion、或 ForcedAligner >250 ms 时，写 `reports/SECOND_STAGE_TRIGGER.md` 并停止，不自行加模型。

## 16. V6 完成条件

只有以下全部满足才算当前 Demo 完成：

```text
Qwen3 70min 完成
Mega-ASR 70min 完成
FireRedASR2 70min 完成
GLM-ASR 70min 完成
四份完整候选逐字稿完成
Qwen3 vs Mega 消融对照完成
唯一 PRIMARY_ASR 已选
最终文字全量时间对齐完成
DiariZen 全量 70min 完成
最终 speaker + time + text 全量融合完成
JSONL + MD + TXT 完成
分歧与漏转 review queue 完成
无未处理源音频区间
```
