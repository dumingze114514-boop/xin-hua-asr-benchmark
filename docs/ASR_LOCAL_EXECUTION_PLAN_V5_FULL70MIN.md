# 心花中文远场双人咨询 ASR 本地部署执行方案 V5

> 本版本替代 V4。当前只有一个约 70 分钟 Demo，因此**必须完整处理整段音频**。任何 30/60 分钟抽样都只能用于质量分析，不能替代完整转录。

## 1. 最终目标

对当前约 70 分钟心理咨询 Demo，生成从 `00:00:00` 到音频结束的完整本地转录：

```text
speaker + start_ms + end_ms + text
```

最终必须同时输出：

```text
outputs/final/demo_full_transcript.jsonl
outputs/final/demo_full_transcript.md
outputs/final/demo_full_transcript.txt
reports/FULL_70MIN_COMPLETENESS.md
```

禁止只做 benchmark 后停止。

## 2. 固定本地模型池

三个 ASR 都必须完整跑满 70 分钟：

1. `Mega-ASR with router`
   - https://github.com/xzf-thu/Mega-ASR
   - https://arxiv.org/abs/2605.19833
   - https://github.com/xzf-thu/Voices-in-the-Wild-Bench
2. `FireRedTeam/FireRedASR2-LLM`
   - https://github.com/FireRedTeam/FireRedASR2S
   - https://arxiv.org/abs/2603.10420
   - 若 GPU 无法稳定运行 8.3B，只允许回退 `FireRedASR2-AED`，不能跳过 FireRed 系列。
3. `zai-org/GLM-ASR-Nano-2512`
   - https://github.com/zai-org/GLM-ASR
   - https://huggingface.co/zai-org/GLM-ASR-Nano-2512

下游固定：

4. `Qwen/Qwen3-ForcedAligner-0.6B`
   - https://github.com/QwenLM/Qwen3-ASR
5. `DiariZen-Large-s80`, `num_speakers=2`
   - https://github.com/ntuspeechlab/Diarizen

唯一条件增强：

6. `MossFormerGAN_SE_16K`
   - https://github.com/modelscope/ClearerVoice-Studio

第一阶段禁止加入 Whisper、Paraformer、SenseVoice、Fun-ASR-Nano、MOSS-Transcribe-Diarize、pyannote、3D-Speaker、云 API、LLM 文本纠错、热词、微调、ROVER、视频 active-speaker、speech separation。

## 3. 原始音频与工作副本

原始文件永久保留。只生成一次标准工作音频：

```bash
ffmpeg -i INPUT -vn -ac 1 -ar 16000 -c:a pcm_s16le data/work/demo_16k_mono.wav
```

禁止默认 denoise、normalize、EQ、AGC、compressor、silence removal。

## 4. 完整 70 分钟 manifest

建立：

```text
outputs/manifests/demo_full_manifest.jsonl
```

外层按约 4.5 分钟切块：

```text
target = 4 min 30 s
min = 3 min 30 s
max = 4 min 50 s
```

在目标边界 ±10 s 内寻找低能量自然边界，相邻块保留 1000 ms overlap。必须保存绝对 `start_ms/end_ms`。

硬性要求：

```text
manifest 覆盖 0 ms 到 source_duration_ms
missing intervals = []
```

模型内部仍应使用各自官方推荐 offline / long-form 路径，不为了统一而破坏其内部处理能力。

## 5. 三个 ASR 必须全量运行

创建：

```text
outputs/mega_asr_full/
outputs/firered2_full/
outputs/glm_asr_full/
```

每个模型都必须处理 manifest 中所有 chunk。每个 chunk 独立落盘，支持断点恢复：

```text
chunk_000.json
chunk_001.json
...
```

重新运行时已成功且输入 hash 匹配的 chunk 不重跑；失败 chunk 自动重试，但不得跳过。

三套最终都必须生成：

```text
transcript_full.jsonl
coverage_report.json
```

其中：

```text
coverage_ratio == 1.0
missing_intervals == []
```

任何“只跑前 30/60 分钟”的行为都算任务未完成。

## 6. Gold 与可听度

如果已有人工 Gold，优先对整个 70 分钟使用。每段增加：

```text
audibility = A / B / C
```

- A：一次正常播放即可确认；
- B：模糊，但耳机/重听后两名标注者可确认；
- C：两名标注者仍无法可靠确定。

主指标只计算 `A+B`，C 只报告时长比例。

如果暂时没有全量 Gold：**不能停止完整转录**。先完成三个模型的 70 分钟结果，并生成模型分歧队列供后续人工复核。

## 7. 全量 ASR 评价

如果全量 Gold 可用，对 70 分钟计算：

```text
CER_AB
CER_A
CER_B
Substitution Rate
Deletion Rate
Insertion Rate
Backchannel Recall
```

中文 CER normalization：Unicode NFKC、英文小写、去标点/空格、简繁统一、常见数字统一；保留“嗯/啊/呃/哦”、重复、自我修正。

可复用 edit distance：
https://github.com/jitsi/jiwer

若暂时没有全量 Gold，只能输出 `provisional_best`，不得宣称最终最佳。

## 8. PRIMARY_ASR 选择

有全量 Gold 时：

1. 先比较 70 分钟 pooled `CER_AB`；
2. 差距 >= 0.5 个绝对百分点：选 CER 最低；
3. 差距 < 0.5：比较 B 类模糊语音 CER；
4. 再比较 Deletion Rate；
5. 再比较 Backchannel Recall；
6. 最终必须选一个 `PRIMARY_ASR`。

不做 ensemble。

## 9. 唯一增强分支

只有当 PRIMARY_ASR：

```text
B 类 CER > 12%
```

或：

```text
Deletion Rate > 6%
```

才测试 `MossFormerGAN_SE_16K`。

先在最困难部分验证。只有同时满足：

```text
CER 下降 >= 1.5 absolute points
Deletion Rate 不增加
Backchannel Recall 下降 <= 5%
```

才保留增强。

一旦保留，必须重新处理**完整 70 分钟**；最终不能只有部分片段增强。否则全量仍用 RAW。

## 10. 时间戳必须覆盖完整 70 分钟

最终 PRIMARY_ASR 文本全部送入：

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

按不超过约 5 分钟的块做 forced alignment，然后恢复绝对时间。

随机 200 token 只用于评估边界误差，不代表只处理 200 token。

判定：

```text
MAE <= 100 ms：接受
100–250 ms：接受，但报告限制
>250 ms：timestamp failure
```

## 11. Diarization 必须覆盖完整 70 分钟

```text
RAW demo_16k_mono.wav
→ DiariZen-Large-s80
→ num_speakers=2
```

完整输出 RTTM/JSON timeline。10 分钟人工抽查只用于 sanity check，不得只 diarize 10 分钟。

若 speaker confusion <=5%，直接接受；若 >5%，停止并报告 speaker failure，不自行增加其它 speaker 模型。

## 12. 全量 speaker-text 融合

对整个 70 分钟：

```text
ForcedAligner token timeline
+
DiariZen speaker timeline
↓
每个 token 分配 speaker
↓
utterance reconstruction
```

规则：

1. token 与哪个 speaker 区间 overlap 最大就归谁；
2. tie 用 token 中点；
3. 空白区前后搜索 300 ms；
4. 仍无法确定标 `speaker=uncertain`；
5. same speaker 且 gap <800 ms 合并；
6. speaker change 立即断开。

每场只需人工一次把 `speaker_0/speaker_1` 映射为 `therapist/client`。

## 13. 必须检测“有语音但没文字”

比较 DiariZen/VAD speech interval 与 ASR text interval。若存在检测到语音但 ASR 无文字的区域，全部写入：

```text
outputs/review/missed_speech_intervals.jsonl
```

不能因为模型没报错就当作完成。

## 14. 必须生成三模型分歧队列

整个 70 分钟生成：

```text
outputs/review/model_disagreement_queue.jsonl
```

优先包括：

- 一个模型为空、另两个有文字；
- 三模型差异大；
- 专有词分歧；
- 模糊/低音量片段；
- 短回应分歧。

这是人工校正的优先队列。

## 15. 最终文件

机器可读：

```text
outputs/final/demo_full_transcript.jsonl
```

人工可读：

```text
outputs/final/demo_full_transcript.md
```

纯文本：

```text
outputs/final/demo_full_transcript.txt
```

Markdown 格式示例：

```text
[00:02:03.456 - 00:02:06.789] 来访者
我最近其实还是睡得不太好。

[00:02:07.100 - 00:02:08.230] 咨询师
嗯。
```

## 16. 完整性报告

生成：

```text
reports/FULL_70MIN_COMPLETENESS.md
```

至少包含：

```text
Source duration
Each ASR covered duration
Final ASR covered duration
Aligned text coverage
Diarization processed full source: yes/no
Final transcript start/end
Missing intervals
Empty-ASR-but-speech intervals
Uncertain speaker intervals
Failed chunks
```

完成条件：

```text
三个 ASR 都完成 70 min
ASR coverage = 100%
Forced alignment 覆盖完整最终文本
DiariZen 处理完整 70 min
Final transcript 到达音频结束
Missing unprocessed intervals = []
```

## 17. 精确执行树

```text
START
 ↓
0. 环境检查
 ↓
1. 原始 70 min → 16 kHz mono PCM16
 ↓
2. 建立覆盖完整 70 min 的 manifest
 ↓
3. 三模型都完整跑 70 min
   ├─ Mega-ASR
   ├─ FireRedASR2
   └─ GLM-ASR
 ↓
4. 三套完整 transcript + coverage report
 ↓
5. 有 Gold → 全量 CER；无全量 Gold → disagreement queue，但不停工
 ↓
6. 选择 PRIMARY_ASR
 ↓
7. 若触发 enhancement → MossFormerGAN 验证；若保留则重跑完整 70 min
 ↓
8. 最终完整 transcript → Qwen3 ForcedAligner
 ↓
9. RAW 完整 70 min → DiariZen speakers=2
 ↓
10. 全量 token ↔ speaker 融合
 ↓
11. speaker0/1 人工一次映射 therapist/client
 ↓
12. 输出完整 JSONL + MD + TXT
 ↓
13. 完整性检查
END
```

## 18. 禁止偷懒规则

以下任何一种都算任务未完成：

```text
只转前 30 min
只转前 60 min
只处理困难片段
只生成 benchmark 表而没有完整 transcript
三个候选中某个只跑部分音频
chunk 报错后直接跳过
只对 200 token 做时间戳
只对 10 min 做 diarization
最终 transcript 没到音频结尾
```

抽样只允许用于**质量评估**，不能替代完整处理。

## 19. 数据安全

`.gitignore` 必须包含：

```gitignore
data/**
private/**
outputs/private/**
.env
*.wav
*.mp3
*.mp4
*.m4a
```

禁止提交真实咨询音频、真实完整转录、被试身份信息、模型权重。

## 20. 最终成功定义

只有同时满足以下条件，才算当前 Demo 完成：

```text
1. Mega-ASR 完整处理 70 min
2. FireRedASR2 完整处理 70 min
3. GLM-ASR 完整处理 70 min
4. 选出 PRIMARY_ASR
5. 最终文本覆盖完整音频
6. Qwen3 ForcedAligner 对完整最终文本完成时间戳
7. DiariZen 对完整 70 min 完成双人 diarization
8. 最终每段都有 speaker + time + text
9. 输出 JSONL + Markdown + TXT
10. 所有空白、失败、分歧区域都进入 review queue
11. completeness report 证明没有未处理时间段
```

一句话要求：

> 当前只有这一个约 70 分钟 Demo，所以 Mega-ASR、FireRedASR2、GLM-ASR 三个本地候选必须全部完整跑满 70 分钟；最终再选唯一 ASR 主干，并对完整 70 分钟执行 forced alignment、双人 diarization 和 speaker-text 融合。任何抽样都只能用于质量评估，不能替代完整转录。