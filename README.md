# 中文远场监控双人心理咨询 ASR：最佳完整路线调研与 Codex 执行任务书

> 更新时间：2026-09-12  
> 目标场景：中文、远场监控/摄像设备录音、单路混合音频、双人心理咨询、音质较差、存在轻声/短反馈/抢话/重叠，需要区分咨询师与来访者，并保留可靠时间戳。  
> 本文目的：直接交给 Codex 继续执行工程验证，不是泛泛综述。

---

## 0. 最终目标

建立一条**可本地运行、可和中国境内云 ASR 公平比较、可量化评估、可复现**的完整 ASR pipeline。

最终输出统一为 JSONL / JSON：

```json
{
  "session_id": "xxx",
  "segments": [
    {
      "start_ms": 15420,
      "end_ms": 18130,
      "speaker": "therapist",
      "speaker_raw": "SPEAKER_00",
      "text": "最近睡眠怎么样？",
      "words": [
        {"text": "最近", "start_ms": 15420, "end_ms": 16080},
        {"text": "睡眠", "start_ms": 16100, "end_ms": 16900},
        {"text": "怎么样", "start_ms": 16920, "end_ms": 18130}
      ]
    }
  ]
}
```

核心不是只追求 CER，而是同时优化：

1. 中文转写准确率
2. 说话人分离准确率
3. speaker-attributed transcription 准确率
4. 时间戳精度
5. 对短反馈词和重叠语音的稳定性
6. 人工后校对成本
7. 对真实“心花”数据的稳健性

---

# 1. 关键判断

## 1.1 不应先验假设“云一定比本地强”

到 2026 年，开源本地方案已经出现三个很强的方向：

### A. 模块化高精度路线

- ASR：Qwen3-ASR-1.7B
- 时间戳：Qwen3-ForcedAligner-0.6B
- diarization：DiariZen-Large-s80
- speaker identity：3D-Speaker / CAM++

Qwen3-ASR 技术报告称 1.7B 在开放与内部测试中达到开源 SOTA，并可与强商业 API 竞争；Qwen3-ForcedAligner 专门用于强制对齐和时间戳。

论文：  
https://arxiv.org/abs/2601.21337

GitHub：  
https://github.com/QwenLM/Qwen3-ASR

---

### B. 工程成熟路线

- ASR：Paraformer-zh
- VAD：FSMN-VAD
- diarization：DiariZen / 3D-Speaker / CAM++
- 标点：CT-Punc
- 时间戳：Paraformer 原生字符级时间戳；必要时 fa-zh 二次对齐

FunASR：  
https://github.com/modelscope/FunASR

Paraformer：  
https://huggingface.co/funasr/paraformer-zh

模型清单：  
https://github.com/modelscope/FunASR/blob/main/model_zoo/readme.md

Paraformer 论文：  
https://arxiv.org/abs/2206.08317  
DOI:  
https://doi.org/10.21437/Interspeech.2022-9996

---

### C. 端到端 speaker-attributed transcription

MOSS-Transcribe-Diarize 0.9B 直接输出：

```text
[start][S01]文本[end]
```

同时完成：

- 长音频 ASR
- speaker diarization
- 时间戳
- speaker attribution

GitHub：  
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

论文：  
https://arxiv.org/abs/2601.01554

其官方 benchmark 包含 AISHELL-4、AliMeeting，并报告 CER/cpCER。

注意：它更适合做“端到端强基线”，但其时间戳粒度不应默认认为比 Paraformer / ForcedAligner 更适合精细音频脱敏。

---

# 2. 本任务最困难的部分不是普通 ASR，而是说话人分离

心理咨询中常见：

```text
来访者：然后我当时其实觉得——
咨询师：嗯。
来访者：——特别难受。
```

对模型而言，“嗯”可能只有 200–500 ms。

因此必须专门测试：

- 极短 backchannel：嗯、对、啊、是、然后呢
- 快速 speaker turn
- 两人抢话
- 重叠发声
- 一人音量显著低于另一人
- 长时间同一 speaker 后突然短切换

普通 CER 无法捕获“文字全对但 speaker 全错”。

---

# 3. 最重要的本地 diarization 候选

## 3.1 第一优先：DiariZen-Large-s80

GitHub：  
https://github.com/ntuspeechlab/Diarizen

另一个代码仓：  
https://github.com/gusaiworld/diarizen

公开结果（SDM / far-field）：

- AISHELL-4：约 9.8% DER
- AliMeeting far：约 12.5% DER
- AMI-SDM：约 14.0% DER

这是目前非常值得优先测试的本地 diarization 方案。

尤其是 AliMeeting far 与本项目较接近：

- 中文
- 多人对话
- 远场
- 自然 turn-taking
- 含 overlap

虽然 AliMeeting 不是心理咨询，但比近讲普通话测试集有意义得多。

---

## 3.2 第二优先：3D-Speaker

GitHub：  
https://github.com/modelscope/3D-Speaker

公开 DER：

- AISHELL-4：10.30%
- AliMeeting：19.73%
- Meeting-CN_ZH-1：18.91%
- Meeting-CN_ZH-2：12.78%

同时支持：

- speaker verification
- speaker embedding
- speaker recognition
- diarization

它最大的附加价值是：

> 可以用“已知咨询师声纹”把匿名 speaker cluster 映射成 therapist/client。

---

## 3.3 第三优先：pyannote Community-1

模型：  
https://huggingface.co/pyannote/speaker-diarization-community-1

GitHub：  
https://github.com/pyannote/pyannote-audio

公开 DER：

- AISHELL-4：11.7%
- AliMeeting channel 1：20.3%

优点：

- 社区成熟
- API 清晰
- 有 exclusive speaker diarization
- 很方便和 ASR timestamp 对齐

缺点：

- 中文远场公开结果不如 DiariZen
- 模型下载需要接受 Hugging Face 条款

因此作为基线，而非当前首选。

---

# 4. 推荐的“最高精度本地完整路线”

推荐实现以下主干。

```text
原始 MP4 / WAV
        │
        ├──────────── 保存原始文件，不覆盖
        │
        ↓
音频规范化
ffmpeg -> 16 kHz / mono / float32 or PCM16
        │
        ├──────────── raw branch
        │
        └──────────── enhancement branch
                       │
                       └─ ClearVoice enhancement
        │
        ↓
质量分析
SNR / clipping / loudness / silence ratio
        │
        ├───────────────────────────────┐
        │                               │
        ↓                               ↓
Diarization                          ASR
DiariZen Large                      Qwen3-ASR-1.7B
固定 speakers=2                    或 Paraformer-zh
        │                               │
        ↓                               ↓
RTTM/segments                       transcript
        │                               │
        └──────────────┬────────────────┘
                       ↓
               Forced Alignment
        Qwen3-ForcedAligner / fa-zh
                       ↓
           char/word timestamps
                       ↓
          speaker × token 对齐
                       ↓
        therapist/client identity
                       ↓
       utterance reconstruction
                       ↓
             final JSONL
```

---

# 5. 为什么 diarization 和 ASR 建议先相对独立

不要把 pipeline 设计成：

```text
VAD -> 切很多小段 -> 每小段 ASR
```

然后再拼起来。

这会让 ASR 丢失上下文，尤其心理咨询中：

- 指代词很多
- 口语残句很多
- 轻声
- 重复
- 自我修正
- 专有词
- 长上下文有助于判断同音字

更推荐：

### 路线 A

1. diarization 对完整音频建立 speaker timeline
2. ASR 对较长语义块/完整音频转写
3. forced align transcript 到原音频
4. 根据 token 时间与 diarization timeline 的 overlap 分配 speaker

也就是说：

> speaker segmentation 与文字识别分开优化，最后按时间轴融合。

这样更适合科研数据工程。

---

# 6. Qwen3-ASR 路线

## 6.1 模型

Qwen3-ASR-1.7B

GitHub：  
https://github.com/QwenLM/Qwen3-ASR

Technical Report：  
https://arxiv.org/abs/2601.21337

特点：

- 中文强
- 支持多种中国方言
- 对复杂声学环境有专门强调
- 可处理长音频
- 可离线部署

---

## 6.2 时间戳

使用：

Qwen3-ForcedAligner-0.6B

可对 text-audio pair 进行词/字符级对齐。

注意：

官方说明单次 alignment 支持的音频长度有限（应按当前官方 README 验证，现有公开说明为最长约 5 分钟）。

因此工程上：

```text
长录音
↓
VAD / diarization boundary
↓
切为 1–4 min chunk
保留绝对 offset
↓
ASR
↓
ForcedAligner
↓
timestamp + absolute offset
```

不要把整小时音频直接送 forced aligner。

---

# 7. Paraformer 路线

Paraformer 最大优势不是“2026 年一定文字最强”，而是：

> 中文成熟 + 快 + 原生细粒度 timestamp + 工程稳定。

推荐组合：

```python
AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc"
)
```

但 speaker diarization 最好另外和：

- DiariZen
- 3D-Speaker

进行独立比较。

不要默认 CAM++ 是最佳 diarization，只因为 FunASR 可以“一行调用”。

官方文档：  
https://github.com/modelscope/FunASR

模型：  
https://huggingface.co/funasr/paraformer-zh

时间戳预测 fa-zh：  
https://huggingface.co/funasr/fa-zh

---

# 8. Fun-ASR-Nano 是否应加入

可以加入 ASR-only benchmark，但不应作为第一条完整生产路线。

原因：

- 中文 ASR 很强
- 复杂口音/难例可能优于旧 Paraformer
- 但 timestamp 能力依具体 checkpoint / hub 而异
- speaker diarization 不是模型原生能力

相关仓库：  
https://github.com/QwenAudio/Fun-ASR

如果 Codex 测试它，必须明确记录：

- checkpoint
- source (ModelScope / HuggingFace)
- commit hash
- 是否真正返回可靠 timestamp

不得把“API 有 timestamp 字段”等同于 timestamp 权重有效。

---

# 9. MOSS-Transcribe-Diarize 路线

应作为本地端到端 baseline。

GitHub：  
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

论文：  
https://arxiv.org/abs/2601.01554

官方报告中 0.9B 在：

- AISHELL-4
- AliMeeting

都有 speaker-aware transcription benchmark。

测试时保留：

- CER
- cpCER
- speaker consistency
- segment timestamp error

MOSS 的意义：

> 检验“端到端联合 ASR + diarization”是否比模块化 pipeline 更适合本项目。

不是默认取代模块化方案。

---

# 10. 音频增强：必须做 A/B，而不是默认开启

远场低质音频很可能需要增强，但增强模型可能：

- 改善 ASR
- 同时损伤声纹
- 改变轻声
- 抹掉 backchannel
- 引入 hallucinated spectral artifacts

因此：

> enhancement 永远作为实验 branch，而不是原始数据的替换。

推荐工具：

ClearerVoice-Studio：  
https://github.com/modelscope/ClearerVoice-Studio

候选：

- FRCRN_SE_16K
- MossFormerGAN_SE_16K
- MossFormer2_SE_48K

对于最终 16 kHz ASR，优先测试：

- FRCRN_SE_16K
- MossFormerGAN_SE_16K

---

## 10.1 推荐实验

每个模型同时测试：

```text
RAW audio
ENHANCED audio
```

分别计算：

- CER
- DER
- cpCER
- timestamp MAE

如果：

- enhancement 降低 CER
- 但提高 DER

则可以：

```text
raw -> diarization
enhanced -> ASR
```

即 diarization 和 ASR 使用不同音频 branch。

这是非常值得验证的一条路线。

---

# 11. 重叠语音处理

不建议一开始对整段音频做 speech separation。

因为 separation 会带来：

- source leakage
- artifact
- speaker permutation
- ASR 退化
- 时间轴复杂化

推荐策略：

```text
diarization / overlap detector
↓
只定位 overlap regions
↓
仅对 overlap 区域做 2-speaker separation
↓
分别 ASR
↓
重新合并 timeline
```

候选：

ClearerVoice MossFormer2_SS_16K

仓库：  
https://github.com/modelscope/ClearerVoice-Studio

这一步属于第二阶段优化，不是 MVP 必需。

---

# 12. 如果视频画面中两个人长期可见：增加音视频路线

由于输入来源是“监控/录像”，如果画面长期能看到咨询师和来访者，音频-only diarization 不一定是最优上限。

可选：

## TalkNet active speaker detection

GitHub：  
https://github.com/TaoRuijie/TalkNet-ASD

功能：

判断画面中的脸在某时间是否正在说话。

---

## AV_MossFormer2 target speaker extraction

ClearerVoice：

https://github.com/modelscope/ClearerVoice-Studio

模型：  
AV_MossFormer2_TSE_16K

可以结合面部/lip cue 提取目标 speaker。

注意：

此路线应作为后续 rescue experiment，不要第一阶段就引入，因为工程复杂度较高。

但如果 audio diarization 卡在：

- overlap
- “嗯”
- 轻声咨询师反馈

视频 active speaker detection 可能非常有价值。

---

# 13. 中国云 ASR：第一阶段必须加入的商业基线

## 13.1 科大讯飞：录音文件转写大模型

官方文档：

https://www.xfyun.cn/doc/spark/asr_llm/Ifasr_llm.html

应测试参数：

```text
roleType=1
roleNum=2
eng_vad_mdn=1
```

含义：

- 开启通用角色分离
- 明确双人
- 远场模式

讯飞支持：

- 16k / 8k
- 长录音
- 说话人分离
- 远场 VAD
- 句级时间戳
- 词级 frame timestamp（10 ms frame）

还应测试：

```text
roleType=3
```

即声纹角色分离。

声纹接口：

https://www.xfyun.cn/doc/spark/asr_llm/voice_print.html

如果咨询师是固定的一批人，可以注册**咨询师声纹**，看是否显著改善 therapist/client identity。

---

# 14. 腾讯云：非常值得纳入云端 benchmark

腾讯 2026 年的大模型 ASR 已明确针对：

- 噪声大
- 回音大
- 人声小
- 人声远

做优化。

尤其：

```text
16k_zh_en_2.0
16k_zh_en_meeting
```

官方文档：

https://cloud.tencent.com/document/api/1093/37823

其中 meeting 模型明确用于：

- 多说话人
- overlap
- speaker diarization

并支持字级时间戳。

因此必须和讯飞一起跑。

不要只做“本地 vs 讯飞”，否则无法判断优势来自：

- 云端商业模型本身
还是
- 讯飞特定实现。

---

# 15. 阿里云百炼也值得作为第三个云候选

2026 年推荐的非实时模型：

```text
qwen-audio-3.0-asr-flash-filetrans
```

官方文档：

https://help.aliyun.com/zh/model-studio/asr-model

非实时转写：

https://help.aliyun.com/zh/model-studio/non-realtime-speech-recognition-user-guide

支持：

- speaker diarization
- speaker_count
- 句级时间戳
- 词级时间戳
- 热词
- 上下文

可设置：

```text
diarization_enabled=true
speaker_count=2
```

API：  
https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api

---

# 16. 推荐最终 benchmark 矩阵

第一阶段不要跑几十个模型。

只跑以下 6 条：

| ID | 类型 | Pipeline |
|---|---|---|
| L1 | 本地模块化高精度 | Qwen3-ASR-1.7B + Qwen3 ForcedAligner + DiariZen-Large |
| L2 | 本地成熟 | Paraformer-zh + DiariZen-Large |
| L3 | 本地端到端 | MOSS-Transcribe-Diarize 0.9B |
| C1 | 中国云 | 讯飞录音文件转写大模型 |
| C2 | 中国云 | 腾讯 16k_zh_en_meeting / 2.0 |
| C3 | 中国云 | 阿里 qwen-audio-3.0-asr-flash-filetrans |

第二阶段再加：

- Paraformer + 3D-Speaker
- pyannote Community-1
- raw vs ClearVoice enhanced
- overlap-only separation
- audio-visual ASD

---

# 17. Benchmark 数据怎么抽

不要随机随便切 10 分钟。

从真实咨询中建立 stratified benchmark。

建议总时长：

```text
第一轮：60–90 分钟
```

分布：

### A. clean-ish

15 min

- 相对清晰
- 正常音量
- 少 overlap

### B. low volume / far-field

15 min

- 来访者小声
- 麦克风距离远
- 混响明显

### C. backchannel-heavy

10–15 min

大量：

- 嗯
- 对
- 是
- 然后呢
- 好
- 啊

### D. rapid turn-taking

10–15 min

快速 speaker switch。

### E. overlap

10–15 min

两人同时发声或抢话。

### F. difficult lexical

10 min

含：

- 人名
- 地名
- 医学词
- 心理咨询术语
- 方言/口音
- 自我修正
- 语气词

---

# 18. Gold annotation 标准

人工真值不要只标文本。

至少标：

```json
{
  "start_ms": 1000,
  "end_ms": 3050,
  "speaker": "client",
  "text_verbatim": "我我觉得……就是有点难受",
  "overlap": false
}
```

必须保留真实口语现象：

- 重复
- 结巴
- 自我修正
- 嗯/啊
- 断句
- 半句话

不要把人工 gold 先“顺滑”。

否则评价 ASR 会失真。

---

# 19. 核心评价指标

## 19.1 CER

Character Error Rate

衡量中文字错多少。

```text
CER = (Substitution + Deletion + Insertion) / Reference characters
```

越低越好。

---

## 19.2 DER

Diarization Error Rate

衡量 speaker timeline 错多少。

包括：

- missed speech
- false alarm
- speaker confusion

越低越好。

---

## 19.3 JER

Jaccard Error Rate

补充 DER。

更关注 reference speaker 与 predicted speaker 时间区间的重叠关系。

---

## 19.4 cpCER

concatenated minimum-permutation CER

非常适合多说话人 ASR。

同时反映：

- 文本识别
- speaker attribution

应作为本项目核心指标之一。

---

## 19.5 SA-CER / speaker-attributed CER

如果工程实现方便，明确计算：

```text
therapist transcript CER
client transcript CER
overall speaker-attributed CER
```

---

## 19.6 时间戳 MAE

人工边界 vs 模型边界：

```text
start MAE
end MAE
word/char timestamp MAE
```

单位 ms。

---

## 19.7 Speaker-turn F1

针对本项目非常建议新增：

speaker change detection precision / recall / F1。

特别统计：

```text
<300 ms
300–800 ms
>800 ms
```

的短 utterance。

---

## 19.8 Backchannel Recall

建立 backchannel 词表，例如：

```text
嗯
嗯嗯
对
是
好
啊
哦
然后呢
```

人工标注其出现次数。

评估：

- 是否识别出来
- speaker 是否正确
- timestamp 是否覆盖

这是心理咨询场景很重要的定制指标。

---

## 19.9 Overlap Recall

在人工标记 overlap 的区间中：

- 是否发现 overlap
- 是否保留双方文本
- 是否只剩一方

---

## 19.10 Human Correction Time

每 10 min 音频人工校正需要多少分钟。

这是最终生产选型的核心指标。

例如：

```text
系统 A：
CER 9.1%
人工修改 3.8 min / 10 min

系统 B：
CER 8.4%
人工修改 6.2 min / 10 min
```

则 A 可能反而更适合数据工程。

---

# 20. 必须做统计分层

不要只报告一个总 CER。

至少报告：

| subset | CER | DER | cpCER | backchannel recall |
|---|---:|---:|---:|---:|
| clean | | | | |
| far/noisy | | | | |
| low-volume client | | | | |
| rapid-turn | | | | |
| overlap | | | | |

这样才能知道模型真正失败在哪里。

---

# 21. Speaker identity：利用“咨询师通常可知”这一先验

本项目比开放会议 diarization 更容易，因为：

- 固定双人
- 一方为咨询师
- 咨询师通常来自有限人员集合

因此不要只做 anonymous diarization。

可以：

```text
diarization
↓
SPEAKER_00 / SPEAKER_01
↓
3D-Speaker embedding
↓
和本地咨询师 enrollment embedding 比较
↓
therapist / client
```

只需注册咨询师。

不一定需要建立来访者长期声纹库。

这既降低复杂度，也减少敏感身份模板数量。

3D-Speaker：  
https://github.com/modelscope/3D-Speaker

---

# 22. 预处理原则

## 22.1 永远保留原文件

```text
data/raw/
```

禁止覆盖。

---

## 22.2 建立标准工作副本

建议：

```bash
ffmpeg -i input.mp4 \
  -vn \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  output.wav
```

但必须记录：

- source sample rate
- source channels
- codec
- conversion command
- ffmpeg version

---

## 22.3 不要自动 loudness normalize 后覆盖

响度归一化必须做 A/B。

因为：

- 轻声片段可能改善
- background noise 也可能同时被放大

---

# 23. 热词与心理咨询领域词汇

所有支持 hotword 的系统要统一建立一个测试词表。

例如：

- 移空疗法
- 承载物
- 象征物
- 来访者
- 躯体感觉
- 焦虑
- 抑郁
- PHQ-9
- GAD-7
- 心花计划

但 benchmark 至少要有两组：

```text
without hotword
with hotword
```

否则云/本地比较不公平。

---

# 24. 云端比较的公平性要求

对于每个云 API 记录：

```yaml
provider:
model:
api_version:
date:
sample_rate:
speaker_diarization:
speaker_count:
far_field_mode:
voiceprint:
hotword:
punctuation:
timestamp_level:
audio_enhancement:
```

云模型可能持续更新。

因此必须记录：

> 调用日期 + API/model version

否则以后无法复现实验。

---

# 25. 推荐 repo 工程结构

Codex 创建：

```text
asr_farfield_dyad/
├── README.md
├── docs/
│   ├── research.md
│   ├── benchmark_protocol.md
│   └── model_matrix.md
├── configs/
│   ├── local_qwen3_diarizen.yaml
│   ├── local_paraformer_diarizen.yaml
│   ├── local_moss.yaml
│   ├── cloud_iflytek.yaml.example
│   ├── cloud_tencent.yaml.example
│   └── cloud_aliyun.yaml.example
├── src/
│   ├── audio/
│   │   ├── preprocess.py
│   │   ├── quality.py
│   │   └── enhance.py
│   ├── asr/
│   │   ├── qwen3_asr.py
│   │   ├── paraformer.py
│   │   └── moss.py
│   ├── diarization/
│   │   ├── diarizen.py
│   │   ├── three_d_speaker.py
│   │   └── pyannote.py
│   ├── alignment/
│   │   ├── qwen_forced_aligner.py
│   │   ├── funasr_fa.py
│   │   └── merge_speaker_words.py
│   ├── cloud/
│   │   ├── iflytek.py
│   │   ├── tencent.py
│   │   └── aliyun.py
│   ├── metrics/
│   │   ├── cer.py
│   │   ├── diarization.py
│   │   ├── cpcer.py
│   │   ├── timestamp.py
│   │   └── backchannel.py
│   └── schema.py
├── scripts/
│   ├── run_benchmark.py
│   ├── normalize_outputs.py
│   └── make_report.py
├── tests/
└── outputs/
```

---

# 26. 统一中间 schema

所有系统必须转成统一 schema。

```json
{
  "audio_id": "demo",
  "duration_ms": 120000,
  "segments": [
    {
      "speaker_raw": "S01",
      "speaker": "therapist",
      "start_ms": 1000,
      "end_ms": 4200,
      "text": "……",
      "tokens": [
        {
          "text": "嗯",
          "start_ms": 1000,
          "end_ms": 1280
        }
      ]
    }
  ],
  "metadata": {
    "system": "qwen3_diarizen",
    "model_versions": {},
    "source_audio_sha256": ""
  }
}
```

禁止下游代码直接依赖讯飞/腾讯/FunASR 原始 JSON。

---

# 27. 数据安全要求

Codex 必须做到：

- `.gitignore` 忽略所有真实音频
- 不把 API key 提交 repo
- cloud config 只提交 `.example`
- 不把真实转录文本作为 unit test fixture
- test fixture 只能使用公开示例音频或人工合成数据

例如：

```gitignore
data/raw/**
data/private/**
outputs/private/**
.env
*.wav
*.mp3
*.mp4
```

必要的公开 demo 音频使用明确许可的数据。

---

# 28. 公开中文参考数据

## AISHELL-4

中文远场会议数据。

论文：  
https://www.isca-archive.org/interspeech_2021/fu21b_interspeech.html

DOI：  
https://doi.org/10.21437/Interspeech.2021-1397

GitHub：  
https://github.com/felixfuyihui/AISHELL-4

OpenSLR：  
https://www.openslr.org/111/

特点：

- 真实会议
- 远场
- overlap
- noise
- 快速 speaker turn
- speaker diarization annotation

---

## AliMeeting / M2MeT

GitHub：  
https://github.com/yufan-aslp/AliMeeting

非常重要，因为其 far-field Mandarin meeting 场景和本项目声学条件较接近。

---

# 29. 为什么公开 benchmark 只能做 sanity check

AISHELL-4 / AliMeeting 不是心理咨询。

本项目可能有：

- 更低音量
- 更长静默
- 更多情绪性声音
- 更少 speaker 数
- 更强的双人先验
- 更固定座位
- 更多“嗯/对/好”
- 不同的摄像设备频响

因此：

> 最终选型必须以真实心花数据的小规模人工 gold 为准。

公开数据只用于：

- 验证代码
- 检查模型是否安装正确
- 排除明显坏配置
- 复现已有 benchmark

---

# 30. Codex 第一阶段执行任务

## Task 1：环境调查

检测：

- OS
- CPU
- RAM
- GPU
- VRAM
- CUDA
- ffmpeg

生成：

```text
outputs/environment.json
```

---

## Task 2：实现 audio inspector

输入任意音视频，输出：

- sample rate
- channels
- duration
- codec
- RMS
- peak
- clipping ratio
- silence ratio

不要修改源文件。

---

## Task 3：跑通本地 3 条 pipeline

### L1

Qwen3-ASR-1.7B  
+ Qwen3-ForcedAligner  
+ DiariZen-Large

### L2

Paraformer-zh  
+ DiariZen-Large  
+ Paraformer native timestamp

### L3

MOSS-Transcribe-Diarize 0.9B

统一输出 schema。

---

## Task 4：云 API adapter

实现但不硬编码 key：

- iFlytek
- Tencent
- Alibaba

从环境变量读取。

示例：

```text
IFLYTEK_APP_ID
IFLYTEK_API_KEY
IFLYTEK_API_SECRET
```

---

## Task 5：评估器

实现：

- CER
- DER
- JER
- cpCER / SA-CER
- timestamp MAE
- speaker-turn F1
- backchannel recall
- overlap recall

优先复用权威公开 evaluation library，不要自己随意重新定义 DER。

推荐：

pyannote.metrics  
https://github.com/pyannote/pyannote-metrics

---

## Task 6：报告生成

自动产生：

```text
outputs/report.md
outputs/results.csv
```

表格：

| system | subset | CER | DER | cpCER | start_MAE | end_MAE | backchannel_recall | runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|

---

# 31. Codex 第二阶段执行任务

第一阶段结果出来以后才做：

1. raw vs enhancement
2. DiariZen vs 3D-Speaker
3. Qwen3-ASR vs Paraformer
4. overlap-only MossFormer separation
5. therapist voice enrollment
6. video active-speaker detection

不要在第一阶段一次性把所有模型拼成复杂系统。

---

# 32. 决策规则

不要直接选择 CER 最低者。

建立综合决策：

```text
speaker-aware accuracy > raw text accuracy
```

建议优先级：

1. cpCER / SA-CER
2. DER / speaker confusion
3. CER
4. backchannel recall
5. timestamp MAE
6. artificial correction time
7. runtime/cost

原因：

对于心理咨询数据：

> 把来访者一句话分给咨询师，通常比错一个汉字更严重。

---

# 33. 推荐的最终生产路线候选

如果 L1 实测最好：

```text
RAW audio
├─> DiariZen Large
└─> Qwen3-ASR
      ↓
Qwen3 ForcedAligner
      ↓
speaker-time fusion
      ↓
3D-Speaker therapist mapping
      ↓
final transcript
```

---

如果 L2 接近 L1，但更稳定：

```text
RAW audio
├─> DiariZen Large
└─> Paraformer
      ↓
native char timestamp
      ↓
speaker-time fusion
      ↓
therapist mapping
```

这很可能是工程上最舒服的纯本地路线。

---

如果云端显著领先：

例如：

```text
local best cpCER = 20%
cloud cpCER = 11%
```

或人工校正时间下降一半以上，则应把：

> “原始咨询音频使用境内云 ASR”

作为正式伦理/数据治理路线论证，而不是为了方便使用云。

---

# 34. 需要 Codex 特别验证的未知点

以下不要直接假设，要实际验证：

- 心花音频真实采样率和 codec
- 是否已有双声道信息被原视频保留
- 是否为真正 mono，还是 stereo duplicate
- 咨询师和来访者是否长期固定坐在两个空间位置
- 是否存在明显设备 AGC
- 是否有持续空调/电流背景声
- overlap 比例
- 每轮平均时长
- 来访者音量是否系统性低于咨询师
- Qwen3-ASR 在这批远场数据上的 CER
- DiariZen 在“双人”固定 speaker count 下是否明显改善
- enhancement 对 speaker embedding 的影响
- 云端声纹角色分离是否比 blind diarization 明显更好

---

# 35. 推荐文献与资料总表

## ASR

### Qwen3-ASR

Paper:  
https://arxiv.org/abs/2601.21337

GitHub:  
https://github.com/QwenLM/Qwen3-ASR

---

### Paraformer

Paper:  
https://arxiv.org/abs/2206.08317

DOI:  
https://doi.org/10.21437/Interspeech.2022-9996

FunASR:  
https://github.com/modelscope/FunASR

---

### MOSS Transcribe Diarize

Paper:  
https://arxiv.org/abs/2601.01554

GitHub:  
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize

---

## Diarization

### DiariZen

https://github.com/ntuspeechlab/Diarizen

https://github.com/gusaiworld/diarizen

---

### 3D-Speaker

https://github.com/modelscope/3D-Speaker

---

### pyannote

https://github.com/pyannote/pyannote-audio

https://huggingface.co/pyannote/speaker-diarization-community-1

---

## Enhancement / Separation

ClearerVoice-Studio:

https://github.com/modelscope/ClearerVoice-Studio

---

## Audio-visual

TalkNet:

https://github.com/TaoRuijie/TalkNet-ASD

ClearerVoice AV target speaker extraction:

https://github.com/modelscope/ClearerVoice-Studio

---

## 中文远场数据

AISHELL-4:

https://github.com/felixfuyihui/AISHELL-4

https://www.openslr.org/111/

Paper:
https://www.isca-archive.org/interspeech_2021/fu21b_interspeech.html

---

AliMeeting / M2MeT:

https://github.com/yufan-aslp/AliMeeting

---

## 云 ASR

### 讯飞

录音文件转写大模型：

https://www.xfyun.cn/doc/spark/asr_llm/Ifasr_llm.html

声纹角色分离：

https://www.xfyun.cn/doc/spark/asr_llm/voice_print.html

---

### 腾讯

录音文件识别：

https://cloud.tencent.com/document/api/1093/37823

产品动态：

https://cloud.tencent.com/document/product/1093/46797

---

### 阿里云百炼

ASR 选型：

https://help.aliyun.com/zh/model-studio/asr-model

非实时识别：

https://help.aliyun.com/zh/model-studio/non-realtime-speech-recognition-user-guide

HTTP API：

https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api

---

# 36. Codex 的最终交付要求

Codex 不应只返回“推荐某模型”。

必须交付：

1. 可运行 repo
2. reproducible environment
3. 三条本地 pipeline
4. 三个云 adapter
5. 统一输出 schema
6. benchmark annotation format
7. 自动 evaluation
8. 自动 report
9. 所有模型/version/commit 记录
10. README 中写明运行方法
11. 不提交任何真实心理咨询音视频或文本
12. 给出第一轮结果后再决定第二阶段优化

---

# 37. 最重要的一条研究原则

本项目最终问题不是：

> “哪个 ASR 模型排行榜最好？”

而是：

> “在中文低质量远场双人心理咨询的真实分布上，哪个完整 pipeline 能最可靠地给出 **谁、在什么时候、说了什么**，并把人工校对成本降到最低？”

因此最终 benchmark unit 必须是：

```text
speaker + text + time
```

而不是只有 text。
