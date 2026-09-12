# 心花中文远场双人咨询 ASR 本地部署执行方案 V4
## 目标：在“模糊、远场、轻声、混响”的真实咨询录音上，把文字转录正确率做到当前本地方案的可验证上限

> 日期：2026-09-12  
> 用途：直接交给 Codex 执行。  
> 本版本替代 V3。  
> 原则：不做开放式模型海选；只测试三个具有互补证据的本地 ASR，然后严格淘汰。

## 1. 核心结论

V3 的候选不完整。当前最值得针对本项目实测的三个本地 ASR 是：

1. **Mega-ASR with router**：Qwen3-ASR-1.7B 的鲁棒性专门适配版本，训练覆盖 noise、far-field、echo/reverberation、recording artifacts、distortion、dropout，与本项目最直接匹配。
2. **FireRedASR2-LLM**：当前公开普通话 benchmark 和 WenetSpeech Meeting 很强；官方 ws-meeting CER 4.32，而 Qwen3-ASR-1.7B 为 5.88。
3. **GLM-ASR-Nano-2512**：官方明确针对 whisper / quiet / low-volume speech 训练，对来访者轻声这一特殊难例有直接相关性。

第一阶段不再单独测试原版 Qwen3-ASR：Mega-ASR 的基础模型就是 Qwen3-ASR-1.7B，并通过 router 在普通音频与鲁棒适配之间切换。

## 2. 必要模型

### A. Mega-ASR — 第一主候选

GitHub：https://github.com/xzf-thu/Mega-ASR  
技术报告：https://arxiv.org/abs/2605.19833  
鲁棒 benchmark：https://github.com/xzf-thu/Voices-in-the-Wild-Bench

必须使用 `Mega-ASR with router`，第一阶段不要强制 `--no-routing`。

### B. FireRedASR2-LLM — 中文普通话强对照

GitHub：https://github.com/FireRedTeam/FireRedASR2S  
技术报告：https://arxiv.org/abs/2603.10420

模型：`FireRedTeam/FireRedASR2-LLM`

唯一硬件回退：如果当前 GPU 无法稳定运行 8.3B LLM，只替换为 `FireRedASR2-AED`，不是增加第四个模型。AED 在同一 ws-meeting 上为 4.53，并自带 word-level timestamps/confidence。

### C. GLM-ASR-Nano-2512 — 低音量专长候选

GitHub：https://github.com/zai-org/GLM-ASR  
模型：https://huggingface.co/zai-org/GLM-ASR-Nano-2512

官方明确强调 low-volume / whisper / quiet speech robustness，所以必须进入困难子集筛选。

## 3. 必要下游组件

### 时间戳

`Qwen/Qwen3-ForcedAligner-0.6B`  
官方：https://github.com/QwenLM/Qwen3-ASR

最终选中任何 ASR 后，都用同一 ForcedAligner 对最终文本做时间对齐。

### 说话人

`DiariZen-Large-s80`，固定 `num_speakers=2`。  
官方：https://github.com/ntuspeechlab/Diarizen

第一阶段不搜索其它 diarization 模型。

## 4. 第一阶段禁止内容

Codex 不得加入：原版 Qwen3-ASR 独立 benchmark、Whisper、Paraformer、SenseVoice、Fun-ASR-Nano、Step-Audio2、MOSS、pyannote、3D-Speaker、云 API、LLM 文本纠错、热词/context、微调、ROVER、视频 active-speaker、speech separation。

## 5. 模型选拔不再强制统一自制 60 秒切块

V4 不再使用 V3 的统一 60 秒 RMS 切块作为模型选拔前提。目标不是论文式公平，而是找到生产效果最好的完整本地 ASR。

每个模型第一阶段必须走其**官方推荐 offline / long-form 路径**。Benchmark 输入统一为相同的 5 分钟原始 WAV；内部怎样 VAD/chunk，由模型官方生产路径决定。

统一输入：16 kHz、mono、PCM16 WAV。禁止前置降噪、归一化、EQ。

## 6. Benchmark 两阶段淘汰

### Stage 1：最难 30 分钟

- B1 远场/混响/模糊：15 min = 3×5 min
- B2 来访者低音量：10 min = 2×5 min
- B3 模糊 + 快速互动：5 min = 1×5 min

三个模型都跑：Mega-ASR with router、FireRedASR2-LLM、GLM-ASR-Nano-2512。

### Gold 可听度

每段标 `audibility=A/B/C`：
- A：一次正常播放可确定；
- B：模糊但两名标注者经重听可确认；
- C：两名标注者仍无法可靠确定。

核心评价只计算 A+B。C 只报告时长比例。

## 7. Stage 1 指标

主指标：`CER_AB`。

同时记录：
- Substitution Rate
- Deletion Rate
- Insertion Rate
- Empty-output duration
- Hallucination cases
- Backchannel Recall

特别关注 Deletion Rate，因为低音量/远场最常见失败是整词或整句漏掉。

## 8. Stage 1 淘汰规则

对三模型按完整 30 分钟 A+B pooled CER 排序，淘汰最高者，只留 TOP_1 / TOP_2。

如果第 2 和第 3 的 Hard CER 差 `<0.5 percentage point`，依次用：
1. B2 low-volume CER
2. Deletion Rate
3. Backchannel Recall

仍然只保留两个。

## 9. Stage 2：Top 2 跑完整 60 分钟

新增：
- A 一般质量 15 min
- C 快速轮替/短回应 10 min
- D 术语/专有词 5 min

合计完整 60 min，只跑 TOP_1 和 TOP_2。

最终选择：
- 总体 CER 差 `>=0.5 point`：选总体更低者；
- 差 `<0.5 point`：依次比较 Hard CER、B2 low-volume CER、Deletion Rate、Backchannel Recall、推理稳定性。

最终必须得到唯一 `PRIMARY_ASR`，不做 ensemble。

## 10. 语音增强不是主路线

只有当：
- PRIMARY_ASR 的 Stage-1 Hard CER `>12%`，或
- Deletion Rate `>6%`

才进入一次增强实验。

### 唯一条件增强：MossFormerGAN_SE_16K

官方：https://github.com/modelscope/ClearerVoice-Studio

V4 不再优先 FRCRN。ClearerVoice 官方 16 kHz benchmark 中，MossFormerGAN_SE_16K 在 VoiceBank+DEMAND 和 DNS-2020 上多数 PESQ/STOI/SI-SDR/SNR 指标优于 FRCRN_SE_16K。

只对 Hard 30 min 做：`RAW -> MossFormerGAN_SE_16K -> PRIMARY_ASR`。

只有同时满足：
- Hard CER 下降 `>=1.5 absolute points`
- Deletion Rate 不增加
- Backchannel Recall 下降 `<=5%`

才保留增强；否则永久丢弃，不搜索更多 enhancer。

## 11. 时间戳

最终 transcript 统一送入 `Qwen3-ForcedAligner-0.6B`。

生产时按不超过 5 min 的自然块对齐。

随机人工检查 200 个 token：
- MAE <=100 ms：接受
- 100–250 ms：接受但报告限制
- >250 ms：timestamp failure

## 12. Diarization

`RAW full audio -> DiariZen-Large-s80, num_speakers=2`。

不要对增强音频做 diarization。

人工抽 10 min，若 speaker confusion `<=5%` 直接接受；若 `>5%`，停止并触发 diarization second-stage，不自行换模型。

## 13. Speaker 与文本融合

ForcedAligner token timeline + DiariZen timeline：
1. token 与哪个 speaker 区间重叠最多，就归谁；
2. overlap 相同，用 token 中点；
3. 空白区域前后搜索 300 ms；
4. 仍无结果：`speaker=uncertain`。

同 speaker 且 gap `<800 ms` 合并为 utterance。每场仅人工一次映射 speaker_0/1 到 therapist/client，不做声纹识别。

## 14. Codex 精确执行树

```text
START
  ↓
0. GPU / CUDA / ffmpeg 检查
  ↓
1. 准备 Hard 30 min Gold（A/B/C audibility）
  ↓
2. RAW 同一 6 个 5-min clips
   ├─ Mega-ASR with router
   ├─ FireRedASR2-LLM
   └─ GLM-ASR-Nano-2512
  ↓
3. Hard CER + S/D/I + low-volume CER + backchannel recall
  ↓
4. 淘汰最差 1 个，只留 Top 2
  ↓
5. 新增 30 min → 完整 60 min
  ↓
6. Top 2 跑完整 60 min
  ↓
7. 固定规则选唯一 PRIMARY_ASR
  ↓
8. Hard CER >12% 或 Del >6%？
   ├─ NO：不做 enhancement
   └─ YES：MossFormerGAN_SE_16K → PRIMARY_ASR
             通过阈值则保留，否则永久丢弃
  ↓
9. 最终 transcript
  ↓
10. Qwen3-ForcedAligner
  ↓
11. DiariZen-Large, speakers=2
  ↓
12. token-speaker 时间融合
  ↓
13. final JSONL
```

## 15. 必须阅读的参考

### 必要模型
- Mega-ASR：https://github.com/xzf-thu/Mega-ASR
- Mega-ASR paper：https://arxiv.org/abs/2605.19833
- Voices-in-the-Wild-Bench：https://github.com/xzf-thu/Voices-in-the-Wild-Bench
- FireRedASR2S：https://github.com/FireRedTeam/FireRedASR2S
- FireRedASR2S paper：https://arxiv.org/abs/2603.10420
- GLM-ASR：https://github.com/zai-org/GLM-ASR
- GLM-ASR-Nano：https://huggingface.co/zai-org/GLM-ASR-Nano-2512
- Qwen3 ForcedAligner：https://github.com/QwenLM/Qwen3-ASR
- DiariZen：https://github.com/ntuspeechlab/Diarizen

### 必要评价准则

CER = `(S + D + I) / N`。

中文统一：Unicode NFKC、英文 lowercase、删标点/空格、简繁统一、数字常规归一化，但保留嗯/啊/呃/哦、重复、自我修正。

Edit distance 可复用：https://github.com/jitsi/jiwer ，但不能使用英文 word-token WER 代替中文 CER。

## 16. 条件触发才读

只有 Enhancement Gate 触发才看 ClearerVoice：
https://github.com/modelscope/ClearerVoice-Studio

只使用 `MossFormerGAN_SE_16K`。

## 17. 没有问题就不要看

以下仅背景，不进入第一阶段：
- Qwen3-ASR 原版独立 benchmark：https://github.com/QwenLM/Qwen3-ASR
- FunASR / Paraformer：https://github.com/modelscope/FunASR
- Step-Audio2：https://github.com/stepfun-ai/Step-Audio2
- MOSS：https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
- pyannote：https://github.com/pyannote/pyannote-audio
- 3D-Speaker：https://github.com/modelscope/3D-Speaker
- AISHELL-4：https://www.openslr.org/111/
- AliMeeting：https://github.com/yufan-aslp/AliMeeting

## 18. 失败出口

只有以下情况才重新调研：

```text
Failure-ASR:
三个模型在 Hard 30 min 上全部 CER >15%
且 MossFormerGAN 改善 <1.5 points
```

或：

```text
Failure-Diarization:
DiariZen speaker confusion >5%
```

或：

```text
Failure-Timestamp:
ForcedAligner MAE >250 ms
```

Codex 到这里必须停，不自行扩展模型池。

## 19. 当前最合理的先验排序

不是预先宣布某一个模型一定最好：

```text
严重 far-field / reverberation / distortion：Mega-ASR 最值得押注
普通话会议整体准确率：FireRedASR2-LLM 最值得押注
极低音量 / whisper-like speech：GLM-ASR-Nano 最值得押注
```

这三个覆盖心花最关键的三种失败来源。

## 20. 一句话任务

> 先在真实心花最难的 30 分钟音频上比较 Mega-ASR、FireRedASR2-LLM、GLM-ASR-Nano，只留下最好的两个；再在完整 60 分钟人工 Gold 上决出唯一 PRIMARY_ASR；只有最佳模型对模糊音频仍明显不足时才测试一次 MossFormerGAN 前端；最后固定使用 Qwen3-ForcedAligner + DiariZen 两人模式恢复时间戳与说话人。