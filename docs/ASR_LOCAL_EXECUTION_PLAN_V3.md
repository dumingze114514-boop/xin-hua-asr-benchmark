# 心花中文远场双人咨询 ASR 本地部署执行方案 V3

> 日期：2026-09-12  
> 用途：直接交给 Codex 执行。  
> 场景：中文、远场监控/摄像设备、单路混合音频、双人心理咨询、音质较差、长录音；最终需要 `speaker + text + start_ms + end_ms`。

## 0. 不允许自由发散

本任务的第一目标是**模糊远场音频本身的文字识别准确率**。Codex 第一阶段不得自行扩大模型池。只允许四个必要组件和一个条件组件：

- ASR-1：`Qwen/Qwen3-ASR-1.7B`
- ASR-2：`FireRedTeam/FireRedASR2-LLM`
- 时间戳：`Qwen/Qwen3-ForcedAligner-0.6B`
- 说话人：`DiariZen-Large-s80`，固定 `num_speakers=2`
- 条件增强：`FRCRN_SE_16K`，仅在本文规定的 Gate 触发后使用

第一阶段禁止加入 Whisper、Paraformer、SenseVoice、Fun-ASR-Nano、MOSS、pyannote、3D-Speaker、CAM++、云 API、LLM 文本纠错、视频 active-speaker、speech separation、自定义微调、热词、ROVER。

---

# 1. 为什么只比较 Qwen3-ASR 与 FireRedASR2-LLM

## Qwen3-ASR-1.7B

官方 2026 技术报告在内部普通话 `ExtremeNoise` 测试中报告：

```text
Qwen3-ASR-1.7B   16.17
Qwen3-ASR-0.6B   17.88
Doubao-ASR        17.04
Fun-ASR           36.55
Whisper-large-v3  63.17
```

这不是心花数据，不能直接外推，但说明它是当前最值得优先验证的本地困难中文 ASR 之一。

必要资料：
- https://github.com/QwenLM/Qwen3-ASR
- https://arxiv.org/abs/2601.21337

## FireRedASR2-LLM

FireRedASR2S 官方 2026 公开结果中，4 个普通话公开 benchmark 平均 CER：

```text
FireRedASR2-LLM   2.89
Qwen3-ASR-1.7B    3.76
Doubao-ASR        3.69
Fun-ASR           4.16
```

公开测试包含 WenetSpeech meeting。它与 Qwen3-ASR 形成很合适的二选一：Qwen3 强调困难声学条件；FireRedASR2-LLM 在公开中文 benchmark 整体很强。

必要资料：
- https://github.com/FireRedTeam/FireRedASR2S
- https://arxiv.org/abs/2603.10420

---

# 2. 固定完整 pipeline

```text
原始视频/音频
    ↓
无损工作副本：16 kHz / mono / PCM16
    ├──────────────────────────┐
    ↓                          ↓
ASR 二选一                    DiariZen-Large-s80
Qwen3-ASR-1.7B               num_speakers=2
FireRedASR2-LLM                    │
    ↓                              │
真实 benchmark 选唯一主干          │
    ↓                              │
Qwen3-ForcedAligner-0.6B           │
文字 → 字/词级时间戳               │
    └──────────────┬───────────────┘
                   ↓
            时间轴融合 speaker
                   ↓
            utterance 重建
                   ↓
      speaker + time + text
```

如果文字仍然不够准，只允许进入一次 `FRCRN_SE_16K` 增强分支。

---

# 3. STEP 1：音频预处理必须保守

原始文件永久保留；工作副本统一：

```bash
ffmpeg -i INPUT -vn -ac 1 -ar 16000 -c:a pcm_s16le OUTPUT.wav
```

第一轮禁止默认 normalize、denoise、compressor、EQ、AGC、silence removal、speed change。

原因：必须先测原始声学信息在当前强模型上的真实上限。

---

# 4. STEP 2：固定 chunk 方案

两个 ASR 必须使用**完全相同的 chunk manifest**。

不要一句一句切；固定：

```text
target = 60 s
min = 30 s
max = 90 s
```

切分算法：

1. 目标切点为 60 s；
2. 在目标切点 ±5 s 搜索；
3. 用 100 ms RMS 窗找最低能量位置；
4. 不短于 30 s、不长于 90 s；
5. 相邻 chunk 保留 500 ms overlap；
6. 保存绝对 `start_ms/end_ms`。

第一轮不引入 VAD 模型做复杂切分。

---

# 5. STEP 3：Qwen3-ASR 固定配置

必须用官方 Python 实现，不使用第三方 wrapper。

```python
Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
)
```

第一轮：

```text
language = Chinese
context = empty
non-streaming
```

禁止热词。Qwen3-ASR 已有公开 issue 报告 context 可能把提示词混入 transcript，弱语音处也可能被近音提示词偏置：
https://github.com/QwenLM/Qwen3-ASR/issues/186

这个 issue 默认不用读；只有未来确定错误主要来自固定术语时再看。

---

# 6. STEP 4：FireRedASR2-LLM 固定配置

必须用官方 `FireRedASR2S` 实现。

模型：

```text
FireRedTeam/FireRedASR2-LLM
```

第一轮用官方默认/推荐附近参数，不做网格调参。基准配置：

```python
decode_min_len = 0
repetition_penalty = 1.0
llm_length_penalty = 0.0
temperature = 1.0
```

若执行时官方 README 的推荐值有变化，以当日官方 README 为准，并将实际参数写入报告。

禁止 temperature sweep、beam/grid search、repetition penalty 搜索。

---

# 7. Benchmark 固定为真实心花 60 分钟

第一阶段不要先跑全部被试。

固定抽样：

```text
A 一般质量                  20 min = 4×5 min
B 模糊/远场/轻声/混响        25 min = 5×5 min
C 快速轮替/短回应/抢话        10 min = 2×5 min
D 术语/专有词                 5 min = 1×5 min
总计                         60 min
```

B 组是核心组，必须优先选择当前转录明显错误的录音。

---

# 8. Gold transcript 与可听度规则

Gold 必须逐字保留口语，不润色。

例如：

```text
音频：我我觉得……就是还是有点那个
Gold：我我觉得就是还是有点那个
```

不能删除“嗯/啊/呃/哦”、重复、自我修正。

每个人工 segment 还必须标：

```text
audibility = A / B / C
```

- A：一次正常播放即可确定；
- B：模糊但可辨，需要重听/耳机/调音量，两名标注者最终可达成明确文本；
- C：两名标注者仍无法可靠确定。

正式主指标只计算：

```text
CER_AB = A + B
```

C 类不进入模型 CER，同时报告 `C_duration_ratio`。

---

# 9. 第一核心指标：CER + S/D/I

```text
CER = (Substitution + Deletion + Insertion) / Reference Characters
```

必须同时报告：

- Substitution rate
- Deletion rate
- Insertion rate

远场轻声尤其关注 `Deletion Rate`，因为模型可能整段漏掉弱语音。

中文 normalization 固定：

## CER_STRICT

1. Unicode NFKC
2. 英文小写
3. 去标点
4. 去空格

## CER_NORMALIZED

在 STRICT 上再做：

5. 简繁统一
6. 常见数字表达统一

仍保留语气词、重复、自我修正。

主指标：`CER_NORMALIZED`，同时保存 strict。

必要准则/工具：
- https://github.com/jitsi/jiwer

可以复用 edit-distance 核心，但中文字符 normalization 必须按本文件实现，不能按英文空格 word tokenization。

---

# 10. 第一轮只跑两个 ASR

同一批 chunk：

```text
Q1 = Qwen3-ASR-1.7B
F1 = FireRedASR2-LLM
```

输出：

```text
outputs/qwen3_raw/
outputs/firered2_raw/
```

必须分组报告：

| Model | Overall A+B | A clear | B blurry | Far/Noisy | Rapid/Backchannel | Terms |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | | | | | | |
| FireRed2 | | | | | | |

以及：

| Model | Sub | Del | Ins |
|---|---:|---:|---:|
| Qwen3 | | | |
| FireRed2 | | | |

---

# 11. Gate 1：只留下一个 PRIMARY_ASR

设：

```text
CER_Q = Qwen3 CER_AB
CER_F = FireRed2 CER_AB
```

若：

```text
abs(CER_Q - CER_F) >= 1.0 percentage point
```

直接选 CER 更低者。

若差距 `<1.0 point`：比较核心 B 组 `Far/Noisy CER`，选 B 组更低者。

若 B 组仍差 `<0.5 point`：默认选 `Qwen3-ASR-1.7B`，因为它与 ForcedAligner 属于同一官方 pipeline，最终系统更简单。

到此必须只留下一个：

```text
PRIMARY_ASR
```

第二名不进入默认生产 pipeline。

---

# 12. Gate 2：是否触发语音增强

检查 PRIMARY_ASR 的 B 组结果。

若：

```text
Group B CER <= 12%
```

且：

```text
Group B Deletion Rate <= 6%
```

则**不测试任何增强**，直接进入 Forced Alignment。

只有当：

```text
Group B CER > 12%
```

或：

```text
Group B Deletion Rate > 6%
```

才触发唯一增强分支：

```text
FRCRN_SE_16K
```

---

# 13. FRCRN 分支只能这样跑

必要模型仓库，仅触发时读：
- https://github.com/modelscope/ClearerVoice-Studio

只对 B 组 25 分钟做：

```text
RAW -> FRCRN_SE_16K -> PRIMARY_ASR
```

禁止同时尝试 MossFormerGAN、MossFormer2、Demucs、RNNoise 等。

保留 FRCRN 的条件必须同时满足：

```text
CER_raw_B - CER_frcrn_B >= 1.5 absolute points
Deletion Rate 不增加
短回应 Recall 下降 <= 5%
```

否则永久弃用增强，不继续搜索其它降噪模型。

如果 B 组通过，再对 A+C+D 35 分钟验证一次；只有当全量 `CER_enhanced <= CER_raw` 且 A 组恶化 `<0.5 point`，才采用“全量 FRCRN -> ASR”。否则生产仍使用 RAW。

不要建立自适应音质路由。

---

# 14. 时间戳固定用 Qwen3-ForcedAligner

无论 PRIMARY_ASR 是 Qwen3 还是 FireRedASR2-LLM，最终文本都统一送入：

```text
Qwen/Qwen3-ForcedAligner-0.6B
```

按 30–90 s chunk 对齐，不整小时对齐。

输出 token/word 相对时间，再加 chunk absolute offset。

人工抽 200 个 token 做边界 sanity check：

```text
start MAE
end MAE
```

判定：

```text
<=100 ms  接受
100–250ms 可用于对话轮次和隐私粗定位，但报告限制
>250ms    触发 timestamp failure exit
```

Qwen3 官方 forced-alignment benchmark 在 human-labeled raw/noisy 数据上报告几十毫秒量级的平均对齐偏差，因此先固定它，不搜索其它 aligner。

必要资料：
- https://github.com/QwenLM/Qwen3-ASR

---

# 15. 说话人固定用 DiariZen，不做搜索

模型：

```text
DiariZen-Large-s80
num_speakers = 2
```

输入永远用 RAW 工作音频；即使文字 ASR 最终用了 FRCRN，也不要把增强音频喂给 diarization。

必要资料：
- https://github.com/ntuspeechlab/Diarizen

只做人工作 10 分钟 sanity check，计算：

```text
speaker confusion time / total speech time
```

若：

```text
<=5%
```

直接接受，不研究其它 speaker 模型。

若：

```text
>5%
```

触发 speaker failure exit，停止并报告，不自动改用 pyannote/3D-Speaker。

---

# 16. Token ↔ speaker 融合固定规则

输入：

```text
ForcedAligner token timeline
DiariZen speaker timeline
```

每个 token：

1. 计算与 `speaker_0`、`speaker_1` 的时间 overlap；
2. 最大 overlap 者为 speaker；
3. 相同则用 token 中点所在 speaker；
4. token 落在空白区则向前后各搜索 300 ms；
5. 仍无法确定则 `speaker="uncertain"`。

utterance 重建：

```text
same speaker AND gap < 800 ms -> 合并
speaker change -> 立即断开
gap >= 800 ms -> 即使同 speaker 也断开
```

每场咨询只需人工一次指定：

```text
speaker_0 = therapist / client
speaker_1 = client / therapist
```

不引入声纹模型。

---

# 17. 必须额外评估短回应召回

固定词表：

```text
嗯
嗯嗯
啊
哦
对
是
好
行
然后呢
```

计算 `Backchannel Recall`。

原因：低质量远场语音和语音增强最容易漏掉轻声短反馈，而这对咨询互动结构很重要。

---

# 18. Codex 严格执行树

```text
START
  ↓
0. 检查 GPU / CUDA / ffmpeg
  ↓
1. 建立真实心花 60 min benchmark manifest
  ↓
2. 读取人工 Gold + audibility A/B/C
  ↓
3. 用同一 chunk manifest 跑：
     ├─ Qwen3-ASR-1.7B
     └─ FireRedASR2-LLM
  ↓
4. 计算 CER_AB + A/B + subsets + S/D/I
  ↓
5. 按 Gate 1 选唯一 PRIMARY_ASR
  ↓
6. B 组 CER >12% 或 Del >6%？
     ├─ NO：跳过增强
     └─ YES：FRCRN_SE_16K -> PRIMARY_ASR
              按 Gate 2 保留或永久弃用
  ↓
7. 得到最终 transcript
  ↓
8. Qwen3-ForcedAligner
  ↓
9. DiariZen-Large, num_speakers=2
  ↓
10. token ↔ speaker 时间融合
  ↓
11. utterance reconstruction
  ↓
12. final JSONL + benchmark report
```

Codex 不得跳出这棵树自行增加模型。

---

# 19. 固定 repo 结构

```text
xin-hua-asr-benchmark/
├── README.md
├── docs/
│   └── ASR_LOCAL_EXECUTION_PLAN_V3.md
├── configs/
│   ├── qwen3.yaml
│   ├── firered2.yaml
│   ├── diarizen.yaml
│   └── frcrn.yaml
├── src/
│   ├── audio/
│   │   ├── inspect.py
│   │   ├── convert.py
│   │   └── chunk.py
│   ├── asr/
│   │   ├── qwen3.py
│   │   └── firered2.py
│   ├── enhance/
│   │   └── frcrn.py
│   ├── align/
│   │   └── qwen3_forced_aligner.py
│   ├── diarization/
│   │   └── diarizen.py
│   ├── fusion/
│   │   ├── assign_speaker.py
│   │   └── build_utterances.py
│   ├── eval/
│   │   ├── normalize_zh.py
│   │   ├── cer.py
│   │   ├── audibility.py
│   │   └── backchannel.py
│   └── schema.py
├── scripts/
│   ├── 00_inspect_env.py
│   ├── 01_make_manifest.py
│   ├── 02_run_qwen3.py
│   ├── 03_run_firered2.py
│   ├── 04_eval_asr.py
│   ├── 05_run_frcrn_if_needed.py
│   ├── 06_force_align.py
│   ├── 07_diarize.py
│   ├── 08_fuse.py
│   └── 09_make_report.py
├── reports/
├── outputs/
└── tests/
```

不创建其它模型 adapter。

---

# 20. 第一阶段报告固定格式

生成：

```text
reports/ASR_LOCAL_BENCHMARK_V3.md
```

Table 1：

| Model | CER_AB | CER_A | CER_B | Sub | Del | Ins |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | | | | | | |
| FireRed2 | | | | | | |

Table 2：

| Model | Normal | Far/Noisy | Rapid/Backchannel | Terms |
|---|---:|---:|---:|---:|
| Qwen3 | | | | |
| FireRed2 | | | | |

Table 3，仅触发增强时：

| Input | Far/Noisy CER | Del | Backchannel Recall |
|---|---:|---:|---:|
| Raw | | | |
| FRCRN | | | |

Table 4：

| Metric | Result |
|---|---:|
| Forced align start MAE | |
| Forced align end MAE | |
| Speaker confusion rate | |
| Backchannel recall | |

必须另外输出至少 30 个典型 ASR 错误，优先来自 B 组，并分类：`low_volume / reverberation / background_noise / homophone / deletion / hallucination / backchannel / overlap / domain_term / proper_noun`。

---

# 21. 数据安全

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

禁止提交真实咨询音频、真实完整 transcript、被试姓名、模型权重。

---

# 22. 参考资料分级

## A. 必须读 / 必须实现

### Qwen3-ASR
- https://github.com/QwenLM/Qwen3-ASR
- https://arxiv.org/abs/2601.21337

必须确认：1.7B 官方推理、Chinese、non-streaming、ForcedAligner、当前依赖。

### FireRedASR2S
- https://github.com/FireRedTeam/FireRedASR2S
- https://arxiv.org/abs/2603.10420

必须确认：FireRedASR2-LLM、官方 inference、输入要求、推荐参数。

### DiariZen
- https://github.com/ntuspeechlab/Diarizen

只实现固定两人 diarization。

### CER/Edit distance
- https://github.com/jitsi/jiwer

只复用 edit-distance 核心；中文 normalization 按本文规则实现。

## B. 条件触发才看

### ClearerVoice / FRCRN
只有 Gate 2 触发才看：
- https://github.com/modelscope/ClearerVoice-Studio

只实现 `FRCRN_SE_16K`。

### Qwen context 风险
只有以后确定错误主要来自领域术语时看：
- https://github.com/QwenLM/Qwen3-ASR/issues/186

当前禁止 context。

## C. 仅供参考，没有问题就不要看

- FunASR / Paraformer: https://github.com/modelscope/FunASR
- MOSS-Transcribe-Diarize: https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
- pyannote: https://github.com/pyannote/pyannote-audio
- 3D-Speaker: https://github.com/modelscope/3D-Speaker
- AISHELL-4: https://www.openslr.org/111/
- AliMeeting: https://github.com/yufan-aslp/AliMeeting

这些不能影响第一阶段实现，也不能替代真实心花 benchmark。

---

# 23. 失败出口：只有这些情况才允许重新调研

Codex 不得自行扩展；只在下列情况生成 `reports/SECOND_STAGE_TRIGGER.md` 后停止：

### Failure 1：ASR

```text
Qwen3 与 FireRed2 的 CER_AB 都 > 15%
且 FRCRN 改善 < 1.5 points
```

### Failure 2：核心模糊组

```text
best Group B CER > 20%
```

### Failure 3：speaker

```text
DiariZen speaker confusion > 5%
```

### Failure 4：timestamp

```text
ForcedAligner MAE > 250 ms
```

触发后只说明哪一项失败，不自动安装更多模型。

---

# 24. 当前预注册假设

```text
H1：Qwen3-ASR-1.7B 在心花 B 组模糊远场数据上优于 FireRedASR2-LLM。
H2：FireRedASR2-LLM 在较清晰普通话段上可能不弱于 Qwen3。
H3：语音增强不会默认提高 ASR，FRCRN 只有真实降低 CER 才保留。
H4：说话人分离不是当前主要瓶颈，所以固定 DiariZen 两人模式，不做模型搜索。
```

这些是假设，不是结论。

---

# 25. 最终允许的唯一成功形态

```text
RAW 或 FRCRN
→ Qwen3-ASR-1.7B / FireRedASR2-LLM 二选一
→ Qwen3-ForcedAligner
→ DiariZen-Large (2 speakers)
→ time-axis fusion
→ final JSONL
```

一句话目标：

> 用真实心花 60 分钟人工 gold 数据，严格二选一确定 Qwen3-ASR-1.7B 或 FireRedASR2-LLM 谁对中文远场模糊咨询录音更准确；只有在二者对核心模糊音频仍明显不足时测试一次 FRCRN_SE_16K；最终统一使用 Qwen3-ForcedAligner + DiariZen 两人模式恢复时间戳与说话人，不允许第一阶段继续扩散到更多模型。
