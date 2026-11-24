# WhisperATC 复现实验与代码核查报告

> 2026-10-23 状态更新：本文保留原日期实验事实和指标；当前主方案已恢复为 sherpa-onnx + SenseVoice，XLS-R/WebRTC 仅保留为比较实验。当前选型与复现入口见[选型说明](../../../docs/模型选型与实验结论说明.md)。

日期：2026-10-21；负责人：李洪（2024141460331，A-4）。

## 1. 结论与适用范围

已下载作者公开的两个 WhisperATC Large v3 检查点，完成本地 GPU 推理、原 49 条录音对照、作者完整 113 条验证集的提示词消融，以及独立评分和官方推理实现核查。

- **目前没有发现此前高 WER 是由评分公式、音频预处理或已核查的模型适配代码接错造成的证据。** 这不等于证明整个系统没有任何缺陷；本次核查范围见第 5 节。
- 模型更大不必然更好：只在 ATCOSIM 微调的 WhisperATC，在本次 ATCO2 跨语料对照中仍弱于现有 XLS-R。
- 联合微调版在作者 113 条验证集上，不加提示词的严格 WER 为 **47.63%**，加入作者使用的元数据提示词后为 **16.96%**；按作者归一化口径分别为 **47.27% / 17.10%**。
- **16.96% 是作者模型选择验证集上、使用额外元数据的结果。它不是 VHHH 成绩，也不是本项目新留出的独立测试成绩。** 本轮没有复现出论文的 13.46%，不能将论文数字直接写成本项目验收指标。
- 原 49 条中有 **38 条出现在该作者公开的 ATCO2 训练划分，10 条在验证划分，1 条未匹配**。联合微调版在全部 49 条上的低 WER 只能用于诊断，不应用于宣称独立泛化能力。

业务 A-4 的生产识别器尚未因本轮研究替换。当前保留 XLS-R 作为跨语料基线，WhisperATC 加元数据作为待独立验证的候选方案。

## 2. 别人的对比实验如何做，哪些代码能复用

### 2.1 WhisperATC

[WhisperATC 的 ICRAT 2024 论文](https://pure.tudelft.nl/ws/portalfiles/portal/218298256/ICRAT2024_paper_83.pdf)比较原始模型、领域微调、文本归一化、提示词及数据划分。ATCO2 是真实无线电录音，ATCOSIM 是模拟环境语料，难度不同。论文中的部分结果如下，均为 WER，越低越好：

| 论文条件 | ATCO2 | ATCOSIM |
|---|---:|---:|
| 原始 Whisper Large v2，未归一化、无提示词 | 71.62% | 79.11% |
| 加文本归一化、无提示词 | 29.05% | 17.98% |
| 加文本归一化及提示词 | 24.03% | 16.74% |
| 两个语料共同微调的报告结果 | 13.46% | 1.17% |

ATCOSIM 改为说话人隔离后，论文报告 3.88%。随机分句的验证成绩、未见说话人成绩和真实无线电跨语料成绩不能直接互换。

[作者仓库](https://github.com/jlvdoorn/WhisperATC)提供数据制作、训练、推理和归一化代码，许可证为 LGPL-3.0。本次固定代码提交 `cd0b71e5e0da9241d589d4bdbfbf16440549e577`，只读取必要源码并适配，没有直接执行作者的下载、训练整段脚本。

| 作者文件 | 可复用部分 | 本次需要处理的差异 |
|---|---|---|
| `CreateDataset/01-Prepare.py` | 音频、文本、上下文信息的数据结构 | 公开脚本按文件遍历顺序分前 80% / 后 20%，没有显式打乱和种子；本次直接采用作者已发布 parquet 的实际成员名单 |
| `Fine-Tune/FineTuning-ATCO2-ATCOSIM.py` | 联合数据训练及验证流程 | 训练是另一个任务，本次只下载已有权重推理 |
| `Evaluate/Evaluate-FineTuned.py` | 音频与元数据提示词构造、全语料 WER | 脚本写有 `large-v2` 初始化及 `.bin` 加载；当前下载的是 Large v3 safetensors，需要正确的 128-mel 架构及严格权重加载 |
| `Evaluate/Normalizer.py` | ATC 数字、NATO、航空公司名称等归一化 | 先安全移除 XML 标记再调用；保留严格评分作为主指标，防止实体内容被误删 |

作者评估脚本默认遍历 `train+validation`，但随后同时输出全量及验证集两种统计。复用时要选验证集输出；这不是论文使用训练集冒充验证集的证据。

### 2.2 Wav2Vec2 / XLS-R

[idiap/w2v2-air-traffic](https://github.com/idiap/w2v2-air-traffic)提供 ATC 数据准备、CTC 训练、评估和 KenLM 语言模型解码代码，可复用其 `src/run_speech_recognition_ctc.py`、`src/eval_model.py`、`src/run_eval_model.sh` 的流程。其公开代码及可得数据不等于论文中所有受限语料都能直接下载。

现用 [XLS-R 300M ATC 模型卡](https://huggingface.co/Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim)的 16.87% 来自 UWB-ATCC 与 ATCOSIM 合并测试集，不能与本项目 ATCO2 的 39.80% 直接相减判断代码错误。比较时还要明确 greedy CTC 与带语言模型解码的区别。

由此可复用的实验结构是：固定划分与录音、固定主评分口径，再分别改变模型、VAD、解码器、提示词；不能同时改变多个条件后把增益都归因于模型。

## 3. 模型与本地环境

项目目录：`D:/course_project/现代软件工程/SEPA2026_TeamBlue/李洪`。

| 本地名称 | 作者模型 | 固定 revision |
|---|---|---|
| `models/whisper-large-v3-atcosim` | [ATCOSIM 微调版](https://huggingface.co/jlvdoorn/whisper-large-v3-atcosim) | `99a8cc0071e479a494372925650737582241070a` |
| `models/whisper-large-v3-atco2-asr-atcosim` | [ATCO2 + ATCOSIM 联合微调版](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr-atcosim) | `efd96bd7862709d2d6aeb55621934a9965db8ed7` |

两份 `model.safetensors` 均为 3,219,908,024 字节，总计约 6.44 GB（十进制），均已验证 SHA-256：

```text
ATCOSIM: 7f1f75dea64ed0f483a62edfcbbfcec40c10ee13b387fa9bfa1466343a86b10b
联合版:  fdd401356e0e8eb5a6d6688a5f2df9f3c52215973151b8eff05b4cf7a152ea29
```

推理完全使用本地文件，无远程 ASR API。运行环境为 `.venv-xlsr`、Python 3.11、Torch `2.14.1+cu130`、Transformers `5.20.2`、OpenAI Whisper `20250625`、jiwer `4.0.0`。完整依赖记录在项目根目录的 `requirements-whisper-atc-gpu.txt`。

GPU 为 RTX 5060 Laptop 8 GB。两模型顺序加载，batch=1，FP16；本轮 PyTorch 显存峰值分配约 **3,300 MiB**、峰值保留 **3,564 MiB**，说明该设置下本机足够完成推理。该数字不代表全部系统显存占用或训练显存需求。

49 条的四种边界共 196 个录音条件，每个模型约 140～144 秒完成，不含约 5 秒模型加载。113 条的两个提示条件共 226 个录音条件，约 167 秒。上述是整轮脚本耗时，不是单条延迟，也不能直接当作实时率。

## 4. 本轮实测

### 4.1 原 49 条固定样本：模型 × 边界

数据沿用上一轮冻结的 ATCO2 真实英语无线电样本：7 个机场、49 条、327.616 秒、66 个人工标注区间、严格参考词数 892。它不是 VHHH，也不是整个 ATCO2 语料。

Whisper 设置：英语、transcribe、greedy、无提示词、FP16、batch=1、最多生成 440 token。XML、Silero、WebRTC 均沿用原有冻结边界；整段条件不切割。每条录音按时间拼接各段输出，再以语料总错误数除以总参考词数。

既有 SenseVoice 为 2024-07-17 CPU int8 版本，XLS-R 为 FP32 greedy CTC、无外部语言模型。本表比较当前可运行的具体配置，没有额外分离量化或数值精度的影响。

| 模型 | XML 人工边界 | Silero 边界 | WebRTC 边界 |
|---|---:|---:|---:|
| SenseVoice（既有基线） | 79.15% | 82.29% | 82.29% |
| XLS-R 300M ATC（既有基线） | **39.80%** | **43.05%** | **40.81%** |
| WhisperATC：只用 ATCOSIM 微调 | 54.93% | 46.41% | 49.78% |
| WhisperATC：ATCO2 + ATCOSIM 联合微调† | 29.26% | 30.72% | 31.39% |

表内为严格 WER；†联合微调版与其中 38 条已知训练成员重合，整行仅作诊断。两种 Whisper 整段输入分别为 55.49% / 33.97%，没有对应的旧 XLS-R 整段实验，不据此作同输入横向结论。

这里能支持的结论：在本轮不含提示词、声明微调语料不含 ATCO2 的跨语料对照里，现有 XLS-R 仍优于 ATCOSIM-only Whisper。基础大模型预训练数据的潜在交集无法从公开资料完全排除。49 条很少，且已经反复查看，不应继续把它作为未接触的最终验收集。

联合版在 49 条中匹配到的 10 条作者验证成员上，XML 严格 WER 为 23.32%；样本仅 193 个参考词，不能用这个小子集替代完整验证集。

### 4.2 作者完整 113 条验证集：提示词消融

固定 [作者发布的数据集](https://huggingface.co/datasets/jlvdoorn/atco2-asr)，revision 为 `3b96f5791582cf6f6f348ca46fd502a981918d85`，实际 446 条训练、113 条验证；未自行重做 80/20 划分。该检查点参与过此验证集的模型选择，因此仍不是新的独立测试。

两组都对整条音频直接推理，无额外 VAD，均为 greedy FP16、英语、最多 220 个输出 token。唯一实验变量是是否加入作者构造的提示信息：机场/信道、航路点、候选呼号等 `info`，以及固定 NATO 和 ATC 词表；未把本条人工转写文本拼入提示。提示按 OpenAI 规则保留最后 223 个上下文 token，加起始标记。

| 条件 | 严格 WER（主指标） | 保守数字归一化 | OpenAI 英语归一化 | 作者 ATC 归一化（安全去标记后） |
|---|---:|---:|---:|---:|
| 不加提示词 | 47.63% | 47.40% | 51.96% | 47.27% |
| 加元数据提示词 | **16.96%** | **16.80%** | **19.48%** | **17.10%** |

严格口径：总参考词 N=2276；无提示词 S/D/I=332/572/180；有提示词 S/D/I=203/105/78。严格 WER 降低 **30.67 个百分点**。这是同一验证集、同一检查点上的条件差异，不能推断任意新机场都有同等提升。

无提示词有 3/113 条输出达到 220 token 上限，有提示词为 0/113；所有条目均纳入评分，没有删掉重复转写或失败案例。本轮没有推理异常。重复转写会增加插入错误，也提示后续产品需单独设计解码异常处理；本轮没有用参考答案选择重试结果。

元数据是额外信息，尤其候选呼号直接提供识别先验。项目若无法在推理前可靠取得对应上下文，就不能用这一组成绩代表纯音频识别能力。元数据来源、时效及真实 VHHH 可得性尚未验证。

本次下载的是作者 Large v3 发布版；论文中的 13.46% 还涉及其具体检查点、解码和归一化协议。本轮的 17.10% 只能称部分复现，不能称精确复现论文或达到当前 SOTA。

## 5. 是否有代码问题：已完成的核查

| 核查项 | 实测证据 | 能排除或发现的范围 |
|---|---|---|
| 旧 WER 实现 | 既有 294 条模型/边界结果 × 严格和数字两种口径，共 588 次与 jiwer 比对，编辑距离和 WER 全部一致；原六组语料 WER 均复算一致 | 没发现评分公式或语料汇总导致的大偏差；最优对齐并列时单独 S/D/I 分配可以不同 |
| 音频及人工参照 | 与作者发布数据匹配的 48 条，PCM 全部逐样本一致、严格参考 token 全部一致；剩余 1 条未匹配 | 没发现这 48 条音频与文本错配；未匹配条不能擅称独立留出 |
| Whisper 特征 | 49 条 HF 128-mel 与 OpenAI 官方特征逐值比较，最大绝对差约 1.19×10⁻⁷ | 没发现采样率、mel 数或补零方式接错 |
| XLS-R 适配器 | 按冻结顺序每机场首个人工区间，共 7 段，用官方 Transformers ASR pipeline CPU FP32 复算 | 7/7 转写与旧适配器完全一致；只覆盖这些片段 |
| Whisper 解码器 | 同一下载权重映射到 OpenAI 原生正确架构，严格加载无缺失/多余键，输出投影与词嵌入共享关系核对通过 | 7 个机场片段 + 首条验证录音的无/有提示条件，9/9 无时间戳输出与 HF 适配器完全一致 |
| 时间戳解码敏感性 | 上述 9 条再启用原生时间戳解码 | 8/9 文本一致；无提示重复案例仍重复，未据此修改主实验条件 |

发现的复用陷阱：

1. **归一化误删实体。** 作者的原始括号清理方式不能直接处理当前 `[...]` 内联 XML 标记。把当前 49 条带标记参照直接送入它，会把中间词语一并删除，归一化后参考总词数只剩 15；先只移除标记本身后为 883。本项目此前严格/数字评分已经保留这些实体内容，故这不是旧高 WER 的原因。本轮作者口径统一先做安全去标记，且只作为补充。
2. **模型结构与文件格式。** 作者评估脚本不能原样加载当前 Large v3 检查点；本轮使用正确 128-mel Large v3 架构、本地 safetensors、固定权重哈希。原生核对还严格检查了所有权重键，没有靠忽略不匹配层来“跑通”。
3. **辅助统计更正。** 新增 113 条评估初版复用了 49 条汇总器，造成额外的 `all49` 标签，并把 token 上限告警阈值误设为 224。已改用 `summary-corrected.json` 仅保留真实验证组，并按生成输出长度 ≥220 统计告警。保留旧计划、执行源码与原始预测，未修改转写或 WER；`postprocessing-audit.json` 记录输入哈希及更正内容。

另须注意：作者归一化可能合并或删除小数分隔、前导零、呼号拼写差异，因此低 WER 不等于航班呼号、跑道、频率等关键实体已正确。WER 也不能简单改写成“准确率=1−WER”。

## 6. 交付物和证据

本目录新增：

- `WhisperATC复现实验与代码核查报告.md`：本报告。
- `WhisperATC模型边界与评分汇总.csv`：14 个模型/边界组合，附全部/训练/验证/未匹配分组及四种评分口径。
- `WhisperATC公开验证集113条汇总.csv`：113 条的两种提示条件、四种口径、编辑错误计数。
- `WhisperATC样本划分核查.csv`：49 条逐条的公开划分、PCM 与参考对齐检查。
- `WhisperATC可复核证据.json`：计划、结果文件校验、计数和核查概览，不包含原始转写正文。

完整原始输出保存在项目的忽略目录：

```text
outputs/atco2-1h-20261021/whisper-atc-20261021/
  experiment-plan.json
  audit.json
  preprocessing-audit.json
  xlsr-official-pipeline-audit.json
  native-decoder-audit.json
  comparison-summary.json
  whisper-atcosim-results.jsonl
  whisper-mixed-results.jsonl
  published-validation-113/
    plan.json
    results.jsonl
    summary-corrected.json
    postprocessing-audit.json
  research/
    asset-provenance.json
    executed-sources/
    published-atco2-dataset/data/
```

49 条方案 SHA-256：`361d74d5b86dcbd45232382d1972811dbebfea4d47f302fc27e213ab4800559d`。方案内记录模型、依赖、音频清单、人工参照及原基线输出哈希，旧结果没有被新实验覆盖。

本次真实执行的原始源码副本位于 `research/executed-sources`；其哈希对应各计划中的 `scriptSha256`。113 条脚本之后只更正辅助统计并增加 `--summarize-only`，当前源码和原执行源码的区别已留痕，不能将新源码哈希冒充原执行哈希。

## 7. 复核和重新运行

在项目目录用 PowerShell 执行。网络代理仅在本机确有对应服务时临时设置；推理不需要网络。

```powershell
# 下载固定提交的两份模型、作者源码及公开划分，并验证哈希。
.\.venv-xlsr\Scripts\python.exe tools/download_whisper_atc_assets_2024141460331.py

# 已完成结果的汇总，不重新推理。
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py summarize
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_published_validation_2024141460331.py --summarize-only
```

以下是从原 49 条基线数据齐备、且本轮结果尚不存在的状态重新执行的顺序。现有结果不要重新 `freeze`；脚本会拒绝覆盖已有实验计划。现有 113 条计划也会拒绝以改变后的脚本继续混写结果，复核现有数据请使用 `--summarize-only`。

```powershell
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py audit
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_preprocessing_audit_2024141460331.py
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py freeze
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py run --model atcosim
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py run --model mixed
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_experiment_2024141460331.py summarize
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_published_validation_2024141460331.py
.\.venv-xlsr\Scripts\python.exe tools/whisper_atc_native_decoder_audit_2024141460331.py
```

## 8. 后续课程项目实验的取舍

本周已有足够证据交付“公开 ATC 基线 + 模型和 VAD 对照 + 提示词消融 + 代码核查”，无需为了追论文数字不断增加模型。下一轮有价值的工作是建立未参与微调和本轮选型的新留出集，并确认实际可用的机场/航班元数据。

若取得真实 VHHH 录音及独立人工参照，在同一份预先冻结的留出集上比较 **XLS-R、WhisperATC 无提示、WhisperATC 真实可用元数据提示**，再固定较好的 ASR 比较 VAD。至少报告严格/数字 WER、呼号与数字实体错误、切割漏词及耗时，保留难例和重复输出。不要把候选呼号列表、人工切割、归一化带来的收益统一写成“换模型的提升”。

正式报告可采用的表述是：“在作者 113 条公开验证录音中，联合微调 WhisperATC 使用元数据提示时严格 WER 为 16.96%；该结果尚待独立 VHHH 数据验证。”

## 9. 完成时验证

- 相关评分、数字归一化、实验输入和划分保护测试：21 项通过。
- 本轮五个工具脚本及新增实验测试文件：Ruff 检查和格式检查均通过。
- `.venv-xlsr` 的 `pip check`：无依赖冲突。
- 新增 Whisper 录音条件结果共 618 条（196 + 196 + 226），计数及唯一键检查通过。
- 两份模型文件哈希、49 条冻结方案的 10 个文件依赖哈希、113 条及原生解码核查的执行源码快照哈希：均验证一致。
- 导出 CSV 的 224 / 8 / 49 个数据行分别对应模型分组评分、公开验证集评分和逐录音划分检查；CSV 使用 UTF-8 BOM，便于本机 Excel 打开。
