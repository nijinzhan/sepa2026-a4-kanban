# ATC 微调 XLS-R 模型对照实验方案

状态：3×2 ASR/VAD 对照已在冻结的 49 条样本上完成，六个单元均无推理错误；完整输出另存于 `outputs/atco2-1h-20261021/model-vad-comparison-20261021/`，首轮评估未覆盖。

## 1. 实验目的

在同一组冻结的 ATCO2 公开录音和人工转写上，交叉比较 SenseVoiceSmall、面向空管通信微调的 XLS-R 300M，以及三种片段输入：人工 XML 参考边界、现有 Silero v5、WebRTC VAD。主问题是 ASR 差异在不同边界下是否保持；辅助问题是 VAD 边界质量与转写质量之间的关系。

本实验是公开语料上的跨语料评估，**不是 VHHH 评估、领域微调或团队服务联调**。不得用这 49 条样本调参或训练后再在同一批上报告无偏测试结果。

## 2. 模型与推理锁定

| 角色 | 模型 | 推理配置 |
|---|---|---|
| 对照基线 | SenseVoiceSmall，2024-07-17，int8 | 既有 A-4 实现；英语、`useItn=true`；复用首轮冻结输出及配置 |
| 候选模型 | `Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim` | `Wav2Vec2Processor` + 本地 Transformers `AutoModelForCTC`；FP32、batch size 1；16 kHz 单声道；CTC greedy；首轮不启用外部语言模型 |

候选检查点固定为 Hugging Face 提交 `01a4853599abf22090b8963e6167b08a890b1e90`，`model.safetensors` 为 1,261,934,532 字节，SHA-256 `8967523e4d6f79daa431dbc4a36374a2c3e67697f70e5860e4614f23fea498b8`。权重存放在本地忽略目录 `models/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim/`。推理时只读本地文件，禁止 `trust_remote_code` 和音频上传/API 调用。

模型卡说明该检查点在 UWB-ATCC 与 ATCOSIM 上微调，许可证标注 Apache-2.0；此来源说明不能替代对 ATCO2 许可和训练语料授权的遵守。主实验先用 CTC greedy 隔离候选检查点的基础解码；不把模型卡上的其他语料分数当成本实验结果。后续若评估仓库附带 KenLM，需作为独立实验臂报告，不能混入主结果。[模型卡](https://huggingface.co/Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim)；[ATC XLS-R 研究与代码](https://github.com/idiap/w2v2-air-traffic)

## 3. 冻结数据与参照

- 样本沿用首轮预注册清单：49 条、327.616 秒、7 个机场（LKPR、LKTB、LSGS、LSZB、LSZH、LZIB、YSSY），每机场 7 条。
- 清单 SHA-256：`d332daf52355def630a3a08e3fb5fd543a8ff19636fedb4efee44e7f7151ccb3`；人工参考 SHA-256：`fd9991448d5536931bdef8f82e560d4a1163512f5c3d7053c5c1e901b89b32c7`。
- 参照文本使用更新 XML 的 `text/start/end`；全程保留全部 49 条，不按模型成败剔除。原始音频、完整转写和逐段预测留在本地，不公开分发。
- 首轮已有 66 个含参考文本的 XML 区间及 67 个 Silero VAD 输出区间。两组片段的源波形均取自同一批已验证的 16 kHz 单声道 PCM。

## 4. 实验矩阵

| 实验臂 | 输入片段 | 目的 | 固定项 |
|---|---|---|---|
| A：边界控制 | 冻结 XML 人工参考边界 | 比较 ASR，尽量排除自动 VAD 的边界影响 | 两模型使用相同起止样本、音频、参考文本、片段顺序和评分器；不运行 VAD |
| B：当前基线 | 冻结的 Silero v5 输出边界 | 测量现有 A-4 切分上的两种 ASR | 完整复用首轮 VAD 片段，不重新调阈值 |
| C：VAD 对照 | 本地 WebRTC VAD 2.0.14.post1 | 测量替换 VAD 后的边界和 ASR 变化 | aggressiveness=2、16 kHz、20 ms 帧；合并≤500 ms静音间隔、丢弃<250 ms语音段；两 ASR 复用同一组生成边界 |

共 6 个单元：SenseVoice 与 XLS-R 分别使用 A/B/C 三种边界。所有单元按录音时序拼接后评分。人工 XML 是转写段时标形成的 oracle 控制，不是可部署 VAD；WebRTC 的 post-processing 参数在评估前锁定。首轮 SenseVoice 参考值：Silero 边界严格 WER `82.29%`、数字归一化 WER `64.97%`；XML 边界严格 WER `79.15%`、数字归一化 WER `59.20%`。

## 5. 指标与分析

1. **主指标：严格词错误率 WER。** 沿用首轮脚本的文本规则、单位代价编辑距离和微平均算法，报告 `N/S/D/I/C` 及 `(S+D+I)/N`。不得在看过候选输出后修改评分器。
2. **补充指标：冻结的数字归一化 WER。** 采用首轮保守规则，仅用于分析数字书写差异；它是自定义敏感性指标，不等同标准官方分数。
3. **分层结果：** 各机场的严格及归一化 WER、每条录音结果、推理失败和空输出数。缺失输出保留在样本中并按删除计分。
4. **配对差异：** 以录音为重抽样单位、固定随机种子做 10,000 次配对 bootstrap，给出 XLS-R 减 SenseVoice 的 WER 差值及 95% 区间。49 条小样本的结果仅支持本批数据结论，不外推为全体 VHHH/ATC 性能。
5. **资源记录：** 分别记录模型加载时长、单片段/总推理时长、RTF、GPU 型号、推理设备、dtype、峰值显存和软件版本。GPU 上的 XLS-R 与 CPU 上的 SenseVoice 耗时只表示本机实际部署路径，不解释为相同硬件下的模型速度排名。
6. **呼号/数字专项：** 只作探索性诊断。首轮选中 65 个呼号 XML 标签均为 `correct_tagging=0`；完成盲法人工核验前不得报告为正式呼号准确率。

## 6. 执行控制与交付物

- 安装和推理使用单独的 Python 3.11 环境 `.venv-xlsr`，依赖锁定在 `requirements-xlsr-gpu.txt`，避免改动运行首轮评测的 `.venv`。候选模型从本地路径加载，固定提交、文件哈希、软件版本和参数。
- 本机执行环境：Python 3.11.9、PyTorch 2.14.1+cu130、Transformers 5.20.2、NumPy 2.4.6、RTX 5060 Laptop 8GB、NVIDIA 驱动 610.47。设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`，用 `Wav2Vec2Processor.from_pretrained(local_path, local_files_only=True)` 与 `AutoModelForCTC.from_pretrained(local_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32)` 加载；`AutoProcessor` 会依据仓库配置选中 KenLM 处理器，因此主实验显式选择无 LM 的 `Wav2Vec2Processor`。
- 冻结样本中首条录音已完成本地推理冒烟：14.144 秒、16 kHz 单声道，GPU 输出非空；模型加载 1.915 秒、推理 0.703 秒、RTF 0.0497，峰值分配显存 1,518.1 MiB，峰值保留显存 1,572.0 MiB。此为单条运行记录，不代表全量评测的平均速度。
- 先用 1 条样本确认解码文本非空、GPU 确实被使用、CUDA 显存有变化且输出可复现；再运行固定的 49 条样本。失败后保留原始错误和样本，不得悄悄跳过。
- 原始预测、运行配置、模型与音频哈希、计分结果及逐录音 CSV 写入被 Git 忽略的 `outputs/atco2-1h-20261021/model-vad-comparison-20261021/`；提交目录只交付汇总结果、可复现实验脚本和报告，不附原始音频或完整转写。
- 本方案及实验输出不覆盖首轮冻结结果或现有《测试分析报告》。三种边界、两种 ASR 的六个单元完成后另写对照结果；下一步先用经授权的 VHHH 录音和人工参照验证 XLS-R，再决定是否测试 KenLM。WebRTC 的边界诊断较好，但转写 WER 差异不确定，暂不据此替换 Silero。

## 7. 执行前后检查

| 顺序 | 检查点 | 通过条件 |
|---|---|---|
| 1 | 模型落盘 | 提交号正确，权重大小与 SHA-256 匹配（已通过） |
| 2 | 实验锁定 | 样本、参照、三种边界输入、指标和排除规则在推理前固定（已记录） |
| 3 | GPU 环境 | PyTorch CUDA 可用，显示 RTX 5060；Transformers 可从本地加载模型（已通过） |
| 4 | 样本冒烟 | 单样本输出非空、GPU 实际推理成功、运行记录完整（已通过） |
| 5 | 全量运行 | 六个实验单元均覆盖全部冻结样本；汇总与逐条结果可复算，异常和空输出保留（已完成） |

## 8. 全量执行结果

实验于 2026-10-21 在固定样本上完成。SenseVoice 的 XML 与 Silero 两臂由首轮结果复用；XLS-R 对三种边界输入重新运行 GPU 推理；SenseVoice 对新增 WebRTC 边界在本地 CPU 推理。六个单元均覆盖 49 条录音、无推理错误。汇总计分复算与首轮 SenseVoice XML/Silero 结果完全一致。

| ASR | 人工 XML 严格/数字归一化 WER | Silero v5 严格/数字归一化 WER | WebRTC 严格/数字归一化 WER |
|---|---:|---:|---:|
| SenseVoiceSmall int8 | 79.15% / 59.20% | 82.29% / 64.97% | 82.29% / 65.85% |
| XLS-R 300M ATC | 39.80% / 39.47% | 43.05% / 42.24% | 40.81% / 40.35% |

XLS-R 相比 SenseVoice 的严格 WER 在三种输入下分别降低 39.35、39.24、41.48 个百分点；按录音进行 10,000 次配对 bootstrap 的 95% 区间均低于 0。该结果支持继续将 XLS-R 作为 ATCO2 公开语料上的本地候选模型，但不代表 VHHH 实测性能。

WebRTC 的参考语音时长覆盖率为 98.10%，高于 Silero 的 89.01%；对应时长精确率为 93.32%，低于 Silero 的 97.36%。两种 ASR 上 WebRTC 减 Silero 的配对 WER 95% 区间均跨过 0，因此本批数据无法证明 WebRTC 稳定改善转写 WER。XML 时间是人工转写片段标注，不能代替独立逐帧 VAD 真值。

XLS-R 使用 RTX 5060 Laptop GPU、FP32；模型加载约 1.71 秒，三种片段输入的 GPU 推理共约 6.92 秒，峰值已分配显存约 1.56 GiB。详细统计见 `对照实验报告.md` 与汇总 CSV；逐录音预测及完整文本继续留在本机忽略目录。
