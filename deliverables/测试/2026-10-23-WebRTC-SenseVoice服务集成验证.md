# WebRTC VAD＋sherpa-onnx SenseVoice 服务集成验证

日期：2026-10-23；负责人：李洪；范围：RQ-A-4-10、RQ-A-4-20 的个人实现。

## 1. 本次决定与改动

按本人确认，当前默认组合为 **WebRTC VAD＋sherpa-onnx SenseVoiceSmall**，两者均在本机 CPU 执行。保留 SenseVoice 模型、词表、英语、ITN 和2线程设置；未微调模型、未接云端识别 API。

- `config.example.json` 与本机私有 `config.local.json` 同步为该组合，后者的服务地址、凭据环境变量等其他字段保持原值。
- 新增独立的 `vad_webrtc_2024141460331.py`，由原 `SherpaEngine` 调用；CLI 识别该组合，既有分段登记、任务、保存和读回入口继续复用。
- `pyproject.toml` 与 `requirements.lock.txt` 增加固定依赖 `webrtcvad-wheels==2.0.14.post1`，默认 `.venv` 已安装并重新安装可编辑项目。
- WebRTC 版本及全部实际分段参数写入执行快照；SenseVoice 权重与词表继续按 SHA-256 核对。版本或文件变化时拒绝按旧快照继续执行，不伪造 Silero 身份。
- `config.silero-sensevoice.example.json` 保留原配置，可显式复核；XLS-R 实验配置、实现、模型和结果保留。未改公共接口结构。

当前 SenseVoice 是官方通用 int8 版，不是已核实的教师 ATC 微调权重。课程要求与权重身份问题沿用[模型选型说明](../../docs/模型选型与实验结论说明.md)单列记录。

## 2. 固定运行条件

| 项目 | 实际配置 |
|---|---|
| 环境 | Windows x64；CPython 3.13.9；sherpa-onnx 1.13.8 |
| VAD | `webrtcvad-wheels==2.0.14.post1`；导入名 `webrtcvad`；本地原生 CPU 检测，不通过 sherpa-onnx 执行 |
| 输入与帧 | 16 kHz 单声道；16位小端 PCM；20ms帧；激进度2 |
| 后处理 | 间隔≤500ms时合并；最短段250ms；padding=0；每条录音新建检测器 |
| 尾帧 | 仅为帧检测补零，输出结束时间收回原始音频时长，不把补零计入真实录音 |
| ASR | SenseVoiceSmall，2024-07-17 int8，英语，`useItn=true`，CPU，2线程 |
| ASR 权重 SHA-256 | `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51` |
| 词表 SHA-256 | `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc` |

选用上述 VAD 参数是为了复用已有比较实验中验证过的条件。本次没有搜索调参，也不声称 WebRTC 在所有指标上优于 Silero。现有 XML 区间仅作边界代理，WebRTC 的覆盖率与边界结果是取舍依据；SenseVoice 严格 WER 在此前两种 VAD 下相同。

## 3. 自动化与真实推理验证

| 检查 | 结果与含义 |
|---|---|
| 新增受控边界测试 | 首轮 **13 failed**，接入后 **13 passed**；覆盖组合分发、无需 Silero 文件、20ms帧、尾点、短语音、静音、合并、精确 ASR 裁剪、模型及版本变化拒绝、保存读回和快照 |
| 当前配置全量回归 | **287 passed, 1 skipped，6.86秒**；显式启用了真实 VAD 静音及 SenseVoice 英语样例测试；唯一跳过项是另行启用的 XLS-R GPU 实验 |
| 原 Silero 路径兼容性 | 显式使用保留配置运行真实模型用例，**1 passed, 27 deselected，3.54秒** |
| 默认 CLI 真实推理 | 官方英语样例7152ms；WebRTC输出20–7060ms；SenseVoice原文非空；`asrInferenceChecked=true` |
| 代码规范 | `src`、`tests` 及两份服务回放脚本 Ruff 检查通过，27个文件格式检查通过 |
| 依赖 | 默认 `.venv` 的 `pip check`：`No broken requirements found.` |

受控边界测试仅替换原生推理边界，不能充当真实模型质量证据。下节使用实际 WebRTC 和 SenseVoice 模型另行回放。

## 4. 固定49条真实 ATCO2 的集成回归

使用之前固定的49条公开英语 ATCO2 录音，共327.616秒、7个机场、892个严格参考词。逐条核对录音哈希、冻结清单和参考文本哈希，再通过实际 `SpeechService` 执行分段、转写、保存、按文本标识读取。A3/A4 HTTP 边界使用 `httpx.MockTransport` 与内存测试服务，模型推理为本地真实执行。

| 项目 | 本次结果 |
|---|---:|
| 成功处理、保存并读回的录音 | 49 / 49 |
| 与接入前 WebRTC/SenseVoice 分段和原始文本逐条相同 | 49 / 49 |
| 分段总数 | 54 |
| 严格词错误 S / D / I | 360 / 366 / 8 |
| 严格参考词 N | 892 |
| 严格 WER | 82.29% |
| 本次回放总耗时 | 113.174秒 |

总耗时包含解码、模型身份复核、真实推理和模拟接口保存读回，不是纯 ASR 延迟，也不用于和 GPU 模型直接比较。本次复用已看过的固定集合，且非 VHHH；验证结论是**新接入方式保留既有实验输出和业务流程**，没有测出新的识别质量提升，不构成独立机场验收。

冻结清单 SHA-256：`d332daf52355def630a3a08e3fb5fd543a8ff19636fedb4efee44e7f7151ccb3`。参考文本 SHA-256：`fd9991448d5536931bdef8f82e560d4a1163512f5c3d7053c5c1e901b89b32c7`。

机器可读结果见[本次回放摘要](2026-10-23-WebRTC-SenseVoice服务集成验证-summary.json)。本机逐录音原始文本保存在忽略目录 `outputs/webrtc-sensevoice-service-20261023/results.jsonl`，不随代码分发数据集内容。摘要中 `resultsSha256` 可用于核对本机原始结果。

## 5. 复现

在个人目录执行，模型文件及冻结 ATCO2 数据已在本机准备，不在运行阶段下载：

```powershell
.\.venv\Scripts\python.exe -m a4_speech --config config.local.json check-model --audio '.\models\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17\test_wavs\en.wav'
$env:A4_REAL_MODEL_CONFIG = (Resolve-Path '.\config.local.json').Path
$env:A4_REAL_TEST_AUDIO = (Resolve-Path '.\models\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17\test_wavs\en.wav').Path
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools/validate_webrtc_sensevoice_service_2024141460331.py
```

新增验证日志在 `outputs/webrtc-sensevoice-integration-20261023/`，包含首次失败、接入后通过、全量回归、原 Silero 兼容性、CLI 实际输出及环境记录。

## 6. 真实团队联调的条件

A-5/团队引擎接口须提供启用的 `webrtc-vad`（`modelName=webrtcvad-wheels`，`version=2.0.14.post1`）与 `sensevoice`（`modelName=SenseVoiceSmall`，`version=2024-07-17`）登记，设备均为 CPU。实际服务地址、凭据、可读取的 A-3 录音及公共数据持久化仍需联调。本次只在模拟 HTTP 边界中验证登记核对，没有替 A-5 写入真实登记。

因此本次完成个人代码的默认组合切换和本地集成回归；不据此关闭真实服务联调、课程微调权重确认或英语 ATC 质量验收事项。
