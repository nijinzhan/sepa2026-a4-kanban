# XLS-R 与 WebRTC 服务集成验证

> 2026-10-23 状态更新：本文保留原日期实验事实和指标；当前主方案已恢复为 sherpa-onnx + SenseVoice，XLS-R/WebRTC 仅保留为比较实验。当前选型与复现入口见[选型说明](../../docs/模型选型与实验结论说明.md)。

日期：2026-10-21。范围：李洪目录内 RQ-A-4-10/20 个人实现。结论：已将确定版本 XLS-R + WebRTC 接入现有 `SpeechService` 和 CLI，引擎经过本地真实模型推理及模拟 HTTP 接口保存读回验证。尚未连接真实团队后端，因此不将本次结果表述为全队服务联调通过。

## 实现与配置

- 新增 `src/a4_speech/models_xlsr_webrtc_2024141460331.py`。复用服务原有的 `snapshot`、`verify_snapshot`、`prepare`、`detect`、`recognize` 引擎协议；业务流程、文本保存及历史读取逻辑仍由 `SpeechService` 执行。
- CLI 根据 `webrtc-vad` / `xlsr-atc` 引擎代码选择新适配器；示例配置和本机配置已切换。历史 `silero-vad` / `sensevoice` 配置仍可选择旧适配器。
- WebRTC 固定为 `webrtcvad-wheels==2.0.14.post1`，16 kHz、20 ms帧、aggressiveness 2、静音合并500 ms、最短段250 ms、padding 0。末尾不足一帧补零，输出终点收至源音频时长。
- XLS-R 固定为 `Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim`，revision `01a4853599abf22090b8963e6167b08a890b1e90`，FP32、`cuda:0`、CTC greedy、无外部语言模型，仅读取本地文件。
- 实际模型权重 SHA-256 为 `8967523e4d6f79daa431dbc4a36374a2c3e67697f70e5860e4614f23fea498b8`，启动时验证；处理器文件哈希写入执行快照。任务期间检查权重文件身份和处理器文件哈希，发生变化即停止。
- 已在 `.venv-xlsr` 中补装 `httpx`/`httpcore`，完整依赖清单为 `requirements-xlsr-gpu.txt`；旧 Sherpa 作为可选依赖，避免 XLS-R 专用环境被要求安装另一推理引擎。

## A-3 VHHH 文件核查与冒烟验证

本地团队仓库中找到 `刘亦含/样例音频_VHHH_TWR_20261005_143000Z.wav`，大小320,044字节、单声道16 kHz、时长10,000 ms。其同目录《成果物6_样例音频文件及元数据登记记录》将该样例标为测试信号 WAV，用于采集存储链路场景；没有找到与该文件对应的人工转写文本。文档提到的30分钟原始 MP3 不在当前本地仓库文件中。

通过 CLI 实际 WebRTC 检测返回 `segments=[]`、`asrInferenceChecked=false`。同一文件经过 `SpeechService` 和模拟 HTTP 边界返回 `status=no_speech` / `code=NO_SPEECH`，创建作业数0、API写入数0。该结果验证无语音分支不会建立空任务或写入空文本，不能用于证明 VHHH 识别质量。

## 真实 ATCO2 模型与服务流程

先使用此前冻结评测集的第一条录音 `LKPR_RUZYNE_Radar_120_520MHz_20201029_075414`，机场 LKPR、时长14,144 ms、PCM16/16 kHz单声道、音频 SHA-256 `b9696f8c306c7a26fe082d357e0fcd66457c459185f1ee6af4c5de5691badf25`。这是公开 ATCO2 数据，不是 VHHH。

真实 WebRTC 分出两段：100—9160 ms、9840—14144 ms。真实 XLS-R 经服务层识别并通过模拟 A-3/A-4 HTTP 接口登记、建作业、保存和按文本ID读取；状态为 `succeeded`，`saveConfirmed=true`，读回原始文本与保存文本一致。

| 段 | 实际原始输出 |
| --- | --- |
| 100—9160 ms | oscar kilo tango uniform romeo praha radar hello identified climb to flight level seven zero and standby for ifr |
| 9840—14144 ms | climbing to flight level seven zero oscar kilo tany uniform romeo |

随后使用 `tools/validate_xlsr_webrtc_service_2024141460331.py` 经新服务适配器重放全部49条冻结 ATCO2 样本。49条均成功保存和读回，共54个分段；分段边界和原始转写逐条与此前冻结的 WebRTC/XLS-R 输出完全相同。严格评分结果为替换276、删除68、插入20、参考词数892，WER 40.81%；最终实现下本次重放耗时36.504秒。输出及校验摘要保存在忽略目录 `outputs/xlsr-webrtc-service-20261021/`，摘要记录了 manifest、人工参考、旧输出和本次输出的SHA-256。

首条录音单条结果：按既有评测器的严格归一化规则，30个参考词中替换1、删除0、插入0，严格WER为3.33%。这是总体集合中的样本，不是独立留出集，也不代表 VHHH 的质量；49条结果同样是复用既有固定集合的复核，不是新采集的独立评测。

## 自动化验证

- 常规全量回归：`.venv` 及 `.venv-xlsr` 均为273通过、2跳过；分别用时8.11秒和8.33秒。两个跳过项均为需要显式启用的真实模型测试。
- 新引擎常规测试覆盖固定 VAD 参数、短静音合并、尾帧补零和时长裁剪、精确ASR时间片、本地加载约束、快照变化、CLI分发、服务任务及文本保存。
- 显式启用真实 XLS-R 单录音服务测试：1通过、其他引擎单元测试取消选取。另有49条全量复核，完整数据在 `outputs/xlsr-webrtc-service-20261021/`；HTTP服务仍为测试替身。
- Ruff：`src`、`tests` 及本验证脚本检查通过；23个Python文件格式检查通过。
- 两个环境 `pip check` 均无依赖冲突。

复现真实模型集成用例：

```powershell
Set-Location 'D:\course_project\现代软件工程\SEPA2026_TeamBlue\李洪'
$env:A4_XLSR_REAL_MODEL_TESTS='1'
.\.venv-xlsr\Scripts\python.exe -m pytest -m real_model -q tests/test_xlsr_webrtc_engine_2024141460331.py
```

## 真实服务联调前仍需具备

1. A-3 提供可访问的真实录音及对应 recording UUID；质量评估另需与音频一一对应的人工转写。
2. A-5 在真实引擎登记中启用 `webrtc-vad`（模型 `webrtcvad-wheels`、版本 `2.0.14.post1`）和 `xlsr-atc`（上述模型名及revision），ASR设备枚举使用 `cuda`，部署方式按实际服务配置。
3. 提供真实 A3/A4 地址、凭据，并核对任务完成后再次提交有效文本的持久化、并发保护及重启读回。
4. 已核对远端 `P-Blue` 最新 `main`（2026-10-20提交 `745db5c`）：FastAPI 后端已有 A-5 实现，但其 A-3/A-4 API 仍是返回501的占位路由。该仓库引擎种子仍为 CPU 上的 sherpa-onnx/SenseVoice，目标资源1vCPU/512MB；XLS-R GPU 推理须在团队设计中确认独立推理主机/sidecar及 A-5 引擎登记，当前配置不满足该目标环境资源上限。本轮没有修改公共后端或其他成员文件，也没有宣称真实团队服务联调完成。

5. 已核对远端 TeamBlue 最新 `main`（2026-10-20提交 `c8c83d8`）：A-3 的郭三媛代码将 VHHH APP LiveATC 流写入本地 `final_storage`，但仓库没有包含采集输出，也没有对应人工文本。刘亦含提交的唯一 WAV 仍是测试信号。不能据此计算 VHHH WER。
