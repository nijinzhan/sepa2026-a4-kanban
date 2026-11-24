# ATCO2 ASR 与 VAD 对照实验报告

实验范围：冻结的 49 条公开 ATCO2 英语录音、327.616 秒、7 个机场。该结果是跨语料模型诊断，不代表 VHHH 业务性能。

## 六组转写结果

严格 WER 是主指标；数字归一化 WER 是既有自定义敏感性指标。S/D/I 分别为替换、删除、插入。呼号标注在这批样本中未通过质量标记核验，不据此发布正式呼号准确率。

| ASR | 边界输入 | 严格 WER | 数字归一化 WER | N | S/D/I | 片段数 | 有推理错误的录音数 |
|---|---|---:|---:|---:|---:|---:|---:|
| sensevoice | xml | 79.15% | 59.20% | 892 | 449/249/8 | 66 | 0 |
| sensevoice | silero | 82.29% | 64.97% | 892 | 365/368/1 | 67 | 0 |
| sensevoice | webrtc | 82.29% | 65.85% | 892 | 360/366/8 | 54 | 0 |
| xlsr | xml | 39.80% | 39.47% | 892 | 267/76/12 | 66 | 0 |
| xlsr | silero | 43.05% | 42.24% | 892 | 274/97/13 | 67 | 0 |
| xlsr | webrtc | 40.81% | 40.35% | 892 | 276/68/20 | 54 | 0 |

XLS-R 与 SenseVoice 的配对差值及按录音重抽样的 95% 区间：

- xml / 严格WER: XLS-R 减 SenseVoice = -39.35%，95%区间 [-45.34%, -32.74%]。
- xml / 数字归一化WER: XLS-R 减 SenseVoice = -19.73%，95%区间 [-25.77%, -13.51%]。
- silero / 严格WER: XLS-R 减 SenseVoice = -39.24%，95%区间 [-45.74%, -32.39%]。
- silero / 数字归一化WER: XLS-R 减 SenseVoice = -22.73%，95%区间 [-28.92%, -16.21%]。
- webrtc / 严格WER: XLS-R 减 SenseVoice = -41.48%，95%区间 [-47.99%, -34.17%]。
- webrtc / 数字归一化WER: XLS-R 减 SenseVoice = -25.50%，95%区间 [-32.53%, -17.50%]。

## VAD 边界诊断

| VAD | 参考区间 | 预测区间 | 一对一匹配 | 未匹配参考/预测 | 语音时长覆盖率 | 起点绝对误差中位数 | 终点绝对误差中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| silero | 52 | 67 | 45 | 7/22 | 89.01% | 318.0 ms | 142.0 ms |
| webrtc | 52 | 54 | 46 | 6/8 | 98.10% | 75.0 ms | 55.0 ms |

限制：XML transcription intervals are a proxy for VAD boundaries, not independently verified frame-level VAD ground truth. 匹配使用首轮固定口径 （参考间隔≤500 ms 合并，IoU≥0.5 一对一匹配）；指标只作诊断。XML 人工边界是控制组，不是可部署 VAD。WebRTC 参数在推理前固定，不基于本批得分调参。

## 运行资源与复现

XLS-R GPU：NVIDIA GeForce RTX 5060 Laptop GPU；PyTorch 2.14.1+cu130 / CUDA 13.0；FP32；模型加载 1.71 秒；峰值已分配显存 1558 MiB。

模型权重 SHA-256：`8967523e4d6f79daa431dbc4a36374a2c3e67697f70e5860e4614f23fea498b8`；样本清单 SHA-256：`d332daf52355def630a3a08e3fb5fd543a8ff19636fedb4efee44e7f7151ccb3`；参考文本 SHA-256：`fd9991448d5536931bdef8f82e560d4a1163512f5c3d7053c5c1e901b89b32c7`。全部输入为本地文件，未调用外部推理 API。SenseVoice 既有结果来自首轮缓存，WebRTC 新臂由同一冻结 SenseVoice 模型和本地 CPU 运行。

原始识别文本只保存在本机忽略目录中。交付目录应仅包含本报告、汇总表、逐录音指标表及可复现实验脚本；ATCO2 原始音频和完整转写不得公开分发。
