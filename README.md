# RQ-A-4-10/20 个人实现与看板成果

本仓库保存李洪负责的两项需求的个人实现副本，并为 [个人 Kanban 看板](https://github.com/users/nijinzhan/projects/6/views/1) 提供成果链接。

所属课程项目为「VHHH可视化ATC地空通话语音标注系统」。正式课程成果仍按要求提交到 [团队仓库的李洪目录](https://github.com/kenzed/SEPA2026_TeamBlue/tree/main/%E6%9D%8E%E6%B4%AA)；本次先发布个人仓库副本，不表示团队仓库已同步提交。

## 范围与进度

- **RQ-A-4-10**：音频读取和校验、VAD通话分段、ASR转写、起止时间关联及结果反馈。
- **RQ-A-4-20**：识别文本的接收、关联、保存、按标识读取及实际模型信息追溯。
- 截至2026-10-17，初版实现和本地验证已完成；单元测试、模拟HTTP接口测试及真实模型用例合计 **156项通过**。真实模型为 Silero VAD v5.0 和 SenseVoice int8。
- 团队真实服务上的「录音→分段→识别→保存→读回」联调、数据库任务关联核验及实际空管语音质量评估尚未完成。
- 当前实现、接口核对和初步测试卡进入 **In review**，表示等待检查；两个总需求为 **In progress**，完整联调卡暂留 **Backlog**。

## 成果入口

| 成果 | 链接 |
| --- | --- |
| 个人需求规格说明书 | [RQ-A-4-10/20 需求规格说明书](docs/A4-RQ-A-4-10_20_需求规格说明书.docx) |
| 数据、接口、时间和职责对应 | [数据与接口对应说明](docs/RQ-A-4-10_20数据与接口对应说明.md) |
| 运行、实现边界和自动化测试 | [代码实现与测试说明](docs/代码实现与测试说明.md) |
| 真实VAD/ASR验证及已知限制 | [真实模型验证记录](docs/真实模型验证记录.md) |
| 实现代码 | [src/a4_speech](src/a4_speech/) |
| 测试代码 | [tests](tests/) |
| 当前个人周报 | [2026-10-13至2026-10-19周报（截至17日）](deliverables/周报/99-项目周报-RQ-A-4-10_20-2026-10-13-2026-10-19.xlsx) |
| 运行配置与依赖 | [配置样例](config.example.json)、[项目配置](pyproject.toml)、[锁定清单](requirements.lock.txt) |

## 安装与验证

以下命令在本仓库根目录执行。已验证 Windows x64、Python 3.13.9；其他Python版本请按实现说明处理依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m a4_speech --help
```

默认运行结果为155项通过、1项真实模型用例跳过。要复现156项全部通过，需按[真实模型验证记录](docs/真实模型验证记录.md)从官方来源准备模型和样例、建立本机配置，并设置 `A4_REAL_MODEL_CONFIG` 与 `A4_REAL_TEST_AUDIO`。配置样例中的服务地址和引擎代码不是团队服务已部署的证明。

本仓库不上传本机凭据、`config.local.json`、模型权重、虚拟环境、缓存和原始运行输出。文档中的本机验证记录用于说明已有结果；模型和实际服务需在运行环境中配置。

## 处理规则与待检查项

- 无语音时不创建空识别任务；重复的相同片段复用已有标识，冲突或重叠停止处理。
- 部分片段识别成功后发生失败时，记录处理情况，**不提交部分正式文本**。保存失败与识别失败分别处理。
- 历史模型信息按任务快照核对；当前公开文本响应缺少 `jobId`，不能据此独立证明数据库中的所属任务外键，需真实服务联调核实。
- 拼接样例第二段存在漏词现象，已记录；当前结果不代表空管领域准确率、时标精度或系统性能已验收。
