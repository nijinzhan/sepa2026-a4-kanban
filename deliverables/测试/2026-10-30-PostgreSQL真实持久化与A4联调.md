# PostgreSQL 真实持久化与 A-4 联调记录

执行日期：2026-10-30。结论：本机真实 PostgreSQL 保存、同批文本查询导出及数据库/转写服务重启读回均通过；当前验证范围为本机隔离部署。

## 基线与证据

| 项目 | 实际使用值 |
|---|---|
| A-5 SQL | P-Blue `745db5cd4d9a52d8bf68adf38290e52f09904aca`；bootstrap、V1–V13、R01–R05，19文件 |
| A-4 同组代码 | SEPA2026_TeamBlue `943d87f0e8e01b546932e33ca6042f91fd631b9a`；刘睿和于天名原始快照 |
| 数据库 | Docker PostgreSQL 16.15，独立持久化卷，本机55433端口 |
| 音频 | 本地 VHHH C15 前20秒；原 WAV SHA-256 `21c83396b32b596f4eee3f57fac51b5325773e4a1d34a875015ce16263dd3b50` |
| 识别模型 | `SenseVoiceSmall-ATC-VHHH / teacher-2025_0706-int8`，实际 CPU 推理 |
| 模型 SHA-256 | `625facec8fb54410d0e34543947cd80c2e0b1e568df00395bd9a31d3d00f23e8` |
| recordingId | `8ff1c260-57ff-439d-89cf-06f06c94db01` |
| businessTaskId | `f4895b6d-7c96-4f40-9c8e-0839424bf3b4`，扩展 JSON 关联，未验证外部业务任务外键 |
| asrJobId | `5feb264d-2d16-4d2d-b47b-e71c346b8378`，真实数据库任务外键 |
| transcriptId | `a3a2c4ca-a622-4b46-9a46-4fae5cc04006`、`cc6d0745-ea61-4430-acda-54e706fd6ef9` |

原始捕获时间未知，音频是真实声音，但导入 UUID 和 UTC 锚点属于本地测试登记，不用于航迹对齐。没有人工参考转写，本记录不报告 VHHH WER。

## 验证结果

- 自动调度通过真实本机 HTTP 调用教师模型服务。显式保存前 `asr.transcript` 为0条，保存后为2条；原文和来源读回一致。
- 于天名查询与导出通过本地只读字段视图直接查询相同 PostgreSQL canonical 行，没有复制文本数据。其既有导出任务写入本地兼容 schema 的 `export_job`，尚未映射团队 `anno.export_job`。
- JSON / CSV / SRT / VTT 各包含2条识别文本、2个音频切片；UUID、文字、时间与实际保存行一致，音频切片时长误差不超过1ms。
- 重启独立数据库容器和本方转写服务后，两个 transcriptId 均读回原文一致。
- 12项真实 PostgreSQL 测试通过：包括保存前无文本、原子回滚、重复保存、原文/语言摘要保护、并发幂等键、音频路径越界、人工修订拒绝，以及部分成功释放失败/未处理片段、旧终态重放不清除新占用。
- 全量回归：**418 passed, 1 skipped，19.00秒**。已启用教师模型真实 CPU 用例及真实 PostgreSQL 用例；跳过项为可选 XLS-R CUDA 实验。Ruff 检查通过。
- 独立审查发现的两项任务占用问题和一项初始化中断续跑问题已修复，复查无剩余 P1/P2 阻断项。复查额外核对了四个 ZIP 哈希及其中的识别文字。

本机详细证据：`outputs/postgres/integration-4384ca6fddfa/summary.json`、`fixture.json`、`automation.json`、`speech.log` 和 `exports/`。不提交真实音频、模型、数据库卷、口令或下载镜像；随库只提交代码、Markdown 和不含口令的结果摘要。

## 尚未完成的跨模块事项

1. A-3 真实采集/下载服务及带真实捕获时间的录音没有因此接通；目前通过已登记的本地手工上传样本执行。
2. A-5 对外部署地址、认证、共同运行环境，以及业务任务标识的真实存在性仍需负责人确认。
3. PostgreSQL 本地模式未接可信 IAM，人工修订明确拒绝。HTTP 原有版本、原因、用户 Bearer 和留痕逻辑保留，不能将其模拟验证算作本轮真实用户修订验收。
4. 查询兼容视图和导出任务元数据目前位于本地适配 schema，需要于天名和 A-5 确认后再决定正式接入方式。

因此，周报中的“真实保存、重启读回”和“A-4 同批数据查询导出”已有本机真实证据；全队部署联调和 VHHH 质量验收继续单独跟踪。原周报 Excel 未在本轮改写。

使用方法见[PostgreSQL 本地接入说明](../../docs/PostgreSQL本地接入与组内联调.md)。
