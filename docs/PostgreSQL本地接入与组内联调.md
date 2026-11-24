# PostgreSQL 本地接入与组内联调

## 当前执行口径（2026-11-07）

严格以团队主仓库已提交的数据库迁移、初始化及部署文件为准。最新核对main为82ef964；当前缺少录音、转写、标注及查询导出等基础迁移，Compose没有有效服务配置，完整部署尚未通过前置检查。本轮不执行后文基于P-Blue的初始化、音频登记或业务重放命令，也不补入主仓库之外的业务SQL来继续联调。

当前进度应写为“主仓库源码及部署前提已核对，数据库初始化前置验证受阻，转写—保存—修订—查询导出业务链路尚未开展”。此前在补充扩展、角色后V1—V4通过、V12缺表报错，是定位缺失文件的诊断证据，不代表仅凭主仓库的部署已通过。SRT的最新两项定向测试已通过。详细记录见[主仓库数据库独立部署复核](../deliverables/测试/2026-11-07-团队主仓库数据库独立部署复核.md)。

后文保留已完成实验的来源、命令和结果，供追溯；其中P-Blue数据库及本机审核快照均不作为本轮主仓库联调的替代环境。待主仓库补齐正式文件并完成部署前提核验后，再进入业务联调。

2026-11-07更新：PostgreSQL适配已支持真实身份人工修订及可配置`maxSegments`，见[本方修复验证](../deliverables/测试/2026-11-07-人工修订与处理上限修复验证.md)。后续历史结果保留原日期，不作为当前测试数量。

2026-10-31：已重放刘睿独立 HTTP 调度，并与于天名新版查询导出完成同批数据联调。新版直接读 `asr.v_transcript_full`、调用 A-5 导出函数并写 `anno.export_job`，不再依赖昨日的字段兼容视图或个人导出任务表。14项新版查询导出检查通过；同学原测试53项通过、1项因 Windows SRT 换行断言失败，含4项真实 PostgreSQL 测试全部通过。详见[新版联调与上游更新核对](../deliverables/测试/2026-10-31-A4新版查询导出与A3-A5更新联调核对.md)。无需另装 Windows PostgreSQL 或 pgAdmin。

首次使用 Docker 可先阅读[Docker 与联调入门操作](Docker与联调入门操作.md)：包含各入口作用、完整联调重放命令及浏览器手动读回步骤。

## 历史实验环境（P-Blue基线）

- 容器：`a4-lihong-postgres-20261031`；仅监听 `127.0.0.1:54329`。
- 数据库：`alpha_atc`；独立卷：`a4-lihong-pg16-20261030`。停止容器不会删除数据库。
- 口令与 DSN：仅保存在 ignored `outputs/postgres/runtime.local.json`。应用使用 `appDsn`，迁移和测试使用 `adminDsn`，不要把该文件加入 Git 或团队文档。
- 上游 SQL：P-Blue 提交 `745db5cd4d9a52d8bf68adf38290e52f09904aca`，bootstrap、V1–V13、R01–R05 原样执行，共19个文件。没有使用 P-Blue 工作区未提交 SQL，也没有要求 A-5 采纳个人 SQL 提案。
- 镜像：Amazon ECR Public 的 Docker Official Image PostgreSQL，Linux amd64 manifest `sha256:cbf42f0edc946c0bc1b8eeed62403734e1f25dc67a3491455d07a92fdb558b28`。下载时经本机现有代理取得原始镜像并校验 manifest、压缩层和展开层摘要；没有修改 Docker 全局代理。

Docker 的 `dockerInference` 启动报错通过在退出 Docker 后保留旧 `run` 目录备份、重新创建运行时端点解决。备份为 `C:\Users\lenovo\AppData\Local\Docker\run-backup-codex-20261030`，未恢复出厂设置、未删除镜像或卷。另一个镜像下载超时属于网络问题，与 PostgreSQL 表结构无关。

2026-10-31，Windows 将55398～55497列为TCP保留端口，原55433无法绑定。已保留原数据卷并换容器映射到54329，原容器 `a4-lihong-postgres-20261030` 停止保留，不能与新容器同时使用。已核对历史文本、新文本和正式导出记录，保留 `pg_stat_statements` 启动参数。运行时配置已更新；初始化工具读取该配置中的容器名和端口。

## 历史实验：启动本方服务

在 PowerShell 中进入本人目录后执行；本机依赖已经安装。

```powershell
Set-Location -LiteralPath 'D:\course_project\现代软件工程\SEPA2026_TeamBlue\李洪'
docker start a4-lihong-postgres-20261031
$taskPg = Get-Content -LiteralPath 'outputs/postgres/runtime.local.json' -Raw | ConvertFrom-Json
$env:A4_DATABASE_URL = $taskPg.appDsn
$env:A4_SERVICE_API_KEY = [guid]::NewGuid().ToString('N')
.\.venv\Scripts\python.exe -m a4_speech.api_2024141460331 --config config.postgres.local.json --host 127.0.0.1 --port 8001
```

调用方仍须使用同一个 `A4_SERVICE_API_KEY`，通过 `X-Api-Key` 访问转写、保存和读取入口。DSN 从环境变量读取；`config.postgres.local.json` 只声明 `storage.mode=postgres`、DSN 环境变量名、音频根目录和模型参数。不设置 PostgreSQL 模式时，原配置继续使用 HTTP 团队接口。

新环境可先安装 `.\.venv\Scripts\python.exe -m pip install -e '.[dev,postgres]'`，再从示例复制 `config.postgres.example.json`。教师模型文件仍需位于示例指定路径。

## 历史实验：初始化和登记本地音频

```powershell
.\.venv\Scripts\python.exe tools/postgres_local_2024141460331.py --source-repo 'D:\course_project\现代软件工程\P-Blue'
```

初始化工具固定上游提交，校验已执行 SQL 的哈希，在专用 `a4_setup_2024141460331` schema 记录迁移日志。每条迁移与日志在同一事务提交；中断可续跑。应用启动不自动建表或迁移。工具不会删除或重建既有卷。

导入音频前设置 `$env:A4_DATABASE_ADMIN_URL = $taskPg.adminDsn`，使用 `tools/postgres_import_audio_2024141460331.py --audio <16k单声道WAV路径> --config config.postgres.local.json --started-at <带时区时间> --test-anchor`。音频必须在配置的 `audioRoot` 内。原始录音时间不明确时必须使用测试锚点标记；不得把它作为真实 UTC 与航迹对齐。

导入工具向 `core.data_source / aud.channel / aud.recording` 登记本地手工上传样本，按实际模型快照登记引擎；这一步不代表已连接 A-3 录音下载服务。

## 历史实验：新版重放命令

在本机已保留的2026-10-31审核快照上执行；首次命令会登记新的20秒真实音频样本并生成新的识别文本，第二条命令使用同一批数据测试新版查询导出。

```powershell
$taskProbe = .\.venv\Scripts\python.exe tools/automation_http_integration_2024141460331.py --audit-dir .doc-sync-work/main-audit-20261031 | ConvertFrom-Json
.\.venv\Scripts\python.exe tools/query_export_integration_2024141460331.py --automation-summary $taskProbe.summaryFile
```

通知仍由本地测试客户端发出；音频导出中的下载方为本地 HTTP 协议替身，提供真实 WAV，测试后关闭。真实 A-3 录音服务及 A-5 Bearer 用户认证不在本次通过范围内。

## 2026-10-30历史链路

1. 本地 C15 真音频前20秒，教师 ATC SenseVoice 模型实际 CPU 推理，识别出2段文本。录音 UUID 和时间锚点为测试值，原始捕获时间未知。
2. 原完整重放脚本直接调用刘睿 `AutomationService.handle()` 业务类，由其 `HttpSpeechGateway` 通过真实本机 HTTP 调用本方识别和保存服务。显式保存前数据库文本行数为0，保存后为2。21:11新增独立自动化 HTTP 入口验证，九项检查通过；通知仍由本地测试客户端发送，真实 A-3 发送端尚未接通。详见[自动转写HTTP入口联调](../deliverables/测试/2026-10-30-自动转写HTTP入口联调.md)。
3. 文本直接写入 A-5 的 `asr.transcript`，具有真实 `segment_id / job_id / engine_code` 关联。业务任务标识保存在 `params/stats`，其外部任务存在性和业务任务外键尚未验证。
4. 于天名查询/导出使用同一个 PostgreSQL 中的只读字段兼容视图 `a4_query_2024141460331.audio_recording/audio_segment/transcript`。没有复制文本到 SQLite 或其他数据库。单独的本地 `export_job` 表保存其导出任务元数据，不冒充 `anno.export_job` 已完成适配。
5. JSON、CSV、SRT、VTT 各导出2条文本和2个音频切片，核对 UUID、文本、起止时间和切片时长。随后重启独立数据库容器和本方转写服务，读回一致。

历史同组代码基线为主仓库 `943d87f0e8e01b546932e33ca6042f91fd631b9a`，使用审核快照，未改动同学目录。旧重放工具为 `tools/postgres_integration_2024141460331.py`，仍固定旧版代码及兼容视图；测试新版应使用上方命令。快照和隔离依赖只保留本机，不复制为本人正式源码；不要对团队生产数据库运行。

验证结果见[真实数据库联调记录](../deliverables/测试/2026-10-30-PostgreSQL真实持久化与A4联调.md)。

## 当前边界

- 已完成本机真实数据库和 A-4 代码联通；同学各自部署环境、网络地址及鉴权仍需双方确认，不能称为全队线上验收。
- PostgreSQL模式人工修订要求配置`a5Url`（例如`http://127.0.0.1:8000/api/v1/a5`），且该服务已启动并连接同一IAM数据库。适配向`GET /auth/me`验证Bearer，按当前数据库用户状态和`a4:transcript:edit`权限控制写入；编辑人不接受客户端指定。未配置、服务不可达或响应无效时仍返回503，失效凭据401、权限不足403、版本过期409。HTTP模式保留原流程。
- 失败及未处理片段可释放后重试；成功但未保存的片段保留本任务占用。旧失败回报不能清除新任务的占用。重复保存只核对已有文本，不覆盖识别原文。
- A-3实际采集、真实捕获UTC及自动通知仍待核验。识别输出属于待人工复核的预标注；已核对的课程需求未给出识别准确率硬门槛，不用未确认的WER阈值判课程验收失败。

## 处理上限与修订验证

配置文件顶层设置`"maxSegments": 500`；省略时默认500，只接受正整数。启动时校验，VAD后、片段登记和ASR推理前检查，超限返回422、`JOB_SEGMENT_LIMIT`、实际`segmentCount`及本次`maxSegments`。任务上下文记录冻结的配置值。服务、HTTP客户端和PostgreSQL适配使用同一配置；没有自动截断或分批。外部A-3段登记接口仍有独立2000条批量约束；HTTP模式提高本方上限前，也须确认上游任务接口允许该规模。

修订在同一事务内锁定记录并检查`expectedRevision`，更新文本或审核状态时推进一次版本，保持`text_raw`不变，利用原触发器记录历史并写入审计；无变化提交不新增版本。适配显式处理旧V7触发器“仅审核状态修改不自动推进版本”的差异，不需要安装个人SQL草案或改动共享函数。

本机已保留审核快照和隔离身份服务依赖，可运行：

```powershell
.\.venv\Scripts\python.exe tools/postgres_revision_integration_2024141460331.py
```

脚本启动最新A-1后端和本方工作服务，生成临时测试账号、取得真实登录令牌并验证修订。样例文本是专用测试夹具，不是ASR质量样本；结束后关闭临时进程并清理本次测试业务行。该脚本依赖本机`.doc-sync-work/a1-audit-20261107/`和`.doc-sync-work/iam-deps-20261107/`，不是新克隆仓库零准备即可运行的部署脚本。日常启动修订仍需把`a5Url`指向实际运行的可信IAM服务；本次验收服务不会常驻。
