# A-4-10 API 服务与 A-5 数据映射核对

核对日期：2026-10-24。范围：本人 RQ-A-4-10 的 HTTP 服务及 RQ-A-4-10/20 已有数据适配，不改变公共数据库或其他成员的模块。

## 1 核对结论与依据

当前实现的录音、片段、识别任务、引擎和文本组织方式，与使用方提供的《A-5-数据表结构.docx》核心数据落点相符。不能据此宣布真实数据库、外键、触发器、事务及服务端并发保护已验证。新增 API 遵循这份结构和主仓库通用 HTTP 约定，复用现有处理逻辑，不自行建表或直接连接 PostgreSQL。

依据为《A-5-数据表结构.docx》§0.2、aud/asr 数据字典、ops.api_endpoint 及附录 A；通用 HTTP 规则来自主仓库 `尹凯臣/openapi/README.md` §1。后者明确“契约定义不代表 HTTP 服务已经实现”。历史 P-Blue YAML 用于解释既有适配器，不当作主仓库已发布契约的证明。

## 2 字段和约束对应

| 公共数据 | 当前处理与 HTTP 字段 | 当前结论与待核验项 |
| --- | --- | --- |
| `aud.recording.id`，UUID | API 请求 `recordingId` 为 UUID；从 A-3 HTTP 读取同一录音 | API 层验证类型；真实录音登记和外键待实际服务 |
| `aud.recording.started_at/duration_ms` | `startedAt/durationMs`；校验带时区起点及解码时长 | 已实现时间与实际音频核对；不使用文件名代替采集起点 |
| `aud.recording.storage_state` | `stored`、`verified` 可执行处理 | V6 CHECK与A-3契约均列出7种状态；数据字典COMMENT只列4种并非完整枚举。更正此前把verified当作非法状态的判断，其余状态不直接处理 |
| `aud.recording.sample_rate_hz/audio_channels` | 解码得到原始属性，记录降混、重采样动作 | 转换记录放入已有任务 `stats` JSON 扩展，不虚构新的录音列 |
| `aud.segment.id/recording_id` | 服务端返回 `segmentId/recordingId`，API 响应为 UUID | 登记后读回核对；不自建一套片段标识 |
| `aud.segment.start_ms/end_ms/start_at/end_at` | `startMs/endMs/startAt/endAt`；整毫秒偏移，响应时间统一 UTC | 校验 `startAt = startedAt + startMs`，以及结束时刻、越界和重叠；数据库触发器/排除约束仍待真实服务 |
| `aud.segment.sequence_no/duration_ms` | 新段提交偏移和语音属性，读回服务端标识及时间 | 由公共服务负责分配序号和计算列；调用端不覆盖已有片段 |
| `asr.engine.code` | `engineCode` 为文本编码，不要求 UUID；当前配置 webrtc-vad / sensevoice | 检查启用状态及模型名/版本；实际引擎登记、部署信息、权限待对接 |
| `asr.job.id/engine_code/recording_id/segment_id` | 调用现有任务接口，使用服务端 `jobId` 与固定目标片段 | 公共完成接口须维护真实外键；创建接口按片段推导录音的行为待联调，不以 params 中有录音 ID 代替外键验证 |
| `asr.job.params/stats` | `a4_10_20` JSON 扩展存放模型快照、目标片段、结果摘要、转换和失败说明 | 使用已有 jsonb 字段；扩展键为个人适配约定，须与消费方对齐 |
| `asr.transcript.segment_id/job_id/engine_code` | A-4-20 经公共完成接口保存，路径携带 jobId，文本项携带 segmentId | 真实服务须写入对应任务与引擎；现有返回未直接证明 job_id，因此保留 `sourceJobLinkVerified=false` |
| `asr.transcript.text_raw/text_norm` | API 返回 `textRaw`；现有保存适配器按历史完成契约发送 `text` | 必须联调确认 `text` 写入 text_raw，且 textNorm、人工修改不覆盖原始结果 |
| `asr.transcript.revision/is_edited/transcript_edit` | 实现修订保存与完整历史读回，要求expectedRevision/reason；编辑者由上游鉴权决定 | 本地验证原文保护与版本冲突；原子CAS及触发器以新增迁移交付，真实数据服务应用与验收待完成 |
| 上层 `businessTaskId` | API 接受可选 UUID，扩展到 job.params/stats | 不是表中已存在的同名正式列，也不是 asr.job.id 的别名；业务任务存在性/外键仍未验证 |
| `ops.api_endpoint` | 本人目录生成 OpenAPI JSON，标明 proposed | 未写入公共注册表；待 A-5 合并确认后按其流程登记，不将个人运行成功当作公共登记完成 |

## 3 服务接口

入口：`POST /api/v1/a4/speech/transcribe`。这是本次实现的个人工作服务路径，**待 A-5、刘睿确认及登记**，不声称它已存在于主仓库公共 A-4 YAML 中。

公共数据接口 `/api/v1/a4/jobs` 等仍由 `TeamApi` 调用。新入口执行实际 VAD/ASR，公共数据接口承担约定的数据读写。两者不能配置成同一个尚未实现的服务地址，也不能将新的工作服务冒充公共任务存储接口。

请求头：

- `X-Api-Key`：入站服务密钥，读取 `A4_SERVICE_API_KEY` 环境变量。
- `Idempotency-Key`：必填，1—64 个可打印 ASCII 字符，不能全为空白。
- `X-Request-Id`：可选，1—128 个字母、数字或 `_.:-` 字符；缺省生成新标识，有效值贯穿响应、处理日志和上游调用。

```json
{
  "recordingId": "00000000-0000-4000-8000-000000000001",
  "businessTaskId": "00000000-0000-4000-8000-000000000002"
}
```

上述 UUID 仅说明格式，不代表团队真实数据；联调必须替换为实际登记的标识。businessTaskId 可省略；保存阶段仍按 A-4-20 的现有规则要求关联。请求不接受任意文件路径、下载地址、模型参数或保存开关，录音与引擎均来自既定配置/公共登记。

本入口同步返回处理结果，HTTP 200 仅说明已获得结构化处理结果。调用方必须检查 `status`：succeeded、partial、failed、no_speech；运行成功不代表识别质量达标。API 的 no_speech 是处理结果，当前不创建空 asr.job，也不向公共任务枚举增加新值。

本入口**不保存转写文本**，正常首次返回 `saveConfirmed=false`、`saveStatus=not_requested`；分段登记、建任务、处理元数据回写仍会发生。有效结果的独立保存沿用已有 A-4-20 `save` 入口，再由公共数据服务落库。若重复请求所对应的文本尚未保存，现有核心会返回 409/SAVE_UNCONFIRMED 等状态，要求按既有任务核查；不会重新推理或自动重发。调用方应保留首次返回的结果。

已保存的历史任务可按原幂等键读回。旧任务未记录 `audioProcessing` 或 `processingTraceId` 时，API 允许缺省这两个诊断字段，不伪造转换动作或原处理标识；当前请求的 `traceId` 仍保留。静态 OpenAPI 与运行中响应模型由测试核对，避免契约与服务分歧。

单进程同一时刻仅处理一条录音，防止并发共享可变模型状态；忙时返回 429/SERVICE_BUSY、Retry-After: 1，本次未执行。不实现自动发现、队列、调度或重试；这些由对应编排功能处理。

错误使用 `application/problem+json`，包含 type/title/status/detail/instance/code/requestId，同时保留中文 message 和 traceId。格式错误为400，未认证401，冲突409，录音时间等语义错误422，上游异常502，模型/引擎登记不可用503，未知内部异常500。原始异常、密钥和输入内容不回显；上游请求标识使用 upstreamRequestId，避免覆盖本次 requestId。

`GET /health` 仅检查进程可响应，公开返回 `{"status":"ok"}`，不表示真实 A-3/A-5、模型质量或持久化已验收。`/docs`、`/openapi.json` 提供接口说明；静态契约草案位于 `openapi/a4-speech.openapi.json`。草案按 A-5 的模块约定使用相对路径 `/speech/transcribe` 和 servers 中的 `/api/v1/a4` 前缀；运行中的 FastAPI 文档展示完整路径，二者对应相同请求 URL。

## 4 安装与启动

在本人目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

将配置中的 a3Url/a4Url 替换为真实公共数据服务地址。API 默认绑定本机8001；原配置里的8000是上游占位地址，不是本 API 已提供公共数据服务的证明。保持当前 WebRTC + sherpa-onnx SenseVoice CPU 模型配置。

在本机设置环境变量 `A4_SERVICE_API_KEY`（调用本工作服务）和 `A4_API_KEY`（本服务访问上游）；不要写入仓库，入站密钥不转发为上游凭据。然后执行：

```powershell
.\.venv\Scripts\python.exe -m a4_speech.api_2024141460331 --config config.local.json --host 127.0.0.1 --port 8001
```

也可使用安装后的 `a4-speech-api` 命令。一个工作进程持有一个模型实例；不要额外启动多个 worker 来绕过本地互斥/回执限制。真实集群互斥、权限点和统一 IAM 授权尚未完成，本次服务密钥校验不能替代这些验证。

团队其他主机调用时，将 `--host` 设置为部署机约定的监听地址（例如 `0.0.0.0`），调用者使用部署机实际 IP 或域名。`127.0.0.1` 仅供本机调用。本次验证运行于本机 Windows CPU 环境，尚未验证 Debian 上的部署、资源占用或目标内存预算。

调用示例（调用者在本机设置 A4_RECORDING_ID，必要时设置 A4_BUSINESS_TASK_ID）：

```powershell
$body = @{ recordingId = $env:A4_RECORDING_ID }
if ($env:A4_BUSINESS_TASK_ID) { $body.businessTaskId = $env:A4_BUSINESS_TASK_ID }
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/api/v1/a4/speech/transcribe' `
  -Headers @{ 'X-Api-Key' = $env:A4_SERVICE_API_KEY; 'Idempotency-Key' = [guid]::NewGuid().ToString() } `
  -ContentType 'application/json' -Body ($body | ConvertTo-Json)
```

## 5 验证与剩余联调

HTTP 测试使用真实 FastAPI/ASGI 入口和既有 SpeechService/TeamApi，外部 HTTP 由 MockTransport 替代。一般测试的模型使用可控替身，另有显式 real_model 用例使用本机实际 CPU 模型和官方英文样例。该样例只验证真实推理与接口贯通，不是 ATC/VHHH 识别准确率验收。

结果及运行证据见 `deliverables/测试/2026-10-24-A4-10-API与A5数据映射验证.md`。仍需：取得 A-3/A-5 真实服务及凭据、统一新增路径/状态语义、核验公共 job/segment/transcript 外键、JSON 持久化、重启读回、并发覆盖和人工修改保护；与刘睿对接触发，与于天名核对同一结果的查询/导出。未完成这些检查前，两项总需求不能标记 Done。
