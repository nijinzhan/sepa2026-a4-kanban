# 教师提供的 ATC 微调 SenseVoice 接入说明

日期：2026-10-29。适用范围：李洪负责的 RQ-A-4-10、RQ-A-4-20 本地识别服务。

> 2026-10-30效果评测更新：同一固定49条ATCO2上，教师版严格WER为36.77%，通用版本轮复跑为82.29%；完整范围、诊断指标和限制见[效果评测报告](../deliverables/测试/2026-10-30-教师SenseVoice模型效果评测.md)。这是非VHHH复用样本对照，尚非目标机场验收。下文第5节保留2026-10-29接入时的验证记录。

## 1. 接入方式

教师提供的 `vhhh_sensevoice.zip` 已含导出的 SenseVoice ONNX 模型和配套词表，可直接由现有 `SherpaEngine` 加载，无需重新训练、转换格式或修改转写／保存／读取接口。继续使用 WebRTC VAD；包内的 Silero VAD 暂不切换，避免同时改变分段和识别两个因素。

按既定处理流程接入，不再自行微调。2026-10-29已将 [config.example.json](../config.example.json) 与本机 `config.local.json` 切换至教师模型；正常启动默认进入此配置。[config.teacher-sensevoice.example.json](../config.teacher-sensevoice.example.json)保留同一教师配置的显式入口，[config.generic-sensevoice.example.json](../config.generic-sensevoice.example.json)保留原通用版用于对照。

主流程为：读取音频 → 解码与16 kHz单声道转换 → WebRTC分段 → 逐段教师SenseVoice识别 → 保留原录音时间、机器原文及执行快照 → 独立确认保存 → 按标识读回。预处理操作与源音频信息另记于`audioProcessing`；后续人工修订沿用版本校验和编辑留痕，保留不可变机器原文。模型接入复用既有接口及异常处理，不新增训练流程。

| 包内文件 | 大小（字节） | 用途 |
| --- | ---: | --- |
| `2025_0706/model.int8.onnx` | 239233833 | 当前选用的 CPU int8 模型 |
| `2025_0706/model.onnx` | 937617178 | 非 int8 导出模型，留在原包中，暂未加载验证 |
| `2025_0706/tokens.txt` | 315894 | 与包内模型配套的词表 |
| `silero_vad.onnx` | 1807522 | 可选分段模型，当前未启用 |

本机部署位置为 `models/vhhh-sensevoice-atc-2025_0706/`。权重及 ZIP 不进入 Git；`/models/` 已忽略。只维护配置、接入说明及必要验证记录。

## 2. 模型身份

| 项目 | 值 |
| --- | --- |
| 来源 | 用户于2026-10-29确认由教师提供的 `D:/vhhh_sensevoice.zip`，作为课程 ATC 微调模型 |
| `engineCode` | `sensevoice`，沿用现有程序支持的引擎类型 |
| `modelName` | `SenseVoiceSmall-ATC-VHHH`，本项目用于区分权重的登记名称 |
| `version` | `teacher-2025_0706-int8`，根据包内目录编制的本地版本标签，不代表独立核实的训练日期 |
| ZIP SHA-256 | `302778e6ccd4a2f6adfc0d9b87bc7a00e173d27e02d412c694f410a2723e35e3` |
| int8 模型 SHA-256 | `625facec8fb54410d0e34543947cd80c2e0b1e568df00395bd9a31d3d00f23e8` |
| 词表 SHA-256 | `f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc` |
| 解码 | 16 kHz 单声道、`language=en`、`useItn=true`、CPU、2线程、greedy search |

新权重与原官方通用 int8 模型的 SHA-256 不同；词表字节相同。仍使用教师包内的词表，不依赖跨包替换。当前 ZIP 没有训练配置、训练数据清单或评测报告；模型来源按教师提供记录，不将这次接入写成本人完成微调或质量达标。

现有引擎会将模型与词表 SHA-256、名称、版本和运行参数写入 `executionSnapshot`；新旧结果据各自快照追溯，不回写历史模型身份。

## 3. 本机复现

在 `李洪/` 下执行。当前机器已完成提取和私有配置生成。其他机器可按以下方式仅提取实际使用的两个文件，不解压较大的备用模型：

```powershell
@'
from pathlib import Path
import shutil, zipfile
target = Path('models/vhhh-sensevoice-atc-2025_0706')
target.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile('D:/vhhh_sensevoice.zip') as archive:
    for name in ('model.int8.onnx', 'tokens.txt'):
        output = target / name
        if output.exists():
            continue
        with archive.open('2025_0706/' + name) as source, output.open('wb') as dest:
            shutil.copyfileobj(source, dest)
'@ | .venv/Scripts/python.exe -
Get-FileHash models/vhhh-sensevoice-atc-2025_0706/model.int8.onnx -Algorithm SHA256
Get-FileHash models/vhhh-sensevoice-atc-2025_0706/tokens.txt -Algorithm SHA256
```

将哈希与第2节核对一致后，设置默认本机配置。有旧配置时先留存备份，保留原服务地址、凭据环境变量名称和VAD设置，只替换ASR；新环境则使用默认示例：

```powershell
@'
from pathlib import Path
import json
path = Path('config.local.json')
source = path if path.exists() else Path('config.example.json')
local = json.loads(source.read_text(encoding='utf-8-sig'))
if path.exists():
    backup = Path('outputs/config-backups/config-before-teacher-20261029.json')
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
teacher = json.loads(Path('config.example.json').read_text(encoding='utf-8-sig'))
local['engine']['asr'] = teacher['engine']['asr']
path.write_text(
    json.dumps(local, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
'@ | .venv/Scripts/python.exe -
```

`config.local.json`与`config.*.local.json`已由个人`.gitignore`忽略。配置位于个人目录，模型相对路径以配置所在目录解析。本机另存`config.generic-sensevoice.local.json`，保留切换前的服务设置和通用权重路径。

先执行无需团队服务的本地检查，`--audio` 选择实际 ATC 样例：

```powershell
.venv/Scripts/python.exe -m a4_speech check-model
.venv/Scripts/python.exe -m a4_speech check-model --audio outputs/atco2-1h-20261021/original/ATCO2-ASRdataset-v1_beta/DATA/LKPR_RUZYNE_Radar_120_520MHz_20201029_075414.wav
```

不带`--audio`只核对配置、文件身份，不证明实际推理成功；带音频且返回`asrInferenceChecked=true`才表示执行了识别。音频示例来自本机已有ATCO2数据，其他机器应替换为实际可读音频。实际服务需先配置原有入站与上游凭据，再正常启动：

```powershell
.venv/Scripts/python.exe -m a4_speech.api_2024141460331 --config config.local.json --port 8001
```

需要比较通用版时，用`--config config.generic-sensevoice.local.json`显式启动。本机已保留该私有配置；其他机器可据通用版示例另建配置并填写真实服务设置。比较时也须使用与通用模型匹配的上游引擎登记，不能复用教师版身份。

## 4. 团队服务联调前的引擎登记

只改模型路径还不够。现有服务先通过公共接口核对引擎登记，若上游 `sensevoice` 仍登记为 `SenseVoiceSmall / 2024-07-17`，会返回 `ENGINE_CONFIG_MISMATCH`。本次已用模拟服务验证该保护生效，不绕过它。

真实联调时需由引擎登记的实际提供方确认：启用的 ASR 登记 `code=sensevoice`、`modelName=SenseVoiceSmall-ATC-VHHH`、`version=teacher-2025_0706-int8` 与本机配置一致，同时保留 `webrtc-vad` 登记。上述名称是本项目标签，若团队采用其他正式标签，应两端一致调整。当前 CLI 只接受既有引擎类型，不能只把配置中的 `engineCode` 改成新名字就期待自动支持。

若团队要让通用版和教师版同时在线运行，需要协商独立登记代码及对应的引擎类型映射，另行修改和验证；本次不修改公共数据库或公共接口，也未向团队服务写入登记。

## 5. 本轮验证与边界

- 教师 int8 模型通过现有 sherpa-onnx 1.13.8 在 CPU 上加载，并使用 WebRTC 分段执行真实识别。
- 默认配置下，两条既有ATCO2音频经真实模型和模拟HTTP边界完成独立转写、显式保存、按标识读回；共3段。转写阶段未写入文本，保存后原文读回一致，任务中保留教师模型的完整执行快照。
- 上游保留旧版模型登记时，流程拒绝执行并返回 `ENGINE_CONFIG_MISMATCH`。
- 切换默认配置后，未指定`--config`的CLI在真实ATCO2音频上返回`asrInferenceChecked=true`，快照中的名称、版本及文件哈希均为教师模型。
- 显式以默认`config.local.json`启用两项真实CPU模型用例，全量回归为 **401 passed, 1 skipped（14.62秒）**；唯一跳过项为未启用的可选XLS-R GPU实验。包含真实教师模型的音频识别及静音检查，其余服务异常与保存读取用例继续使用受控测试边界。
- 对既有 VHHH C15 音频的两个片段做了预览；没有可核验人工参照，输出仍为未复核机器稿，不能计算或宣称 VHHH 准确率。
- 通用英语演示音频虽可正常运行，但输出与原通用模型明显不同；领域模型应以实际 ATC 音频和人工参照验收，不能把“非空输出”当作质量合格。

本机详细证据保存在被忽略的`outputs/teacher-sensevoice-20261029/`，包含`smoke-result.json`、`service-probe.json`、`vhhh-preview.json`、`default-cli-model-check.json`与`default-pytest.txt`。模型来源和文件摘要另存于模型目录的`teacher-package-manifest.json`。本轮未运行49条全量质量评测，未比较备用ONNX模型，也未完成真实团队HTTP联调。旧通用模型的WER不代表本次教师模型。

官方接口资料：[sherpa-onnx SenseVoice Python API](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html)、[SenseVoice ONNX 导出说明](https://k2-fsa.github.io/sherpa/onnx/sense-voice/export.html)。
