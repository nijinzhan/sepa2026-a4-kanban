"""Reproducible manual entry points; service credentials stay in environment variables."""

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from .audio_2024141460331 import decode_audio
from .client_2024141460331 import ApiError, TeamApi
from .models_2024141460331 import SherpaEngine
from .service_2024141460331 import ProcessingError_2024141460331, SpeechService_2024141460331


class ConfigError_2024141460331(ValueError):
    pass


def load_config_2024141460331(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError_2024141460331("configuration cannot be read") from exc
    if not isinstance(config, dict):
        raise ConfigError_2024141460331("configuration must be an object")
    config["_directory"] = str(path.resolve().parent)
    return config


def _result_file_2024141460331(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(result, dict):
            raise ValueError("result must be an object")
    except (OSError, ValueError) as exc:
        raise ProcessingError_2024141460331("INVALID_RESULT_FILE") from exc
    return result


def _endpoint_2024141460331(config: dict, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ConfigError_2024141460331(f"{key} is required")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError_2024141460331(
            f"{key} must be an HTTP endpoint without credentials or query"
        )
    return value


def _local_path_2024141460331(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_directory"]) / path


def _engine_2024141460331(config: dict) -> SherpaEngine:
    model = config.get("engine")
    if not isinstance(model, dict):
        raise ConfigError_2024141460331("engine manifest is required for model execution")
    # Copy before resolving paths: path strings are never submitted as model history.
    model = json.loads(json.dumps(model))
    for kind in ("vad", "asr"):
        item = model.get(kind)
        if not isinstance(item, dict):
            raise ConfigError_2024141460331(f"engine.{kind} is required")
        for key in ("modelPath", "tokensPath"):
            if key in item and isinstance(item[key], str):
                item[key] = str(_local_path_2024141460331(config, item[key]))
    return SherpaEngine(model)


def _parser_2024141460331() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ-A-4-10/20 个人实现入口")
    parser.add_argument("--config", type=Path, default=Path("config.local.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    run_2024141460331 = commands.add_parser(
        "transcribe", help="读取团队录音并执行一次分段转写和保存"
    )
    run_2024141460331.add_argument("--recording-id", required=True)
    run_2024141460331.add_argument("--idempotency-key", required=True)
    run_2024141460331.add_argument(
        "--business-task-id", help="关联上层业务任务；一步保存必需，只转写可省略"
    )
    run_2024141460331.add_argument(
        "--no-save", action="store_true", help="只取得转写结果，留待显式保存"
    )
    save = commands.add_parser("save", help="校验并保存已取得的转写结果，无需本地模型")
    save.add_argument("--result-file", type=Path, required=True, help="调用方保存的结果 JSON 对象")
    save.add_argument("--business-task-id", help="为未绑定的结果补充业务任务，不得修改已有任务关联")
    read = commands.add_parser("read", help="按文本标识读取并核对历史来源；可指定任务消除歧义")
    read.add_argument("--transcript-id", required=True)
    read.add_argument("--job-id")
    check = commands.add_parser("check-model", help="检查实际模型清单；提供音频时实际推理")
    check.add_argument("--audio", type=Path)
    return parser


def main_2024141460331(argv: list[str] | None = None) -> int:
    args = _parser_2024141460331().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        config = load_config_2024141460331(args.config)
        if args.command == "check-model":
            engine = _engine_2024141460331(config)
            output = {"executionSnapshot": engine.snapshot(), "inferenceChecked": False}
            if args.audio is not None:
                audio = decode_audio(args.audio.read_bytes())
                spans = engine.detect(audio)
                output.update(
                    inferenceChecked=True,
                    durationMs=audio.duration_ms,
                    segments=[
                        {
                            "startMs": start,
                            "endMs": end,
                            "textRaw": engine.recognize(audio, start, end),
                        }
                        for start, end in spans
                    ],
                )
                output["asrInferenceChecked"] = bool(spans)
        else:
            a3, a4 = (
                _endpoint_2024141460331(config, "a3Url"),
                _endpoint_2024141460331(config, "a4Url"),
            )
            env_name = config.get("apiKeyEnv", "A4_API_KEY")
            if not isinstance(env_name, str):
                raise ConfigError_2024141460331("apiKeyEnv must name an environment variable")
            secret = os.environ.get(env_name)
            if not secret:
                print(json.dumps({"code": "AUTH_ENV_MISSING"}))
                return 2
            timeout = config.get("timeoutSeconds", 60)
            if type(timeout) not in (int, float) or not 0 < timeout <= 600:
                raise ConfigError_2024141460331("timeoutSeconds must be in (0,600]")
            with httpx.Client(
                headers={"X-Api-Key": secret, "X-Request-Id": str(uuid4())},
                timeout=timeout,
                trust_env=False,
            ) as http:
                api = TeamApi(http, a3, a4)
                engine = _engine_2024141460331(config) if args.command == "transcribe" else None
                receipt_dir = config.get("receiptDirectory", "outputs/requests")
                if not isinstance(receipt_dir, str):
                    raise ConfigError_2024141460331("receiptDirectory must be a path")
                service = SpeechService_2024141460331(
                    api, engine, receipt_dir=_local_path_2024141460331(config, receipt_dir)
                )
                if args.command == "transcribe":
                    execute = (
                        service.transcribe_2024141460331
                        if args.no_save
                        else service.run_2024141460331
                    )
                    output = execute(
                        args.recording_id,
                        args.idempotency_key,
                        business_task_id=args.business_task_id,
                    )
                elif args.command == "save":
                    result = _result_file_2024141460331(args.result_file)
                    if args.business_task_id is not None:
                        if result.get("businessTaskId") not in (None, "", args.business_task_id):
                            raise ProcessingError_2024141460331("RESULT_CONTEXT_MISMATCH")
                        result["businessTaskId"] = args.business_task_id
                    output = service.save_result_2024141460331(result)
                else:
                    output = service.read_transcript_2024141460331(args.transcript_id, args.job_id)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except ConfigError_2024141460331:
        error = {"code": "CONFIG_ERROR"}
    except ProcessingError_2024141460331 as exc:
        error = exc.as_dict_2024141460331()
    except ApiError as exc:
        error = {"code": exc.code, "httpStatus": exc.status, "requestId": exc.request_id}
    except (ValueError, OSError):
        # Native decoder/model errors can contain paths; avoid leaking config/credentials.
        error = {"code": "MODEL_OR_INPUT_ERROR"}
    print(json.dumps(error, ensure_ascii=False))
    return 2
