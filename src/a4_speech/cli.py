"""Reproducible manual entry points; service credentials stay in environment variables."""

import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from .audio import decode_audio
from .client import ApiError, TeamApi
from .models import SherpaEngine
from .service import ProcessingError, SpeechService


class ConfigError(ValueError):
    pass


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError("configuration cannot be read") from exc
    if not isinstance(config, dict):
        raise ConfigError("configuration must be an object")
    config["_directory"] = str(path.resolve().parent)
    return config


def _endpoint(config: dict, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ConfigError(f"{key} is required")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ConfigError(f"{key} must be an HTTP endpoint without credentials or query")
    return value


def _local_path(config: dict, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_directory"]) / path


def _engine(config: dict) -> SherpaEngine:
    model = config.get("engine")
    if not isinstance(model, dict):
        raise ConfigError("engine manifest is required for model execution")
    # Copy before resolving paths: path strings are never submitted as model history.
    model = json.loads(json.dumps(model))
    for kind in ("vad", "asr"):
        item = model.get(kind)
        if not isinstance(item, dict):
            raise ConfigError(f"engine.{kind} is required")
        for key in ("modelPath", "tokensPath"):
            if key in item and isinstance(item[key], str):
                item[key] = str(_local_path(config, item[key]))
    return SherpaEngine(model)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ-A-4-10/20 个人实现入口")
    parser.add_argument("--config", type=Path, default=Path("config.local.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("transcribe", help="读取团队录音并执行一次分段转写和保存")
    run.add_argument("--recording-id", required=True)
    run.add_argument("--idempotency-key", required=True)
    read = commands.add_parser("read", help="按文本标识读取；历史模型另需明确任务上下文")
    read.add_argument("--transcript-id", required=True)
    read.add_argument("--job-id")
    check = commands.add_parser("check-model", help="检查实际模型清单；提供音频时实际推理")
    check.add_argument("--audio", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        config = load_config(args.config)
        if args.command == "check-model":
            engine = _engine(config)
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
            a3, a4 = _endpoint(config, "a3Url"), _endpoint(config, "a4Url")
            env_name = config.get("apiKeyEnv", "A4_API_KEY")
            if not isinstance(env_name, str):
                raise ConfigError("apiKeyEnv must name an environment variable")
            secret = os.environ.get(env_name)
            if not secret:
                print(json.dumps({"code": "AUTH_ENV_MISSING"}))
                return 2
            timeout = config.get("timeoutSeconds", 60)
            if type(timeout) not in (int, float) or not 0 < timeout <= 600:
                raise ConfigError("timeoutSeconds must be in (0,600]")
            with httpx.Client(
                headers={"X-Api-Key": secret, "X-Request-Id": str(uuid4())},
                timeout=timeout,
                trust_env=False,
            ) as http:
                api = TeamApi(http, a3, a4)
                engine = _engine(config) if args.command == "transcribe" else None
                receipt_dir = config.get("receiptDirectory", "outputs/requests")
                if not isinstance(receipt_dir, str):
                    raise ConfigError("receiptDirectory must be a path")
                service = SpeechService(api, engine, receipt_dir=_local_path(config, receipt_dir))
                if args.command == "transcribe":
                    output = service.run(args.recording_id, args.idempotency_key)
                else:
                    output = service.read_transcript(args.transcript_id, args.job_id)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except ConfigError:
        error = {"code": "CONFIG_ERROR"}
    except ProcessingError as exc:
        error = exc.as_dict()
    except ApiError as exc:
        error = {"code": exc.code, "httpStatus": exc.status, "requestId": exc.request_id}
    except (ValueError, OSError):
        # Native decoder/model errors can contain paths; avoid leaking config/credentials.
        error = {"code": "MODEL_OR_INPUT_ERROR"}
    print(json.dumps(error, ensure_ascii=False))
    return 2
