"""Chinese feedback and bounded JSON diagnostics for personal service operations."""

import json
import logging
import re
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from time import perf_counter
from uuid import uuid4

_TRACE: ContextVar[str | None] = ContextVar("a4_trace_id", default=None)
_MESSAGES = {
    "CONFIG_ERROR": "配置文件无法读取或配置项不合法，请检查本地配置。",
    "AUTH_ENV_MISSING": "未配置服务凭据，请设置配置文件指定的环境变量。",
    "POSTGRES_UNAVAILABLE": "PostgreSQL 暂不可用，请检查本地数据库容器和连接配置。",
    "POSTGRES_DRIVER_UNAVAILABLE": "未安装 PostgreSQL 驱动，请安装 postgres 可选依赖。",
    "REVISION_AUTH_UNAVAILABLE": "可信用户身份服务未配置、不可用或响应无效，修订尚未保存。",
    "REVISION_AUTH_DENIED": "登录身份无效或没有文本修订权限，修订未保存。",
    "REVISION_HISTORY_INVALID": "修订历史校验失败，本次事务已回滚。",
    "MODEL_OR_INPUT_ERROR": "模型或输入音频不可用，请检查文件与运行依赖。",
    "MODEL_UNAVAILABLE": "识别模型无法加载，请检查模型文件和运行依赖。",
    "MODEL_MISMATCH": "模型文件或参数与本次执行快照不一致，已停止处理。",
    "ENGINE_UNAVAILABLE": "团队服务未登记或未启用所需引擎，请核对引擎登记。",
    "ENGINE_CONFIG_MISMATCH": "团队引擎登记与本地模型配置不一致。",
    "INVALID_RECORDING": "录音信息不完整或存储状态不可用。",
    "AUDIO_SIZE_MISMATCH": "下载音频大小与录音登记不一致。",
    "AUDIO_CHECKSUM_MISMATCH": "下载音频校验值与录音登记不一致。",
    "AUDIO_DURATION_MISMATCH": "实际音频时长与录音登记不一致。",
    "NO_SPEECH": "未检测到符合当前分段条件的语音，本次未创建识别任务。",
    "ASR_FAILED": "片段识别失败，有效结果与失败、未处理片段已分别列出，请核对后处理。",
    "JOB_SEGMENT_LIMIT": (
        "语音片段数量超过本次配置上限，请核对数量与资源配置；分批处理须保留来源关联。"
    ),
    "SEGMENT_BUSY": "片段已有排队或处理中的任务，请先核对现有任务。",
    "INVALID_TIME": "时间格式不合法，请使用带时区的ISO8601时间。",
    "INVALID_SEGMENT": "语音片段时间无效、越界或重叠，请核对分段结果。",
    "SEGMENT_ASSOCIATION": "片段与录音的关联不一致。",
    "SEGMENT_CONFLICT": "已有片段时间区间冲突，已停止写入以保护现有片段。",
    "SEGMENT_LIMIT": "待登记片段数量超过接口单次上限。",
    "SEGMENT_UNCONFIRMED": "登记片段未能从服务端完整读回，请核对已有写入。",
    "TIME_MISMATCH": "片段绝对时间与录音起始时间及偏移不一致。",
    "RESULT_EXISTS": "片段已有转写结果，已停止写入以保护现有结果。",
    "BUSINESS_TASK_REQUIRED": "请提供有效的业务任务标识。",
    "BUSINESS_TASK_MISMATCH": "业务任务标识与已登记的任务关联不一致。",
    "INVALID_IDEMPOTENCY_KEY": "请求幂等标识无效，请使用不超过64个可打印ASCII字符。",
    "IDEMPOTENCY_CONFLICT": "该请求标识已用于其他录音，请核对调用参数。",
    "RECEIPT_DIRECTORY_REQUIRED": "未配置本地请求回执目录，无法安全记录写入状态。",
    "REQUEST_RECEIPT_UNREADABLE": "本地请求回执无法读取，请先核对已有任务状态。",
    "REQUEST_OUTCOME_UNKNOWN": "已有请求的执行结果未知，请核对服务端状态后再处理。",
    "REQUEST_ALREADY_SUBMITTED": "该请求已提交，请查询现有任务，避免重复执行。",
    "JOB_CONTEXT_MISMATCH": "任务上下文与录音、目标片段或模型信息不一致。",
    "INVALID_RESULT_FILE": "待保存的结果文件无法读取或内容格式不合法。",
    "INVALID_RESULT": "待保存的结果不完整或格式不合法。",
    "INVALID_OUTCOME": "处理状态、片段计数或有效结果清单不一致。",
    "RESULT_CONTEXT_MISMATCH": "待保存结果与服务端任务记录不一致，已停止保存。",
    "SAVE_UNCONFIRMED": "尚未确认保存结果，请按任务标识读回核对，勿盲目重复提交。",
    "SAVE_FAILED": "服务端拒绝保存且未查到有效文本，请核对错误码和任务状态。",
    "SAVE_RECEIPT_FAILED": "本地保存回执写入失败，请先读回核对服务端状态。",
    "READBACK_INVALID": "读回文本与录音或片段时间不一致。",
    "TRANSCRIPT_ASSOCIATION": "文本与录音、片段或原文摘要的关联不一致。",
    "TRANSCRIPT_TIME_MISMATCH": "文本时间与对应片段时间不一致。",
    "HISTORY_ASSOCIATION": "文本与指定历史任务的关联无法核实。",
    "HISTORY_SNAPSHOT_MISMATCH": "历史任务的创建模型快照与执行模型快照不一致。",
    "HISTORY_UNAVAILABLE": "上游服务未返回修订留痕，请核对接口版本，不能据此认定没有历史。",
    "INVALID_REVISION": "修订参数不合法，请提交待修改文本或复核状态、当前修订号及修改原因。",
    "REVISION_AUTH_REQUIRED": "人工修订需要用户身份凭据，服务密钥不能替代编辑者身份。",
    "REVISION_CONFLICT": "文本已被修改，请重新读取当前版本后再提交，勿覆盖他人修订。",
    "REVISION_UNCHANGED": "文本和复核状态均未改变，本次没有新增修订记录。",
    "REVISION_UNCONFIRMED": "尚未核实修订及留痕，请按文本标识读回核对，勿盲目重复提交。",
    "TRANSPORT_ERROR": "无法连接团队服务或请求超时，请检查服务状态并核对已有写入。",
    "HTTP_ERROR": "团队服务返回非预期状态，请结合状态码和请求标识核查。",
    "INVALID_RESPONSE": "团队服务响应不符合接口约定，请核对接口与服务版本。",
    "INVALID_DATA": "输入数据或处理结果不符合约定，请核对录音与任务信息。",
    "UNAUTHORIZED": "服务密钥缺失或无效。",
    "VALIDATION_ERROR": "请求参数缺失或格式错误，请对照接口说明检查。",
    "INVALID_REQUEST_ID": "请求追踪标识格式错误，请使用不超过128个安全ASCII字符。",
    "SERVICE_BUSY": "识别服务正在处理其他录音，请稍后由调用方安排提交。",
    "INTERNAL_ERROR": "服务内部处理异常，请使用请求标识核查日志。",
    "NOT_FOUND": "请求的接口或资源不存在。",
    "METHOD_NOT_ALLOWED": "该接口不支持当前请求方法。",
}


def current_trace_id() -> str | None:
    return _TRACE.get()


def new_trace_id() -> str:
    return str(uuid4())


def chinese_message(code: str) -> str:
    return _MESSAGES.get(code, "处理未完成，请根据错误码、任务标识及追踪标识核查。")


def feedback(result: dict) -> dict:
    """Add per-call diagnostics without replacing machine-readable state or codes."""
    result = {**result, "traceId": current_trace_id() or new_trace_id()}
    if result.get("code"):
        result["message"] = chinese_message(result["code"])
    elif result.get("status") in {"partial", "failed"} and result.get("failedSegments"):
        result["message"] = chinese_message("ASR_FAILED")
    return result


@contextmanager
def trace_scope(trace_id: str | None = None):
    """Nested run/transcribe/save/read operations share their caller's trace."""
    if trace_id is not None and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", trace_id):
        raise ValueError("INVALID_REQUEST_ID")
    token = _TRACE.set(current_trace_id() or trace_id or new_trace_id())
    try:
        yield current_trace_id()
    finally:
        _TRACE.reset(token)


def log_event(logger: logging.Logger, log_code: str, *, level=logging.INFO, **fields) -> None:
    """Explicit allowlist: never serialize exception bodies, paths, text or headers."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "logCode": log_code,
        "traceId": current_trace_id(),
    }
    for key in ("jobId", "recordingId", "requestId", "code", "status", "operation", "method"):
        value = fields.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            record[key] = value
    for key in (
        "processedSegments",
        "failedSegments",
        "totalSegments",
        "httpStatus",
        "durationMs",
        "sourceSampleRateHz",
        "sourceChannels",
        "outputSampleRateHz",
        "outputChannels",
        "sourceSamples",
        "outputSamples",
    ):
        value = fields.get(key)
        if type(value) in (int, float):
            record[key] = value
    # Conversion values are generated by our decoder, not input metadata.
    if "actions" in fields:
        record["actions"] = [
            action for action in fields["actions"] if action in {"downmix_to_mono", "resample"}
        ]
    logger.log(level, json.dumps(record, ensure_ascii=False, allow_nan=False))


def traced_operation(operation: str):
    def decorate(function):
        @wraps(function)
        def execute(*args, **kwargs):
            logger = logging.getLogger(function.__module__)
            with trace_scope():
                started = perf_counter()
                log_event(logger, "A4.OPERATION.STARTED", operation=operation)
                try:
                    result = feedback(function(*args, **kwargs))
                except Exception as exc:
                    log_event(
                        logger,
                        "A4.OPERATION.FAILED",
                        level=logging.WARNING,
                        operation=operation,
                        code=getattr(exc, "code", "INTERNAL_ERROR"),
                        jobId=getattr(exc, "job_id", None),
                        requestId=getattr(exc, "request_id", None),
                        httpStatus=getattr(exc, "status", getattr(exc, "http_status", None)),
                        durationMs=round((perf_counter() - started) * 1000, 3),
                    )
                    raise
                log_event(
                    logger,
                    "A4.OPERATION.COMPLETED",
                    level=logging.WARNING
                    if result.get("status") in {"partial", "failed"}
                    else logging.INFO,
                    operation=operation,
                    durationMs=round((perf_counter() - started) * 1000, 3),
                    **{
                        key: result.get(key)
                        for key in (
                            "jobId",
                            "recordingId",
                            "status",
                            "code",
                            "processedSegments",
                            "failedSegments",
                        )
                    },
                )
                return result

        return execute

    return decorate


@contextmanager
def stderr_diagnostics():
    """CLI owns its temporary handler; importing this module changes no root logger."""
    logger = logging.getLogger("a4_speech")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
