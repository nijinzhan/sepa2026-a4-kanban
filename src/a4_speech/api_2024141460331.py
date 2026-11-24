"""RQ-A-4-10/20 worker API with explicit HTTP or A-5 PostgreSQL storage.

The /speech/transcribe path is a personal integration proposal, not an existing
A-5 registry entry. Recognition and explicit text saving are separate operations.
"""

import argparse
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from starlette.exceptions import HTTPException

from .cli_2024141460331 import (
    ConfigError_2024141460331,
    _api_context_2024141460331,
    _engine_2024141460331,
    _local_path_2024141460331,
    load_config_2024141460331,
)
from .client_2024141460331 import ApiError
from .diagnostics_2024141460331 import (
    chinese_message,
    current_trace_id,
    feedback,
    log_event,
    stderr_diagnostics,
    trace_scope,
)
from .service_2024141460331 import ProcessingError_2024141460331, SpeechService_2024141460331

LOG = logging.getLogger(__name__)
PATH = "/api/v1/a4/speech/transcribe"
LOCAL_UNAVAILABLE_CODES = {
    "POSTGRES_UNAVAILABLE",
    "POSTGRES_DRIVER_UNAVAILABLE",
    "REVISION_AUTH_UNAVAILABLE",
    "REVISION_HISTORY_INVALID",
}


class TranscribeRequest_2024141460331(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recordingId: UUID = Field(description="A-3 已登记录音的 aud.recording.id")
    businessTaskId: UUID | None = Field(
        default=None, description="可选上层业务任务标识；通过 job.params/stats 扩展关联，非新增外键"
    )


class TranscriptResult_2024141460331(BaseModel):
    model_config = ConfigDict(extra="allow")

    recordingId: UUID
    segmentId: UUID
    startMs: int = Field(ge=0)
    endMs: int = Field(gt=0)
    startAt: AwareDatetime
    endAt: AwareDatetime
    textRaw: str
    language: str

    @field_validator("startAt", "endAt")
    @classmethod
    def utc_timestamp_2024141460331(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)


class TranscriptionResult_2024141460331(BaseModel):
    model_config = ConfigDict(extra="allow")

    schemaVersion: Literal[1]
    recordingId: UUID
    businessTaskId: UUID | None = None
    jobId: UUID | None
    status: Literal["succeeded", "partial", "failed", "no_speech"]
    processedSegments: int = Field(ge=0)
    failedSegments: int = Field(ge=0)
    transcripts: list[TranscriptResult_2024141460331]
    failedSegmentsDetails: list[dict]
    unprocessedSegmentIds: list[UUID]
    targetSegmentIds: list[UUID] = Field(default_factory=list)
    saveConfirmed: bool
    saveStatus: str
    executionSnapshot: dict | None = None
    audioProcessing: dict | None = None
    processingTraceId: str | None = None
    traceId: str


class SaveRequest_2024141460331(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: dict = Field(
        description=(
            "分段转写返回的完整结果；businessTaskId 必须为 UUID 字符串，服务核验来源后独立保存"
        ),
        json_schema_extra={
            "properties": {"businessTaskId": {"type": "string", "format": "uuid"}},
            "required": ["businessTaskId"],
        },
    )

    @field_validator("result")
    @classmethod
    def business_task_uuid_2024141460331(cls, value: dict) -> dict:
        task = value.get("businessTaskId")
        if not isinstance(task, str):
            raise ValueError("businessTaskId must be a UUID string")
        # Use the response model's parser while preserving the existing business task key.
        TypeAdapter(UUID).validate_python(task)
        return value


class RevisionRequest_2024141460331(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = Field(default=None, max_length=4000)
    reviewStatus: (
        Literal["unreviewed", "accepted", "rejected", "needs_review", "skipped"] | None
    ) = None
    expectedRevision: int = Field(ge=1, strict=True)
    reason: str = Field(min_length=1, max_length=200)

    @field_validator("reason")
    @classmethod
    def nonblank_2024141460331(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("blank revision input")
        return value

    @model_validator(mode="after")
    def has_change_2024141460331(self):
        if self.text is None and self.reviewStatus is None:
            raise ValueError("text or reviewStatus is required")
        return self


class SavedResult_2024141460331(BaseModel):
    model_config = ConfigDict(extra="allow")

    recordingId: UUID
    jobId: UUID
    businessTaskId: UUID
    saveConfirmed: Literal[True]
    saveStatus: Literal["confirmed"]
    transcripts: list[dict]


class TranscriptDetail_2024141460331(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: UUID
    segmentId: UUID
    recordingId: UUID
    jobId: UUID | None = None
    resolvedJobId: UUID | None = Field(
        default=None, description="由上下文与摘要核验的任务；不等于数据库外键已验证"
    )
    text: str
    textRaw: str
    revision: int = Field(ge=1)
    reviewStatus: Literal["unreviewed", "accepted", "rejected", "needs_review", "skipped"]
    isEdited: bool
    startMs: int = Field(ge=0)
    endMs: int = Field(gt=0)
    startAt: AwareDatetime
    endAt: AwareDatetime
    edits: list[dict]
    reviewHistoryStatus: Literal["recorded", "partial", "not_exposed", "not_applicable"] | None = (
        Field(
            default=None,
            description="审核前后状态历史的返回情况；not_exposed不表示没有真实编辑历史",
        )
    )


def _transcript_detail_2024141460331(result: dict) -> dict:
    """Keep the existing P-Blue transcript detail shape at the HTTP boundary."""
    if "transcript" not in result:
        return result
    metadata = {key: value for key, value in result.items() if key != "transcript"}
    context_job = metadata.pop("jobId", None)
    if context_job is not None and context_job != result["transcript"].get("jobId"):
        metadata["resolvedJobId"] = context_job
    detail = {**metadata, **result["transcript"]}
    if "reviewHistoryStatus" in metadata:
        detail["reviewHistoryStatus"] = metadata["reviewHistoryStatus"]
    return detail


def _problem_2024141460331(
    request: Request,
    status: int,
    code: str,
    *,
    extra: dict | None = None,
    headers: dict | None = None,
) -> JSONResponse:
    message = chinese_message(code)
    return JSONResponse(
        status_code=status,
        media_type="application/problem+json",
        headers=headers,
        content={
            **(extra or {}),
            "type": "about:blank",
            "title": message,
            "status": status,
            "detail": message,
            "instance": request.url.path,
            "code": code,
            "message": message,
            "requestId": current_trace_id(),
            "traceId": current_trace_id(),
        },
    )


def _error_status_2024141460331(error: ProcessingError_2024141460331) -> int:
    if error.code in LOCAL_UNAVAILABLE_CODES:
        return 503
    if error.code in {"TRANSPORT_ERROR", "INVALID_RESPONSE", "HISTORY_UNAVAILABLE"}:
        return 502
    if error.http_status is not None:
        return error.http_status if error.http_status in {400, 401, 403, 404, 409, 422} else 502
    if error.code in {
        "MODEL_UNAVAILABLE",
        "MODEL_MISMATCH",
        "ENGINE_UNAVAILABLE",
        "ENGINE_CONFIG_MISMATCH",
        "RECEIPT_DIRECTORY_REQUIRED",
        "REQUEST_RECEIPT_UNREADABLE",
    }:
        return 503
    if error.code in {
        "IDEMPOTENCY_CONFLICT",
        "REQUEST_ALREADY_SUBMITTED",
        "REQUEST_OUTCOME_UNKNOWN",
        "RESULT_EXISTS",
        "SEGMENT_BUSY",
        "SEGMENT_CONFLICT",
        "SAVE_UNCONFIRMED",
        "BUSINESS_TASK_MISMATCH",
        "JOB_CONTEXT_MISMATCH",
        "REVISION_CONFLICT",
        "REVISION_UNCONFIRMED",
        "REVISION_UNCHANGED",
    }:
        return 409
    if error.code in {"INVALID_IDEMPOTENCY_KEY", "BUSINESS_TASK_REQUIRED"}:
        return 400
    return 422


def create_app_2024141460331(
    *,
    service: SpeechService_2024141460331 | None = None,
    api_key: str | None = None,
    config_path: Path | str | None = None,
) -> FastAPI:
    """Create one worker; injection also supports hosting inside another service."""
    credential = api_key if api_key is not None else os.environ.get("A4_SERVICE_API_KEY")
    if not isinstance(credential, str) or not credential.strip():
        raise ConfigError_2024141460331("inbound service credential is required")
    gate = Lock()

    @asynccontextmanager
    async def lifespan_2024141460331(app: FastAPI):
        if service is not None:
            app.state.speech_service = service
            yield
            return
        path = Path(config_path or os.environ.get("A4_CONFIG", "config.local.json"))
        config = load_config_2024141460331(path)
        receipt = config.get("receiptDirectory", "outputs/requests")
        if not isinstance(receipt, str) or not receipt.strip():
            raise ConfigError_2024141460331("receiptDirectory must be a nonempty path")
        engine = _engine_2024141460331(config)
        with _api_context_2024141460331(config) as upstream:
            app.state.speech_service = SpeechService_2024141460331(
                upstream,
                engine,
                receipt_dir=_local_path_2024141460331(config, receipt),
                max_segments=config["maxSegments"],
            )
            yield

    app = FastAPI(
        title="A-4-10/20 分段转写与文本服务",
        version="0.2.0",
        lifespan=lifespan_2024141460331,
        description=(
            "RQ-A-4-10/20 独立 CPU 工作服务。同步转写、显式保存、按标识读取与修订。"
            "沿用 A-5 表结构及 P-Blue 路径；新增复合接口待 A-5/调用方确认登记。"
        ),
    )
    app.state.speech_service = service

    @app.middleware("http")
    async def request_context_2024141460331(request: Request, call_next):
        supplied = request.headers.get("X-Request-Id")
        valid = supplied is None or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", supplied)
        with trace_scope(supplied if valid else None):
            log_event(LOG, "A4.API.RECEIVED", method=request.method)
            if not valid:
                response = _problem_2024141460331(request, 400, "INVALID_REQUEST_ID")
            else:
                try:
                    response = await call_next(request)
                except Exception:
                    log_event(LOG, "A4.API.FAILED", level=logging.ERROR, code="INTERNAL_ERROR")
                    response = _problem_2024141460331(request, 500, "INTERNAL_ERROR")
            response.headers["X-Request-Id"] = current_trace_id()
            log_event(LOG, "A4.API.COMPLETED", httpStatus=response.status_code)
            return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_2024141460331(request: Request, error: RequestValidationError):
        # Never echo raw input: it can include credential headers or private paths.
        return _problem_2024141460331(
            request,
            400,
            "VALIDATION_ERROR",
            extra={
                "errors": [{"location": list(e["loc"]), "type": e["type"]} for e in error.errors()]
            },
        )

    @app.exception_handler(ProcessingError_2024141460331)
    async def processing_error_2024141460331(
        request: Request,
        error: ProcessingError_2024141460331,
    ):
        return _problem_2024141460331(
            request,
            _error_status_2024141460331(error),
            error.code,
            extra={**error.as_dict_2024141460331(), "upstreamRequestId": error.request_id},
        )

    @app.exception_handler(ApiError)
    async def upstream_error_2024141460331(request: Request, error: ApiError):
        return _problem_2024141460331(
            request,
            503
            if error.code in LOCAL_UNAVAILABLE_CODES
            else error.status
            if error.status in {400, 401, 403, 404, 409, 422}
            else 502,
            error.code,
            extra={"httpStatus": error.status, "upstreamRequestId": error.request_id},
        )

    @app.exception_handler(HTTPException)
    async def http_error_2024141460331(request: Request, error: HTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(error.status_code, "HTTP_ERROR")
        return _problem_2024141460331(request, error.status_code, code, headers=error.headers)

    security = APIKeyHeader(name="X-Api-Key", scheme_name="ServiceApiKey", auto_error=False)
    user_security = HTTPBearer(scheme_name="UserBearer", auto_error=False)

    def authorize_2024141460331(key: Annotated[str | None, Depends(security)]):
        if key is None or not hmac.compare_digest(key.encode(), credential.encode()):
            raise _Unauthorized_2024141460331()

    @app.exception_handler(_Unauthorized_2024141460331)
    async def unauthorized_2024141460331(request: Request, error):
        return _problem_2024141460331(request, 401, "UNAUTHORIZED")

    @app.get("/health", operation_id="speechHealth_2024141460331", tags=["health"])
    def health_2024141460331():
        return {"status": "ok"}

    problem_schema = {
        "type": "object",
        "required": ["type", "title", "status", "code", "requestId"],
        "properties": {
            **{
                k: {"type": "string"}
                for k in (
                    "type",
                    "title",
                    "detail",
                    "instance",
                    "code",
                    "requestId",
                    "traceId",
                    "message",
                )
            },
            "status": {"type": "integer"},
            "saveConfirmed": {"type": "boolean"},
            "saveStatus": {"type": "string", "enum": ["failed", "unconfirmed"]},
            "writeOutcome": {
                "type": "string",
                "enum": ["not_written", "rolled_back", "unknown"],
                "description": "修订失败的本地提交证据；unknown须读回核对，不自动重发",
            },
        },
        "additionalProperties": True,
    }

    @app.post(
        PATH,
        operation_id="transcribeSpeech_2024141460331",
        tags=["RQ-A-4-10"],
        dependencies=[Depends(authorize_2024141460331)],
        response_model=TranscriptionResult_2024141460331,
        response_model_exclude_unset=True,
        responses={
            status: {
                "description": description,
                "content": {
                    "application/problem+json": {"schema": problem_schema},
                },
            }
            for status, description in {
                400: "请求格式错误",
                401: "服务密钥缺失或无效",
                404: "上游录音不存在",
                409: "已有结果、重复提交或状态未确认",
                422: "录音或时间语义不符合约定",
                429: "单进程正在推理；本次未提交",
                500: "内部异常",
                502: "上游接口异常",
                503: "模型、引擎登记或本地配置不可用",
            }.items()
        },
        openapi_extra={
            "x-alpha-module": "A-4",
            "x-alpha-requirements": ["RQ-A-4-10"],
            "x-contract-status": "proposed",
        },
    )
    def transcribe_2024141460331(
        request: Request,
        body: TranscribeRequest_2024141460331,
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                min_length=1,
                max_length=64,
                pattern=r"^[\x20-\x7e]{1,64}$",
            ),
        ],
    ):
        if not idempotency_key.strip():
            return _problem_2024141460331(request, 400, "VALIDATION_ERROR")
        if not gate.acquire(blocking=False):
            return _problem_2024141460331(
                request,
                429,
                "SERVICE_BUSY",
                headers={"Retry-After": "1"},
            )
        try:
            worker = app.state.speech_service
            if worker is None:
                return _problem_2024141460331(request, 503, "MODEL_UNAVAILABLE")
            return feedback(
                worker.transcribe_2024141460331(
                    str(body.recordingId),
                    idempotency_key,
                    business_task_id=str(body.businessTaskId) if body.businessTaskId else None,
                )
            )
        finally:
            gate.release()

    transcript_responses = {
        status: {
            "description": description,
            "content": {"application/problem+json": {"schema": problem_schema}},
        }
        for status, description in {
            400: "参数错误",
            401: "凭据缺失或无效",
            403: "无编辑权限",
            404: "文本不存在",
            409: "版本冲突或保存结果未确认",
            422: "来源关联或业务校验不通过",
            500: "内部异常",
            502: "上游异常",
            503: "本地存储、修订身份不可用或修订历史校验失败且已回滚",
        }.items()
    }

    @app.post(
        "/api/v1/a4/speech/results/save",
        operation_id="saveSpeechResult_2024141460331",
        tags=["RQ-A-4-20"],
        dependencies=[Depends(authorize_2024141460331)],
        response_model=SavedResult_2024141460331,
        responses=transcript_responses,
        openapi_extra={
            "x-alpha-module": "A-4",
            "x-alpha-requirements": ["RQ-A-4-20"],
            "x-contract-status": "proposed",
        },
    )
    def save_result_2024141460331(request: Request, body: SaveRequest_2024141460331):
        worker = app.state.speech_service
        if worker is None:
            return _problem_2024141460331(request, 503, "MODEL_UNAVAILABLE")
        return feedback(worker.save_result_2024141460331(body.result))

    @app.get(
        "/api/v1/a4/transcripts/{transcriptId}",
        operation_id="getTranscript_2024141460331",
        tags=["RQ-A-4-20"],
        dependencies=[Depends(authorize_2024141460331)],
        response_model=TranscriptDetail_2024141460331,
        responses=transcript_responses,
        openapi_extra={
            "x-alpha-module": "A-4",
            "x-alpha-requirements": ["RQ-A-4-20"],
            "x-contract-status": "local-compatible-revision",
        },
    )
    def read_transcript_2024141460331(
        request: Request,
        transcriptId: UUID,
        jobId: Annotated[UUID | None, Query(description="可选来源任务标识")] = None,
    ):
        worker = app.state.speech_service
        if worker is None:
            return _problem_2024141460331(request, 503, "MODEL_UNAVAILABLE")
        result = worker.read_transcript_2024141460331(
            str(transcriptId),
            str(jobId) if jobId else None,
            include_history=True,
        )
        return feedback(_transcript_detail_2024141460331(result))

    @app.patch(
        "/api/v1/a4/transcripts/{transcriptId}",
        operation_id="updateTranscript_2024141460331",
        tags=["RQ-A-4-20"],
        dependencies=[Depends(authorize_2024141460331)],
        response_model=TranscriptDetail_2024141460331,
        responses=transcript_responses,
        openapi_extra={
            "x-alpha-module": "A-4",
            "x-alpha-requirements": ["RQ-A-4-20"],
            "x-contract-status": "local-compatible-revision",
            "security": [{"ServiceApiKey": [], "UserBearer": []}],
        },
    )
    def revise_transcript_2024141460331(
        request: Request,
        transcriptId: UUID,
        body: RevisionRequest_2024141460331,
        user: Annotated[HTTPAuthorizationCredentials | None, Depends(user_security)] = None,
    ):
        if not user or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", user.credentials):
            return _problem_2024141460331(request, 401, "UNAUTHORIZED")
        worker = app.state.speech_service
        if worker is None:
            return _problem_2024141460331(request, 503, "MODEL_UNAVAILABLE")
        result = worker.revise_transcript_2024141460331(
            str(transcriptId),
            text=body.text,
            review_status=body.reviewStatus,
            expected_revision=body.expectedRevision,
            reason=body.reason,
            bearer_token=user.credentials,
        )
        return feedback(_transcript_detail_2024141460331(result))

    generate_openapi = app.openapi

    def openapi_2024141460331():
        schema = generate_openapi()
        # FastAPI collects independent security dependencies as OR alternatives.
        # This worker requires BOTH its service key and the upstream user's Bearer.
        schema["paths"]["/api/v1/a4/transcripts/{transcriptId}"]["patch"]["security"] = [
            {"ServiceApiKey": [], "UserBearer": []}
        ]
        return schema

    app.openapi = openapi_2024141460331
    return app


class _Unauthorized_2024141460331(Exception):
    pass


def main_2024141460331(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="RQ-A-4-10 独立分段转写 HTTP 服务")
    parser.add_argument("--config", type=Path, default=Path("config.local.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    with stderr_diagnostics():
        app = create_app_2024141460331(config_path=args.config)
        # One worker owns one mutable engine and its durable execution receipts.
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_2024141460331())
