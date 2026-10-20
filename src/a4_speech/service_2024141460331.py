"""Manual execution of the two personal requirements, without a scheduler or database."""

import hashlib
import json
import logging
import os
import tempfile
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

from .audio_2024141460331 import decode_audio
from .client_2024141460331 import ApiError
from .results_2024141460331 import result_digest, validate_outcome
from .segments_2024141460331 import ensure_segments, utc_time, verify_segment

LOG = logging.getLogger(__name__)


class ProcessingError_2024141460331(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        job_id: str | None = None,
        processed_segments: int = 0,
        failed_segments: int = 0,
        request_id: str | None = None,
        http_status: int | None = None,
    ):
        super().__init__(code)
        self.code = code
        self.job_id = job_id
        self.processed_segments = processed_segments
        self.failed_segments = failed_segments
        self.request_id = request_id
        self.http_status = http_status
        self.processing_result: dict | None = None

    def as_dict_2024141460331(self) -> dict:
        result = {
            **deepcopy(self.processing_result or {}),
            "code": self.code,
            "jobId": self.job_id,
            "processedSegments": self.processed_segments,
            "failedSegments": self.failed_segments,
            "saveConfirmed": False,
            "requestId": self.request_id,
            "httpStatus": self.http_status,
        }
        if self.processing_result is not None or self.code.startswith("SAVE_"):
            result["saveStatus"] = "unconfirmed" if self.code == "SAVE_UNCONFIRMED" else "failed"
        return result


class SpeechService_2024141460331:
    def __init__(self, api: Any, engine: Any = None, *, receipt_dir: Path | None = None):
        self.api, self.engine = api, engine
        self.receipt_dir = Path(receipt_dir) if receipt_dir is not None else None

    def _reserve_2024141460331(self, recording_id: str, key: str) -> tuple[Path, dict | None]:
        if not key or len(key) > 64 or any(ord(c) < 32 or ord(c) > 126 for c in key):
            raise ProcessingError_2024141460331("INVALID_IDEMPOTENCY_KEY")
        if self.receipt_dir is None:
            raise ProcessingError_2024141460331("RECEIPT_DIRECTORY_REQUIRED")
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        path = self.receipt_dir / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"recordingId": recording_id, "jobId": None}, handle)
        except FileExistsError:
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(receipt, dict) or not isinstance(receipt.get("recordingId"), str):
                    raise ValueError("invalid receipt object")
                if receipt.get("jobId") is not None and not isinstance(receipt["jobId"], str):
                    raise ValueError("invalid receipt job id")
            except (OSError, ValueError) as exc:
                raise ProcessingError_2024141460331("REQUEST_RECEIPT_UNREADABLE") from exc
            if receipt.get("recordingId") != recording_id:
                raise ProcessingError_2024141460331("IDEMPOTENCY_CONFLICT") from None
            return path, receipt
        return path, None

    @staticmethod
    def _record_job_2024141460331(path: Path, recording_id: str, job_id: str) -> None:
        # Only a local execution receipt, never a second transcript/mission database.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temp = Path(handle.name)
            json.dump({"recordingId": recording_id, "jobId": job_id}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _context_2024141460331(job: dict) -> dict:
        context = SpeechService_2024141460331._personal_data_2024141460331(job, "params")
        targets = context.get("targetSegmentIds", [])
        if not isinstance(targets, list) or any(not isinstance(t, str) or not t for t in targets):
            raise ProcessingError_2024141460331("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        return context

    @staticmethod
    def _personal_data_2024141460331(job: dict, field: str) -> dict:
        value = (job.get(field) or {}).get("a4_10_20", {})
        if not isinstance(value, dict):
            raise ProcessingError_2024141460331("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        snapshot = value.get("executionSnapshot")
        if snapshot is not None and (
            not isinstance(snapshot, dict)
            or any(not isinstance(snapshot.get(kind), dict) for kind in ("vad", "asr"))
        ):
            raise ProcessingError_2024141460331("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        if any(
            not isinstance(value.get(key, {}), dict)
            for key in ("transcriptDigests", "resultDigests")
        ):
            raise ProcessingError_2024141460331("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        return value

    def run_2024141460331(
        self, recording_id: str, idempotency_key: str, business_task_id: str | None = None
    ) -> dict:
        if not isinstance(business_task_id, str) or not business_task_id.strip():
            raise ProcessingError_2024141460331("BUSINESS_TASK_REQUIRED")
        result = self.transcribe_2024141460331(recording_id, idempotency_key, business_task_id)
        if result.get("saveConfirmed") or result["status"] in {"failed", "no_speech"}:
            return result
        try:
            return self.save_result_2024141460331(result)
        except ApiError as exc:
            error = ProcessingError_2024141460331(
                "SAVE_UNCONFIRMED",
                job_id=result["jobId"],
                processed_segments=result["processedSegments"],
                failed_segments=result["failedSegments"],
                request_id=exc.request_id,
                http_status=exc.status,
            )
            error.processing_result = result
            raise error from exc
        except ProcessingError_2024141460331 as exc:
            exc.processing_result = result
            exc.job_id = result["jobId"]
            exc.processed_segments = result["processedSegments"]
            exc.failed_segments = result["failedSegments"]
            raise

    def transcribe_2024141460331(
        self, recording_id: str, idempotency_key: str, business_task_id: str | None = None
    ) -> dict:
        """Execute once locally; receipts prevent blind replay after an uncertain write."""
        if business_task_id is not None and (
            not isinstance(business_task_id, str) or not business_task_id.strip()
        ):
            raise ProcessingError_2024141460331("BUSINESS_TASK_REQUIRED")
        path, old = self._reserve_2024141460331(recording_id, idempotency_key)
        if old is not None:
            jid = old.get("jobId")
            if not jid:
                raise ProcessingError_2024141460331("REQUEST_OUTCOME_UNKNOWN")
            job = self.api.get_job(jid)
            if job["status"] not in {"succeeded", "partial"}:
                raise ProcessingError_2024141460331("REQUEST_ALREADY_SUBMITTED", job_id=jid)
            try:
                result = self._confirmed_result_2024141460331(job)
            except (ApiError, ValueError, ProcessingError_2024141460331, KeyError) as exc:
                raise ProcessingError_2024141460331(
                    "SAVE_UNCONFIRMED",
                    job_id=jid,
                    processed_segments=job.get("processedSegments", 0),
                    failed_segments=job.get("failedSegments", 0),
                ) from exc
            if business_task_id is not None and result.get("businessTaskId") != business_task_id:
                raise ProcessingError_2024141460331("BUSINESS_TASK_MISMATCH", job_id=jid)
            return result

        job_id = None
        processed = failed = 0
        try:
            recording = self.api.get_recording(recording_id)
            duration = recording.get("durationMs")
            if (
                type(duration) is not int
                or duration <= 0
                or recording.get("id") != recording_id
                or recording.get("storageState") not in {"stored", "verified"}
            ):
                raise ProcessingError_2024141460331("INVALID_RECORDING")
            utc_time(recording["startedAt"])
            raw = self.api.get_audio(recording_id)
            if (
                recording.get("fileSizeBytes") is not None
                and len(raw) != recording["fileSizeBytes"]
            ):
                raise ProcessingError_2024141460331("AUDIO_SIZE_MISMATCH")
            if (
                recording.get("checksumSha256")
                and hashlib.sha256(raw).hexdigest() != recording["checksumSha256"].lower()
            ):
                raise ProcessingError_2024141460331("AUDIO_CHECKSUM_MISMATCH")
            audio = decode_audio(raw)
            if abs(audio.duration_ms - duration) > 50:
                raise ProcessingError_2024141460331("AUDIO_DURATION_MISMATCH")
            snapshot = deepcopy(self.engine.snapshot())
            self.engine.verify_snapshot(snapshot)
            self._verify_engines_2024141460331(snapshot)
            spans = self.engine.detect(audio)
            # VAD rounds sample endpoints outward; the recording stores integer ms.
            # Only reconcile the <=1 ms quantization tail, never arbitrary overflow.
            if 0 < ceil(audio.duration_ms) - duration <= 1:
                spans = [
                    (start, duration if duration < end <= ceil(audio.duration_ms) else end)
                    for start, end in spans
                ]
            if not spans:
                return {
                    "schemaVersion": 1,
                    "recordingId": recording_id,
                    "businessTaskId": business_task_id,
                    "jobId": None,
                    "status": "no_speech",
                    "code": "NO_SPEECH",
                    "processedSegments": 0,
                    "failedSegments": 0,
                    "transcripts": [],
                    "failedSegmentsDetails": [],
                    "unprocessedSegmentIds": [],
                    "saveConfirmed": False,
                    "saveStatus": "not_requested",
                }
            if len(spans) > 500:
                raise ProcessingError_2024141460331("JOB_SEGMENT_LIMIT")
            try:
                self.engine.prepare()
            except ValueError as exc:
                raise ProcessingError_2024141460331("MODEL_UNAVAILABLE") from exc
            segments = ensure_segments(self.api, recording, spans, snapshot["vad"]["engineCode"])
            for segment in segments:
                if segment.get("hasTranscript") or segment.get("transcriptStatus") in {
                    "done",
                    "edited",
                }:
                    raise ProcessingError_2024141460331("RESULT_EXISTS")
                if segment.get("transcriptStatus") in {"queued", "processing"}:
                    raise ProcessingError_2024141460331("SEGMENT_BUSY")
                if self.api.get_transcript(segment["id"]) is not None:
                    raise ProcessingError_2024141460331("RESULT_EXISTS")
            targets = [s["id"] for s in segments]
            context = {
                "recordingId": recording_id,
                "targetSegmentIds": targets,
                "executionSnapshot": snapshot,
            }
            if business_task_id is not None:
                context["businessTaskId"] = business_task_id
            job = self.api.create_job(
                targets, snapshot["asr"]["engineCode"], {"a4_10_20": context}, idempotency_key
            )
            job_id = job["id"]
            self._record_job_2024141460331(path, recording_id, job_id)
            LOG.info(
                "job created jobId=%s recordingId=%s totalSegments=%d",
                job_id,
                recording_id,
                len(targets),
            )
            if (
                self._context_2024141460331(job) != context
                or job.get("engineCode") != snapshot["asr"]["engineCode"]
            ):
                raise ProcessingError_2024141460331("JOB_CONTEXT_MISMATCH", job_id=job_id)
            if job["status"] in {"succeeded", "partial"}:
                return self._confirmed_result_2024141460331(job, expected_context=context)
            if job["status"] != "queued":
                raise ProcessingError_2024141460331("REQUEST_ALREADY_SUBMITTED", job_id=job_id)
            stats = {"a4_10_20": {**context, "totalSegments": len(targets)}}
            try:
                self.engine.verify_snapshot(snapshot)
            except ValueError as exc:
                self._fail_job_2024141460331(job_id, 0, 0, "MODEL_MISMATCH", stats)
                raise ProcessingError_2024141460331("MODEL_MISMATCH", job_id=job_id) from exc
            transcripts, failures = [], []
            for segment in segments:
                try:
                    text = self.engine.recognize(audio, segment["startMs"], segment["endMs"])
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("empty recognition result")
                except Exception:
                    processed += 1
                    failed += 1
                    failures.append({"segmentId": segment["id"], "code": "ASR_FAILED"})
                    break
                processed += 1
                transcripts.append(
                    {
                        **{
                            k: segment[k]
                            for k in ("recordingId", "startMs", "endMs", "startAt", "endAt")
                        },
                        "segmentId": segment["id"],
                        "textRaw": text,
                        "language": self._language_2024141460331(snapshot),
                    }
                )
            try:
                self.engine.verify_snapshot(snapshot)
            except ValueError as exc:
                self._fail_job_2024141460331(job_id, processed, failed, "MODEL_MISMATCH", stats)
                raise ProcessingError_2024141460331(
                    "MODEL_MISMATCH",
                    job_id=job_id,
                    processed_segments=processed,
                    failed_segments=failed,
                ) from exc
            status = (
                "succeeded"
                if len(transcripts) == len(targets)
                else ("partial" if transcripts else "failed")
            )
            details = stats["a4_10_20"]
            details.update(
                processingStatus=status,
                successfulSegmentIds=[t["segmentId"] for t in transcripts],
                failedSegmentsDetails=failures,
                unprocessedSegmentIds=targets[processed:],
                transcriptDigests={
                    t["segmentId"]: self._text_digest_2024141460331(t) for t in transcripts
                },
                resultDigests={t["segmentId"]: result_digest(t) for t in transcripts},
            )
            result = {
                "schemaVersion": 1,
                "jobId": job_id,
                "recordingId": recording_id,
                "businessTaskId": business_task_id,
                "status": status,
                "processedSegments": processed,
                "failedSegments": failed,
                "targetSegmentIds": targets,
                "transcripts": transcripts,
                "failedSegmentsDetails": failures,
                "unprocessedSegmentIds": targets[processed:],
                "executionSnapshot": snapshot,
                "saveConfirmed": False,
                "saveStatus": "not_requested",
                "processingRecorded": False,
            }
            # Record outcome metadata, without publishing any transcript text.
            try:
                self.api.complete_job(
                    job_id,
                    {
                        "status": status,
                        "processedSegments": processed,
                        "failedSegments": failed,
                        "errorMessage": "ASR_FAILED" if failures else None,
                        "stats": stats,
                    },
                )
            except ApiError:
                pass
            try:
                recorded = self.api.get_job(job_id)
                result["processingRecorded"] = (
                    recorded.get("status") == status
                    and recorded.get("processedSegments") == processed
                    and recorded.get("failedSegments") == failed
                    and self._personal_data_2024141460331(recorded, "stats") == details
                )
            except (ApiError, ProcessingError_2024141460331):
                pass
            return result
        except ProcessingError_2024141460331 as exc:
            LOG.warning(
                "processing failed jobId=%s code=%s processed=%d failed=%d",
                exc.job_id,
                exc.code,
                exc.processed_segments,
                exc.failed_segments,
            )
            raise
        except ApiError as exc:
            raise ProcessingError_2024141460331(
                exc.code,
                job_id=job_id,
                processed_segments=processed,
                failed_segments=failed,
                request_id=exc.request_id,
                http_status=exc.status,
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            code = str(exc).split(":", 1)[0] if isinstance(exc, ValueError) else "INVALID_DATA"
            if not code or len(code) > 64 or not code.replace("_", "").isalnum():
                code = "INVALID_DATA"
            raise ProcessingError_2024141460331(
                code, job_id=job_id, processed_segments=processed, failed_segments=failed
            ) from exc

    @staticmethod
    def _language_2024141460331(snapshot: dict) -> str:
        language = snapshot.get("asr", {}).get("parameters", {}).get("language", "auto")
        return language if language != "auto" else "und"

    def _verify_engines_2024141460331(self, snapshot: dict) -> None:
        for kind in ("vad", "asr"):
            fixed = snapshot[kind]
            entries = self.api.get_engines(kind)
            entry = next((e for e in entries if e["code"] == fixed["engineCode"]), None)
            if entry is None or entry.get("isEnabled") is not True:
                raise ProcessingError_2024141460331("ENGINE_UNAVAILABLE")
            for key in ("modelName", "version"):
                if entry.get(key) and fixed.get(key) and entry[key] != fixed[key]:
                    raise ProcessingError_2024141460331("ENGINE_CONFIG_MISMATCH")

    def _fail_job_2024141460331(
        self, job_id: str, processed: int, failed: int, code: str, stats: dict
    ) -> None:
        try:
            self.api.complete_job(
                job_id,
                {
                    "status": "failed",
                    "processedSegments": processed,
                    "failedSegments": failed,
                    "errorMessage": code,
                    "stats": stats,
                },
            )
        except ApiError:
            LOG.warning("failure status unconfirmed jobId=%s code=%s", job_id, code)

    @staticmethod
    def _text_digest_2024141460331(text: dict) -> str:
        fields = {
            key: text.get(key)
            for key in ("segmentId", "recordingId", "startMs", "endMs", "textRaw")
        }
        return hashlib.sha256(
            json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def save_result_2024141460331(self, result: dict) -> dict:
        """Save a previously obtained result; never performs model inference."""
        try:
            if (
                not isinstance(result, dict)
                or type(result.get("schemaVersion")) is not int
                or result["schemaVersion"] != 1
            ):
                raise ValueError("INVALID_RESULT")
            task = result.get("businessTaskId")
            if not isinstance(task, str) or not task.strip():
                raise ProcessingError_2024141460331("BUSINESS_TASK_REQUIRED")
            jid = result.get("jobId")
            if not isinstance(jid, str) or not jid:
                raise ValueError("INVALID_RESULT")
            job = self.api.get_job(jid)
            context = self._context_2024141460331(job)
            stats = self._personal_data_2024141460331(job, "stats")
            if not stats.get("processingStatus"):
                raise ProcessingError_2024141460331(
                    "SAVE_UNCONFIRMED",
                    job_id=jid,
                    processed_segments=result.get("processedSegments", 0),
                    failed_segments=result.get("failedSegments", 0),
                )
            if (
                context.get("businessTaskId")
                and stats.get("businessTaskId")
                and context["businessTaskId"] != stats["businessTaskId"]
            ):
                raise ProcessingError_2024141460331("BUSINESS_TASK_MISMATCH", job_id=jid)
            bound = stats.get("businessTaskId") or context.get("businessTaskId")
            if bound is not None and bound != task:
                raise ProcessingError_2024141460331("BUSINESS_TASK_MISMATCH", job_id=jid)
            successful = validate_outcome(
                context,
                stats,
                job["status"],
                job.get("processedSegments"),
                job.get("failedSegments"),
            )
            if (
                not successful
                or result.get("status") != job["status"]
                or result.get("recordingId") != context.get("recordingId")
                or result.get("targetSegmentIds") != context.get("targetSegmentIds")
                or result.get("executionSnapshot") != context.get("executionSnapshot")
                or stats.get("executionSnapshot") != context.get("executionSnapshot")
                or not context.get("executionSnapshot")
                or job.get("kind") != "asr"
                or job.get("engineCode") != context["executionSnapshot"]["asr"].get("engineCode")
                or result.get("processedSegments") != job.get("processedSegments")
                or type(result.get("processedSegments")) is not int
                or result.get("failedSegments") != job.get("failedSegments")
                or type(result.get("failedSegments")) is not int
                or result.get("failedSegmentsDetails") != stats.get("failedSegmentsDetails")
                or result.get("unprocessedSegmentIds") != stats.get("unprocessedSegmentIds")
            ):
                raise ValueError("RESULT_CONTEXT_MISMATCH")
            rows = result.get("transcripts")
            if (
                not isinstance(rows, list)
                or len(rows) != len(successful)
                or any(not isinstance(t, dict) for t in rows)
                or [t.get("segmentId") for t in rows] != successful
            ):
                raise ValueError("INVALID_RESULT")
            recording = self.api.get_recording(context["recordingId"])
            segments = {s["id"]: s for s in self.api.list_segments(recording["id"])}
            for text in rows:
                sid = text["segmentId"]
                verify_segment(recording, {**text, "id": sid})
                segment = segments.get(sid)
                if (
                    segment is None
                    or any(text[k] != segment[k] for k in ("recordingId", "startMs", "endMs"))
                    or any(utc_time(text[k]) != utc_time(segment[k]) for k in ("startAt", "endAt"))
                    or not isinstance(text.get("textRaw"), str)
                    or not text["textRaw"].strip()
                    or (stats.get("resultDigests") or {}).get(sid) != result_digest(text)
                ):
                    raise ValueError("RESULT_CONTEXT_MISMATCH")
            existing = [self.api.get_transcript(sid) for sid in successful]
            if all(t is not None for t in existing):
                return self._confirmed_result_2024141460331(job, expected_task=task)
            if any(t is not None for t in existing) or any(
                segments[sid].get("hasTranscript")
                or segments[sid].get("transcriptStatus") in {"done", "edited"}
                for sid in successful
            ):
                raise ProcessingError_2024141460331("RESULT_EXISTS", job_id=jid)
            if self.receipt_dir is None:
                raise ProcessingError_2024141460331("RECEIPT_DIRECTORY_REQUIRED")
            self.receipt_dir.mkdir(parents=True, exist_ok=True)
            marker = self.receipt_dir / (
                "save-" + hashlib.sha256(jid.encode()).hexdigest() + ".json"
            )
            # Reserve before writing: any interrupted attempt requires readback, never reposting.
            try:
                with marker.open("x", encoding="utf-8") as handle:
                    json.dump({"jobId": jid, "businessTaskId": task}, handle)
            except FileExistsError:
                try:
                    return self._confirmed_result_2024141460331(job, expected_task=task)
                except (ValueError, ProcessingError_2024141460331, ApiError, KeyError) as exc:
                    raise ProcessingError_2024141460331(
                        "SAVE_UNCONFIRMED",
                        job_id=jid,
                        processed_segments=result["processedSegments"],
                        failed_segments=result["failedSegments"],
                    ) from exc
            metadata = {**stats, "businessTaskId": task}
            save_error = None
            try:
                self.api.complete_job(
                    jid,
                    {
                        "status": result["status"],
                        "processedSegments": result["processedSegments"],
                        "failedSegments": result["failedSegments"],
                        "errorMessage": "ASR_FAILED" if result["failedSegments"] else None,
                        "stats": {**job.get("stats", {}), "a4_10_20": metadata},
                        "transcripts": [
                            {
                                "segmentId": t["segmentId"],
                                "text": t["textRaw"],
                                "language": t["language"],
                            }
                            for t in rows
                        ],
                    },
                )
            except ApiError as exc:
                save_error = exc
            try:
                return self._confirmed_result_2024141460331(
                    self.api.get_job(jid), expected_task=task
                )
            except (ValueError, ProcessingError_2024141460331, ApiError, KeyError) as exc:
                if save_error is not None and save_error.status in {
                    400,
                    401,
                    403,
                    404,
                    409,
                    413,
                    422,
                }:
                    try:
                        none_saved = all(self.api.get_transcript(sid) is None for sid in successful)
                    except ApiError:
                        none_saved = False
                    if none_saved:
                        raise ProcessingError_2024141460331(
                            "SAVE_FAILED",
                            job_id=jid,
                            processed_segments=result["processedSegments"],
                            failed_segments=result["failedSegments"],
                            request_id=save_error.request_id,
                            http_status=save_error.status,
                        ) from exc
                raise ProcessingError_2024141460331(
                    "SAVE_UNCONFIRMED",
                    job_id=jid,
                    processed_segments=result["processedSegments"],
                    failed_segments=result["failedSegments"],
                ) from exc
        except OSError as exc:
            raise ProcessingError_2024141460331(
                "SAVE_RECEIPT_FAILED",
                job_id=result.get("jobId"),
                processed_segments=result.get("processedSegments", 0),
                failed_segments=result.get("failedSegments", 0),
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ProcessingError_2024141460331(
                "RESULT_CONTEXT_MISMATCH",
                job_id=result.get("jobId") if isinstance(result, dict) else None,
            ) from exc

    def _confirmed_result_2024141460331(
        self, job: dict, expected_context: dict | None = None, expected_task: str | None = None
    ) -> dict:
        context = self._context_2024141460331(job)
        stats = self._personal_data_2024141460331(job, "stats")
        snapshot = context.get("executionSnapshot")
        task = stats.get("businessTaskId") or context.get("businessTaskId")
        try:
            successful = validate_outcome(
                context,
                stats,
                job.get("status"),
                job.get("processedSegments"),
                job.get("failedSegments"),
            )
            if (
                not successful
                or job.get("kind") != "asr"
                or not isinstance(snapshot, dict)
                or not snapshot
                or job.get("engineCode") != snapshot.get("asr", {}).get("engineCode")
                or (expected_context is not None and context != expected_context)
                or stats.get("executionSnapshot") != snapshot
                or not isinstance(task, str)
                or not task.strip()
                or (expected_task is not None and task != expected_task)
                or (
                    context.get("businessTaskId")
                    and stats.get("businessTaskId")
                    and context["businessTaskId"] != stats["businessTaskId"]
                )
            ):
                raise ValueError("SAVE_UNCONFIRMED")
            recording = self.api.get_recording(context["recordingId"])
            segments = {s["id"]: s for s in self.api.list_segments(recording["id"])}
            texts = []
            for sid in successful:
                text = self.api.get_transcript(sid)
                if text is None or text.get("segmentId") != sid or sid not in segments:
                    raise ValueError("TRANSCRIPT_ASSOCIATION")
                segment = segments[sid]
                verify_segment(recording, segment)
                verify_segment(recording, {**text, "id": sid})
                if (
                    any(text.get(k) != segment[k] for k in ("startMs", "endMs"))
                    or segment.get("transcriptStatus") not in {"done", "edited"}
                    or (stats.get("transcriptDigests") or {}).get(sid)
                    != self._text_digest_2024141460331(text)
                    or (stats.get("resultDigests") or {}).get(sid) != result_digest(text)
                ):
                    raise ValueError("TRANSCRIPT_ASSOCIATION")
                texts.append(text)
        except (ValueError, KeyError, TypeError) as exc:
            raise ProcessingError_2024141460331("SAVE_UNCONFIRMED", job_id=job.get("id")) from exc
        return {
            "schemaVersion": 1,
            "jobId": job["id"],
            "recordingId": context["recordingId"],
            "businessTaskId": task,
            "status": job["status"],
            "processingStatus": job["status"],
            "processedSegments": job["processedSegments"],
            "failedSegments": job["failedSegments"],
            "targetSegmentIds": context["targetSegmentIds"],
            "failedSegmentsDetails": stats["failedSegmentsDetails"],
            "unprocessedSegmentIds": stats["unprocessedSegmentIds"],
            "saveConfirmed": True,
            "saveStatus": "confirmed",
            "transcripts": texts,
            "executionSnapshot": snapshot,
            "businessTaskAssociationVerifiedBy": "job_metadata_roundtrip",
            "businessTaskForeignKeyVerified": False,
            "sourceJobLinkVerified": False,
        }

    def read_transcript_2024141460331(self, transcript_id: str, job_id: str | None = None) -> dict:
        text = self.api.get_transcript_by_id(transcript_id)
        try:
            recording = self.api.get_recording(text["recordingId"])
            verify_segment(recording, {**text, "id": text["segmentId"]})
            segments = self.api.list_segments(recording["id"])
            matching = next((s for s in segments if s["id"] == text["segmentId"]), None)
            if matching is None:
                raise ValueError("TRANSCRIPT_ASSOCIATION")
            verify_segment(recording, matching)
            if any(text.get(k) != matching[k] for k in ("startMs", "endMs")):
                raise ValueError("TRANSCRIPT_TIME_MISMATCH")
        except (KeyError, ValueError, TypeError) as exc:
            raise ProcessingError_2024141460331("READBACK_INVALID", job_id=job_id) from exc
        result = {"transcript": text, "executionSnapshot": None, "historyStatus": "not_recorded"}
        if job_id is None:
            matches = []
            for candidate in self.api.list_jobs(text["recordingId"]):
                if candidate.get("status") not in {"succeeded", "partial"}:
                    continue
                try:
                    history = self.read_transcript_2024141460331(transcript_id, candidate["id"])
                except ProcessingError_2024141460331:
                    continue
                if history.get("historyStatus") == "recorded":
                    matches.append(history)
            if len(matches) == 1:
                return matches[0]
            result["historyStatus"] = "ambiguous" if matches else "job_context_required"
            result["sourceJobLinkVerified"] = False
            return result
        job = self.api.get_job(job_id)
        context = self._context_2024141460331(job)
        if (
            job.get("kind") != "asr"
            or job.get("status") not in {"succeeded", "partial"}
            or text["segmentId"] not in context.get("targetSegmentIds", [])
            or text.get("recordingId") != context.get("recordingId")
        ):
            raise ProcessingError_2024141460331("HISTORY_ASSOCIATION", job_id=job_id)
        snapshot = context.get("executionSnapshot")
        stats = self._personal_data_2024141460331(job, "stats")
        actual = stats.get("executionSnapshot")
        if (
            context.get("businessTaskId")
            and stats.get("businessTaskId")
            and context["businessTaskId"] != stats["businessTaskId"]
        ) or (snapshot and job.get("engineCode") != snapshot["asr"].get("engineCode")):
            raise ProcessingError_2024141460331("HISTORY_ASSOCIATION", job_id=job_id)
        if snapshot is not None and actual is not None and snapshot != actual:
            raise ProcessingError_2024141460331("HISTORY_SNAPSHOT_MISMATCH", job_id=job_id)
        if snapshot or actual:
            digest = (stats.get("transcriptDigests") or {}).get(text["segmentId"])
            if digest is None:
                result.update(jobId=job_id, historyStatus="source_unverified")
                return result
            if digest != self._text_digest_2024141460331(text):
                raise ProcessingError_2024141460331("HISTORY_ASSOCIATION", job_id=job_id)
            # New outcomes also bind language and absolute times to the original result.
            if stats.get("processingStatus") is not None and (
                stats.get("resultDigests", {}).get(text["segmentId"]) != result_digest(text)
            ):
                raise ProcessingError_2024141460331("HISTORY_ASSOCIATION", job_id=job_id)
            if not snapshot or not actual:
                result.update(jobId=job_id, historyStatus="source_unverified")
                return result
        result.update(
            jobId=job_id,
            executionSnapshot=snapshot or actual,
            historyStatus="recorded" if snapshot or actual else "not_recorded",
        )
        # The public transcript schema does not return jobId: target membership is
        # checked, but server-side transcript.job_id linkage still needs live validation.
        if stats.get("successfulSegmentIds") is not None:
            try:
                successful = validate_outcome(
                    context,
                    stats,
                    job["status"],
                    job.get("processedSegments"),
                    job.get("failedSegments"),
                )
                if text["segmentId"] not in successful:
                    raise ValueError("HISTORY_ASSOCIATION")
            except ValueError as exc:
                raise ProcessingError_2024141460331("HISTORY_ASSOCIATION", job_id=job_id) from exc
        result["processingStatus"] = stats.get("processingStatus", job["status"])
        result["businessTaskId"] = stats.get("businessTaskId") or context.get("businessTaskId")
        result["businessTaskAssociationVerifiedBy"] = "job_metadata_roundtrip"
        result["businessTaskForeignKeyVerified"] = False
        result["associationVerifiedBy"] = "explicit_job_targets_and_raw_text_digest"
        result["sourceJobLinkVerified"] = False
        return result
