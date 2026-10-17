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

from .audio import decode_audio
from .client import ApiError
from .segments import ensure_segments, utc_time, verify_segment

LOG = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
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

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "jobId": self.job_id,
            "processedSegments": self.processed_segments,
            "failedSegments": self.failed_segments,
            "saveConfirmed": False,
            "requestId": self.request_id,
            "httpStatus": self.http_status,
        }


class SpeechService:
    def __init__(self, api: Any, engine: Any = None, *, receipt_dir: Path | None = None):
        self.api, self.engine = api, engine
        self.receipt_dir = Path(receipt_dir) if receipt_dir is not None else None

    def _reserve(self, recording_id: str, key: str) -> tuple[Path, dict | None]:
        if not key or len(key) > 64 or any(ord(c) < 32 or ord(c) > 126 for c in key):
            raise ProcessingError("INVALID_IDEMPOTENCY_KEY")
        if self.receipt_dir is None:
            raise ProcessingError("RECEIPT_DIRECTORY_REQUIRED")
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
                raise ProcessingError("REQUEST_RECEIPT_UNREADABLE") from exc
            if receipt.get("recordingId") != recording_id:
                raise ProcessingError("IDEMPOTENCY_CONFLICT") from None
            return path, receipt
        return path, None

    @staticmethod
    def _record_job(path: Path, recording_id: str, job_id: str) -> None:
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
    def _context(job: dict) -> dict:
        context = SpeechService._personal_data(job, "params")
        targets = context.get("targetSegmentIds", [])
        if not isinstance(targets, list) or any(not isinstance(t, str) or not t for t in targets):
            raise ProcessingError("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        return context

    @staticmethod
    def _personal_data(job: dict, field: str) -> dict:
        value = (job.get(field) or {}).get("a4_10_20", {})
        if not isinstance(value, dict):
            raise ProcessingError("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        snapshot = value.get("executionSnapshot")
        if snapshot is not None and (
            not isinstance(snapshot, dict)
            or any(not isinstance(snapshot.get(kind), dict) for kind in ("vad", "asr"))
        ):
            raise ProcessingError("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        digests = value.get("transcriptDigests", {})
        if not isinstance(digests, dict):
            raise ProcessingError("JOB_CONTEXT_MISMATCH", job_id=job.get("id"))
        return value

    def run(self, recording_id: str, idempotency_key: str) -> dict:
        """Execute once locally; receipts prevent blind replay after an uncertain write."""
        path, old = self._reserve(recording_id, idempotency_key)
        if old is not None:
            jid = old.get("jobId")
            if not jid:
                raise ProcessingError("REQUEST_OUTCOME_UNKNOWN")
            job = self.api.get_job(jid)
            if job["status"] != "succeeded":
                raise ProcessingError("REQUEST_ALREADY_SUBMITTED", job_id=jid)
            try:
                return self._confirmed_result(job)
            except (ApiError, ValueError, ProcessingError, KeyError) as exc:
                raise ProcessingError(
                    "SAVE_UNCONFIRMED",
                    job_id=jid,
                    processed_segments=job.get("processedSegments", 0),
                ) from exc

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
                raise ProcessingError("INVALID_RECORDING")
            utc_time(recording["startedAt"])
            raw = self.api.get_audio(recording_id)
            if (
                recording.get("fileSizeBytes") is not None
                and len(raw) != recording["fileSizeBytes"]
            ):
                raise ProcessingError("AUDIO_SIZE_MISMATCH")
            if (
                recording.get("checksumSha256")
                and hashlib.sha256(raw).hexdigest() != recording["checksumSha256"].lower()
            ):
                raise ProcessingError("AUDIO_CHECKSUM_MISMATCH")
            audio = decode_audio(raw)
            if abs(audio.duration_ms - duration) > 50:
                raise ProcessingError("AUDIO_DURATION_MISMATCH")
            snapshot = deepcopy(self.engine.snapshot())
            self.engine.verify_snapshot(snapshot)
            self._verify_engines(snapshot)
            spans = self.engine.detect(audio)
            # VAD rounds sample endpoints outward; the recording stores integer ms.
            # Only reconcile the <=1 ms quantization tail, never arbitrary overflow.
            if 0 < ceil(audio.duration_ms) - duration <= 1:
                spans = [
                    (start, duration if duration < end <= ceil(audio.duration_ms) else end)
                    for start, end in spans
                ]
            if not spans:
                raise ProcessingError("NO_SPEECH")
            if len(spans) > 500:
                raise ProcessingError("JOB_SEGMENT_LIMIT")
            try:
                self.engine.prepare()
            except ValueError as exc:
                raise ProcessingError("MODEL_UNAVAILABLE") from exc
            segments = ensure_segments(self.api, recording, spans, snapshot["vad"]["engineCode"])
            for segment in segments:
                if segment.get("hasTranscript") or segment.get("transcriptStatus") in {
                    "done",
                    "edited",
                }:
                    raise ProcessingError("RESULT_EXISTS")
                if segment.get("transcriptStatus") in {"queued", "processing"}:
                    raise ProcessingError("SEGMENT_BUSY")
                if self.api.get_transcript(segment["id"]) is not None:
                    raise ProcessingError("RESULT_EXISTS")
            targets = [s["id"] for s in segments]
            context = {
                "recordingId": recording_id,
                "targetSegmentIds": targets,
                "executionSnapshot": snapshot,
            }
            job = self.api.create_job(
                targets, snapshot["asr"]["engineCode"], {"a4_10_20": context}, idempotency_key
            )
            job_id = job["id"]
            self._record_job(path, recording_id, job_id)
            LOG.info(
                "job created jobId=%s recordingId=%s totalSegments=%d",
                job_id,
                recording_id,
                len(targets),
            )
            if (
                self._context(job) != context
                or job.get("engineCode") != snapshot["asr"]["engineCode"]
            ):
                raise ProcessingError("JOB_CONTEXT_MISMATCH", job_id=job_id)
            if job["status"] == "succeeded":
                return self._confirmed_result(job, expected_context=context)
            if job["status"] != "queued":
                raise ProcessingError("REQUEST_ALREADY_SUBMITTED", job_id=job_id)
            stats = {"a4_10_20": {**context, "totalSegments": len(targets)}}
            try:
                self.engine.verify_snapshot(snapshot)
            except ValueError as exc:
                self._fail_job(job_id, 0, 0, "MODEL_MISMATCH", stats)
                raise ProcessingError("MODEL_MISMATCH", job_id=job_id) from exc
            transcripts = []
            for segment in segments:
                try:
                    text = self.engine.recognize(audio, segment["startMs"], segment["endMs"])
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("empty recognition result")
                except Exception as exc:
                    processed += 1
                    failed += 1
                    self._fail_job(job_id, processed, failed, "ASR_FAILED", stats)
                    raise ProcessingError(
                        "ASR_FAILED",
                        job_id=job_id,
                        processed_segments=processed,
                        failed_segments=failed,
                    ) from exc
                processed += 1
                transcripts.append(
                    {"segmentId": segment["id"], "text": text, "language": self._language(snapshot)}
                )
            stats["a4_10_20"]["transcriptDigests"] = {
                s["id"]: self._text_digest({**s, "segmentId": s["id"], "textRaw": t["text"]})
                for s, t in zip(segments, transcripts, strict=True)
            }
            # Model files must still match the fixed configuration before publishing.
            try:
                self.engine.verify_snapshot(snapshot)
            except ValueError as exc:
                self._fail_job(job_id, processed, failed, "MODEL_MISMATCH", stats)
                raise ProcessingError(
                    "MODEL_MISMATCH", job_id=job_id, processed_segments=processed
                ) from exc
            payload = {
                "status": "succeeded",
                "processedSegments": processed,
                "failedSegments": 0,
                "stats": stats,
                "transcripts": transcripts,
            }
            try:
                self.api.complete_job(job_id, payload)
            except ApiError:
                # Do not retry: the completion may already have committed.
                pass
            try:
                return self._confirmed_result(self.api.get_job(job_id), transcripts, context)
            except (ApiError, ValueError, ProcessingError, KeyError) as exc:
                raise ProcessingError(
                    "SAVE_UNCONFIRMED", job_id=job_id, processed_segments=processed
                ) from exc
        except ProcessingError as exc:
            LOG.warning(
                "processing failed jobId=%s code=%s processed=%d failed=%d",
                exc.job_id,
                exc.code,
                exc.processed_segments,
                exc.failed_segments,
            )
            raise
        except ApiError as exc:
            raise ProcessingError(
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
            raise ProcessingError(
                code, job_id=job_id, processed_segments=processed, failed_segments=failed
            ) from exc

    @staticmethod
    def _language(snapshot: dict) -> str:
        language = snapshot.get("asr", {}).get("parameters", {}).get("language", "auto")
        return language if language != "auto" else "und"

    def _verify_engines(self, snapshot: dict) -> None:
        for kind in ("vad", "asr"):
            fixed = snapshot[kind]
            entries = self.api.get_engines(kind)
            entry = next((e for e in entries if e["code"] == fixed["engineCode"]), None)
            if entry is None or entry.get("isEnabled") is not True:
                raise ProcessingError("ENGINE_UNAVAILABLE")
            for key in ("modelName", "version"):
                if entry.get(key) and fixed.get(key) and entry[key] != fixed[key]:
                    raise ProcessingError("ENGINE_CONFIG_MISMATCH")

    def _fail_job(self, job_id: str, processed: int, failed: int, code: str, stats: dict) -> None:
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
    def _text_digest(text: dict) -> str:
        fields = {
            key: text.get(key)
            for key in ("segmentId", "recordingId", "startMs", "endMs", "textRaw")
        }
        return hashlib.sha256(
            json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()

    def _confirmed_result(
        self, job: dict, expected: list[dict] | None = None, expected_context: dict | None = None
    ) -> dict:
        context = self._context(job)
        targets = context.get("targetSegmentIds", [])
        stats = self._personal_data(job, "stats")
        snapshot = context.get("executionSnapshot")
        if (
            job.get("status") != "succeeded"
            or not targets
            or len(set(targets)) != len(targets)
            or not isinstance(snapshot, dict)
            or not snapshot
            or job.get("engineCode") != snapshot.get("asr", {}).get("engineCode")
            or (expected_context is not None and context != expected_context)
            or job.get("processedSegments") != len(targets)
            or job.get("failedSegments") != 0
            or stats.get("totalSegments") != len(targets)
            or stats.get("executionSnapshot") != snapshot
        ):
            raise ProcessingError("SAVE_UNCONFIRMED", job_id=job["id"])
        recording = self.api.get_recording(context["recordingId"])
        segments = {s["id"]: s for s in self.api.list_segments(recording["id"])}
        expected_text = {t["segmentId"]: t["text"] for t in expected or []}
        texts = []
        for sid in targets:
            text = self.api.get_transcript(sid)
            if text is None or text.get("segmentId") != sid or sid not in segments:
                raise ValueError("TRANSCRIPT_ASSOCIATION")
            segment = segments[sid]
            verify_segment(recording, segment)
            verify_segment(recording, {**text, "id": sid})
            if any(text.get(k) != segment[k] for k in ("startMs", "endMs")):
                raise ValueError("TRANSCRIPT_TIME_MISMATCH")
            if segment.get("transcriptStatus") not in {"done", "edited"}:
                raise ValueError("TRANSCRIPT_STATUS_MISMATCH")
            if sid in expected_text and text.get("textRaw") != expected_text[sid]:
                raise ValueError("TRANSCRIPT_TEXT_MISMATCH")
            if not isinstance(text.get("textRaw"), str) or (
                stats.get("transcriptDigests") or {}
            ).get(sid) != self._text_digest(text):
                raise ValueError("TRANSCRIPT_DIGEST_MISMATCH")
            texts.append(text)
        LOG.info("save confirmed jobId=%s processedSegments=%d", job["id"], len(targets))
        return {
            "jobId": job["id"],
            "status": "succeeded",
            "processedSegments": len(targets),
            "failedSegments": 0,
            "saveConfirmed": True,
            "transcripts": texts,
            "executionSnapshot": context.get("executionSnapshot"),
        }

    def read_transcript(self, transcript_id: str, job_id: str | None = None) -> dict:
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
            raise ProcessingError("READBACK_INVALID", job_id=job_id) from exc
        result = {"transcript": text, "executionSnapshot": None, "historyStatus": "not_recorded"}
        if job_id is None:
            result["historyStatus"] = "job_context_required"
            return result
        job = self.api.get_job(job_id)
        context = self._context(job)
        if (
            job.get("status") != "succeeded"
            or text["segmentId"] not in context.get("targetSegmentIds", [])
            or text.get("recordingId") != context.get("recordingId")
        ):
            raise ProcessingError("HISTORY_ASSOCIATION", job_id=job_id)
        snapshot = context.get("executionSnapshot")
        stats = self._personal_data(job, "stats")
        actual = stats.get("executionSnapshot")
        if snapshot is not None and actual is not None and snapshot != actual:
            raise ProcessingError("HISTORY_SNAPSHOT_MISMATCH", job_id=job_id)
        if snapshot or actual:
            digest = (stats.get("transcriptDigests") or {}).get(text["segmentId"])
            if digest is None:
                result.update(jobId=job_id, historyStatus="source_unverified")
                return result
            if digest != self._text_digest(text):
                raise ProcessingError("HISTORY_ASSOCIATION", job_id=job_id)
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
        result["associationVerifiedBy"] = "explicit_job_targets_and_raw_text_digest"
        result["sourceJobLinkVerified"] = False
        return result
