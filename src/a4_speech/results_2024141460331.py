"""Validation of processing outcomes independently of persistence confirmation."""

import hashlib
import json

from .segments_2024141460331 import utc_time


def result_digest(text: dict) -> str:
    fields = {
        k: text.get(k)
        for k in (
            "segmentId",
            "recordingId",
            "startMs",
            "endMs",
            "startAt",
            "endAt",
            "textRaw",
            "language",
        )
    }
    # Equivalent timezone offsets and Z/+00:00 must produce the same digest.
    for key in ("startAt", "endAt"):
        fields[key] = utc_time(fields[key]).isoformat()
    return hashlib.sha256(
        json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def validate_outcome(context: dict, stats: dict, status: str, processed: int, failed: int):
    targets = context.get("targetSegmentIds")
    successful = stats.get("successfulSegmentIds")
    failures = stats.get("failedSegmentsDetails")
    pending = stats.get("unprocessedSegmentIds")
    if (
        not isinstance(targets, list)
        or not targets
        or not isinstance(successful, list)
        or not isinstance(failures, list)
        or not isinstance(pending, list)
        or type(processed) is not int
        or type(failed) is not int
        or processed < 0
        or failed < 0
        or any(not isinstance(f, dict) or f.get("code") != "ASR_FAILED" for f in failures)
    ):
        raise ValueError("INVALID_OUTCOME")
    failed_ids = [f.get("segmentId") for f in failures]
    groups = [targets, successful, failed_ids, pending]
    if any(
        any(not isinstance(s, str) or not s for s in g) or len(set(g)) != len(g) for g in groups
    ):
        raise ValueError("INVALID_OUTCOME")
    combined = successful + failed_ids + pending
    if (
        len(combined) != len(set(combined))
        or set(combined) != set(targets)
        or processed != len(successful) + len(failed_ids)
        or failed != len(failed_ids)
        or type(stats.get("totalSegments")) is not int
        or stats.get("totalSegments") != len(targets)
        or stats.get("processingStatus") != status
    ):
        raise ValueError("INVALID_OUTCOME")
    expected = (
        "succeeded" if len(successful) == len(targets) else ("partial" if successful else "failed")
    )
    if status != expected:
        raise ValueError("INVALID_OUTCOME")
    return successful
