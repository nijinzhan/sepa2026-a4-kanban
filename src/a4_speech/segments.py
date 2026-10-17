"""Non-destructive VAD segment registration against the shared audio API."""

from datetime import UTC, datetime, timedelta
from typing import Any


def utc_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.astimezone(UTC)
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError("INVALID_TIME: timezone-aware ISO8601 required") from exc


def verify_segment(recording: dict, segment: dict) -> None:
    start, end = segment.get("startMs"), segment.get("endMs")
    duration = recording["durationMs"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= duration:
        raise ValueError("INVALID_SEGMENT: invalid millisecond interval")
    if segment.get("recordingId") != recording["id"] or not segment.get("id"):
        raise ValueError("SEGMENT_ASSOCIATION: incorrect recording or missing segment id")
    origin = utc_time(recording["startedAt"])
    for key, offset in (("startAt", start), ("endAt", end)):
        if utc_time(segment.get(key)) != origin + timedelta(milliseconds=offset):
            raise ValueError("TIME_MISMATCH: segment UTC does not match recording offset")


def plan_segments(
    recording: dict, existing: list[dict], spans: list[tuple[int, int]]
) -> tuple[list[dict], list[dict]]:
    """Preflight the entire batch before the first write; adjacent spans are legal."""
    duration = recording["durationMs"]
    ordered = sorted(spans)
    last_end = -1
    reused, new = [], []
    for start, end in ordered:
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= duration
            or start < last_end
        ):
            raise ValueError("INVALID_SEGMENT: invalid or overlapping VAD intervals")
        last_end = end
        same = None
        for old in existing:
            if old.get("recordingId") != recording["id"]:
                raise ValueError("SEGMENT_ASSOCIATION: response belongs to another recording")
            left, right = old["startMs"], old["endMs"]
            if left == start and right == end and old.get("isSpeech") is True:
                if same is not None:
                    raise ValueError("SEGMENT_CONFLICT: duplicate server intervals")
                same = old
            elif start < right and left < end:
                raise ValueError("SEGMENT_CONFLICT: existing interval cannot be overwritten")
        if same is not None:
            reused.append(same)
        else:
            new.append({"startMs": start, "endMs": end, "isSpeech": True, "speakerRole": "unknown"})
    return reused, new


def ensure_segments(
    api: Any, recording: dict, spans: list[tuple[int, int]], engine_code: str
) -> list[dict]:
    from .client import ApiError

    existing = api.list_segments(recording["id"])
    _, new = plan_segments(recording, existing, spans)
    if len(new) > 2000:
        raise ValueError("SEGMENT_LIMIT: batch exceeds 2000 segments")
    if new:
        try:
            api.add_segments(recording["id"], new, engine_code)
        except ApiError as exc:
            if exc.status not in (0, 409):
                raise
            # A timeout can follow a committed write. Reconcile instead of resubmitting.
        existing = api.list_segments(recording["id"])
    reused, missing = plan_segments(recording, existing, spans)
    if missing:
        raise ValueError("SEGMENT_UNCONFIRMED: registered intervals absent from readback")
    for item in reused:
        verify_segment(recording, item)
    return sorted(reused, key=lambda item: item["startMs"])
