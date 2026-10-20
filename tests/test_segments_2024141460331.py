from datetime import UTC, datetime

import pytest

from a4_speech.segments_2024141460331 import ensure_segments, plan_segments, verify_segment

RECORDING = {"id": "r1", "durationMs": 3000, "startedAt": "2026-10-17T00:00:00Z"}


def segment(start=0, end=1000, **extra):
    return {
        "id": "s1",
        "recordingId": "r1",
        "startMs": start,
        "endMs": end,
        "isSpeech": True,
        "sequenceNo": 7,
        "startAt": "2026-10-17T00:00:00Z",
        "endAt": "2026-10-17T00:00:01Z",
        **extra,
    }


def test_exact_segment_reuses_id_and_sequence():
    existing = segment(isEdited=True)
    reused, new = plan_segments(RECORDING, [existing], [(0, 1000), (1000, 2000)])
    assert reused == [existing]
    assert new == [{"startMs": 1000, "endMs": 2000, "isSpeech": True, "speakerRole": "unknown"}]
    assert existing["sequenceNo"] == 7


@pytest.mark.parametrize(
    "spans",
    [
        [(-1, 100)],
        [(1000, 1000)],
        [(0, 3001)],
        [(0, 1000), (900, 1100)],
        [(0, 1000), (0, 1000)],
        [(False, 1000)],
    ],
)
def test_invalid_vad_spans_rejected_before_write(spans):
    with pytest.raises(ValueError):
        plan_segments(RECORDING, [], spans)


@pytest.mark.parametrize(
    "existing", [segment(0, 1100), segment(500, 1500), segment(isSpeech=False)]
)
def test_existing_overlap_never_changes_boundary(existing):
    with pytest.raises(ValueError, match="SEGMENT_CONFLICT"):
        plan_segments(RECORDING, [existing], [(0, 1000)])


def test_absolute_timestamp_is_utc_recording_start_plus_offset():
    verify_segment(RECORDING, segment())
    assert datetime.fromisoformat(RECORDING["startedAt"].replace("Z", "+00:00")).tzinfo == UTC
    with pytest.raises(ValueError, match="TIME_MISMATCH"):
        verify_segment(RECORDING, segment(endAt="2026-10-17T00:00:02Z"))


class SegmentApi:
    def __init__(self, items, failure=None, readback=None):
        self.items = items
        self.failure = failure
        self.readback = readback
        self.writes = []

    def list_segments(self, recording_id):
        return self.items

    def add_segments(self, recording_id, new, engine_code):
        self.writes.append(new)
        self.items = self.readback if self.readback is not None else [segment()]
        if self.failure:
            raise self.failure
        return {"recordingId": recording_id, "inserted": len(new), "skipped": 0}


def test_all_existing_segments_avoid_empty_post():
    api = SegmentApi([segment()])
    assert ensure_segments(api, RECORDING, [(0, 1000)], "vad")[0]["id"] == "s1"
    assert api.writes == []


def test_uncertain_write_is_read_back_without_retry():
    from a4_speech.client_2024141460331 import ApiError

    api = SegmentApi([], failure=ApiError(0, "TRANSPORT_ERROR"))
    assert ensure_segments(api, RECORDING, [(0, 1000)], "vad")[0]["id"] == "s1"
    assert len(api.writes) == 1


def test_missing_segment_after_write_cannot_be_claimed_registered():
    api = SegmentApi([], readback=[])
    with pytest.raises(ValueError, match="SEGMENT_UNCONFIRMED"):
        ensure_segments(api, RECORDING, [(0, 1000)], "vad")
