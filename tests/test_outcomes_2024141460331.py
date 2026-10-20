"""Public result, replay, and provenance boundaries over the shared HTTP contract."""

import json
from copy import deepcopy

import httpx
import pytest
from support_2024141460331 import RID, Engine, SharedServer

from a4_speech.client_2024141460331 import TeamApi
from a4_speech.service_2024141460331 import (
    ProcessingError_2024141460331,
    SpeechService_2024141460331,
)


@pytest.fixture
def outcome_service(tmp_path):
    server, engine = SharedServer(), Engine(fail_at=2)
    with httpx.Client(transport=httpx.MockTransport(server), trust_env=False) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        yield (
            SpeechService_2024141460331(api, engine, receipt_dir=tmp_path / "receipts"),
            server,
            engine,
        )


def test_partial_save_timeout_after_commit_preserves_outcome_without_duplicate_text(
    outcome_service,
):
    service, server, engine = outcome_service
    result = service.transcribe_2024141460331(RID, "partial-timeout", business_task_id="business-1")
    assert result["status"] == "partial"
    server.complete_mode = "timeout_after_commit"

    saved = service.save_result_2024141460331(result)
    replayed = service.save_result_2024141460331(result)

    for confirmed in (saved, replayed):
        assert confirmed["status"] == "partial"
        assert confirmed["saveConfirmed"] is True
        assert (confirmed["processedSegments"], confirmed["failedSegments"]) == (2, 1)
        assert confirmed["failedSegmentsDetails"] == result["failedSegmentsDetails"]
        assert confirmed["unprocessedSegmentIds"] == result["unprocessedSegmentIds"]
    assert len(server.texts) == 1
    assert next(iter(server.texts.values()))["textRaw"] == "flight 1 cleared"
    text_writes = [body for _, body in server.writes if "transcripts" in body]
    assert len(text_writes) == 1
    assert len(text_writes[0]["transcripts"]) == 1
    assert engine.calls == 2


def test_repeated_partial_key_reads_saved_result_without_new_inference(outcome_service):
    service, server, engine = outcome_service
    first = service.run_2024141460331(RID, "same-partial", business_task_id="business-1")
    before = deepcopy(server.writes)

    # A fresh caller with no local model can only recover by the existing receipt.
    recovered_service = SpeechService_2024141460331(service.api, receipt_dir=service.receipt_dir)
    second = recovered_service.run_2024141460331(RID, "same-partial", business_task_id="business-1")

    assert first["status"] == second["status"] == "partial"
    assert second["saveConfirmed"] is True
    assert second["jobId"] == first["jobId"]
    assert (second["processedSegments"], second["failedSegments"]) == (2, 1)
    assert engine.calls == 2
    assert len(server.jobs) == 1
    assert len(server.texts) == 1
    assert server.writes == before


def test_unsaved_partial_replay_does_not_infer_again_or_publish_text(outcome_service):
    service, server, engine = outcome_service
    first = service.transcribe_2024141460331(RID, "unsaved-partial", business_task_id="business-1")
    assert first["status"] == "partial" and first["saveConfirmed"] is False
    before = deepcopy(server.writes)

    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.transcribe_2024141460331(RID, "unsaved-partial", business_task_id="business-1")

    assert caught.value.code == "SAVE_UNCONFIRMED"
    assert engine.calls == 2
    assert len(server.jobs) == 1
    assert server.texts == {}
    assert server.writes == before


def test_read_by_id_reports_ambiguous_when_two_jobs_match_the_same_raw_result(outcome_service):
    service, server, engine = outcome_service
    saved = service.run_2024141460331(RID, "ambiguous-history", business_task_id="business-1")
    duplicate = deepcopy(server.jobs[saved["jobId"]])
    duplicate["id"] = "another-matching-job"
    server.jobs[duplicate["id"]] = duplicate
    before = deepcopy(server.writes)

    history = service.read_transcript_2024141460331(saved["transcripts"][0]["id"])

    assert history["transcript"]["textRaw"] == "flight 1 cleared"
    assert history["historyStatus"] == "ambiguous"
    assert history["executionSnapshot"] is None
    assert history.get("jobId") is None
    assert history.get("businessTaskId") is None
    assert history["sourceJobLinkVerified"] is False
    assert engine.calls == 2
    assert server.writes == before


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        pytest.param(("status",), "succeeded", id="status"),
        pytest.param(("failedSegments",), True, id="boolean-failed-counter"),
        pytest.param(("failedSegmentsDetails",), [], id="failed-details"),
        pytest.param(("unprocessedSegmentIds",), [], id="pending-segments"),
        pytest.param(("executionSnapshot", "asr", "version"), "changed", id="snapshot"),
        pytest.param(("recordingId",), "different-recording", id="recording-id"),
    ],
)
def test_modified_result_file_is_rejected_before_any_text_write(
    outcome_service, tmp_path, field_path, replacement
):
    service, server, _ = outcome_service
    result = service.transcribe_2024141460331(RID, "file-validation", business_task_id="business-1")
    current = result
    for name in field_path[:-1]:
        current = current[name]
    current[field_path[-1]] = replacement
    result_file = tmp_path / "modified-result.json"
    result_file.write_text(json.dumps(result), encoding="utf-8")
    before = deepcopy(server.writes)

    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.save_result_2024141460331(json.loads(result_file.read_text(encoding="utf-8")))

    assert caught.value.code == "RESULT_CONTEXT_MISMATCH"
    assert server.texts == {}
    assert server.writes == before


def test_boolean_processed_counter_cannot_equal_a_single_successful_segment(
    outcome_service, tmp_path
):
    service, server, engine = outcome_service
    engine.spans = [(0, 1000)]
    engine.fail_at = None
    result = service.transcribe_2024141460331(RID, "boolean-counter", business_task_id="business-1")
    assert result["processedSegments"] == 1
    result["processedSegments"] = True
    result_file = tmp_path / "boolean-result.json"
    result_file.write_text(json.dumps(result), encoding="utf-8")
    before = deepcopy(server.writes)

    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.save_result_2024141460331(json.loads(result_file.read_text(encoding="utf-8")))

    assert caught.value.code == "RESULT_CONTEXT_MISMATCH"
    assert server.texts == {}
    assert server.writes == before


@pytest.mark.parametrize(
    ("start_at", "end_at"),
    [
        ("2026-10-17T00:00:00Z", "2026-10-17T00:00:01Z"),
        ("2026-10-17T08:00:00+08:00", "2026-10-17T08:00:01+08:00"),
    ],
    ids=["utc-z", "equivalent-offset"],
)
def test_saved_text_with_equivalent_absolute_timestamp_can_still_be_confirmed(
    outcome_service, start_at, end_at
):
    service, server, engine = outcome_service
    result = service.transcribe_2024141460331(RID, "normalized-text", business_task_id="business-1")
    saved = service.save_result_2024141460331(result)
    assert saved["saveConfirmed"] is True
    text = server.texts[result["transcripts"][0]["segmentId"]]
    text.update(startAt=start_at, endAt=end_at)
    before = deepcopy(server.writes)

    confirmed = service.save_result_2024141460331(result)

    assert confirmed["saveConfirmed"] is True
    assert confirmed["status"] == "partial"
    assert engine.calls == 2
    assert server.writes == before


@pytest.mark.parametrize(
    ("start_at", "end_at"),
    [
        ("2026-10-17T00:00:00Z", "2026-10-17T00:00:01Z"),
        ("2026-10-17T08:00:00+08:00", "2026-10-17T08:00:01+08:00"),
    ],
    ids=["utc-z", "equivalent-offset"],
)
def test_independent_save_accepts_segment_timestamp_normalization(
    outcome_service, start_at, end_at
):
    service, server, engine = outcome_service
    result = service.transcribe_2024141460331(
        RID, "normalized-segment", business_task_id="business-1"
    )
    segment = server.segments[0]
    assert segment["id"] == result["transcripts"][0]["segmentId"]
    segment.update(startAt=start_at, endAt=end_at)

    saved = service.save_result_2024141460331(result)

    assert saved["saveConfirmed"] is True
    assert saved["status"] == "partial"
    assert len(server.texts) == 1
    assert next(iter(server.texts.values()))["textRaw"] == "flight 1 cleared"
    assert engine.calls == 2


def test_confirmed_replay_result_can_be_saved_again_without_publishing_text_twice(outcome_service):
    service, server, engine = outcome_service
    first = service.run_2024141460331(RID, "save-after-replay", business_task_id="business-1")
    replayed = service.transcribe_2024141460331(
        RID, "save-after-replay", business_task_id="business-1"
    )
    before = deepcopy(server.writes)

    confirmed = service.save_result_2024141460331(replayed)

    assert confirmed["saveConfirmed"] is True
    assert confirmed["jobId"] == first["jobId"]
    assert confirmed["status"] == "partial"
    assert replayed["schemaVersion"] == 1
    assert replayed["targetSegmentIds"] == [segment["id"] for segment in server.segments]
    assert len(server.texts) == 1
    assert engine.calls == 2
    assert server.writes == before
