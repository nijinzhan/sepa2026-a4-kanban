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
def setup_service(tmp_path):
    server, engine = SharedServer(), Engine()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        yield SpeechService_2024141460331(api, engine, receipt_dir=tmp_path), server, engine


def test_combined_call_requires_business_task_before_any_processing(setup_service):
    service, server, engine = setup_service
    with pytest.raises(ProcessingError_2024141460331, match="BUSINESS_TASK_REQUIRED"):
        service.run_2024141460331(RID, "missing-task")
    assert engine.calls == 0 and not server.writes
    assert not list(service.receipt_dir.glob("*.json"))


def test_combined_save_failure_returns_reusable_partial_result(setup_service):
    service, server, engine = setup_service
    engine.fail_at = 2
    original = service.save_result_2024141460331

    def unavailable(result):
        server.complete_mode = "reject"
        return original(result)

    service.save_result_2024141460331 = unavailable
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED") as caught:
        service.run_2024141460331(RID, "retain-partial", business_task_id="business-1")
    output = caught.value.as_dict_2024141460331()
    assert output["status"] == "partial"
    assert output["schemaVersion"] == 1 and len(output["transcripts"]) == 1
    assert output["saveStatus"] == "unconfirmed" and output["saveConfirmed"] is False
    assert (output["processedSegments"], output["failedSegments"]) == (2, 1)
    assert server.texts == {}


def test_definite_save_rejection_is_distinct_from_uncertain_delivery(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "rejected", business_task_id="business-1")

    def rejected(*args):
        from a4_speech.client_2024141460331 import ApiError

        raise ApiError(403, "FORBIDDEN", "req-reject")

    service.api.complete_job = rejected
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_FAILED") as caught:
        service.save_result_2024141460331(result)
    assert caught.value.http_status == 403 and caught.value.request_id == "req-reject"
    assert not server.texts


def test_malformed_result_digest_map_is_a_structured_error(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "bad-digests", business_task_id="business-1")
    server.jobs[result["jobId"]]["stats"]["a4_10_20"]["resultDigests"] = ["invalid"]
    before = len(server.writes)
    with pytest.raises(ProcessingError_2024141460331):
        service.save_result_2024141460331(result)
    assert len(server.writes) == before and not server.texts


def test_history_cannot_claim_conflicting_business_task_binding(setup_service):
    service, server, _ = setup_service
    saved = service.run_2024141460331(RID, "task-conflict", business_task_id="business-1")
    server.jobs[saved["jobId"]]["stats"]["a4_10_20"]["businessTaskId"] = "business-2"
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
        service.read_transcript_2024141460331(saved["transcripts"][0]["id"], saved["jobId"])


def test_new_history_verifies_language_as_part_of_result_digest(setup_service):
    service, server, _ = setup_service
    saved = service.run_2024141460331(RID, "language-conflict", business_task_id="business-1")
    text = saved["transcripts"][0]
    server.texts[text["segmentId"]]["language"] = "zh"
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
        service.read_transcript_2024141460331(text["id"], saved["jobId"])


def test_replay_cannot_silently_rebind_business_task(setup_service):
    service, server, engine = setup_service
    saved = service.run_2024141460331(RID, "task-replay", business_task_id="business-1")
    before = deepcopy(server.texts)
    with pytest.raises(ProcessingError_2024141460331, match="BUSINESS_TASK_MISMATCH"):
        service.run_2024141460331(RID, "task-replay", business_task_id="business-2")
    assert engine.calls == 3 and server.texts == before
    assert saved["businessTaskId"] == "business-1"


def test_combined_call_retains_result_when_save_receipt_cannot_be_written(setup_service):
    service, server, _ = setup_service
    original = service.save_result_2024141460331

    def broken_receipt(result):
        blocked = service.receipt_dir / "blocked"
        blocked.write_text("existing file", encoding="utf-8")
        service.receipt_dir = blocked
        return original(result)

    service.save_result_2024141460331 = broken_receipt
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_RECEIPT_FAILED") as caught:
        service.run_2024141460331(RID, "receipt-io", business_task_id="business-1")
    output = caught.value.as_dict_2024141460331()
    assert output["status"] == "succeeded" and len(output["transcripts"]) == 3
    assert output["saveConfirmed"] is False and output["saveStatus"] == "failed"
    assert not server.texts


def test_save_checks_job_engine_before_publishing_any_text(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "engine-conflict", business_task_id="business-1")
    server.jobs[result["jobId"]]["engineCode"] = "another-engine"
    before = len(server.writes)
    with pytest.raises(ProcessingError_2024141460331):
        service.save_result_2024141460331(result)
    assert len(server.writes) == before and not server.texts


def test_definite_http_rejection_with_partial_write_is_still_unconfirmed(setup_service):
    from a4_speech.client_2024141460331 import ApiError

    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "partial-commit", business_task_id="business-1")
    original = service.api.complete_job

    def partial_commit(jid, payload):
        original(jid, {**payload, "transcripts": payload["transcripts"][:1]})
        raise ApiError(409, "CONFLICT")

    service.api.complete_job = partial_commit
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED"):
        service.save_result_2024141460331(result)
    assert len(server.texts) == 1
