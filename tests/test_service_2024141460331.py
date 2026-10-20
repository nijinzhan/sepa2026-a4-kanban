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


def test_success_saves_all_and_reads_back_text_time_and_history(setup_service):
    service, server, engine = setup_service
    result = service.run_2024141460331(RID, "request-1", business_task_id="business-1")
    assert result["status"] == "succeeded"
    assert (result["processedSegments"], result["failedSegments"]) == (3, 0)
    assert [t["textRaw"] for t in result["transcripts"]] == [
        "flight 1 cleared",
        "flight 2 cleared",
        "flight 3 cleared",
    ]
    job = server.jobs[result["jobId"]]
    assert len(job["params"]["a4_10_20"]["targetSegmentIds"]) == 3
    assert job["stats"]["a4_10_20"]["totalSegments"] == 3
    engine.version = "2"
    history = service.read_transcript_2024141460331(result["transcripts"][0]["id"], result["jobId"])
    assert history["executionSnapshot"]["asr"]["version"] == "1"


def test_no_speech_creates_no_empty_segments_or_asr_job(setup_service):
    service, server, engine = setup_service
    engine.spans = []
    result = service.transcribe_2024141460331(RID, "silent")
    assert result["status"] == "no_speech"
    assert result["processedSegments"] == result["failedSegments"] == 0
    assert not server.jobs and not server.writes


def test_second_segment_failure_retains_partial_text_and_saves_valid_subset(setup_service):
    service, server, engine = setup_service
    engine.fail_at = 2
    result = service.transcribe_2024141460331(RID, "fail", business_task_id="business-1")
    assert result["status"] == "partial"
    assert (result["processedSegments"], result["failedSegments"], engine.calls) == (2, 1, 2)
    assert len(result["transcripts"]) == 1
    assert len(result["unprocessedSegmentIds"]) == 1
    assert server.texts == {}
    assert "transcripts" not in server.writes[-1][1]
    saved = service.save_result_2024141460331(result)
    assert saved["status"] == "partial" and saved["saveConfirmed"] is True
    assert len(server.texts) == 1
    history = service.read_transcript_2024141460331(saved["transcripts"][0]["id"], saved["jobId"])
    assert history["processingStatus"] == "partial"
    assert history["businessTaskId"] == "business-1"


def test_save_timeout_after_commit_is_confirmed_by_readback_without_repost(setup_service):
    service, server, _ = setup_service
    server.complete_mode = "timeout_after_commit"
    result = service.run_2024141460331(RID, "timeout", business_task_id="business-1")
    assert result["status"] == "succeeded"
    assert result["saveConfirmed"] is True
    assert (
        len([p for p, body in server.writes if p.endswith("/complete") and "transcripts" in body])
        == 1
    )


def test_save_failure_cannot_be_reported_as_success_or_as_asr_failure(setup_service):
    service, server, _ = setup_service
    server.complete_mode = "reject"
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "db-down", business_task_id="business-1")
    assert caught.value.code == "SAVE_UNCONFIRMED"
    assert (caught.value.processed_segments, caught.value.failed_segments) == (3, 0)
    assert server.texts == {}
    assert len([p for p, _ in server.writes if p.endswith("/complete")]) == 1


def test_existing_edited_text_is_never_overwritten(setup_service):
    service, server, engine = setup_service
    seg = server.add_segment(0, 1000, transcriptStatus="edited", hasTranscript=True)
    server.texts[seg["id"]] = {"id": "edited", "text": "human corrected"}
    before = deepcopy(server.texts)
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "edited", business_task_id="business-1")
    assert caught.value.code == "RESULT_EXISTS"
    assert server.texts == before
    assert engine.calls == 0
    assert not server.jobs


def test_readback_wrong_absolute_time_is_not_completed(setup_service):
    service, server, _ = setup_service
    server.break_time = True
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "bad-readback", business_task_id="business-1")
    assert caught.value.code == "SAVE_UNCONFIRMED"


def test_unrelated_job_cannot_be_used_as_model_history(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "one", business_task_id="business-1")
    job = deepcopy(server.jobs[result["jobId"]])
    job["id"] = "wrong-job"
    job["params"]["a4_10_20"]["targetSegmentIds"] = ["unrelated-segment"]
    server.jobs["wrong-job"] = job
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
        service.read_transcript_2024141460331(result["transcripts"][0]["id"], "wrong-job")


def test_old_job_without_snapshot_does_not_invent_current_model(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "old", business_task_id="business-1")
    job = server.jobs[result["jobId"]]
    job["params"]["a4_10_20"].pop("executionSnapshot")
    job["stats"] = {}
    history = service.read_transcript_2024141460331(result["transcripts"][0]["id"], result["jobId"])
    assert history["executionSnapshot"] is None
    assert history["historyStatus"] == "not_recorded"


def test_repeated_key_reads_completed_result_without_new_inference(setup_service):
    service, server, engine = setup_service
    first = service.run_2024141460331(RID, "same-key", business_task_id="business-1")
    second = service.run_2024141460331(RID, "same-key", business_task_id="business-1")
    assert second["jobId"] == first["jobId"]
    assert engine.calls == 3
    assert len(server.jobs) == 1


def test_queued_receipt_is_not_automatically_claimed(setup_service):
    service, server, engine = setup_service
    first = service.run_2024141460331(RID, "same-key", business_task_id="business-1")
    server.jobs[first["jobId"]]["status"] = "queued"
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "same-key", business_task_id="business-1")
    assert caught.value.code == "REQUEST_ALREADY_SUBMITTED"
    assert engine.calls == 3


def test_model_load_failure_is_not_a_failed_segment(setup_service):
    service, server, engine = setup_service

    def broken():
        raise ValueError("bad model")

    engine.prepare = broken
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "bad-model", business_task_id="business-1")
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert caught.value.processed_segments == caught.value.failed_segments == 0
    assert not server.jobs


def test_failed_job_cannot_claim_text_model_provenance(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "history", business_task_id="business-1")
    server.jobs[result["jobId"]]["status"] = "failed"
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
        service.read_transcript_2024141460331(result["transcripts"][0]["id"], result["jobId"])


def test_changed_raw_text_cannot_borrow_an_earlier_jobs_model(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "history", business_task_id="business-1")
    text = result["transcripts"][0]
    server.texts[text["segmentId"]]["textRaw"] = "new recognition by another task"
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
        service.read_transcript_2024141460331(text["id"], result["jobId"])


def test_save_confirmation_requires_snapshot_not_two_missing_values(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "lost-snapshot", business_task_id="business-1")
    job = server.jobs[result["jobId"]]
    job["params"]["a4_10_20"].pop("executionSnapshot")
    job["stats"]["a4_10_20"].pop("executionSnapshot")
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED"):
        service.run_2024141460331(RID, "lost-snapshot", business_task_id="business-1")


def test_model_changed_after_job_creation_marks_job_failed_with_zero_counts(setup_service):
    service, server, engine = setup_service
    original_create = service.api.create_job

    def create_then_change(*args):
        job = original_create(*args)
        engine.version = "2"
        return job

    service.api.create_job = create_then_change
    with pytest.raises(ProcessingError_2024141460331) as caught:
        service.run_2024141460331(RID, "changed", business_task_id="business-1")
    assert caught.value.code == "MODEL_MISMATCH"
    assert server.jobs[caught.value.job_id]["status"] == "failed"
    assert (caught.value.processed_segments, caught.value.failed_segments) == (0, 0)
    assert not server.texts


def test_receipt_readback_wrong_raw_text_is_structured_unconfirmed_error(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "repeat", business_task_id="business-1")
    server.texts[result["transcripts"][0]["segmentId"]]["textRaw"] = "different"
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED"):
        service.run_2024141460331(RID, "repeat", business_task_id="business-1")


def test_fractional_audio_tail_quantizes_to_recording_millisecond_bound(setup_service):
    from support_2024141460331 import wav_bytes

    service, server, engine = setup_service
    server.audio = wav_bytes(48001)
    engine.spans[-1] = (2000, 3001)
    result = service.run_2024141460331(RID, "fractional-tail", business_task_id="business-1")
    assert result["status"] == "succeeded"
    assert result["transcripts"][-1]["endMs"] == 3000


def test_unregistered_engine_stops_before_creating_job(setup_service):
    service, server, _ = setup_service
    server.engines["asr"]["code"] = "unrelated"
    with pytest.raises(ProcessingError_2024141460331, match="ENGINE_UNAVAILABLE"):
        service.run_2024141460331(RID, "engine-not-found", business_task_id="business-1")
    assert not server.jobs


def test_registry_version_conflict_does_not_silently_change_model(setup_service):
    service, server, _ = setup_service
    server.engines["asr"]["version"] = "2"
    with pytest.raises(ProcessingError_2024141460331, match="ENGINE_CONFIG_MISMATCH"):
        service.run_2024141460331(RID, "engine-upgraded", business_task_id="business-1")
    assert not server.jobs


def test_read_by_id_verifies_time_even_without_job_context(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "read-time", business_task_id="business-1")
    text = result["transcripts"][0]
    server.texts[text["segmentId"]]["endAt"] = "2026-10-17T00:00:00Z"
    with pytest.raises(ProcessingError_2024141460331, match="READBACK_INVALID"):
        service.read_transcript_2024141460331(text["id"])


def test_creation_snapshot_without_execution_confirmation_is_not_provenance(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "missing-actual", business_task_id="business-1")
    server.jobs[result["jobId"]]["stats"]["a4_10_20"].pop("executionSnapshot")
    history = service.read_transcript_2024141460331(result["transcripts"][0]["id"], result["jobId"])
    assert history["historyStatus"] == "source_unverified"
    assert history["executionSnapshot"] is None


def test_fractional_tail_does_not_hide_large_vad_overflow(setup_service):
    from support_2024141460331 import wav_bytes

    service, server, engine = setup_service
    server.audio = wav_bytes(48001)
    engine.spans[-1] = (2000, 4000)
    with pytest.raises(ProcessingError_2024141460331, match="INVALID_SEGMENT"):
        service.run_2024141460331(RID, "not-quantization", business_task_id="business-1")
    assert not server.jobs


def test_foreign_job_namespace_is_rejected_as_structured_error(setup_service):
    service, server, _ = setup_service
    result = service.run_2024141460331(RID, "foreign-context", business_task_id="business-1")
    server.jobs[result["jobId"]]["params"]["a4_10_20"] = "foreign extension"
    with pytest.raises(ProcessingError_2024141460331, match="JOB_CONTEXT_MISMATCH"):
        service.read_transcript_2024141460331(result["transcripts"][0]["id"], result["jobId"])


def test_corrupt_local_receipt_does_not_allow_reexecution(setup_service):
    service, server, engine = setup_service
    service.run_2024141460331(RID, "corrupt-receipt", business_task_id="business-1")
    next(service.receipt_dir.glob("*.json")).write_text("[]", encoding="utf-8")
    with pytest.raises(ProcessingError_2024141460331, match="REQUEST_RECEIPT_UNREADABLE"):
        service.run_2024141460331(RID, "corrupt-receipt", business_task_id="business-1")
    assert engine.calls == 3
    assert len(server.jobs) == 1


def test_transcription_does_not_save_and_can_be_saved_without_models(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "independent")
    assert result["status"] == "succeeded"
    assert result["saveConfirmed"] is False
    assert not server.texts
    result["businessTaskId"] = "caller-task"
    reader = SpeechService_2024141460331(service.api, receipt_dir=service.receipt_dir)
    saved = reader.save_result_2024141460331(result)
    assert saved["saveConfirmed"] is True
    assert saved["businessTaskId"] == "caller-task"
    history = reader.read_transcript_2024141460331(saved["transcripts"][0]["id"])
    assert history["businessTaskId"] == "caller-task"
    assert history["processingStatus"] == "succeeded"
    assert history["executionSnapshot"]["asr"]["version"] == "1"


def test_first_segment_failure_is_not_success_or_no_speech(setup_service):
    service, server, engine = setup_service
    engine.fail_at = 1
    result = service.transcribe_2024141460331(RID, "all-failed")
    assert result["status"] == "failed"
    assert result["transcripts"] == []
    assert (result["processedSegments"], result["failedSegments"]) == (1, 1)
    assert len(result["unprocessedSegmentIds"]) == 2
    assert not server.texts


@pytest.mark.parametrize("field", ["textRaw", "startMs", "segmentId", "language"])
def test_modified_result_is_rejected_before_any_text_write(setup_service, field):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "altered", business_task_id="business-1")
    result["transcripts"][0][field] = 100 if field == "startMs" else "modified"
    before = len(server.writes)
    with pytest.raises(ProcessingError_2024141460331):
        service.save_result_2024141460331(result)
    assert len(server.writes) == before and not server.texts


def test_missing_business_task_is_rejected_and_existing_task_cannot_be_rebound(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "task-binding")
    with pytest.raises(ProcessingError_2024141460331, match="BUSINESS_TASK_REQUIRED"):
        service.save_result_2024141460331(result)
    result["businessTaskId"] = "business-1"
    saved = service.save_result_2024141460331(result)
    result["businessTaskId"] = "business-2"
    before = len(server.writes)
    with pytest.raises(ProcessingError_2024141460331, match="BUSINESS_TASK_MISMATCH"):
        service.save_result_2024141460331(result)
    assert len(server.writes) == before and saved["businessTaskId"] == "business-1"


def test_repeated_independent_save_reads_back_without_posting_again(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "save-twice", business_task_id="business-1")
    first = service.save_result_2024141460331(result)
    before = len(server.writes)
    second = service.save_result_2024141460331(result)
    assert first["transcripts"] == second["transcripts"]
    assert len(server.writes) == before


def test_unconfirmed_save_repetition_never_blindly_reposts(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "save-unknown", business_task_id="business-1")
    server.complete_mode = "reject"
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED"):
        service.save_result_2024141460331(result)
    before = len(server.writes)
    server.complete_mode = "normal"
    with pytest.raises(ProcessingError_2024141460331, match="SAVE_UNCONFIRMED"):
        service.save_result_2024141460331(result)
    assert len(server.writes) == before and not server.texts


def test_independent_save_protects_human_edits_added_after_transcription(setup_service):
    service, server, _ = setup_service
    result = service.transcribe_2024141460331(RID, "edit-after", business_task_id="business-1")
    sid = result["transcripts"][0]["segmentId"]
    segment = next(s for s in server.segments if s["id"] == sid)
    segment.update(transcriptStatus="edited", hasTranscript=True)
    server.texts[sid] = {
        **segment,
        "id": "human-text",
        "segmentId": sid,
        "text": "human change",
        "textRaw": "different source",
        "revision": 2,
        "reviewStatus": "accepted",
    }
    before = deepcopy(server.texts)
    with pytest.raises(ProcessingError_2024141460331, match="RESULT_EXISTS"):
        service.save_result_2024141460331(result)
    assert server.texts == before
