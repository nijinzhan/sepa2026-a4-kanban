from copy import deepcopy

import httpx
import pytest
from support import RID, Engine, SharedServer

from a4_speech.client import TeamApi
from a4_speech.service import ProcessingError, SpeechService


@pytest.fixture
def setup_service(tmp_path):
    server, engine = SharedServer(), Engine()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        yield SpeechService(api, engine, receipt_dir=tmp_path), server, engine


def test_success_saves_all_and_reads_back_text_time_and_history(setup_service):
    service, server, engine = setup_service
    result = service.run(RID, "request-1")
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
    history = service.read_transcript(result["transcripts"][0]["id"], result["jobId"])
    assert history["executionSnapshot"]["asr"]["version"] == "1"


def test_no_speech_creates_no_empty_segments_or_asr_job(setup_service):
    service, server, engine = setup_service
    engine.spans = []
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "silent")
    assert caught.value.code == "NO_SPEECH"
    assert caught.value.processed_segments == caught.value.failed_segments == 0
    assert not server.jobs and not server.writes


def test_second_segment_failure_never_publishes_partial_text(setup_service):
    service, server, engine = setup_service
    engine.fail_at = 2
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "fail")
    error = caught.value
    assert error.code == "ASR_FAILED"
    assert (error.processed_segments, error.failed_segments, engine.calls) == (2, 1, 2)
    assert server.jobs[error.job_id]["status"] == "failed"
    assert server.texts == {}
    assert "transcripts" not in server.writes[-1][1]


def test_save_timeout_after_commit_is_confirmed_by_readback_without_repost(setup_service):
    service, server, _ = setup_service
    server.complete_mode = "timeout_after_commit"
    result = service.run(RID, "timeout")
    assert result["status"] == "succeeded"
    assert result["saveConfirmed"] is True
    assert len([p for p, _ in server.writes if p.endswith("/complete")]) == 1


def test_save_failure_cannot_be_reported_as_success_or_as_asr_failure(setup_service):
    service, server, _ = setup_service
    server.complete_mode = "reject"
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "db-down")
    assert caught.value.code == "SAVE_UNCONFIRMED"
    assert (caught.value.processed_segments, caught.value.failed_segments) == (3, 0)
    assert server.texts == {}
    assert len([p for p, _ in server.writes if p.endswith("/complete")]) == 1


def test_existing_edited_text_is_never_overwritten(setup_service):
    service, server, engine = setup_service
    seg = server.add_segment(0, 1000, transcriptStatus="edited", hasTranscript=True)
    server.texts[seg["id"]] = {"id": "edited", "text": "human corrected"}
    before = deepcopy(server.texts)
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "edited")
    assert caught.value.code == "RESULT_EXISTS"
    assert server.texts == before
    assert engine.calls == 0
    assert not server.jobs


def test_readback_wrong_absolute_time_is_not_completed(setup_service):
    service, server, _ = setup_service
    server.break_time = True
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "bad-readback")
    assert caught.value.code == "SAVE_UNCONFIRMED"


def test_unrelated_job_cannot_be_used_as_model_history(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "one")
    job = deepcopy(server.jobs[result["jobId"]])
    job["id"] = "wrong-job"
    job["params"]["a4_10_20"]["targetSegmentIds"] = ["unrelated-segment"]
    server.jobs["wrong-job"] = job
    with pytest.raises(ProcessingError, match="HISTORY_ASSOCIATION"):
        service.read_transcript(result["transcripts"][0]["id"], "wrong-job")


def test_old_job_without_snapshot_does_not_invent_current_model(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "old")
    job = server.jobs[result["jobId"]]
    job["params"]["a4_10_20"].pop("executionSnapshot")
    job["stats"] = {}
    history = service.read_transcript(result["transcripts"][0]["id"], result["jobId"])
    assert history["executionSnapshot"] is None
    assert history["historyStatus"] == "not_recorded"


def test_repeated_key_reads_completed_result_without_new_inference(setup_service):
    service, server, engine = setup_service
    first = service.run(RID, "same-key")
    second = service.run(RID, "same-key")
    assert second["jobId"] == first["jobId"]
    assert engine.calls == 3
    assert len(server.jobs) == 1


def test_queued_receipt_is_not_automatically_claimed(setup_service):
    service, server, engine = setup_service
    first = service.run(RID, "same-key")
    server.jobs[first["jobId"]]["status"] = "queued"
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "same-key")
    assert caught.value.code == "REQUEST_ALREADY_SUBMITTED"
    assert engine.calls == 3


def test_model_load_failure_is_not_a_failed_segment(setup_service):
    service, server, engine = setup_service

    def broken():
        raise ValueError("bad model")

    engine.prepare = broken
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "bad-model")
    assert caught.value.code == "MODEL_UNAVAILABLE"
    assert caught.value.processed_segments == caught.value.failed_segments == 0
    assert not server.jobs


def test_failed_job_cannot_claim_text_model_provenance(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "history")
    server.jobs[result["jobId"]]["status"] = "failed"
    with pytest.raises(ProcessingError, match="HISTORY_ASSOCIATION"):
        service.read_transcript(result["transcripts"][0]["id"], result["jobId"])


def test_changed_raw_text_cannot_borrow_an_earlier_jobs_model(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "history")
    text = result["transcripts"][0]
    server.texts[text["segmentId"]]["textRaw"] = "new recognition by another task"
    with pytest.raises(ProcessingError, match="HISTORY_ASSOCIATION"):
        service.read_transcript(text["id"], result["jobId"])


def test_save_confirmation_requires_snapshot_not_two_missing_values(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "lost-snapshot")
    job = server.jobs[result["jobId"]]
    job["params"]["a4_10_20"].pop("executionSnapshot")
    job["stats"]["a4_10_20"].pop("executionSnapshot")
    with pytest.raises(ProcessingError, match="SAVE_UNCONFIRMED"):
        service.run(RID, "lost-snapshot")


def test_model_changed_after_job_creation_marks_job_failed_with_zero_counts(setup_service):
    service, server, engine = setup_service
    original_create = service.api.create_job

    def create_then_change(*args):
        job = original_create(*args)
        engine.version = "2"
        return job

    service.api.create_job = create_then_change
    with pytest.raises(ProcessingError) as caught:
        service.run(RID, "changed")
    assert caught.value.code == "MODEL_MISMATCH"
    assert server.jobs[caught.value.job_id]["status"] == "failed"
    assert (caught.value.processed_segments, caught.value.failed_segments) == (0, 0)
    assert not server.texts


def test_receipt_readback_wrong_raw_text_is_structured_unconfirmed_error(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "repeat")
    server.texts[result["transcripts"][0]["segmentId"]]["textRaw"] = "different"
    with pytest.raises(ProcessingError, match="SAVE_UNCONFIRMED"):
        service.run(RID, "repeat")


def test_fractional_audio_tail_quantizes_to_recording_millisecond_bound(setup_service):
    from support import wav_bytes

    service, server, engine = setup_service
    server.audio = wav_bytes(48001)
    engine.spans[-1] = (2000, 3001)
    result = service.run(RID, "fractional-tail")
    assert result["status"] == "succeeded"
    assert result["transcripts"][-1]["endMs"] == 3000


def test_unregistered_engine_stops_before_creating_job(setup_service):
    service, server, _ = setup_service
    server.engines["asr"]["code"] = "unrelated"
    with pytest.raises(ProcessingError, match="ENGINE_UNAVAILABLE"):
        service.run(RID, "engine-not-found")
    assert not server.jobs


def test_registry_version_conflict_does_not_silently_change_model(setup_service):
    service, server, _ = setup_service
    server.engines["asr"]["version"] = "2"
    with pytest.raises(ProcessingError, match="ENGINE_CONFIG_MISMATCH"):
        service.run(RID, "engine-upgraded")
    assert not server.jobs


def test_read_by_id_verifies_time_even_without_job_context(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "read-time")
    text = result["transcripts"][0]
    server.texts[text["segmentId"]]["endAt"] = "2026-10-17T00:00:00Z"
    with pytest.raises(ProcessingError, match="READBACK_INVALID"):
        service.read_transcript(text["id"])


def test_creation_snapshot_without_execution_confirmation_is_not_provenance(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "missing-actual")
    server.jobs[result["jobId"]]["stats"]["a4_10_20"].pop("executionSnapshot")
    history = service.read_transcript(result["transcripts"][0]["id"], result["jobId"])
    assert history["historyStatus"] == "source_unverified"
    assert history["executionSnapshot"] is None


def test_fractional_tail_does_not_hide_large_vad_overflow(setup_service):
    from support import wav_bytes

    service, server, engine = setup_service
    server.audio = wav_bytes(48001)
    engine.spans[-1] = (2000, 4000)
    with pytest.raises(ProcessingError, match="INVALID_SEGMENT"):
        service.run(RID, "not-quantization")
    assert not server.jobs


def test_foreign_job_namespace_is_rejected_as_structured_error(setup_service):
    service, server, _ = setup_service
    result = service.run(RID, "foreign-context")
    server.jobs[result["jobId"]]["params"]["a4_10_20"] = "foreign extension"
    with pytest.raises(ProcessingError, match="JOB_CONTEXT_MISMATCH"):
        service.read_transcript(result["transcripts"][0]["id"], result["jobId"])


def test_corrupt_local_receipt_does_not_allow_reexecution(setup_service):
    service, server, engine = setup_service
    service.run(RID, "corrupt-receipt")
    next(service.receipt_dir.glob("*.json")).write_text("[]", encoding="utf-8")
    with pytest.raises(ProcessingError, match="REQUEST_RECEIPT_UNREADABLE"):
        service.run(RID, "corrupt-receipt")
    assert engine.calls == 3
    assert len(server.jobs) == 1
