"""HTTP service boundary for the personal RQ-A-4-10 implementation."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from support_2024141460331 import RID, Engine, SharedServer

from a4_speech.api_2024141460331 import create_app_2024141460331
from a4_speech.audio_2024141460331 import decode_audio
from a4_speech.cli_2024141460331 import (
    ConfigError_2024141460331,
    _engine_2024141460331,
    load_config_2024141460331,
)
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.service_2024141460331 import SpeechService_2024141460331

PATH = "/api/v1/a4/speech/transcribe"
BID = "00000000-0000-4000-8000-000000000002"
KEY = "http-test-service-key"
HEADERS = {"X-Api-Key": KEY, "Idempotency-Key": "http-request-1"}


@pytest.fixture
def http_service(tmp_path):
    server, engine = SharedServer(), Engine()
    with httpx.Client(transport=httpx.MockTransport(server)) as upstream:
        api = TeamApi(upstream, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path)
        app = create_app_2024141460331(service=service, api_key=KEY)
        with TestClient(app) as client:
            yield client, server, engine


def test_health_is_public_without_claiming_shared_services_are_verified(http_service):
    client, server, engine = http_service
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert not server.writes and engine.calls == 0


def test_http_transcribe_returns_text_and_time_without_publishing_transcripts(http_service):
    client, server, _ = http_service
    response = client.post(
        PATH,
        headers={**HEADERS, "X-Request-Id": "caller-http-1"},
        json={"recordingId": RID, "businessTaskId": BID},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["recordingId"] == RID and result["businessTaskId"] == BID
    assert result["status"] == "succeeded"
    assert result["saveConfirmed"] is False and result["saveStatus"] == "not_requested"
    assert len(result["transcripts"]) == 3
    first = result["transcripts"][0]
    assert first["textRaw"] == "flight 1 cleared"
    assert (first["startMs"], first["endMs"]) == (0, 1000)
    assert first["startAt"] == "2026-10-17T00:00:00Z"
    assert first["endAt"] == "2026-10-17T00:00:01Z"
    assert result["traceId"] == result["processingTraceId"] == "caller-http-1"
    assert response.headers["X-Request-Id"] == "caller-http-1"
    assert not server.texts
    assert all("transcripts" not in payload for _, payload in server.writes)
    context = server.jobs[result["jobId"]]["params"]["a4_10_20"]
    assert context["businessTaskId"] == BID


@pytest.mark.parametrize("key", [None, "incorrect-key"])
def test_unauthenticated_request_cannot_start_processing(http_service, key):
    client, server, engine = http_service
    headers = {"Idempotency-Key": "no-auth"}
    if key is not None:
        headers["X-Api-Key"] = key
    response = client.post(PATH, headers=headers, json={"recordingId": RID})
    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "UNAUTHORIZED"
    assert not server.writes and engine.calls == 0


@pytest.mark.parametrize(
    "body,headers",
    [
        ({"recordingId": "not-a-uuid"}, HEADERS),
        ({"recordingId": RID, "businessTaskId": "not-a-uuid"}, HEADERS),
        ({"recordingId": RID, "modelPath": "untrusted-model"}, HEADERS),
        ({"recordingId": RID, "save": True}, HEADERS),
        ({}, HEADERS),
        ({"recordingId": RID}, {"X-Api-Key": KEY}),
        ({"recordingId": RID}, {**HEADERS, "Idempotency-Key": "x" * 65}),
        ({"recordingId": RID}, {**HEADERS, "Idempotency-Key": "   "}),
    ],
)
def test_invalid_http_input_is_problem_response_before_any_write(http_service, body, headers):
    client, server, engine = http_service
    response = client.post(PATH, headers=headers, json=body)
    assert response.status_code == 400
    result = response.json()
    assert result["code"] == "VALIDATION_ERROR" and result["status"] == 400
    assert result["instance"] == PATH
    assert result["requestId"] == response.headers["X-Request-Id"]
    assert KEY not in response.text and "untrusted-model" not in response.text
    assert not server.writes and engine.calls == 0


def test_invalid_request_id_is_not_echoed_into_logs_or_response(http_service):
    client, server, engine = http_service
    response = client.post(
        PATH,
        headers={**HEADERS, "X-Request-Id": "bad request id"},
        json={"recordingId": RID},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_REQUEST_ID"
    assert response.headers["X-Request-Id"] != "bad request id"
    assert not server.writes and engine.calls == 0


def test_partial_result_preserves_successful_and_unprocessed_segments(http_service):
    client, _, engine = http_service
    engine.fail_at = 2
    response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "partial" and result["saveConfirmed"] is False
    assert len(result["transcripts"]) == len(result["failedSegmentsDetails"]) == 1
    assert len(result["unprocessedSegmentIds"]) == 1


def test_no_speech_is_a_distinct_result_without_an_empty_asr_job(http_service):
    client, server, engine = http_service
    engine.spans = []
    response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "no_speech" and result["jobId"] is None
    assert result["code"] == "NO_SPEECH" and result["transcripts"] == []
    assert not server.jobs and not server.writes


def test_failed_asr_does_not_become_successful_recognition(http_service):
    client, _, engine = http_service
    engine.fail_at = 1
    response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed" and result["transcripts"] == []
    assert result["failedSegments"] == 1 and result["saveConfirmed"] is False


def test_repeated_unsaved_request_reports_conflict_without_repeating_inference(http_service):
    client, server, engine = http_service
    first = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert first.status_code == 200
    second = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert second.status_code == 409
    assert second.json()["code"] == "SAVE_UNCONFIRMED"
    assert second.json()["jobId"] == first.json()["jobId"]
    assert engine.calls == 3 and len(server.jobs) == 1


def test_saved_legacy_result_can_be_replayed_without_fabricated_diagnostics(http_service):
    client, server, engine = http_service
    saved = client.app.state.speech_service.run_2024141460331(RID, HEADERS["Idempotency-Key"], BID)
    stats = server.jobs[saved["jobId"]]["stats"]["a4_10_20"]
    stats.pop("audioProcessing")
    stats.pop("processingTraceId")

    response = client.post(
        PATH,
        headers={**HEADERS, "X-Request-Id": "legacy-replay-1"},
        json={"recordingId": RID, "businessTaskId": BID},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["saveConfirmed"] is True and result["jobId"] == saved["jobId"]
    for key in ("id", "segmentId", "textRaw", "startMs", "endMs"):
        assert [t[key] for t in result["transcripts"]] == [t[key] for t in saved["transcripts"]]
    assert result.get("audioProcessing") is None and result.get("processingTraceId") is None
    assert result["traceId"] == "legacy-replay-1"
    assert engine.calls == 3 and len(server.jobs) == 1


def test_transport_failure_is_502_with_safe_problem_details(http_service):
    client, server, _ = http_service

    def unavailable(request):
        raise httpx.ConnectError("secret-path-and-credentials", request=request)

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as upstream:
        client.app.state.speech_service.api = TeamApi(
            upstream,
            "https://team.test/api/v1/a3",
            "https://team.test/api/v1/a4",
        )
        response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 502
    result = response.json()
    assert result["code"] == "TRANSPORT_ERROR" and result["httpStatus"] == 0
    assert "secret-path-and-credentials" not in response.text
    assert not server.writes


def test_engine_registration_missing_is_503_not_an_empty_success(http_service):
    client, server, engine = http_service
    server.engines["asr"]["code"] = "another-engine"
    response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 503
    assert response.json()["code"] == "ENGINE_UNAVAILABLE"
    assert engine.calls == 0


def test_parallel_request_is_rejected_before_sharing_mutable_model_state(http_service):
    client, _, engine = http_service
    started, release = Event(), Event()
    recognize = engine.recognize

    def blocking_recognize(*args):
        started.set()
        assert release.wait(5)
        return recognize(*args)

    engine.recognize = blocking_recognize
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, PATH, headers=HEADERS, json={"recordingId": RID})
        try:
            assert started.wait(5)
            response = client.post(
                PATH,
                headers={**HEADERS, "Idempotency-Key": "parallel-2"},
                json={"recordingId": RID},
            )
            assert response.status_code == 429
            assert response.json()["code"] == "SERVICE_BUSY"
            assert response.headers["Retry-After"] == "1"
        finally:
            release.set()
        assert pending.result(timeout=5).status_code == 200


def test_openapi_documents_security_uuid_and_non_persistence_boundary(http_service):
    client, _, _ = http_service
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][PATH]["post"]
    assert operation["security"] == [{"ServiceApiKey": []}]
    request_schema = schema["components"]["schemas"]["TranscribeRequest_2024141460331"]
    assert request_schema["properties"]["recordingId"]["format"] == "uuid"
    assert request_schema["additionalProperties"] is False
    save_schema = schema["components"]["schemas"]["SaveRequest_2024141460331"]
    save_result = save_schema["properties"]["result"]
    assert save_result["required"] == ["businessTaskId"]
    assert save_result["properties"]["businessTaskId"] == {"type": "string", "format": "uuid"}
    assert "/api/v1/a4/jobs" not in schema["paths"]
    assert "application/problem+json" in operation["responses"]["400"]["content"]
    assert "422" in operation["responses"]
    checked_in = json.loads(
        (Path(__file__).resolve().parents[1] / "openapi/a4-speech.openapi.json").read_text(
            encoding="utf-8"
        )
    )
    assert checked_in["components"] == schema["components"]
    assert checked_in["paths"] == {
        path.removeprefix("/api/v1/a4"): operation
        for path, operation in schema["paths"].items()
        if path.startswith("/api/v1/a4/")
    }


def test_service_never_starts_with_implicit_or_missing_inbound_credentials(monkeypatch):
    monkeypatch.delenv("A4_SERVICE_API_KEY", raising=False)
    with pytest.raises(ConfigError_2024141460331, match="service credential"):
        create_app_2024141460331()


def test_upstream_request_id_is_preserved_separately_from_caller_trace(http_service):
    client, _, _ = http_service

    def failed_upstream(request):
        return httpx.Response(503, json={"code": "DATABASE_UNAVAILABLE", "requestId": "upstream-7"})

    with httpx.Client(transport=httpx.MockTransport(failed_upstream)) as upstream:
        client.app.state.speech_service.api = TeamApi(
            upstream,
            "https://team.test/api/v1/a3",
            "https://team.test/api/v1/a4",
        )
        response = client.post(
            PATH,
            headers={**HEADERS, "X-Request-Id": "caller-7"},
            json={"recordingId": RID},
        )
    assert response.status_code == 502
    result = response.json()
    assert result["requestId"] == "caller-7"
    assert result["upstreamRequestId"] == "upstream-7"
    assert result["httpStatus"] == 503


def test_unexpected_internal_exception_is_sanitized_and_releases_worker(http_service):
    client, _, engine = http_service
    snapshot = engine.snapshot

    def broken_snapshot():
        raise RuntimeError("private-model-path-and-secret")

    engine.snapshot = broken_snapshot
    response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "private-model-path-and-secret" not in response.text
    engine.snapshot = snapshot
    recovered = client.post(
        PATH,
        headers={**HEADERS, "Idempotency-Key": "after-exception"},
        json={"recordingId": RID},
    )
    assert recovered.status_code == 200


@pytest.mark.real_model
def test_actual_cpu_model_through_http_with_simulated_shared_services(tmp_path):
    config = os.environ.get("A4_REAL_MODEL_CONFIG")
    audio = os.environ.get("A4_REAL_TEST_AUDIO")
    if not config or not audio:
        pytest.skip("set A4_REAL_MODEL_CONFIG and A4_REAL_TEST_AUDIO to local assets")
    engine = _engine_2024141460331(load_config_2024141460331(Path(config)))
    server = SharedServer()
    server.audio = Path(audio).read_bytes()
    server.recording["durationMs"] = round(decode_audio(server.audio).duration_ms)
    fixed = engine.snapshot()
    for kind in ("vad", "asr"):
        server.engines[kind].update(
            code=fixed[kind]["engineCode"],
            modelName=fixed[kind]["modelName"],
            version=fixed[kind]["version"],
        )
    with httpx.Client(transport=httpx.MockTransport(server)) as upstream:
        service = SpeechService_2024141460331(
            TeamApi(upstream, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path,
        )
        with TestClient(create_app_2024141460331(service=service, api_key=KEY)) as client:
            response = client.post(PATH, headers=HEADERS, json={"recordingId": RID})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "succeeded" and result["transcripts"]
    assert all(t["textRaw"].strip() for t in result["transcripts"])
    assert result["executionSnapshot"]["asr"]["parameters"]["provider"] == "cpu"
    assert result["saveConfirmed"] is False and not server.texts
