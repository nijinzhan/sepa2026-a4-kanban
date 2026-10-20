"""HTTP contract tests use real httpx requests with a simulated network boundary."""

import json

import httpx
import pytest

from a4_speech.client_2024141460331 import ApiError, TeamApi

RECORDING = "11111111-1111-4111-8111-111111111111"
SEGMENT = "22222222-2222-4222-8222-222222222222"
TRANSCRIPT = "33333333-3333-4333-8333-333333333333"
JOB = "44444444-4444-4444-8444-444444444444"
START = "2026-10-17T00:00:00Z"


def recording():
    return {
        "id": RECORDING,
        "channelId": "55555555-5555-4555-8555-555555555555",
        "startedAt": START,
        "storageState": "stored",
    }


def segment(segment_id=SEGMENT):
    return {
        "id": segment_id,
        "recordingId": RECORDING,
        "startMs": 0,
        "endMs": 1000,
        "durationMs": 1000,
        "startAt": START,
        "endAt": "2026-10-17T00:00:01Z",
    }


def job(status="queued"):
    return {"id": JOB, "kind": "asr", "status": status, "submittedAt": START}


def transcript():
    return {
        "id": TRANSCRIPT,
        "segmentId": SEGMENT,
        "text": "cleared to land",
        "revision": 1,
        "reviewStatus": "unreviewed",
    }


@pytest.fixture
def api_factory():
    clients = []

    def create(handler):
        client = httpx.Client(
            transport=httpx.MockTransport(handler),
            headers={"X-Api-Key": "private-key-not-for-errors"},
        )
        clients.append(client)
        return TeamApi(client, "https://team.test/api/v1/a3/", "https://team.test/api/v1/a4")

    yield create
    for client in clients:
        client.close()


def test_recording_get_uses_a3_and_keeps_client_auth(api_factory):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == f"/api/v1/a3/recordings/{RECORDING}"
        assert request.headers["X-Api-Key"] == "private-key-not-for-errors"
        return httpx.Response(200, json=recording())

    assert api_factory(handler).get_recording(RECORDING) == recording()


def test_recording_mismatched_id_is_not_accepted(api_factory):
    data = recording() | {"id": JOB}
    api = api_factory(lambda request: httpx.Response(200, json=data))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api.get_recording(RECORDING)


@pytest.mark.parametrize("body", [[], None, {"id": RECORDING}])
def test_malformed_recording_response_is_not_accepted(api_factory, body):
    api = api_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ApiError) as error:
        api.get_recording(RECORDING)
    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.status == 200


def test_download_requests_full_audio(api_factory):
    def handler(request):
        assert request.url.path.endswith("/content")
        assert "Range" not in request.headers
        return httpx.Response(200, content=b"real-encoded-audio")

    assert api_factory(handler).get_audio(RECORDING) == b"real-encoded-audio"


@pytest.mark.parametrize("status,content", [(206, b"part"), (200, b"")])
def test_partial_or_empty_audio_is_rejected(api_factory, status, content):
    api = api_factory(lambda request: httpx.Response(status, content=content))
    with pytest.raises(ApiError) as error:
        api.get_audio(RECORDING)
    assert error.value.code == "INVALID_RESPONSE"


def test_segments_request_includes_non_speech_and_uses_declared_envelope(api_factory):
    def handler(request):
        assert dict(request.url.params) == {"onlySpeech": "false"}
        return httpx.Response(200, json={"recordingId": RECORDING, "items": [segment()]})

    assert api_factory(handler).list_segments(RECORDING) == [segment()]


def test_segments_read_all_pages_when_page_metadata_is_present(api_factory):
    ids = [SEGMENT, JOB]

    def handler(request):
        page = int(request.url.params.get("page", "1"))
        assert request.url.params["onlySpeech"] == "false"
        if page == 2:
            assert request.url.params["pageSize"] == "1"
        return httpx.Response(
            200,
            json={
                "recordingId": RECORDING,
                "items": [segment(ids[page - 1])],
                "page": {
                    "page": page,
                    "pageSize": 1,
                    "total": 2,
                    "totalPages": 2,
                    "hasNext": page == 1,
                },
            },
        )

    assert [row["id"] for row in api_factory(handler).list_segments(RECORDING)] == ids


def test_repeated_page_cannot_loop_or_silently_duplicate(api_factory):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(
            200,
            json={
                "recordingId": RECORDING,
                "items": [segment()],
                "page": {"page": 1, "pageSize": 1, "total": 2, "totalPages": 2, "hasNext": True},
            },
        )

    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(handler).list_segments(RECORDING)
    assert len(attempts) == 2


@pytest.mark.parametrize(
    "data",
    [
        {"recordingId": JOB, "items": []},
        {"recordingId": RECORDING, "items": [segment() | {"recordingId": JOB}]},
        {"recordingId": RECORDING, "items": ["not a segment"]},
        {"recordingId": RECORDING, "items": {}},
    ],
)
def test_bad_segment_envelope_is_rejected(api_factory, data):
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=data)).list_segments(RECORDING)


def test_segment_write_never_replaces_existing_segments(api_factory):
    rows = [{"startMs": 100, "endMs": 200, "isSpeech": True}]

    def handler(request):
        assert request.method == "POST"
        assert json.loads(request.content) == {
            "segments": rows,
            "engineCode": "silero",
            "replaceAll": False,
        }
        return httpx.Response(200, json={"recordingId": RECORDING, "inserted": 1, "skipped": 0})

    assert api_factory(handler).add_segments(RECORDING, rows, "silero")["inserted"] == 1


@pytest.mark.parametrize("count", [0, 2001])
def test_segment_batch_limits_rejected_before_network(api_factory, count):
    def unexpected(request):
        pytest.fail("invalid segment batch reached the network")

    with pytest.raises(ValueError):
        api_factory(unexpected).add_segments(RECORDING, [{"startMs": 0, "endMs": 1}] * count, "vad")


def test_create_job_uses_only_segments_and_idempotency_header(api_factory):
    params = {"a4_10_20": {"executionSnapshot": {"version": "1"}}}

    def handler(request):
        assert request.url.path == "/api/v1/a4/jobs"
        assert request.method == "POST"
        assert request.headers["Idempotency-Key"] == "same-operation"
        assert json.loads(request.content) == {
            "kind": "asr",
            "engineCode": "sensevoice",
            "segmentIds": [SEGMENT],
            "params": params,
            "origin": "api",
        }
        return httpx.Response(202, json=job())

    assert (
        api_factory(handler).create_job([SEGMENT], "sensevoice", params, "same-operation") == job()
    )


@pytest.mark.parametrize(
    "ids,key", [([], "k"), ([SEGMENT] * 501, "k"), ([SEGMENT], "x" * 65), ([SEGMENT], "")]
)
def test_job_invalid_request_rejected_before_network(api_factory, ids, key):
    def unexpected(request):
        pytest.fail("invalid task creation reached the network")

    with pytest.raises(ValueError):
        api_factory(unexpected).create_job(ids, "sensevoice", {}, key)


def test_get_job_and_complete_preserve_payload(api_factory):
    payload = {
        "status": "succeeded",
        "processedSegments": 1,
        "failedSegments": 0,
        "stats": {"x": 1},
    }

    def handler(request):
        if request.method == "GET":
            assert request.url.path == f"/api/v1/a4/jobs/{JOB}"
            return httpx.Response(200, json=job())
        assert request.url.path == f"/api/v1/a4/jobs/{JOB}/complete"
        assert json.loads(request.content) == payload
        return httpx.Response(200, json=job("succeeded"))

    api = api_factory(handler)
    assert api.get_job(JOB) == job()
    assert api.complete_job(JOB, payload) == job("succeeded")


def test_job_id_mismatch_is_rejected(api_factory):
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=job() | {"id": SEGMENT})).get_job(JOB)


def job_page(items, *, page=1, total=1, total_pages=1, has_next=False):
    return {
        "items": items,
        "page": {
            "page": page,
            "pageSize": 100,
            "total": total,
            "totalPages": total_pages,
            "hasNext": has_next,
        },
    }


def test_list_jobs_reads_all_pages_with_recording_and_asr_filters(api_factory):
    first = [job("succeeded") | {"id": f"job-{i}", "recordingId": RECORDING} for i in range(100)]
    last = job("partial") | {"id": "last-job", "recordingId": RECORDING}
    pages = []

    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/a4/jobs"
        page = int(request.url.params["page"])
        assert dict(request.url.params) == {
            "recordingId": RECORDING,
            "kind": "asr",
            "page": str(page),
            "pageSize": "100",
        }
        pages.append(page)
        if page == 1:
            body = job_page(first, total=101, total_pages=2, has_next=True)
        else:
            assert page == 2
            body = job_page([last], page=2, total=101, total_pages=2)
        return httpx.Response(200, json=body)

    rows = api_factory(handler).list_jobs(RECORDING)
    assert len(rows) == 101
    assert rows[0]["id"] == "job-0"
    assert rows[-1] == last
    assert pages == [1, 2]


@pytest.mark.parametrize("total_pages", [0, 1])
def test_list_jobs_accepts_declared_empty_page(api_factory, total_pages):
    body = job_page([], total=0, total_pages=total_pages)
    assert api_factory(lambda request: httpx.Response(200, json=body)).list_jobs(RECORDING) == []


@pytest.mark.parametrize("recording_id", ["", " ", ".", "..", None])
def test_list_jobs_invalid_recording_rejected_before_network(api_factory, recording_id):
    def unexpected(request):
        pytest.fail("invalid recording reached the network")

    with pytest.raises(ValueError):
        api_factory(unexpected).list_jobs(recording_id)


@pytest.mark.parametrize("body", [[], {}, {"items": []}, {"items": {}, "page": {}}])
def test_list_jobs_requires_items_and_page_envelope(api_factory, body):
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=body)).list_jobs(RECORDING)


@pytest.mark.parametrize(
    "changes",
    [
        {"page": True},
        {"page": 2},
        {"pageSize": 0},
        {"pageSize": 50},
        {"pageSize": True},
        {"total": -1},
        {"total": True},
        {"totalPages": -1},
        {"totalPages": True},
        {"totalPages": 2},
        {"hasNext": 0},
        {"hasNext": True},
    ],
)
def test_list_jobs_rejects_invalid_or_inconsistent_page_metadata(api_factory, changes):
    body = job_page([job() | {"recordingId": RECORDING}])
    body["page"].update(changes)
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=body)).list_jobs(RECORDING)


@pytest.mark.parametrize(
    "item",
    [
        [],
        job(),
        job() | {"recordingId": None},
        job() | {"recordingId": SEGMENT},
        job() | {"recordingId": RECORDING, "kind": "vad"},
        job() | {"recordingId": RECORDING, "status": "unknown"},
        job() | {"recordingId": RECORDING, "id": ""},
        job() | {"recordingId": RECORDING, "params": []},
        job() | {"recordingId": RECORDING, "processedSegments": True},
        job() | {"recordingId": RECORDING, "submittedAt": None},
    ],
)
def test_list_jobs_rejects_invalid_or_unbound_job_evidence(api_factory, item):
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=job_page([item]))).list_jobs(RECORDING)


def test_list_jobs_rejects_duplicate_ids_within_page(api_factory):
    item = job() | {"recordingId": RECORDING}
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(
            lambda request: httpx.Response(200, json=job_page([item, item], total=2))
        ).list_jobs(RECORDING)


@pytest.mark.parametrize(
    "fault", ["repeated_page", "duplicate_id", "changed_total", "missing_page"]
)
def test_list_jobs_refuses_ambiguous_or_changed_later_pages(api_factory, fault):
    first = [job() | {"id": f"job-{i}", "recordingId": RECORDING} for i in range(100)]
    calls = []

    def handler(request):
        page = int(request.url.params["page"])
        calls.append(page)
        if page == 1:
            return httpx.Response(
                200, json=job_page(first, total=101, total_pages=2, has_next=True)
            )
        assert page == 2
        body = job_page([job() | {"recordingId": RECORDING}], page=2, total=101, total_pages=2)
        if fault == "repeated_page":
            body["page"]["page"] = 1
        elif fault == "duplicate_id":
            body["items"][0]["id"] = "job-0"
        elif fault == "changed_total":
            body["page"]["total"] = 102
        else:
            del body["page"]
        return httpx.Response(200, json=body)

    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(handler).list_jobs(RECORDING)
    assert calls == [1, 2]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_list_jobs_does_not_accept_missing_or_extra_rows(api_factory, count):
    items = [job() | {"id": f"job-{i}", "recordingId": RECORDING} for i in range(count)]
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api_factory(lambda request: httpx.Response(200, json=job_page(items, total=2))).list_jobs(
            RECORDING
        )


def test_list_jobs_propagates_later_error_instead_of_partial_evidence(api_factory):
    first = [job() | {"id": f"job-{i}", "recordingId": RECORDING} for i in range(100)]
    calls = []

    def handler(request):
        page = int(request.url.params["page"])
        calls.append(page)
        if page == 1:
            return httpx.Response(
                200, json=job_page(first, total=101, total_pages=2, has_next=True)
            )
        assert page == 2
        return httpx.Response(503, json={"code": "UNAVAILABLE"})

    with pytest.raises(ApiError) as error:
        api_factory(handler).list_jobs(RECORDING)
    assert error.value.status == 503
    assert error.value.code == "UNAVAILABLE"
    assert calls == [1, 2]


def test_only_404_means_no_transcript(api_factory):
    assert api_factory(lambda request: httpx.Response(404)).get_transcript(SEGMENT) is None


@pytest.mark.parametrize("status", [401, 403, 409, 500])
def test_transcript_errors_are_not_treated_as_missing(api_factory, status):
    api = api_factory(
        lambda request: httpx.Response(
            status,
            json={
                "code": "TEAM_ERROR",
                "requestId": "trace-1",
                "detail": "private-key-not-for-errors",
            },
        )
    )
    with pytest.raises(ApiError) as error:
        api.get_transcript(SEGMENT)
    assert error.value.status == status
    assert error.value.code == "TEAM_ERROR"
    assert error.value.request_id == "trace-1"
    assert "private-key" not in str(error.value)


def test_transcript_read_paths_and_identity_validation(api_factory):
    def handler(request):
        assert request.url.path in {
            f"/api/v1/a4/segments/{SEGMENT}/transcript",
            f"/api/v1/a4/transcripts/{TRANSCRIPT}",
        }
        return httpx.Response(200, json=transcript())

    api = api_factory(handler)
    assert api.get_transcript(SEGMENT) == transcript()
    assert api.get_transcript_by_id(TRANSCRIPT) == transcript()


@pytest.mark.parametrize("lookup", ["segment", "transcript"])
def test_transcript_mismatched_lookup_rejected(api_factory, lookup):
    api = api_factory(lambda request: httpx.Response(200, json=transcript()))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        if lookup == "segment":
            api.get_transcript(JOB)
        else:
            api.get_transcript_by_id(JOB)


def test_text_by_id_404_is_error(api_factory):
    with pytest.raises(ApiError) as error:
        api_factory(lambda request: httpx.Response(404)).get_transcript_by_id(TRANSCRIPT)
    assert error.value.status == 404


def test_transport_failure_does_not_retry_write_or_leak_request(api_factory):
    attempts = []

    def handler(request):
        attempts.append(request)
        raise httpx.ReadTimeout("private-key-not-for-errors", request=request)

    with pytest.raises(ApiError) as error:
        api_factory(handler).complete_job(JOB, {"status": "failed"})
    assert error.value.status == 0
    assert error.value.code == "TRANSPORT_ERROR"
    assert "private-key" not in str(error.value)
    assert error.value.__cause__ is None
    assert len(attempts) == 1


def test_non_json_failure_keeps_http_status_and_request_header(api_factory):
    api = api_factory(
        lambda request: httpx.Response(
            502, text="<h1>upstream secret</h1>", headers={"X-Request-Id": "trace-header"}
        )
    )
    with pytest.raises(ApiError) as error:
        api.get_job(JOB)
    assert error.value.status == 502
    assert error.value.code == "HTTP_ERROR"
    assert error.value.request_id == "trace-header"
    assert "upstream secret" not in str(error.value)


def test_redirect_does_not_forward_api_key_to_untrusted_location(api_factory):
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(307, headers={"Location": "https://untrusted.test/"})

    with pytest.raises(ApiError):
        api_factory(handler).get_job(JOB)
    assert len(attempts) == 1


@pytest.mark.parametrize("field", ["kind", "status"])
def test_malformed_job_enum_is_structured_protocol_failure(api_factory, field):
    api = api_factory(lambda request: httpx.Response(200, json=job() | {field: []}))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api.get_job(JOB)


def test_malformed_review_status_is_structured_protocol_failure(api_factory):
    api = api_factory(lambda request: httpx.Response(200, json=transcript() | {"reviewStatus": {}}))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api.get_transcript(SEGMENT)


def test_malformed_write_status_rejected_before_network(api_factory):
    def unexpected(request):
        pytest.fail("malformed completion reached the network")

    with pytest.raises(ValueError):
        api_factory(unexpected).complete_job(JOB, {"status": []})


def engine(kind="asr"):
    return {
        "code": "sensevoice" if kind == "asr" else "silero",
        "name": "Registered speech model",
        "kind": kind,
        "device": "cpu",
        "deployMode": "embedded",
        "isEnabled": True,
        "modelName": "SenseVoiceSmall" if kind == "asr" else "silero-vad",
        "version": "2024-07-17",
    }


@pytest.mark.parametrize("kind", ["vad", "asr"])
def test_engines_request_enabled_kind_and_preserve_registered_metadata(api_factory, kind):
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/api/v1/a4/engines"
        assert dict(request.url.params) == {"kind": kind, "enabledOnly": "true"}
        return httpx.Response(200, json={"items": [engine(kind)]})

    assert api_factory(handler).get_engines(kind) == [engine(kind)]


def test_engine_missing_optional_version_is_not_fabricated(api_factory):
    item = engine()
    del item["version"]
    item["modelName"] = None
    api = api_factory(lambda request: httpx.Response(200, json={"items": [item]}))
    assert api.get_engines("asr") == [item]
    assert "version" not in api.get_engines("asr")[0]


@pytest.mark.parametrize("kind", ["speaker", "", [], None])
def test_non_speech_engine_kind_rejected_before_network(api_factory, kind):
    def unexpected(request):
        pytest.fail("unsupported engine kind reached the network")

    with pytest.raises(ValueError):
        api_factory(unexpected).get_engines(kind)


@pytest.mark.parametrize(
    "item",
    [
        [],
        {"code": "sensevoice", "kind": "asr", "isEnabled": True},
        engine() | {"code": ""},
        engine() | {"kind": "vad"},
        engine() | {"kind": []},
        engine() | {"isEnabled": False},
        engine() | {"isEnabled": 1},
        engine() | {"version": 1},
        engine() | {"modelName": {}},
        engine() | {"device": "invalid"},
        engine() | {"deployMode": []},
    ],
)
def test_engine_malformed_or_wrong_kind_or_disabled_rejected(api_factory, item):
    api = api_factory(lambda request: httpx.Response(200, json={"items": [item]}))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api.get_engines("asr")


@pytest.mark.parametrize("body", [[], {}, {"items": {}}, {"items": [engine(), engine()]}])
def test_engine_bad_envelope_or_duplicate_code_rejected(api_factory, body):
    api = api_factory(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ApiError, match="INVALID_RESPONSE"):
        api.get_engines("asr")


@pytest.mark.parametrize("code", ["secret response contents", "X" * 81, "<html>", [], "BAD\nCODE"])
def test_problem_code_is_short_identifier_or_falls_back(api_factory, code):
    api = api_factory(lambda request: httpx.Response(500, json={"code": code}))
    with pytest.raises(ApiError) as error:
        api.get_job(JOB)
    assert error.value.code == "HTTP_ERROR"
    assert error.value.status == 500
    assert "secret response contents" not in str(error.value)


@pytest.mark.parametrize(
    "request_id", ["X" * 129, "trace\nsecret", {}, "response detail with spaces"]
)
def test_invalid_problem_request_id_does_not_override_safe_header(api_factory, request_id):
    api = api_factory(
        lambda request: httpx.Response(
            500,
            json={"code": "SAVE_FAILED", "requestId": request_id},
            headers={"X-Request-Id": "trace-header"},
        )
    )
    with pytest.raises(ApiError) as error:
        api.get_job(JOB)
    assert error.value.code == "SAVE_FAILED"
    assert error.value.request_id == "trace-header"
