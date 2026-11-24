"""Consumer-visible save/read/revision routes; team HTTP remains an external boundary."""

import json
from copy import deepcopy
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from support_2024141460331 import RID, Engine, SharedServer

from a4_speech.api_2024141460331 import create_app_2024141460331
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.service_2024141460331 import SpeechService_2024141460331

BASE = "/api/v1/a4"
KEY = "http-save-key"
HEADERS = {"X-Api-Key": KEY}
BID = "00000000-0000-4000-8000-000000000002"


class TranscriptServer(SharedServer):
    """The existing P-Blue detail/PATCH wire contract, including actual edit history."""

    def __call__(self, request):
        if "/transcripts/" in request.url.path:
            tid = request.url.path.rsplit("/", 1)[-1]
            item = next((t for t in self.texts.values() if t["id"] == tid), None)
            if not item:
                return httpx.Response(404, json={"code": "NOT_FOUND"})
            if request.method == "PATCH":
                assert request.headers.get("Authorization") == "Bearer editor-token"
                assert "X-Api-Key" not in request.headers
                body = json.loads(request.content)
                if body["expectedRevision"] != item["revision"]:
                    return httpx.Response(409, json={"code": "REVISION_CONFLICT"})
                if (
                    body.get("text", item["text"]) == item["text"]
                    and body.get("reviewStatus", item["reviewStatus"]) == item["reviewStatus"]
                ):
                    return httpx.Response(
                        200,
                        json={
                            "transcriptId": tid,
                            "revision": item["revision"],
                            "reviewStatus": item["reviewStatus"],
                            "isEdited": item["isEdited"],
                        },
                    )
                before = deepcopy(item)
                item["revision"] += 1
                item["text"] = body.get("text", item["text"])
                item["reviewStatus"] = body.get("reviewStatus", item["reviewStatus"])
                item["isEdited"] = True
                item["editedAt"] = "2026-10-24T01:00:00Z"
                item.setdefault("edits", []).append(
                    {
                        "id": str(uuid4()),
                        "revision": item["revision"],
                        "editorUserId": "00000000-0000-4000-8000-000000000003",
                        "beforeText": before["text"],
                        "afterText": item["text"],
                        "beforeReviewStatus": before["reviewStatus"],
                        "afterReviewStatus": item["reviewStatus"],
                        "reason": body["reason"],
                        "editedAt": item["editedAt"],
                    }
                )
                return httpx.Response(
                    200,
                    json={
                        "transcriptId": tid,
                        "revision": item["revision"],
                        "reviewStatus": item["reviewStatus"],
                        "isEdited": True,
                        "editedAt": item["editedAt"],
                    },
                )
            return httpx.Response(200, json={**item, "edits": item.get("edits", [])})
        return super().__call__(request)


@pytest.fixture
def transcript_http(tmp_path):
    server, engine = TranscriptServer(), Engine()
    with httpx.Client(
        transport=httpx.MockTransport(server), headers={"X-Api-Key": "upstream-service-key"}
    ) as upstream:
        worker = SpeechService_2024141460331(
            TeamApi(upstream, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path,
        )
        with TestClient(
            create_app_2024141460331(service=worker, api_key=KEY),
            raise_server_exceptions=False,
        ) as client:
            yield client, server, engine


def transcribe(client):
    response = client.post(
        BASE + "/speech/transcribe",
        headers={**HEADERS, "Idempotency-Key": "save-api-1"},
        json={"recordingId": RID, "businessTaskId": BID},
    )
    assert response.status_code == 200, response.text
    return response.json()


def save(client):
    result = transcribe(client)
    response = client.post(BASE + "/speech/results/save", headers=HEADERS, json={"result": result})
    assert response.status_code == 200, response.text
    assert response.json()["saveConfirmed"] is True
    return result, response.json()


@pytest.mark.parametrize(
    "business_field",
    [
        pytest.param({"businessTaskId": "bad-business-id"}, id="malformed-uuid"),
        pytest.param(
            {"businessTaskId": "abcdefab--1234--4abc--8def--1234567890ab"},
            id="extra-hyphens",
        ),
        pytest.param(
            {"businessTaskId": "--abcdefab-1234-4abc-8def-1234567890ab"},
            id="prefix-hyphens",
        ),
        pytest.param(
            {"businessTaskId": "{abcdefab-1234-4abc-8def-1234567890ab"},
            id="unmatched-brace",
        ),
        pytest.param({"businessTaskId": ""}, id="empty"),
        pytest.param({"businessTaskId": " "}, id="blank"),
        pytest.param({"businessTaskId": None}, id="null"),
        pytest.param({"businessTaskId": 42}, id="integer"),
        pytest.param({"businessTaskId": 1.5}, id="float"),
        pytest.param({"businessTaskId": True}, id="boolean"),
        pytest.param({"businessTaskId": []}, id="array"),
        pytest.param({"businessTaskId": {}}, id="object"),
        pytest.param({}, id="missing"),
    ],
)
def test_save_rejects_invalid_business_id_before_text_or_metadata_write(
    transcript_http, business_field
):
    client, server, engine = transcript_http
    response = client.post(
        BASE + "/speech/transcribe",
        headers={**HEADERS, "Idempotency-Key": "unbound-save"},
        json={"recordingId": RID},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    result.pop("businessTaskId")
    result.update(business_field)
    writes_before, jobs_before = deepcopy(server.writes), deepcopy(server.jobs)

    response = client.post(BASE + "/speech/results/save", headers=HEADERS, json={"result": result})

    assert response.status_code == 400, response.text
    assert not server.texts
    assert server.writes == writes_before
    assert server.jobs == jobs_before
    assert engine.calls == 3


@pytest.mark.parametrize(
    "business_task_id",
    [
        "abcdefab-1234-4abc-8def-1234567890ab",
        "ABCDEFAB-1234-4ABC-8DEF-1234567890AB",
        "{abcdefab-1234-4abc-8def-1234567890ab}",
        "abcdefab12344abc8def1234567890ab",
        "urn:uuid:abcdefab-1234-4abc-8def-1234567890ab",
    ],
)
def test_save_binds_valid_uuid_without_rewriting_business_id_or_repeating_inference(
    transcript_http, business_task_id
):
    client, server, engine = transcript_http
    response = client.post(
        BASE + "/speech/transcribe",
        headers={**HEADERS, "Idempotency-Key": "bind-on-save"},
        json={"recordingId": RID},
    )
    assert response.status_code == 200, response.text
    result = {**response.json(), "businessTaskId": business_task_id}

    response = client.post(BASE + "/speech/results/save", headers=HEADERS, json={"result": result})

    assert response.status_code == 200, response.text
    assert response.json()["saveConfirmed"] is True
    assert response.json()["businessTaskId"] == str(UUID(business_task_id))
    assert len(server.texts) == 3
    assert server.jobs[result["jobId"]]["stats"]["a4_10_20"]["businessTaskId"] == business_task_id
    writes_before = deepcopy(server.writes)
    response = client.post(BASE + "/speech/results/save", headers=HEADERS, json={"result": result})
    assert response.status_code == 200, response.text
    assert server.writes == writes_before
    assert engine.calls == 3


@pytest.mark.parametrize("source_job_field", ["missing", None])
def test_independent_save_then_read_keeps_raw_source_and_never_repeats_inference(
    transcript_http, source_job_field
):
    client, server, engine = transcript_http
    result, _ = save(client)
    assert engine.calls == 3
    first = next(iter(server.texts.values()))
    if source_job_field is None:
        first["jobId"] = None
    response = client.get(BASE + f"/transcripts/{first['id']}", headers=HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["textRaw"] == body["text"] == "flight 1 cleared"
    assert body["revision"] == 1 and body["edits"] == []
    assert body["segmentId"] == result["transcripts"][0]["segmentId"]
    assert body["businessTaskId"] == BID
    assert body["jobId"] is None
    assert body["resolvedJobId"] == result["jobId"]
    assert body["sourceJobLinkVerified"] is False
    assert engine.calls == 3


def test_patch_preserves_original_and_exposes_versioned_audit_history(transcript_http):
    client, server, engine = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "flight one cleared", "expectedRevision": 1, "reason": "校正数字"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["textRaw"] == "flight 1 cleared" and body["text"] == "flight one cleared"
    assert body["revision"] == 2 and body["edits"][-1]["reason"] == "校正数字"
    assert body["edits"][-1]["editorUserId"].endswith("000003")
    assert engine.calls == 3
    conflict = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "stale edit", "expectedRevision": 1, "reason": "过期请求"},
    )
    assert conflict.status_code == 409
    assert first["text"] == "flight one cleared" and first["revision"] == 2


def test_revision_requires_user_bearer_not_just_service_credential(transcript_http):
    client, server, _ = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers=HEADERS,
        json={"text": "unauthorized", "expectedRevision": 1, "reason": "attempt"},
    )
    assert response.status_code == 401
    assert first["revision"] == 1


def test_review_status_only_edit_uses_declared_a5_enum_and_increments_revision(transcript_http):
    client, server, _ = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"reviewStatus": "accepted", "expectedRevision": 1, "reason": "人工核对通过"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 2
    assert response.json()["textRaw"] == response.json()["text"] == "flight 1 cleared"
    assert response.json()["edits"][-1]["afterReviewStatus"] == "accepted"


@pytest.mark.parametrize(
    "body",
    [
        {"text": "x", "reason": "why"},
        {"text": "x", "expectedRevision": 1},
        {"text": "x", "reason": " ", "expectedRevision": 1},
        {"reason": "why", "expectedRevision": 1},
        {"text": "x", "reason": "why", "expectedRevision": True},
        {"text": "x", "reason": "why", "expectedRevision": 1, "editorUserId": str(uuid4())},
    ],
)
def test_invalid_revision_is_rejected_before_upstream_write(transcript_http, body):
    client, server, _ = transcript_http
    response = client.patch(
        BASE + "/transcripts/00000000-0000-4000-8000-000000000004",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json=body,
    )
    assert response.status_code == 400
    assert not server.writes


def test_read_and_save_require_service_credential(transcript_http):
    client, server, _ = transcript_http
    for method, path, kwargs in [
        (client.get, BASE + "/transcripts/00000000-0000-4000-8000-000000000004", {}),
        (client.post, BASE + "/speech/results/save", {"json": {"result": {}}}),
    ]:
        response = method(path, **kwargs)
        assert response.status_code == 401
    assert not server.writes


def test_revision_openapi_requires_both_service_and_user_credentials(transcript_http):
    client, _, _ = transcript_http
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][BASE + "/transcripts/{transcriptId}"]["patch"]
    assert operation["security"] == [{"ServiceApiKey": [], "UserBearer": []}]


def test_http_reads_and_revises_legacy_review_history_without_renumbering(transcript_http):
    from test_revision_service_2024141460331 import legacy_review_event

    client, server, _ = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    first.update(reviewStatus="accepted", edits=[legacy_review_event()])
    response = client.get(BASE + f"/transcripts/{first['id']}", headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["legacyRevisionHistory"] is True
    assert response.json()["editHistoryStatus"] == "legacy_recorded"
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "flight one cleared", "expectedRevision": 1, "reason": "校对"},
    )
    assert response.status_code == 200, response.text
    assert [edit["revision"] for edit in response.json()["edits"]] == [1, 2]


def test_http_no_change_patch_returns_current_version_without_new_history(transcript_http):
    client, server, _ = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "flight 1 cleared", "expectedRevision": 1, "reason": "复核"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 1
    assert response.json()["changed"] is False and response.json()["edits"] == []


def test_empty_current_text_is_allowed_by_public_contract_and_preserves_raw(transcript_http):
    client, server, _ = transcript_http
    save(client)
    first = next(iter(server.texts.values()))
    response = client.patch(
        BASE + f"/transcripts/{first['id']}",
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "", "expectedRevision": 1, "reason": "人工清除误识别文本"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == ""
    assert response.json()["textRaw"] == "flight 1 cleared"


def test_partial_transcription_save_read_revision_and_stale_edit_keep_each_state(transcript_http):
    client, server, engine = transcript_http
    engine.fail_at = 2
    result, saved = save(client)
    assert result["status"] == saved["status"] == "partial"
    assert len(saved["transcripts"]) == 1 and engine.calls == 2
    tid = saved["transcripts"][0]["id"]
    path = BASE + f"/transcripts/{tid}"
    initial = client.get(path, headers=HEADERS)
    assert initial.status_code == 200
    assert initial.json()["textRaw"] == initial.json()["text"] == "flight 1 cleared"
    assert initial.json()["revision"] == 1 and initial.json()["edits"] == []
    revised = client.patch(
        path,
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "flight one cleared", "expectedRevision": 1, "reason": "核听数字"},
    )
    assert revised.status_code == 200
    body = revised.json()
    assert body["textRaw"] == "flight 1 cleared" and body["text"] == "flight one cleared"
    assert body["revision"] == 2 and body["edits"][-1]["reason"] == "核听数字"
    assert body["processingStatus"] == "partial"
    stale = client.patch(
        path,
        headers={**HEADERS, "Authorization": "Bearer editor-token"},
        json={"text": "stale overwrite", "expectedRevision": 1, "reason": "旧版本"},
    )
    assert stale.status_code == 409
    reread = client.get(path, headers=HEADERS).json()
    assert reread["text"] == "flight one cleared" and reread["revision"] == 2
    assert reread["textRaw"] == "flight 1 cleared" and len(reread["edits"]) == 1
    assert engine.calls == 2 and len(server.texts) == 1
