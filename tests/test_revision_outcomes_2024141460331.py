"""RQ-A-4-20: distinguish local write evidence from an uncertain remote response."""

from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from test_revision_service_2024141460331 import (
    TID,
    RevisionBoundary,
    setup_revision,
)

from a4_speech.api_2024141460331 import create_app_2024141460331
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.postgres_2024141460331 import PostgresTeamApi_2024141460331
from a4_speech.service_2024141460331 import SpeechService_2024141460331

revision_case = setup_revision
ACTOR = "00000000-0000-4000-8000-000000000004"
KEY = "revision-outcome-test"
HEADERS = {
    "X-Api-Key": KEY,
    "Authorization": "Bearer editor-token",
    "X-Request-Id": "revision-outcome-20261110",
}
PATCH = {"text": "climb five thousand", "expectedRevision": 1, "reason": "核听数字"}


def patch_revision(service):
    app = create_app_2024141460331(service=service, api_key=KEY)
    with TestClient(app) as client:
        return client.patch(f"/api/v1/a4/transcripts/{TID}", headers=HEADERS, json=PATCH)


@pytest.mark.parametrize("identity", ["unconfigured", "timeout", "invalid", "unavailable"])
def test_local_identity_failure_reports_not_written_without_starting_transaction(
    revision_case, monkeypatch, tmp_path, identity
):
    service, boundary = revision_case
    original = deepcopy(boundary.text)
    adapter = PostgresTeamApi_2024141460331("unused-test-dsn", tmp_path)

    def unexpected_transaction():
        pytest.fail("identity failure must not start a storage transaction")

    monkeypatch.setattr(adapter, "_transaction_2024141460331", unexpected_transaction)
    monkeypatch.setattr(service.api, "revise_transcript", adapter.revise_transcript)

    def identify(request):
        if identity == "timeout":
            raise httpx.ReadTimeout("identity timeout", request=request)
        if identity == "invalid":
            return httpx.Response(200, json={"id": "invalid", "status": "active"})
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(identify)) as http:
        if identity != "unconfigured":
            adapter.a5_url, adapter.identity_client = "https://identity.test", http
        response = patch_revision(service)

    body = response.json()
    assert response.status_code == 503
    assert body["code"] == "REVISION_AUTH_UNAVAILABLE"
    assert body["saveConfirmed"] is False and body["saveStatus"] == "failed"
    assert body["writeOutcome"] == "not_written"
    assert "尚未保存" in body["message"]
    assert body["requestId"] == HEADERS["X-Request-Id"]
    assert boundary.text == original
    assert not boundary.patch_payloads


@pytest.mark.parametrize(
    "status,code,expected_status",
    [
        (401, "REVISION_AUTH_REQUIRED", 401),
        (401, "REVISION_AUTH_DENIED", 401),
        (403, "REVISION_AUTH_DENIED", 403),
        (409, "REVISION_CONFLICT", 409),
        (422, "INVALID_REVISION", 422),
    ],
)
def test_explicit_local_rejection_reports_not_written(status, code, expected_status):
    class RejectedAdapter:
        def get_transcript_by_id(self, transcript_id):
            return {
                "id": TID,
                "text": "climb four thousand",
                "textRaw": "climb four thousand",
                "revision": 1,
                "reviewStatus": "unreviewed",
            }

        def revise_transcript(self, *args, **kwargs):
            from a4_speech.client_2024141460331 import ApiError

            raise ApiError(status, code, write_outcome="not_written")

    response = patch_revision(SpeechService_2024141460331(RejectedAdapter()))
    body = response.json()
    assert response.status_code == expected_status and body["code"] == code
    assert body["saveConfirmed"] is False and body["saveStatus"] == "failed"
    assert body["writeOutcome"] == "not_written"


class HistoryFailureConnection:
    """SQL boundary double: real adapter and transaction wrapper, no actual database."""

    def __init__(self, boundary, *, rollback_fails):
        self.boundary = boundary
        self.before = deepcopy(boundary.text)
        self.rollback_fails = rollback_fails
        self.rollback_completed = False
        self.updates = 0
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        assert exc is not None, "history failure must exit the transaction exceptionally"
        if self.rollback_fails:
            import psycopg

            raise psycopg.OperationalError("rollback confirmation lost")
        self.boundary.text.clear()
        self.boundary.text.update(self.before)
        self.rollback_completed = True

    def execute(self, sql, params=None):
        if sql.startswith("SET LOCAL") or sql.startswith("SELECT set_config"):
            self.row = None
        elif sql.startswith("SELECT status,iam.fn_has_permission"):
            self.row = {"status": "active", "allowed": True}
        elif sql.startswith("SELECT * FROM asr.transcript WHERE"):
            self.row = {
                "text_raw": self.before["textRaw"],
                "text_norm": None,
                "revision": 1,
                "review_status": "unreviewed",
            }
        elif sql.startswith("UPDATE asr.transcript SET"):
            self.updates += 1
            self.boundary.text.update(text=params[0], revision=2, isEdited=True)
            self.boundary.text["edits"].append({"revision": 2, "reason": "wrong history"})
            self.row = {"revision": 2}
        elif sql.startswith("SELECT * FROM asr.transcript_edit"):
            self.row = []  # Missing mandatory edit triggers real adapter validation.
        else:
            raise AssertionError(f"Unexpected SQL operation: {sql}")
        return self

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.row


@pytest.mark.parametrize("rollback_fails", [False, True], ids=["rolled_back", "rollback_unknown"])
def test_history_failure_is_failed_only_after_confirmed_rollback(
    revision_case, monkeypatch, tmp_path, rollback_fails
):
    import psycopg

    service, boundary = revision_case
    original = deepcopy(boundary.text)
    connection = HistoryFailureConnection(boundary, rollback_fails=rollback_fails)
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"id": ACTOR, "status": "active"})
        )
    ) as identity:
        adapter = PostgresTeamApi_2024141460331(
            "unused-test-dsn", tmp_path, a5_url="https://identity.test", identity_client=identity
        )
        monkeypatch.setattr(service.api, "revise_transcript", adapter.revise_transcript)
        response = patch_revision(service)

    body = response.json()
    assert connection.updates == 1  # No retry after either failure.
    assert body["saveConfirmed"] is False
    if rollback_fails:
        assert response.status_code == 502
        assert body["code"] == "REVISION_UNCONFIRMED"
        assert body["saveStatus"] == "unconfirmed" and body["writeOutcome"] == "unknown"
        assert not connection.rollback_completed
    else:
        assert response.status_code == 503
        assert body["code"] == "REVISION_HISTORY_INVALID"
        assert body["saveStatus"] == "failed" and body["writeOutcome"] == "rolled_back"
        assert "已回滚" in body["message"]
        assert connection.rollback_completed and boundary.text == original


@pytest.mark.parametrize(
    "status,code",
    [
        (503, "REVISION_AUTH_UNAVAILABLE"),
        (503, "REVISION_HISTORY_INVALID"),
        (400, "INVALID_RESPONSE"),
    ],
)
def test_remote_error_code_and_claimed_outcome_do_not_prove_no_write(status, code):
    boundary = RevisionBoundary()
    attempts = []

    def upstream(request):
        if request.method == "PATCH":
            attempts.append(request)
            # The write happened despite a server error. Do not trust its claimed outcome.
            boundary(request)
            return httpx.Response(
                status,
                json={"code": code, "writeOutcome": "rolled_back", "saveStatus": "failed"},
                headers={"X-Request-Id": "upstream-revision-failure"},
            )
        return boundary(request)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        )
        response = patch_revision(service)
    body = response.json()
    assert response.status_code == 502 and body["code"] == "REVISION_UNCONFIRMED"
    assert body["saveConfirmed"] is False and body["saveStatus"] == "unconfirmed"
    assert body["writeOutcome"] == "unknown"
    assert body["upstreamRequestId"] == "upstream-revision-failure"
    assert len(attempts) == 1
    assert boundary.text["textRaw"] == "climb four thousand"
    assert boundary.text["text"] == "climb five thousand" and boundary.text["revision"] == 2


def test_response_lost_after_commit_stays_unconfirmed_and_can_be_read_back(revision_case):
    service, boundary = revision_case
    boundary.mode = "lost_response"
    response = patch_revision(service)
    body = response.json()
    assert response.status_code == 502 and body["code"] == "REVISION_UNCONFIRMED"
    assert body["saveStatus"] == "unconfirmed" and body["writeOutcome"] == "unknown"
    assert len(boundary.patch_payloads) == 1
    reread = service.read_transcript_2024141460331(TID, include_history=True)
    assert reread["transcript"]["revision"] == 2
    assert reread["transcript"]["textRaw"] == "climb four thousand"
    assert reread["edits"][-1]["reason"] == PATCH["reason"]


def test_two_successive_revisions_keep_original_and_history_then_reject_stale_version(
    revision_case,
):
    service, boundary = revision_case
    for revision, text in [(1, "climb five thousand"), (2, "climb six thousand")]:
        result = service.revise_transcript_2024141460331(
            TID,
            text=text,
            expected_revision=revision,
            reason="核听数字",
            bearer_token="editor-token",
        )
        assert result["saveConfirmed"] is True
        assert result["transcript"]["revision"] == revision + 1
        assert result["transcript"]["textRaw"] == "climb four thousand"
        assert [edit["revision"] for edit in result["edits"]] == list(range(2, revision + 2))
    before = deepcopy(boundary.text)
    response = patch_revision(service)  # expectedRevision=1 is now stale.
    assert response.status_code == 409 and response.json()["code"] == "REVISION_CONFLICT"
    assert boundary.text == before and len(boundary.patch_payloads) == 2
