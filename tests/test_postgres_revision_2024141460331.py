"""Real storage, IAM permissions, concurrency and history; HTTP identity boundary double."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
import test_postgres_2024141460331 as pg_support
from test_postgres_2024141460331 import (
    job_2024141460331,
    payload_2024141460331,
)

from a4_speech.client_2024141460331 import ApiError
from a4_speech.postgres_2024141460331 import PostgresTeamApi_2024141460331

pg_case = pg_support.pg_2024141460331


@pytest.fixture
def editable(pg_case):
    import psycopg

    api, _, tag, dsn = pg_case
    job, ids, _ = job_2024141460331(pg_case)
    api.complete_job(job["id"], payload_2024141460331(ids))
    transcript = api.get_transcript(ids[0])
    users = [str(uuid4()), str(uuid4()), str(uuid4())]
    with psycopg.connect(dsn) as db:
        for i, uid in enumerate(users):
            db.execute(
                "INSERT INTO iam.user_account(id,username,display_name,password_hash,status) "
                "VALUES (%s,%s,%s,'unused-test-password','active')",
                (uid, "u" + tag[:28] + str(i), "test editor"),
            )
        role = db.execute(
            "INSERT INTO iam.role(code,name) VALUES (%s,%s) RETURNING id", ("r" + tag[:30], tag)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO iam.role_permission(role_id,permission_id) "
            "SELECT %s,id FROM iam.permission WHERE code='a4:transcript:edit'",
            (role,),
        )
        for uid in users[:2]:
            db.execute("INSERT INTO iam.user_role(user_id,role_id) VALUES (%s,%s)", (uid, role))

    def identify(request):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if token not in {"editor1", "editor2", "viewer"}:
            return httpx.Response(401)
        uid = users[{"editor1": 0, "editor2": 1, "viewer": 2}[token]]
        return httpx.Response(200, json={"id": uid, "status": "active"})

    with httpx.Client(transport=httpx.MockTransport(identify)) as http:
        adapter = PostgresTeamApi_2024141460331(dsn, api.audio_root)
        adapter.a5_url = "https://iam.test/api/v1/a5"
        adapter.identity_client = http
        try:
            yield adapter, transcript, users, dsn
        finally:
            with psycopg.connect(dsn) as db:
                db.execute(
                    "DELETE FROM iam.audit_log WHERE actor_user_id=ANY(%s::uuid[])", (users,)
                )
                db.execute(
                    "UPDATE asr.transcript SET edited_by=NULL WHERE edited_by=ANY(%s::uuid[])",
                    (users,),
                )
                db.execute(
                    "UPDATE asr.transcript_edit SET editor_user_id=NULL "
                    "WHERE editor_user_id=ANY(%s::uuid[])",
                    (users,),
                )
                db.execute("DELETE FROM iam.user_account WHERE id=ANY(%s::uuid[])", (users,))
                db.execute("DELETE FROM iam.role WHERE id=%s", (role,))


def revise(api, tid, revision=1, token="editor1", **changes):
    return api.revise_transcript(
        tid, expected_revision=revision, reason="verified edit", bearer_token=token, **changes
    )


def test_authenticated_text_review_noop_and_restart_history(editable):
    api, initial, users, dsn = editable
    tid = initial["id"]
    assert revise(api, tid, text="corrected call")["revision"] == 2
    assert revise(api, tid, revision=2, token="editor2", review_status="accepted")["revision"] == 3
    same = revise(
        api, tid, revision=3, token="editor2", text="corrected call", review_status="accepted"
    )
    assert same["revision"] == 3 and same["changed"] is False
    fresh = PostgresTeamApi_2024141460331(dsn, api.audio_root).get_transcript_by_id(tid)
    assert fresh["textRaw"] == "cleared 0" and fresh["text"] == "corrected call"
    assert fresh["reviewStatus"] == "accepted"
    assert [e["revision"] for e in fresh["edits"]] == [2, 3]
    assert [e["editorUserId"] for e in fresh["edits"]] == users[:2]
    assert fresh["edits"][0]["reason"] == "verified edit"
    with pytest.raises(ApiError) as caught:
        revise(api, tid, revision=2, text="stale")
    assert caught.value.status == 409


@pytest.mark.parametrize("token,status", [(None, 401), ("invalid", 401), ("viewer", 403)])
def test_denied_identity_does_not_change_text(editable, token, status):
    api, initial, _, _ = editable
    with pytest.raises(ApiError) as caught:
        revise(api, initial["id"], token=token, text="unauthorized")
    assert caught.value.status == status
    assert api.get_transcript_by_id(initial["id"]) == initial


def test_concurrent_editors_cannot_silently_overwrite(editable):
    api, initial, _, _ = editable

    def edit(token):
        try:
            return revise(api, initial["id"], token=token, text=token)["revision"]
        except ApiError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, ["editor1", "editor2"]))
    assert sorted(results) == [2, 409]
    assert len(api.get_transcript_by_id(initial["id"])["edits"]) == 1


def test_audit_failure_rolls_back_text_and_history(editable):
    api, initial, _, _ = editable
    # A trigger is not installed: a transaction wrapper injects failure after real SQL.
    from contextlib import contextmanager

    import psycopg

    original = api._transaction_2024141460331

    @contextmanager
    def fail_commit():
        with original() as db:
            yield db
            raise psycopg.OperationalError("simulated commit failure")

    api._transaction_2024141460331 = fail_commit
    with pytest.raises(ApiError):
        revise(api, initial["id"], text="must roll back")
    api._transaction_2024141460331 = original
    assert api.get_transcript_by_id(initial["id"]) == initial


def test_disabled_user_is_rejected_even_if_identity_cache_says_active(editable):
    api, initial, users, dsn = editable
    import psycopg

    with psycopg.connect(dsn) as db:
        db.execute("UPDATE iam.user_account SET status='disabled' WHERE id=%s", (users[0],))
    with pytest.raises(ApiError) as caught:
        revise(api, initial["id"], text="must not change")
    assert caught.value.status == 403
    assert api.get_transcript_by_id(initial["id"]) == initial


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(302, headers={"Location": "https://other.test"}),
        httpx.Response(200, json={"id": "not-a-uuid", "status": "active"}),
        httpx.Response(500),
    ],
)
def test_untrusted_identity_response_cannot_write(editable, response):
    api, initial, _, _ = editable
    with httpx.Client(transport=httpx.MockTransport(lambda request: response)) as http:
        api.identity_client = http
        with pytest.raises(ApiError) as caught:
            revise(api, initial["id"], text="must not change")
    assert caught.value.status == 503
    assert api.get_transcript_by_id(initial["id"]) == initial


def test_postgres_task_creation_honors_limit_above_500(pg_case):
    api, rid, tag, dsn = pg_case
    adapter = PostgresTeamApi_2024141460331(dsn, api.audio_root, max_segments=501)
    adapter.add_segments(
        rid, [{"startMs": i * 5, "endMs": (i + 1) * 5} for i in range(501)], tag + "vad"
    )
    ids = [s["id"] for s in adapter.list_segments(rid)]
    params = {"a4_10_20": {"recordingId": rid, "targetSegmentIds": ids, "maxSegments": 501}}
    result = adapter.create_job(ids, tag + "asr", params, "limit-501")
    assert len(result["params"]["a4_10_20"]["targetSegmentIds"]) == 501


def test_history_validation_failure_reports_rollback_and_preserves_real_storage(editable):
    from contextlib import contextmanager

    from a4_speech.service_2024141460331 import (
        ProcessingError_2024141460331,
        SpeechService_2024141460331,
    )

    api, initial, _, _ = editable
    original = api._transaction_2024141460331

    class HistoryHidden:
        def __init__(self, db):
            self.db = db

        def execute(self, sql, params=None):
            cursor = self.db.execute(sql, params)
            if sql.startswith("SELECT * FROM asr.transcript_edit WHERE transcript_id=%s AND"):
                return self
            return cursor

        def fetchall(self):
            return []

    @contextmanager
    def hide_new_history():
        with original() as db:
            yield HistoryHidden(db)

    api._transaction_2024141460331 = hide_new_history
    try:
        with pytest.raises(ProcessingError_2024141460331) as caught:
            SpeechService_2024141460331(api).revise_transcript_2024141460331(
                initial["id"],
                text="must roll back",
                expected_revision=1,
                reason="verify missing history",
                bearer_token="editor1",
            )
        assert caught.value.code == "REVISION_HISTORY_INVALID"
        assert caught.value.as_dict_2024141460331()["writeOutcome"] == "rolled_back"
    finally:
        api._transaction_2024141460331 = original
    assert api.get_transcript_by_id(initial["id"]) == initial
