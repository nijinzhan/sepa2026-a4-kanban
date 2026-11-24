"""Revision boundaries exercise the real adapter/service, never a substitute service."""

import json
from copy import deepcopy

import httpx
import pytest

from a4_speech.client_2024141460331 import ApiError, TeamApi
from a4_speech.service_2024141460331 import (
    ProcessingError_2024141460331,
    SpeechService_2024141460331,
)

TID = "00000000-0000-4000-8000-000000000003"
SID = "00000000-0000-4000-8000-000000000002"
RID = "00000000-0000-4000-8000-000000000001"


class RevisionBoundary:
    def __init__(self):
        self.text = {
            "id": TID,
            "segmentId": SID,
            "recordingId": RID,
            "text": "climb four thousand",
            "textRaw": "climb four thousand",
            "revision": 1,
            "reviewStatus": "unreviewed",
            "isEdited": False,
            "startMs": 0,
            "endMs": 1000,
            "startAt": "2026-10-17T00:00:00Z",
            "endAt": "2026-10-17T00:00:01Z",
            "language": "en",
            "edits": [],
        }
        self.patch_payloads = []
        self.patch_headers = []
        self.mode = "normal"

    def __call__(self, request):
        if request.url.path == f"/api/v1/a4/transcripts/{TID}":
            if request.method == "GET":
                return httpx.Response(200, json=self.text)
            assert request.method == "PATCH"
            payload = json.loads(request.content)
            self.patch_payloads.append(payload)
            self.patch_headers.append(dict(request.headers))
            if self.mode == "forbidden":
                return httpx.Response(403, json={"code": "FORBIDDEN"})
            if self.mode == "conflict" or payload["expectedRevision"] != self.text["revision"]:
                return httpx.Response(409, json={"code": "VERSION_CONFLICT"})
            if (
                payload.get("text", self.text["text"]) == self.text["text"]
                and payload.get("reviewStatus", self.text["reviewStatus"])
                == self.text["reviewStatus"]
            ):
                return httpx.Response(
                    200,
                    json={
                        "transcriptId": TID,
                        "revision": self.text["revision"],
                        "reviewStatus": self.text["reviewStatus"],
                        "isEdited": self.text["isEdited"],
                    },
                )
            old = deepcopy(self.text)
            self.text["text"] = payload.get("text", self.text["text"])
            self.text["reviewStatus"] = payload.get("reviewStatus", self.text["reviewStatus"])
            self.text["revision"] += 1
            self.text["isEdited"] = True
            self.text["edits"].append(
                {
                    "revision": self.text["revision"],
                    "beforeText": old["text"],
                    "afterText": self.text["text"],
                    "beforeReviewStatus": old["reviewStatus"],
                    "afterReviewStatus": self.text["reviewStatus"],
                    "editorUserId": "00000000-0000-4000-8000-000000000004",
                    "reason": payload["reason"],
                    "editedAt": "2026-10-24T02:00:00Z",
                }
            )
            if self.mode == "lost_response":
                raise httpx.ReadTimeout("response lost", request=request)
            if self.mode == "corrupt_raw":
                self.text["textRaw"] = "overwritten original"
            if self.mode == "missing_history":
                self.text["edits"] = []
            return httpx.Response(
                200,
                json={
                    "transcriptId": TID,
                    "revision": self.text["revision"],
                    "reviewStatus": self.text["reviewStatus"],
                    "isEdited": True,
                },
            )
        if request.url.path == f"/api/v1/a3/recordings/{RID}":
            return httpx.Response(
                200,
                json={
                    "id": RID,
                    "channelId": "channel-1",
                    "startedAt": "2026-10-17T00:00:00Z",
                    "durationMs": 1000,
                    "storageState": "verified",
                },
            )
        if request.url.path == f"/api/v1/a3/recordings/{RID}/segments":
            return httpx.Response(
                200,
                json={"recordingId": RID, "items": [{**self.text, "id": SID}]},
            )
        if request.url.path == "/api/v1/a4/jobs":
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "page": {
                        "page": 1,
                        "pageSize": 100,
                        "total": 0,
                        "totalPages": 0,
                        "hasNext": False,
                    },
                },
            )
        raise AssertionError(f"Unexpected boundary {request.method} {request.url.path}")


@pytest.fixture
def setup_revision():
    boundary = RevisionBoundary()
    with httpx.Client(
        transport=httpx.MockTransport(boundary), headers={"X-Api-Key": "service-key"}
    ) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        yield SpeechService_2024141460331(api), boundary


def test_revision_preserves_raw_and_returns_verified_edit_history(setup_revision):
    service, boundary = setup_revision
    result = service.revise_transcript_2024141460331(
        TID,
        text="climb five thousand",
        expected_revision=1,
        reason="人工核听数字",
        bearer_token="user-token",
    )
    assert result["saveConfirmed"] is True
    assert result["transcript"]["text"] == "climb five thousand"
    assert result["transcript"]["textRaw"] == "climb four thousand"
    assert result["transcript"]["revision"] == 2
    assert result["edits"][0]["reason"] == "人工核听数字"
    assert result["sourceJobLinkVerified"] is False
    assert boundary.patch_payloads == [
        {"text": "climb five thousand", "expectedRevision": 1, "reason": "人工核听数字"}
    ]
    assert boundary.patch_headers[0]["authorization"] == "Bearer user-token"
    assert "x-api-key" not in boundary.patch_headers[0]


def test_review_only_revision_advances_version_and_records_state_transition(setup_revision):
    service, _ = setup_revision
    result = service.revise_transcript_2024141460331(
        TID,
        review_status="accepted",
        expected_revision=1,
        reason="核听通过",
        bearer_token="user-token",
    )
    assert result["transcript"]["revision"] == 2
    assert result["edits"][0]["beforeReviewStatus"] == "unreviewed"
    assert result["edits"][0]["afterReviewStatus"] == "accepted"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"text": 9},
        {"text": "x" * 4001},
        {"text": "new", "expected_revision": True},
        {"text": "new", "expected_revision": 0},
        {"text": "new", "reason": "  "},
        {"text": "new", "reason": "x" * 201},
        {"review_status": "invalid"},
    ],
)
def test_invalid_revision_never_writes(setup_revision, changes):
    service, boundary = setup_revision
    kwargs = {"expected_revision": 1, "reason": "核对", "bearer_token": "user-token", **changes}
    with pytest.raises(ProcessingError_2024141460331, match="INVALID_REVISION"):
        service.revise_transcript_2024141460331(TID, **kwargs)
    assert boundary.patch_payloads == []


def test_stale_revision_is_conflict_and_cannot_overwrite(setup_revision):
    service, boundary = setup_revision
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_CONFLICT") as caught:
        service.revise_transcript_2024141460331(
            TID, text="new", expected_revision=2, reason="核对", bearer_token="user-token"
        )
    assert caught.value.http_status == 409
    assert boundary.text["revision"] == 1
    assert boundary.patch_payloads == []


def test_atomic_upstream_conflict_is_not_retried(setup_revision):
    service, boundary = setup_revision
    boundary.mode = "conflict"
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_CONFLICT"):
        service.revise_transcript_2024141460331(
            TID, text="new", expected_revision=1, reason="核对", bearer_token="user-token"
        )
    assert len(boundary.patch_payloads) == 1
    assert boundary.text["revision"] == 1


def test_revision_forbidden_preserves_authorization_failure(setup_revision):
    service, boundary = setup_revision
    boundary.mode = "forbidden"
    with pytest.raises(ApiError) as caught:
        service.revise_transcript_2024141460331(
            TID, text="new", expected_revision=1, reason="核对", bearer_token="user-token"
        )
    assert caught.value.status == 403
    assert caught.value.code == "FORBIDDEN"


@pytest.mark.parametrize("mode", ["lost_response", "corrupt_raw", "missing_history"])
def test_uncertain_or_unverifiable_revision_is_not_reported_confirmed(setup_revision, mode):
    service, boundary = setup_revision
    boundary.mode = mode
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_UNCONFIRMED") as caught:
        service.revise_transcript_2024141460331(
            TID, text="new", expected_revision=1, reason="核对", bearer_token="user-token"
        )
    assert caught.value.as_dict_2024141460331()["saveConfirmed"] is False
    assert caught.value.as_dict_2024141460331()["saveStatus"] == "unconfirmed"
    assert caught.value.as_dict_2024141460331()["writeOutcome"] == "unknown"
    assert len(boundary.patch_payloads) == 1


def test_no_change_revision_does_not_create_duplicate_history(setup_revision):
    service, boundary = setup_revision
    result = service.revise_transcript_2024141460331(
        TID,
        text="climb four thousand",
        expected_revision=1,
        reason="核对",
        bearer_token="user-token",
    )
    assert result["revisionConfirmed"] is True and result["changed"] is False
    assert result["transcript"]["revision"] == 1 and result["edits"] == []
    assert len(boundary.patch_payloads) == 1  # Upstream must still authorize the user.


def test_revision_requires_user_identity_even_when_service_has_api_key(setup_revision):
    service, boundary = setup_revision
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_AUTH_REQUIRED") as caught:
        service.revise_transcript_2024141460331(TID, text="new", expected_revision=1, reason="核对")
    assert caught.value.http_status == 401
    assert boundary.patch_payloads == []


def test_explicit_history_read_rejects_missing_history_instead_of_inventing_empty(setup_revision):
    service, boundary = setup_revision
    boundary.text.pop("edits")
    with pytest.raises(ProcessingError_2024141460331, match="HISTORY_UNAVAILABLE"):
        service.read_transcript_2024141460331(TID, include_history=True)


def test_history_read_requires_original_text(setup_revision):
    service, boundary = setup_revision
    boundary.text["textRaw"] = None
    with pytest.raises(ProcessingError_2024141460331, match="READBACK_INVALID"):
        service.read_transcript_2024141460331(TID, include_history=True)


def test_verified_recording_is_eligible_for_recognition(tmp_path):
    from support_2024141460331 import Engine, SharedServer

    boundary, engine = SharedServer(), Engine()
    boundary.recording["storageState"] = "verified"
    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path,
        )
        result = service.transcribe_2024141460331(RID, "verified-recording")
    assert result["status"] == "succeeded"
    assert result["processedSegments"] == 3


@pytest.mark.parametrize("explicit_job", [True, False])
def test_returned_job_id_matches_recorded_job_link(tmp_path, explicit_job):
    from support_2024141460331 import Engine, SharedServer

    boundary = SharedServer()
    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(),
            receipt_dir=tmp_path,
        )
        saved = service.run_2024141460331(RID, "job-link", "business-1")
        text = saved["transcripts"][0]
        boundary.texts[text["segmentId"]]["jobId"] = saved["jobId"]
        result = service.read_transcript_2024141460331(
            text["id"], saved["jobId"] if explicit_job else None
        )
    assert result["sourceJobLinkVerified"] is True
    assert result["jobId"] == saved["jobId"]
    assert result["businessTaskForeignKeyVerified"] is False


def test_returned_job_id_cannot_be_overridden_with_another_job(tmp_path):
    from support_2024141460331 import Engine, SharedServer

    boundary = SharedServer()
    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(),
            receipt_dir=tmp_path,
        )
        saved = service.run_2024141460331(RID, "wrong-job-link", "business-1")
        text = saved["transcripts"][0]
        boundary.texts[text["segmentId"]]["jobId"] = "different-job"
        with pytest.raises(ProcessingError_2024141460331, match="HISTORY_ASSOCIATION"):
            service.read_transcript_2024141460331(text["id"], saved["jobId"])


def legacy_review_event():
    # V7 incremented only text changes but still logged status changes at revision 1.
    return {
        "revision": 1,
        "beforeText": None,
        "afterText": None,
        "beforeReviewStatus": "unreviewed",
        "afterReviewStatus": "accepted",
        "editorUserId": "00000000-0000-4000-8000-000000000004",
        "reason": "旧版人工核对",
        "editedAt": "2026-10-23T02:00:00Z",
    }


def test_legacy_v7_review_history_is_preserved_and_explicitly_marked(setup_revision):
    service, boundary = setup_revision
    boundary.text.update(reviewStatus="accepted", edits=[legacy_review_event()])
    result = service.read_transcript_2024141460331(TID, include_history=True)
    assert result["transcript"]["revision"] == 1
    assert result["edits"][0]["revision"] == 1
    assert result["editHistoryStatus"] == "legacy_recorded"
    assert result["legacyRevisionHistory"] is True


def test_new_cas_edit_after_legacy_history_can_be_confirmed_without_renumbering(setup_revision):
    service, boundary = setup_revision
    boundary.text.update(reviewStatus="accepted", edits=[legacy_review_event()])
    result = service.revise_transcript_2024141460331(
        TID,
        text="climb five thousand",
        expected_revision=1,
        reason="按新版本校正数字",
        bearer_token="user-token",
    )
    assert result["revisionConfirmed"] is True
    assert result["transcript"]["revision"] == 2
    assert [edit["revision"] for edit in result["edits"]] == [1, 2]
    assert result["edits"][0] == legacy_review_event()
    assert result["editHistoryStatus"] == "legacy_recorded"


@pytest.mark.parametrize("corruption", ["text_change", "no_state_change", "missing_revision_two"])
def test_legacy_exception_cannot_hide_text_edits_or_missing_revisions(setup_revision, corruption):
    service, boundary = setup_revision
    legacy = legacy_review_event()
    boundary.text.update(reviewStatus="accepted", edits=[legacy])
    if corruption == "text_change":
        legacy["afterText"] = "not a status-only legacy event"
    elif corruption == "no_state_change":
        legacy["afterReviewStatus"] = "unreviewed"
    else:
        boundary.text["revision"] = 3
        boundary.text["edits"].append({**legacy_review_event(), "revision": 3})
    with pytest.raises(ProcessingError_2024141460331, match="READBACK_INVALID"):
        service.read_transcript_2024141460331(TID, include_history=True)
