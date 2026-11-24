"""Historical HTTP shapes exercise the production adapter and revision service."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from support_2024141460331 import Engine, SharedServer
from test_revision_service_2024141460331 import RID, TID, RevisionBoundary

from a4_speech.api_2024141460331 import create_app_2024141460331
from a4_speech.client_2024141460331 import ApiError, TeamApi
from a4_speech.service_2024141460331 import (
    ProcessingError_2024141460331,
    SpeechService_2024141460331,
)


class HistoricalRevisionBoundary(RevisionBoundary):
    def __init__(self):
        super().__init__()
        self.review_states = "missing"

    def __call__(self, request):
        if request.method == "PATCH":
            payload = json.loads(request.content)
            if "text" not in payload:
                return httpx.Response(422, json={"code": "TEXT_REQUIRED"})
        response = super().__call__(request)
        if request.method == "PATCH" and response.status_code == 200:
            for edit in self.text["edits"]:
                if self.review_states == "missing":
                    edit.pop("beforeReviewStatus", None)
                    edit.pop("afterReviewStatus", None)
                elif self.review_states == "wrong":
                    edit["beforeReviewStatus"] = "rejected"
                elif self.review_states == "null":
                    edit["beforeReviewStatus"] = None
                elif self.review_states == "partial":
                    edit.pop("afterReviewStatus", None)
        return response


@pytest.fixture
def historical_revision():
    boundary = HistoricalRevisionBoundary()
    with httpx.Client(
        transport=httpx.MockTransport(boundary), headers={"X-Api-Key": "service-key"}
    ) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        yield SpeechService_2024141460331(api), boundary


def test_review_only_edit_sends_observed_text_to_historical_required_text_endpoint(
    historical_revision,
):
    service, boundary = historical_revision
    result = service.revise_transcript_2024141460331(
        TID,
        review_status="accepted",
        expected_revision=1,
        reason="人工核听",
        bearer_token="user-token",
    )
    assert result["revisionConfirmed"] is True
    assert result["transcript"]["textRaw"] == "climb four thousand"
    assert boundary.patch_payloads == [
        {
            "text": "climb four thousand",
            "reviewStatus": "accepted",
            "expectedRevision": 1,
            "reason": "人工核听",
        }
    ]
    assert "x-api-key" not in boundary.patch_headers[0]
    assert boundary.patch_headers[0]["authorization"] == "Bearer user-token"


def test_text_edit_confirms_real_old_history_without_inventing_review_states(historical_revision):
    service, boundary = historical_revision
    result = service.revise_transcript_2024141460331(
        TID,
        text="climb five thousand",
        expected_revision=1,
        reason="核对数字",
        bearer_token="user-token",
    )
    assert result["revisionConfirmed"] is True
    assert result["reviewHistoryStatus"] == "not_exposed"
    assert result["edits"] == boundary.text["edits"]
    assert "beforeReviewStatus" not in result["edits"][0]
    assert "afterReviewStatus" not in result["edits"][0]
    assert result["transcript"]["revision"] == 2


@pytest.mark.parametrize("shape", ["wrong", "null"])
def test_optional_history_fields_are_checked_when_the_upstream_exposes_them(
    historical_revision, shape
):
    service, boundary = historical_revision
    boundary.review_states = shape
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_UNCONFIRMED"):
        service.revise_transcript_2024141460331(
            TID, text="new", expected_revision=1, reason="核对", bearer_token="user-token"
        )
    assert len(boundary.patch_payloads) == 1


def test_compatibility_does_not_replace_a_stale_client_revision(historical_revision):
    service, boundary = historical_revision
    with pytest.raises(ProcessingError_2024141460331, match="REVISION_CONFLICT"):
        service.revise_transcript_2024141460331(
            TID,
            review_status="accepted",
            expected_revision=2,
            reason="核对",
            bearer_token="user-token",
        )
    assert boundary.patch_payloads == []


@pytest.mark.parametrize("shape,status", [("present", "recorded"), ("partial", "partial")])
def test_review_history_availability_describes_the_fields_actually_returned(
    historical_revision, shape, status
):
    service, boundary = historical_revision
    boundary.review_states = shape
    result = service.revise_transcript_2024141460331(
        TID, text="new", expected_revision=1, reason="核对", bearer_token="user-token"
    )
    assert result["reviewHistoryStatus"] == status
    assert result["edits"] == boundary.text["edits"]


@pytest.mark.parametrize("upstream_mark", [None, "recorded"])
def test_legacy_http_edit_returns_honest_history_availability(historical_revision, upstream_mark):
    service, boundary = historical_revision
    if upstream_mark is not None:
        boundary.text["reviewHistoryStatus"] = upstream_mark
    app = create_app_2024141460331(service=service, api_key="worker-key")
    with TestClient(app) as http:
        response = http.patch(
            f"/api/v1/a4/transcripts/{TID}",
            headers={"X-Api-Key": "worker-key", "Authorization": "Bearer user-token"},
            json={"reviewStatus": "accepted", "expectedRevision": 1, "reason": "人工核听"},
        )
    assert response.status_code == 200
    assert response.json()["reviewHistoryStatus"] == "not_exposed"
    assert response.json()["revision"] == 2
    assert "beforeReviewStatus" not in response.json()["edits"][0]


class HistoricalPipelineBoundary(SharedServer):
    def __init__(self):
        super().__init__()
        self.revisions = HistoricalRevisionBoundary()

    def __call__(self, request):
        if request.method == "PATCH":
            self.revisions.text = next(iter(self.texts.values()))
            return self.revisions(request)
        response = super().__call__(request)
        if request.method == "POST" and request.url.path.endswith("/complete"):
            for text in self.texts.values():
                text["id"] = TID
                text.setdefault("edits", [])
        return response


@pytest.mark.parametrize("fail_at,expected_status", [(None, "succeeded"), (2, "partial")])
def test_process_save_read_and_edit_pipeline_with_historical_http_shapes(
    tmp_path, fail_at, expected_status
):
    boundary = HistoricalPipelineBoundary()
    spans = [(0, 1000)] if fail_at is None else [(0, 1000), (1000, 2000)]
    engine = Engine(spans=spans, fail_at=fail_at)
    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path,
        )
        processed = service.transcribe_2024141460331(
            RID, "historical-pipeline", business_task_id="business-compat"
        )
        assert processed["status"] == expected_status
        assert not boundary.texts
        calls_after_processing = engine.calls
        saved = service.save_result_2024141460331(processed)
        assert saved["saveConfirmed"] is True
        assert saved["status"] == expected_status
        detail = service.read_transcript_2024141460331(TID, include_history=True)
        assert detail["processingStatus"] == expected_status
        assert detail["transcript"]["textRaw"] == "flight 1 cleared"
        revised = service.revise_transcript_2024141460331(
            TID,
            text="flight one cleared",
            expected_revision=1,
            reason="核对呼号",
            bearer_token="user-token",
        )
        assert revised["revisionConfirmed"] is True
        assert revised["reviewHistoryStatus"] == "not_exposed"
        assert revised["processingStatus"] == expected_status
        assert revised["transcript"]["textRaw"] == "flight 1 cleared"
        assert revised["transcript"]["text"] == "flight one cleared"
        assert engine.calls == calls_after_processing
        assert len(boundary.revisions.patch_payloads) == 1


def test_jobs_follow_actual_smaller_page_size_without_losing_source_candidates():
    requests = []
    jobs = [
        {
            "id": f"job-{i}",
            "kind": "asr",
            "status": "succeeded",
            "recordingId": RID,
            "submittedAt": "2026-10-24T00:00:00Z",
        }
        for i in range(3)
    ]

    def handle(request):
        page = int(request.url.params["page"])
        requests.append((page, int(request.url.params["pageSize"])))
        return httpx.Response(
            200,
            json={
                "items": jobs[(page - 1) * 2 : page * 2],
                "page": {
                    "page": page,
                    "pageSize": 2,
                    "total": 3,
                    "totalPages": 2,
                    "hasNext": page == 1,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        assert api.list_jobs(RID) == jobs
    assert requests == [(1, 100), (2, 2)]


def test_jobs_reject_page_size_changes_after_accepting_the_first_actual_size():
    requests = []

    def handle(request):
        page = int(request.url.params["page"])
        requests.append(page)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": f"job-{page}",
                        "kind": "asr",
                        "status": "succeeded",
                        "recordingId": RID,
                        "submittedAt": "2026-10-24T00:00:00Z",
                    }
                ],
                "page": {
                    "page": page,
                    "pageSize": 1 if page == 1 else 2,
                    "total": 2,
                    "totalPages": 2,
                    "hasNext": page == 1,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        with pytest.raises(ApiError, match="INVALID_RESPONSE"):
            api.list_jobs(RID)
    assert requests == [1, 2]
