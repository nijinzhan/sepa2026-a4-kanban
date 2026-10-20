"""Strict HTTP adapter for the existing team A3/A4 OpenAPI contracts.

Authentication and transport lifetime belong to the injected httpx client.
Writes are sent once: recovery from an uncertain response belongs to the caller.
"""

import re
from typing import Any
from urllib.parse import quote

import httpx

JOB_STATUSES = {"queued", "running", "succeeded", "partial", "failed", "canceled"}
JOB_KINDS = {"vad", "asr", "postprocess", "batch", "rescore"}


def _error_code(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        return value
    return None


def _request_id(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        return value
    return None


class ApiError(Exception):
    """Failure with structured identifiers, never request credentials or response body."""

    def __init__(self, status: int, code: str, request_id: str | None = None) -> None:
        self.status = status
        self.code = _error_code(code) or "HTTP_ERROR"
        self.request_id = _request_id(request_id)
        super().__init__(f"Team API failed: status={status}, code={self.code}")


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _enum(value: Any, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


def _identifier(value: str) -> str:
    if not _string(value) or value in {".", ".."}:
        raise ValueError("A nonempty identifier is required")
    return quote(value, safe="")


class TeamApi:
    """Calls only declared team endpoints; creates no shared schemas or tables."""

    def __init__(self, client: httpx.Client, a3_url: str, a4_url: str) -> None:
        self.client = client
        self.a3_url = a3_url.rstrip("/")
        self.a4_url = a4_url.rstrip("/")

    @staticmethod
    def _invalid(response: httpx.Response) -> ApiError:
        return ApiError(
            response.status_code, "INVALID_RESPONSE", response.headers.get("X-Request-Id")
        )

    def _send(
        self,
        method: str,
        url: str,
        *,
        expected: int = 200,
        missing_ok: bool = False,
        full_audio: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            request = self.client.build_request(method, url, **kwargs)
            if full_audio:
                request.headers.pop("Range", None)
            response = self.client.send(request, follow_redirects=False)
        except httpx.RequestError:
            raise ApiError(0, "TRANSPORT_ERROR") from None
        if full_audio and response.status_code == 206:
            raise self._invalid(response)
        if response.status_code == expected or (missing_ok and response.status_code == 404):
            return response
        code = "HTTP_ERROR"
        request_id = _request_id(response.headers.get("X-Request-Id"))
        try:
            problem = response.json()
        except ValueError:
            problem = None
        if isinstance(problem, dict):
            code = _error_code(problem.get("code")) or code
            request_id = _request_id(problem.get("requestId")) or request_id
        raise ApiError(response.status_code, code, request_id)

    def _object(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError:
            raise self._invalid(response) from None
        if not isinstance(data, dict):
            raise self._invalid(response)
        return data

    def _job(self, response: httpx.Response, expected_id: str | None = None) -> dict[str, Any]:
        return self._job_data(response, self._object(response), expected_id)

    def _job_data(
        self, response: httpx.Response, data: Any, expected_id: str | None = None
    ) -> dict[str, Any]:
        if (
            not isinstance(data, dict)
            or not _string(data.get("id"))
            or (expected_id is not None and data["id"] != expected_id)
            or not _enum(data.get("kind"), JOB_KINDS)
            or not _enum(data.get("status"), JOB_STATUSES)
            or not _string(data.get("submittedAt"))
            or any(key in data and not isinstance(data[key], dict) for key in ("params", "stats"))
        ):
            raise self._invalid(response)
        for key in ("processedSegments", "failedSegments", "totalSegments"):
            if key in data and not _integer(data[key]):
                raise self._invalid(response)
        return data

    def _transcript(
        self, response: httpx.Response, lookup: str, expected_id: str
    ) -> dict[str, Any]:
        data = self._object(response)
        if (
            not _string(data.get("id"))
            or not _string(data.get("segmentId"))
            or data.get(lookup) != expected_id
            or not isinstance(data.get("text"), str)
            or not _integer(data.get("revision"), 1)
            or not _enum(
                data.get("reviewStatus"),
                {"unreviewed", "accepted", "rejected", "needs_review", "skipped"},
            )
        ):
            raise self._invalid(response)
        return data

    def get_recording(self, recording_id: str) -> dict[str, Any]:
        response = self._send("GET", f"{self.a3_url}/recordings/{_identifier(recording_id)}")
        data = self._object(response)
        if (
            data.get("id") != recording_id
            or not _string(data.get("channelId"))
            or not _string(data.get("startedAt"))
            or not _string(data.get("storageState"))
        ):
            raise self._invalid(response)
        return data

    def get_audio(self, recording_id: str) -> bytes:
        response = self._send(
            "GET",
            f"{self.a3_url}/recordings/{_identifier(recording_id)}/content",
            full_audio=True,
            missing_ok=False,
        )
        if not response.content or "Content-Range" in response.headers:
            raise self._invalid(response)
        return response.content

    def get_engines(self, kind: str) -> list[dict[str, Any]]:
        """Return enabled VAD/ASR registry entries exactly as supplied by the team."""
        if not _enum(kind, {"vad", "asr"}):
            raise ValueError("Engine kind must be vad or asr")
        response = self._send(
            "GET",
            f"{self.a4_url}/engines",
            params={"kind": kind, "enabledOnly": "true"},
        )
        data = self._object(response)
        items = data.get("items")
        if not isinstance(items, list):
            raise self._invalid(response)
        codes: set[str] = set()
        nullable_strings = (
            "engineFamily",
            "version",
            "modelName",
            "runtime",
            "language",
            "endpointUrl",
            "notes",
        )
        for item in items:
            if (
                not isinstance(item, dict)
                or not _string(item.get("code"))
                or item["code"] in codes
                or not _string(item.get("name"))
                or item.get("kind") != kind
                or item.get("isEnabled") is not True
                or not _enum(item.get("device"), {"cpu", "cuda", "npu"})
                or not _enum(item.get("deployMode"), {"sidecar", "remote", "batch", "embedded"})
            ):
                raise self._invalid(response)
            for field in nullable_strings:
                if field in item and item[field] is not None and not isinstance(item[field], str):
                    raise self._invalid(response)
            for field in ("batchSize", "timeoutS"):
                if field in item and item[field] is not None and type(item[field]) is not int:
                    raise self._invalid(response)
            if "config" in item and not isinstance(item["config"], dict):
                raise self._invalid(response)
            codes.add(item["code"])
        return items

    def list_segments(self, recording_id: str) -> list[dict[str, Any]]:
        url = f"{self.a3_url}/recordings/{_identifier(recording_id)}/segments"
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        params: dict[str, str | int] = {"onlySpeech": "false"}
        expected_page = 1
        fixed_meta: tuple[int, int, int] | None = None
        while True:
            response = self._send("GET", url, params=params)
            data = self._object(response)
            items = data.get("items")
            if data.get("recordingId") != recording_id or not isinstance(items, list):
                raise self._invalid(response)
            for item in items:
                if (
                    not isinstance(item, dict)
                    or not _string(item.get("id"))
                    or item["id"] in seen
                    or item.get("recordingId") != recording_id
                    or not _integer(item.get("startMs"))
                    or not _integer(item.get("endMs"), 1)
                ):
                    raise self._invalid(response)
                seen.add(item["id"])
                rows.append(item)
            if "page" not in data:
                if expected_page != 1:
                    raise self._invalid(response)
                return rows
            page = data["page"]
            if (
                not isinstance(page, dict)
                or not _integer(page.get("page"), 1)
                or page["page"] != expected_page
                or not _integer(page.get("pageSize"), 1)
                or not _integer(page.get("total"))
                or not _integer(page.get("totalPages"))
                or page["totalPages"] > 10000
                or type(page.get("hasNext")) is not bool
                or page["hasNext"] != (page["page"] < page["totalPages"])
            ):
                raise self._invalid(response)
            meta = (page["pageSize"], page["total"], page["totalPages"])
            if fixed_meta is not None and meta != fixed_meta:
                raise self._invalid(response)
            fixed_meta = meta
            if not page["hasNext"]:
                if len(rows) != page["total"]:
                    raise self._invalid(response)
                return rows
            if not items:
                raise self._invalid(response)
            expected_page += 1
            params = {"onlySpeech": "false", "page": expected_page, "pageSize": page["pageSize"]}

    def add_segments(
        self, recording_id: str, segments: list[dict[str, Any]], engine_code: str
    ) -> dict[str, Any]:
        if not isinstance(segments, list) or not 1 <= len(segments) <= 2000:
            raise ValueError("Segment batch must contain 1..2000 segments")
        if not _string(engine_code):
            raise ValueError("An engine code is required")
        response = self._send(
            "POST",
            f"{self.a3_url}/recordings/{_identifier(recording_id)}/segments",
            json={"segments": segments, "engineCode": engine_code, "replaceAll": False},
        )
        data = self._object(response)
        if (
            data.get("recordingId") != recording_id
            or not _integer(data.get("inserted"))
            or not _integer(data.get("skipped"))
            or data["inserted"] + data["skipped"] != len(segments)
        ):
            raise self._invalid(response)
        return data

    def create_job(
        self,
        segment_ids: list[str],
        engine_code: str,
        params: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(segment_ids, list)
            or not 1 <= len(segment_ids) <= 500
            or any(not _string(item) for item in segment_ids)
            or len(set(segment_ids)) != len(segment_ids)
        ):
            raise ValueError("Task creation requires 1..500 distinct segment identifiers")
        if (
            not _string(idempotency_key)
            or len(idempotency_key) > 64
            or not idempotency_key.isascii()
            or not idempotency_key.isprintable()
        ):
            raise ValueError("Idempotency-Key must contain 1..64 printable ASCII characters")
        if not _string(engine_code) or not isinstance(params, dict):
            raise ValueError("An engine code and object parameters are required")
        response = self._send(
            "POST",
            f"{self.a4_url}/jobs",
            expected=202,
            headers={"Idempotency-Key": idempotency_key},
            json={
                "kind": "asr",
                "segmentIds": segment_ids,
                "engineCode": engine_code,
                "params": params,
                "origin": "api",
            },
        )
        return self._job(response)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._job(self._send("GET", f"{self.a4_url}/jobs/{_identifier(job_id)}"), job_id)

    def list_jobs(self, recording_id: str) -> list[dict[str, Any]]:
        """Read every ASR job page, rejecting incomplete or unbound source evidence."""
        _identifier(recording_id)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        expected_page = 1
        fixed_meta: tuple[int, int] | None = None
        while True:
            response = self._send(
                "GET",
                f"{self.a4_url}/jobs",
                params={
                    "recordingId": recording_id,
                    "kind": "asr",
                    "page": expected_page,
                    "pageSize": 100,
                },
            )
            data = self._object(response)
            items, page = data.get("items"), data.get("page")
            if (
                not isinstance(items, list)
                or not isinstance(page, dict)
                or not _integer(page.get("page"), 1)
                or page["page"] != expected_page
                or not _integer(page.get("pageSize"), 1)
                or page["pageSize"] != 100
                or not _integer(page.get("total"))
                or not _integer(page.get("totalPages"))
                or type(page.get("hasNext")) is not bool
            ):
                raise self._invalid(response)
            total, total_pages = page["total"], page["totalPages"]
            if (
                (total == 0 and total_pages not in {0, 1})
                or (total > 0 and total_pages != (total + 99) // 100)
                or (total_pages > 0 and expected_page > total_pages)
                or page["hasNext"] != (expected_page < total_pages)
                or len(items) != min(100, max(0, total - (expected_page - 1) * 100))
            ):
                raise self._invalid(response)
            meta = (total, total_pages)
            if fixed_meta is not None and meta != fixed_meta:
                raise self._invalid(response)
            fixed_meta = meta
            for item in items:
                item = self._job_data(response, item)
                if (
                    item["id"] in seen
                    or item["kind"] != "asr"
                    or item.get("recordingId") != recording_id
                ):
                    raise self._invalid(response)
                seen.add(item["id"])
                rows.append(item)
            if not page["hasNext"]:
                if len(rows) != total:
                    raise self._invalid(response)
                return rows
            expected_page += 1

    def complete_job(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not _enum(payload.get("status"), JOB_STATUSES):
            raise ValueError("A declared job status is required")
        for field in ("processedSegments", "failedSegments"):
            if field in payload and not _integer(payload[field]):
                raise ValueError("Task counters must be nonnegative integers")
        return self._job(
            self._send("POST", f"{self.a4_url}/jobs/{_identifier(job_id)}/complete", json=payload),
            job_id,
        )

    def get_transcript(self, segment_id: str) -> dict[str, Any] | None:
        response = self._send(
            "GET",
            f"{self.a4_url}/segments/{_identifier(segment_id)}/transcript",
            missing_ok=True,
        )
        if response.status_code == 404:
            return None
        return self._transcript(response, "segmentId", segment_id)

    def get_transcript_by_id(self, transcript_id: str) -> dict[str, Any]:
        response = self._send("GET", f"{self.a4_url}/transcripts/{_identifier(transcript_id)}")
        return self._transcript(response, "id", transcript_id)
