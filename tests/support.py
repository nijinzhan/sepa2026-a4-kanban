"""Stateful shared-HTTP boundary double; no production persistence is replaced."""

import io
import json
import wave
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx

RID = "00000000-0000-4000-8000-000000000001"
ORIGIN = datetime(2026, 10, 17, tzinfo=UTC)


def wav_bytes(frames=48000):
    data = io.BytesIO()
    with wave.open(data, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * frames)
    return data.getvalue()


class Engine:
    def __init__(self, spans=None, fail_at=None):
        self.spans = [(0, 1000), (1000, 2000), (2000, 3000)] if spans is None else spans
        self.fail_at = fail_at
        self.calls = 0
        self.version = "1"

    def snapshot(self):
        return {
            "schemaVersion": 1,
            "vad": {"engineCode": "silero", "version": "1"},
            "asr": {"engineCode": "sense", "version": self.version},
        }

    def verify_snapshot(self, expected):
        if expected != self.snapshot():
            raise ValueError("MODEL_MISMATCH")

    def prepare(self):
        pass

    def detect(self, audio):
        return self.spans

    def recognize(self, audio, start_ms, end_ms):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("inference failure")
        return f"flight {start_ms // 1000 + 1} cleared"


class SharedServer:
    def __init__(self):
        self.recording = {
            "id": RID,
            "channelId": str(uuid4()),
            "startedAt": ORIGIN.isoformat(),
            "storageState": "stored",
            "durationMs": 3000,
            "sampleRateHz": 16000,
            "audioChannels": 1,
        }
        self.segments = []
        self.jobs = {}
        self.texts = {}
        self.keys = {}
        self.writes = []
        self.complete_mode = "normal"
        self.break_time = False
        self.audio = wav_bytes()
        self.engines = {
            "vad": {
                "code": "silero",
                "name": "Silero",
                "kind": "vad",
                "isEnabled": True,
                "device": "cpu",
                "deployMode": "embedded",
                "version": "1",
            },
            "asr": {
                "code": "sense",
                "name": "SenseVoice",
                "kind": "asr",
                "isEnabled": True,
                "device": "cpu",
                "deployMode": "embedded",
                "version": "1",
            },
        }

    def add_segment(self, start, end, **extra):
        item = {
            "id": str(uuid4()),
            "recordingId": RID,
            "sequenceNo": len(self.segments) + 1,
            "startMs": start,
            "endMs": end,
            "durationMs": end - start,
            "startAt": (ORIGIN + timedelta(milliseconds=start)).isoformat(),
            "endAt": (ORIGIN + timedelta(milliseconds=end)).isoformat(),
            "isSpeech": True,
            "speakerRole": "unknown",
            "transcriptStatus": "none",
            "hasTranscript": False,
            **extra,
        }
        self.segments.append(item)
        return item

    def __call__(self, request):
        path = request.url.path
        if request.method == "POST":
            payload = json.loads(request.content)
            self.writes.append((path, payload))
        if path == f"/api/v1/a3/recordings/{RID}":
            return httpx.Response(200, json=self.recording)
        if path.endswith("/content"):
            return httpx.Response(200, content=self.audio)
        if path == "/api/v1/a4/engines":
            kind = request.url.params["kind"]
            return httpx.Response(200, json={"items": [self.engines[kind]]})
        if path == f"/api/v1/a3/recordings/{RID}/segments":
            if request.method == "POST":
                for item in payload["segments"]:
                    self.add_segment(item["startMs"], item["endMs"])
                return httpx.Response(
                    200,
                    json={"recordingId": RID, "inserted": len(payload["segments"]), "skipped": 0},
                )
            return httpx.Response(200, json={"recordingId": RID, "items": self.segments})
        if path == "/api/v1/a4/jobs":
            key = request.headers["Idempotency-Key"]
            if key not in self.keys:
                jid = str(uuid4())
                self.keys[key] = jid
                self.jobs[jid] = {
                    "id": jid,
                    "kind": "asr",
                    "status": "queued",
                    "engineCode": payload["engineCode"],
                    "recordingId": RID,
                    "params": deepcopy(payload["params"]),
                    "stats": {},
                    "submittedAt": ORIGIN.isoformat(),
                    "processedSegments": 0,
                    "failedSegments": 0,
                }
            return httpx.Response(202, json=self.jobs[self.keys[key]])
        if "/jobs/" in path:
            jid = path.split("/jobs/")[1].split("/")[0]
            job = self.jobs[jid]
            if path.endswith("/complete"):
                if self.complete_mode == "reject":
                    return httpx.Response(500, json={"code": "DATABASE_UNAVAILABLE"})
                job.update({k: v for k, v in payload.items() if k != "transcripts"})
                for text in payload.get("transcripts", []):
                    seg = next(s for s in self.segments if s["id"] == text["segmentId"])
                    seg.update(transcriptStatus="done", hasTranscript=True)
                    tid = str(uuid4())
                    self.texts[seg["id"]] = {
                        **deepcopy(seg),
                        "id": tid,
                        "segmentId": seg["id"],
                        "text": text["text"],
                        "textRaw": text["text"],
                        "language": "en",
                        "revision": 1,
                        "isEdited": False,
                        "reviewStatus": "unreviewed",
                        "qualityFlags": [],
                    }
                if self.complete_mode == "timeout_after_commit":
                    raise httpx.ReadTimeout("connection lost", request=request)
                if self.break_time and self.texts:
                    next(iter(self.texts.values()))["endAt"] = ORIGIN.isoformat()
            return httpx.Response(200, json=job)
        if "/segments/" in path and path.endswith("/transcript"):
            sid = path.split("/segments/")[1].split("/")[0]
            return (
                httpx.Response(200, json=self.texts[sid])
                if sid in self.texts
                else httpx.Response(404)
            )
        if "/transcripts/" in path:
            tid = path.split("/transcripts/")[1]
            item = next((t for t in self.texts.values() if t["id"] == tid), None)
            return httpx.Response(200, json=item) if item else httpx.Response(404)
        raise AssertionError(f"Unexpected boundary request: {request.method} {request.url}")
