"""Real local A4 automation → speech HTTP → A5 PostgreSQL → query/export probe.

Requires the pinned teammate snapshot captured by the repository audit. Query views
read canonical rows; only export task metadata has a local compatibility table.
"""

import argparse
import hashlib
import io
import json
import os
import secrets
import subprocess
import sys
import time
import wave
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".doc-sync-work/main-audit-20261030"


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=AUDIT / "snapshot")
    parser.add_argument("--port", type=int, default=18041)
    args = parser.parse_args()
    manifest = json.loads((AUDIT / "snapshot-manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        if row["path"].startswith(("刘睿/src/a4_auto/", "于天名/app/")):
            assert (
                hashlib.sha256((args.snapshot / row["path"]).read_bytes()).hexdigest()
                == row["sha256"]
            )
    sys.path[:0] = [
        str(AUDIT / "query-export-deps"),
        str(args.snapshot / "刘睿/src"),
        str(args.snapshot / "于天名"),
    ]
    from a4_auto.gateways import HttpSpeechGateway
    from a4_auto.models import AudioReadyEvent
    from a4_auto.repository import Repository
    from a4_auto.service import AutomationService
    from app.main import create_app as create_query_app
    from app.tables import export_job
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    runtime = json.loads((ROOT / "outputs/postgres/runtime.local.json").read_text())
    run = ROOT / "outputs/postgres" / ("integration-" + uuid4().hex[:12])
    run.mkdir()
    source = ROOT / "outputs/vhhh-c15-20261023/C15.16k.mono.wav"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "21c83396b32b596f4eee3f57fac51b5325773e4a1d34a875015ce16263dd3b50"
    )
    audio, rate = sf.read(source, frames=20 * 16000, dtype="int16")
    clip = run / "c15-first-20s.wav"
    sf.write(clip, audio, rate, subtype="PCM_16")
    config = ROOT / "config.postgres.local.json"
    if not config.exists():
        config.write_bytes((ROOT / "config.postgres.example.json").read_bytes())
    anchor = datetime(2026, 10, 30, 0, 0, tzinfo=UTC) + timedelta(seconds=int(uuid4().hex[:6], 16))
    env = dict(
        os.environ,
        A4_DATABASE_URL=runtime["appDsn"],
        A4_DATABASE_ADMIN_URL=runtime["adminDsn"],
        A4_SERVICE_API_KEY=secrets.token_hex(24),
    )
    imported = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/postgres_import_audio_2024141460331.py"),
            "--audio",
            str(clip),
            "--config",
            str(config),
            "--started-at",
            anchor.isoformat(),
            "--test-anchor",
        ],
        env=env,
        capture_output=True,
        check=True,
    )
    fixture = json.loads(imported.stdout)
    (run / "fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    rid = fixture["recordingId"]
    base = f"http://127.0.0.1:{args.port}"
    log = (run / "speech.log").open("wb")
    process = None

    def start_2024141460331():
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "a4_speech.api_2024141460331",
                "--config",
                str(config),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(60):
            if worker.poll() is not None:
                raise RuntimeError("speech worker exited; inspect local speech.log")
            try:
                if httpx.get(base + "/health", trust_env=False, timeout=1).status_code == 200:
                    return worker
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        worker.terminate()
        raise RuntimeError("speech worker startup timed out")

    class ObservedGateway_2024141460331:
        is_mock = False
        model_version = "actual-teacher-see-snapshot"

        def __init__(self):
            self.gateway = HttpSpeechGateway(base + "/api/v1/a4", env["A4_SERVICE_API_KEY"])
            self.result = None
            self.unsaved_count = None

        def transcribe(self, *a, **kw):
            self.result = self.gateway.transcribe(*a, **kw)
            with psycopg.connect(runtime["adminDsn"]) as conn:
                self.unsaved_count = conn.execute(
                    "SELECT count(*) FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
                    "WHERE s.recording_id=%s",
                    (rid,),
                ).fetchone()[0]
            assert self.unsaved_count == 0
            return self.result

        def save(self, *a, **kw):
            return self.gateway.save(*a, **kw)

    try:
        process = start_2024141460331()
        gateway = ObservedGateway_2024141460331()
        event = AudioReadyEvent(
            audioId=rid,
            startTime=anchor.isoformat().replace("+00:00", "Z"),
            endTime=(anchor + timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
            source="local-real-C15-test-UTC-anchor-original-capture-time-unknown",
            isSynthetic=False,
        )
        auto = AutomationService(Repository(run / "automation.sqlite3"), gateway, gateway)
        task = auto.handle(event, "real-pg-" + uuid4().hex[:12])
        (run / "automation.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
        assert task["status"] in {"succeeded", "partial"} and task["saveConfirmed"], task
        assert gateway.result and gateway.result["transcripts"]
        details = []
        with psycopg.connect(runtime["adminDsn"]) as conn:
            rows = conn.execute(
                "SELECT t.id::text,t.segment_id::text,t.job_id::text,t.text_raw "
                "FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
                "WHERE s.recording_id=%s ORDER BY s.start_ms",
                (rid,),
            ).fetchall()
            assert len(rows) == len(gateway.result["transcripts"])
            conn.execute("CREATE SCHEMA IF NOT EXISTS a4_query_2024141460331")
            conn.execute(
                "CREATE OR REPLACE VIEW a4_query_2024141460331.audio_recording AS "
                "SELECT id::text AS recording_id,file_path,"
                "started_at AT TIME ZONE 'UTC' AS started_at FROM aud.recording"
            )
            conn.execute(
                "CREATE OR REPLACE VIEW a4_query_2024141460331.audio_segment AS "
                "SELECT id::text AS segment_id,recording_id::text,start_ms,end_ms,"
                "start_at AT TIME ZONE 'UTC' AS start_at,"
                "end_at AT TIME ZONE 'UTC' AS end_at FROM aud.segment"
            )
            conn.execute(
                "CREATE OR REPLACE VIEW a4_query_2024141460331.transcript AS "
                "SELECT id::text AS transcript_id,segment_id::text,text_raw,"
                "coalesce(text_norm,text_raw) AS text_norm,review_status FROM asr.transcript"
            )
            conn.execute("GRANT USAGE ON SCHEMA a4_query_2024141460331 TO alpha_app")
            conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA a4_query_2024141460331 TO alpha_app")
        query_options = {"options": "-csearch_path=a4_query_2024141460331,public"}
        admin_url = make_url(runtime["adminDsn"]).set(
            drivername="postgresql+psycopg", query=query_options
        )
        engine = create_engine(admin_url)
        export_job.create(engine, checkfirst=True)
        engine.dispose()
        with psycopg.connect(runtime["adminDsn"]) as conn:
            conn.execute(
                "GRANT SELECT,INSERT,UPDATE ON a4_query_2024141460331.export_job TO alpha_app"
            )
        query_url = make_url(runtime["appDsn"]).set(
            drivername="postgresql+psycopg", query=query_options
        )
        query_app = create_query_app(
            query_url.render_as_string(hide_password=False), run / "exports"
        )
        headers = {"X-Api-Key": env["A4_SERVICE_API_KEY"]}
        for tid, _sid, jid, text in rows:
            body = httpx.get(
                base + "/api/v1/a4/transcripts/" + tid, headers=headers, trust_env=False
            ).json()
            assert body["id"] == tid and body["textRaw"] == text and body["jobId"] == jid
            details.append(body)
        formats = []
        with TestClient(query_app) as client:
            page = client.get(
                "/api/v1/a3/transcripts", params={"start": event.startTime, "end": event.endTime}
            )
            assert page.status_code == 200, page.text
            assert {r["transcriptId"] for r in page.json()["items"]} == {r[0] for r in rows}
            for format_name in ("json", "csv", "srt", "vtt"):
                response = client.post(
                    "/api/v1/a3/exports",
                    json={
                        "start": event.startTime,
                        "end": event.endTime,
                        "format": format_name,
                        "includeAudio": True,
                    },
                )
                assert response.status_code == 202, response.text
                jid = response.json()["exportJobId"]
                status = client.get("/api/v1/a3/exports/" + jid).json()
                assert status["status"] == "done", status
                content = client.get("/api/v1/a3/exports/" + jid + "/download").content
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    exported = json.loads(archive.read("manifest.json"))
                    assert exported["itemCount"] == len(rows)
                    for item in exported["items"]:
                        assert item["audioError"] is None
                        with wave.open(io.BytesIO(archive.read(item["audioFile"]))) as wav:
                            actual_ms = wav.getnframes() / wav.getframerate() * 1000
                            source_row = next(
                                r for r in details if r["segmentId"] == item["segmentId"]
                            )
                            assert (
                                abs(actual_ms - (source_row["endMs"] - source_row["startMs"])) <= 1
                            )
                formats.append(
                    {
                        "format": format_name,
                        "itemCount": len(rows),
                        "zipSha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        query_app.state.engine.dispose()
        process.terminate()
        process.wait(timeout=15)
        subprocess.run(
            [
                r"C:\Program Files\Docker\Docker\resources\bin\docker.exe",
                "restart",
                runtime["container"],
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        for _ in range(40):
            try:
                with psycopg.connect(runtime["adminDsn"], connect_timeout=1):
                    break
            except psycopg.OperationalError:
                time.sleep(0.5)
        process = start_2024141460331()
        for row in details:
            response = httpx.get(
                base + "/api/v1/a4/transcripts/" + row["id"], headers=headers, trust_env=False
            )
            assert response.status_code == 200 and response.json()["textRaw"] == row["textRaw"]
        summary = {
            "status": "passed",
            "recordingId": rid,
            "businessTaskId": task["taskId"],
            "asrJobId": gateway.result["jobId"],
            "transcriptIds": [r[0] for r in rows],
            "beforeSaveTranscriptCount": gateway.unsaved_count,
            "savedTranscriptCount": len(rows),
            "teammateCommit": manifest.get("ref", manifest.get("commit")),
            "sharedCanonicalTables": ["aud.recording", "aud.segment", "asr.job", "asr.transcript"],
            "queryCompatibilityViews": "a4_query_2024141460331",
            "textCopiedToQueryDatabase": False,
            "databaseAndSpeechServiceRestartReadback": True,
            "formats": formats,
            "originalCaptureTimeKnown": False,
            "executionSnapshot": fixture["executionSnapshot"],
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "summaryFile": str(run / "summary.json")}))
    finally:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    main_2024141460331()
