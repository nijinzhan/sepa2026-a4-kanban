"""Verify updated query/export HTTP against canonical A-5 and actual ASR results.

Uses unmodified, hash-checked teammate sources. The optional audio HTTP sender is
a local protocol fixture serving the real WAV, not a live A-3 implementation.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import wave
import zipfile
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automation-summary", type=Path, required=True)
    parser.add_argument(
        "--audit-dir", type=Path, default=ROOT / ".doc-sync-work/main-audit-20261031"
    )
    parser.add_argument("--port", type=int, default=18053)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    audit = args.audit_dir.resolve()
    snapshot = audit / "snapshot"
    manifest = json.loads((audit / "snapshot-manifest.json").read_text(encoding="utf-8"))
    verified = 0
    for entry in manifest["files"]:
        if entry["path"].startswith("于天名/app/"):
            data = (snapshot / entry["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            verified += 1
    assert verified > 0
    automation = json.loads(args.automation_summary.read_text(encoding="utf-8"))
    assert automation["status"] == "passed" and automation["teammateCommit"] == manifest["ref"]
    rid = str(UUID(automation["recordingId"]))
    runtime = json.loads((ROOT / "outputs/postgres/runtime.local.json").read_text())
    deps = ROOT / ".doc-sync-work/main-audit-20261030/query-export-deps"
    sys.path.insert(0, str(deps))
    from sqlalchemy.engine import make_url

    with psycopg.connect(runtime["appDsn"], row_factory=psycopg.rows.dict_row) as db:
        recording = db.execute("SELECT * FROM aud.recording WHERE id=%s", (rid,)).fetchone()
        rows = db.execute(
            "SELECT t.id::text,t.text_raw,s.id::text AS segment_id,s.start_ms,s.end_ms,"
            "s.start_at,s.end_at FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
            "WHERE s.recording_id=%s ORDER BY s.start_at,s.id",
            (rid,),
        ).fetchall()
    assert rows and {r["id"] for r in rows} == set(automation["transcriptIds"])
    audio_path = Path(recording["file_path"]).resolve()
    assert audio_path.is_relative_to((ROOT / "outputs").resolve()) and audio_path.is_file()
    run = ROOT / "outputs/postgres" / ("query-export-" + uuid4().hex[:12])
    run.mkdir()
    key, audio_key = secrets.token_hex(24), secrets.token_hex(24)
    env = dict(os.environ)
    env.update(
        DATABASE_URL=make_url(runtime["appDsn"])
        .set(drivername="postgresql+psycopg")
        .render_as_string(hide_password=False),
        ATC_EXPORT_DIR=str(run / "exports"),
        ATC_API_KEYS=key,
        PYTHONPATH=os.pathsep.join([str(deps), str(snapshot / "于天名")]),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for name in ("ATC_A3_URL", "ATC_A3_API_KEY", "ATC_A5_URL"):
        env.pop(name, None)
    checks, exports, handles = [], [], []
    process = None
    base = f"http://127.0.0.1:{args.port}"
    params = {
        "recordingId": rid,
        "from": recording["started_at"].isoformat(),
        "to": rows[-1]["end_at"].isoformat(),
    }
    expected = {r["id"]: r for r in rows}
    fixture_state = {"requests": 0, "fail": False}

    class AudioFixture_2024141460331(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("X-Api-Key") != audio_key:
                self.send_error(401)
                return
            if self.path != f"/api/v1/a3/recordings/{rid}/content":
                self.send_error(404)
                return
            fixture_state["requests"] += 1
            if fixture_state["fail"]:
                self.send_error(503)
                return
            data = audio_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), AudioFixture_2024141460331)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def start_2024141460331():
        log = (run / "query.log").open("ab")
        handles.append(log)
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            env=env,
            cwd=run,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(60):
            if child.poll() is not None:
                raise RuntimeError("query service exited; inspect ignored query.log")
            try:
                if httpx.get(base + "/docs", trust_env=False, timeout=1).status_code == 200:
                    return child
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        child.terminate()
        child.wait(timeout=15)
        raise RuntimeError("query service startup timed out")

    def stop_2024141460331():
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=15)

    def export_2024141460331(client, format_name, *, audio=False, review=None):
        strategy = {**params}
        if review:
            strategy["reviewStatus"] = [review]
        body = {
            "name": "local-integration-2024141460331",
            "format": format_name,
            "strategy": strategy,
            "includeAudio": audio,
            "audioFormat": "wav",
        }
        response = client.post("/api/v1/a4/exports", json=body)
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        for _ in range(120):
            response = client.get("/api/v1/a4/exports/" + job_id)
            assert response.status_code == 200, response.text
            job = response.json()
            if job["status"] not in {"queued", "running"}:
                break
            time.sleep(0.1)
        assert job["status"] in {"succeeded", "partial"}, job
        artifact = client.get(f"/api/v1/a4/exports/{job_id}/download")
        assert artifact.status_code == 200
        content = artifact.content
        assert hashlib.sha256(content).hexdigest() == job["checksumSha256"]
        assert len(content) == job["fileSizeBytes"]
        detail = client.get(f"/api/v1/a4/exports/{job_id}/manifest", params={"format": "json"})
        assert detail.status_code == 200
        assert detail.json()["total"] == (0 if review else len(rows))
        with psycopg.connect(runtime["appDsn"]) as db:
            db_job = db.execute(
                "SELECT status,segment_count FROM anno.export_job WHERE id=%s", (job_id,)
            ).fetchone()
            assert db_job == (job["status"], job["segmentCount"])
        exports.append(
            {
                "id": job_id,
                "format": format_name,
                "includeAudio": audio,
                "status": job["status"],
                "segmentCount": job["segmentCount"],
                "checksumSha256": job["checksumSha256"],
            }
        )
        return job, content

    try:
        process = start_2024141460331()
        with httpx.Client(
            base_url=base, trust_env=False, timeout=120, headers={"X-Api-Key": key}
        ) as client:
            unauth = httpx.get(base + "/api/v1/a4/transcripts", trust_env=False)
            assert unauth.status_code == 401
            checks.append({"case": "missing_credential", "status": 401})
            page = client.get("/api/v1/a4/transcripts", params=params)
            assert page.status_code == 200, page.text
            assert {r["id"] for r in page.json()["items"]} == set(expected)
            for row in page.json()["items"]:
                assert row["textRaw"] == expected[row["id"]]["text_raw"]
            checks.append(
                {"case": "canonical_view_query_matches_saved_transcripts", "count": len(rows)}
            )
            boundary = client.get(
                "/api/v1/a4/transcripts",
                params={
                    **params,
                    "from": rows[-1]["start_at"].isoformat(),
                    "to": rows[-1]["start_at"].isoformat(),
                },
            )
            assert boundary.status_code == 200 and [r["id"] for r in boundary.json()["items"]] == [
                rows[-1]["id"]
            ]
            checks.append({"case": "query_includes_both_time_endpoints", "count": 1})
            page_ids = []
            for number in range(1, len(rows) + 1):
                part = client.get(
                    "/api/v1/a4/transcripts", params={**params, "pageSize": 1, "page": number}
                ).json()
                assert part["page"]["total"] == len(rows)
                page_ids.extend(r["id"] for r in part["items"])
            assert page_ids == [r["id"] for r in rows]
            checks.append({"case": "pagination_no_loss_or_duplicates", "count": len(rows)})
            invalid = client.get("/api/v1/a4/transcripts", params={"recordingId": "invalid"})
            assert invalid.status_code == 400 and invalid.headers["content-type"].startswith(
                "application/problem+json"
            )
            checks.append({"case": "invalid_query_problem_json", "status": 400})
            unavailable = client.post(
                "/api/v1/a4/exports",
                json={"name": "missing-a3", "format": "bundle_zip", "includeAudio": True},
            )
            assert unavailable.status_code == 503
            checks.append({"case": "audio_export_without_a3_configuration", "status": 503})
            for format_name in ("jsonl", "csv", "bundle_zip"):
                job, content = export_2024141460331(client, format_name)
                assert job["status"] == "succeeded"
                if format_name == "jsonl":
                    items = [json.loads(line) for line in content.decode().splitlines()]
                    assert {item["transcript"]["id"] for item in items} == set(expected)
                elif format_name == "csv":
                    items = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
                    assert {item["transcriptId"] for item in items} == set(expected)
                else:
                    with zipfile.ZipFile(io.BytesIO(content)) as archive:
                        assert {
                            "transcripts.jsonl",
                            "transcripts.csv",
                            "transcripts.srt",
                            "transcripts.vtt",
                            "manifest.json",
                        } <= set(archive.namelist())
                        assert json.loads(archive.read("manifest.json"))["itemCount"] == len(rows)
                        for row in rows:
                            assert row["text_raw"] in archive.read("transcripts.srt").decode()
                checks.append(
                    {
                        "case": "canonical_export_" + format_name,
                        "status": job["status"],
                        "count": job["segmentCount"],
                    }
                )
            empty_job, _ = export_2024141460331(client, "jsonl", review="accepted")
            assert empty_job["segmentCount"] == 0
            checks.append({"case": "a5_manifest_review_filter_and_statistics", "count": 0})
        stop_2024141460331()
        env.update(
            ATC_A3_URL=f"http://127.0.0.1:{server.server_port}/api/v1/a3", ATC_A3_API_KEY=audio_key
        )
        process = start_2024141460331()
        with httpx.Client(
            base_url=base, trust_env=False, timeout=120, headers={"X-Api-Key": key}
        ) as client:
            before = fixture_state["requests"]
            job, content = export_2024141460331(client, "bundle_zip", audio=True)
            assert job["status"] == "succeeded" and fixture_state["requests"] - before == 1
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                items = json.loads(archive.read("manifest.json"))["items"]
                for item in items:
                    assert item["audioError"] is None
                    source = expected[item["transcriptId"]]
                    with wave.open(io.BytesIO(archive.read(item["audioFile"]))) as wav:
                        actual = wav.getnframes() / wav.getframerate() * 1000
                        assert abs(actual - (source["end_ms"] - source["start_ms"])) <= 1
            checks.append(
                {
                    "case": "real_wav_via_local_audio_http_fixture",
                    "clips": len(rows),
                    "downloads": 1,
                    "liveA3": False,
                }
            )
            successful_job = job["id"]
            successful_hash = job["checksumSha256"]
            fixture_state["fail"] = True
            job, content = export_2024141460331(client, "bundle_zip", audio=True)
            assert job["status"] == "partial"
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                assert all(
                    item["audioError"]
                    for item in json.loads(archive.read("manifest.json"))["items"]
                )
            checks.append(
                {"case": "audio_dependency_failure_preserves_text_export", "status": "partial"}
            )
        stop_2024141460331()
        process = start_2024141460331()
        with httpx.Client(
            base_url=base, trust_env=False, timeout=120, headers={"X-Api-Key": key}
        ) as client:
            job = client.get(f"/api/v1/a4/exports/{successful_job}")
            assert job.status_code == 200 and job.json()["status"] == "succeeded"
            artifact = client.get(f"/api/v1/a4/exports/{successful_job}/download")
            assert (
                artifact.status_code == 200
                and hashlib.sha256(artifact.content).hexdigest() == successful_hash
            )
            assert (
                client.post(
                    f"/api/v1/a4/exports/{successful_job}/complete", json={"status": "succeeded"}
                ).status_code
                == 409
            )
            checks.append(
                {
                    "case": "query_service_restart_persists_export_and_download",
                    "status": "succeeded",
                }
            )
            checks.append({"case": "completed_export_rejects_second_completion", "status": 409})
        summary = {
            "status": "passed",
            "verifiedAt": datetime.now(UTC).isoformat(),
            "teammateCommit": manifest["ref"],
            "databaseSourceCommit": runtime["sourceRef"],
            "recordingId": rid,
            "transcriptIds": list(expected),
            "checks": checks,
            "exports": exports,
            "querySource": "asr.v_transcript_full",
            "exportTable": "anno.export_job",
            "exportFunctions": ["anno.fn_build_export_items", "ui.fn_export_manifest"],
            "legacyCompatibilityViewsUsed": False,
            "audioSender": "local HTTP protocol fixture with real WAV",
            "a3LiveProviderVerified": False,
            "a5BearerAuthenticationVerified": False,
            "notificationSender": automation["notificationSender"],
            "originalCaptureTimeKnown": False,
        }
        (run / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "checks": len(checks),
                    "summaryFile": str(run / "summary.json"),
                }
            )
        )
    finally:
        stop_2024141460331()
        server.shutdown()
        server.server_close()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main_2024141460331()
