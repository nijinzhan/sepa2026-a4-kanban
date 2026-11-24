"""Probe Liu Rui's standalone HTTP automation service and the real speech/PG path.

The notification sender is this local test client; no live A-3 capture is claimed.
Uses the pinned, hash-checked teammate snapshot without modifying their files.
"""

import argparse
import hashlib
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
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
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--speech-port", type=int, default=18051)
    parser.add_argument("--automation-port", type=int, default=18052)
    args = parser.parse_args()
    snapshot = args.audit_dir / "snapshot"
    if args.speech_port == args.automation_port:
        parser.error("services need different ports")
    for port in (args.speech_port, args.automation_port):
        if not 1 <= port <= 65535:
            parser.error("ports must be in 1..65535")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    source_manifest = json.loads(
        (args.audit_dir / "snapshot-manifest.json").read_text(encoding="utf-8")
    )
    verified = []
    for item in source_manifest["files"]:
        if item["path"].startswith("刘睿/src/a4_auto/"):
            data = (snapshot / item["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise RuntimeError("teammate snapshot checksum mismatch")
            verified.append(item)
    if not verified:
        raise RuntimeError("automation snapshot is missing")
    runtime = json.loads((ROOT / "outputs/postgres/runtime.local.json").read_text())
    run = ROOT / "outputs/postgres" / ("automation-http-" + uuid4().hex[:12])
    run.mkdir()
    checks = []
    processes = []
    handles = []
    speech_base = f"http://127.0.0.1:{args.speech_port}"
    auto_base = f"http://127.0.0.1:{args.automation_port}"
    env = dict(os.environ)
    env.update(
        A4_DATABASE_URL=runtime["appDsn"],
        A4_DATABASE_ADMIN_URL=runtime["adminDsn"],
        A4_SERVICE_API_KEY=secrets.token_hex(24),
        A4_AUTO_API_KEY=secrets.token_hex(24),
        A4_AUTO_MODE="http",
        A4_AUTO_DB=str(run / "automation.sqlite3"),
        A4_SPEECH_URL=speech_base + "/api/v1/a4",
        A4_AUTO_TIMEOUT_SECONDS="120",
        PYTHONPATH=os.pathsep.join([str(snapshot / "刘睿/src"), str(ROOT / "src")]),
    )
    env["A4_SPEECH_API_KEY"] = env["A4_SERVICE_API_KEY"]
    config = ROOT / "config.postgres.local.json"
    source = ROOT / "outputs/vhhh-c15-20261023/C15.16k.mono.wav"
    if hashlib.sha256(source.read_bytes()).hexdigest() != (
        "21c83396b32b596f4eee3f57fac51b5325773e4a1d34a875015ce16263dd3b50"
    ):
        raise RuntimeError("C15 source audio checksum mismatch")
    audio, rate = sf.read(source, frames=20 * 16000, dtype="int16")
    clip = run / "c15-first-20s.wav"
    sf.write(clip, audio, rate, subtype="PCM_16")
    anchor = datetime.now(UTC).replace(microsecond=0)
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
        cwd=ROOT,
        capture_output=True,
    )
    if imported.returncode:
        (run / "import-error.log").write_bytes(imported.stderr)
        raise RuntimeError("audio import failed; see local import-error.log")
    fixture = json.loads(imported.stdout)
    rid = fixture["recordingId"]
    event = {
        "audioId": rid,
        "startTime": anchor.isoformat().replace("+00:00", "Z"),
        "endTime": (anchor + timedelta(seconds=20)).isoformat().replace("+00:00", "Z"),
        "audioFormat": "wav",
        "source": "local-http-test-C15-original-UTC-unknown",
        "isSynthetic": False,
    }
    (run / "fixture.json").write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    (run / "event.json").write_text(json.dumps(event, indent=2), encoding="utf-8")

    def counts_2024141460331():
        with psycopg.connect(runtime["appDsn"]) as db:
            jobs = db.execute("SELECT count(*) FROM asr.job WHERE recording_id=%s", (rid,))
            job_count = jobs.fetchone()[0]
            count = db.execute(
                "SELECT count(*) FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
                "WHERE s.recording_id=%s",
                (rid,),
            ).fetchone()[0]
            return {"asrJobs": job_count, "transcripts": count}

    def start_2024141460331(module, arguments, base, log_name):
        log = (run / log_name).open("ab")
        handles.append(log)
        child = subprocess.Popen(
            [sys.executable, "-m", module, *arguments],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        processes.append(child)
        for _ in range(60):
            if child.poll() is not None:
                raise RuntimeError("service exited; inspect " + log_name)
            try:
                response = httpx.get(base + "/health", trust_env=False, timeout=1)
                if response.status_code == 200:
                    return child, response.json()
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        raise RuntimeError("service startup timed out; inspect " + log_name)

    try:
        start_2024141460331(
            "a4_speech.api_2024141460331",
            ["--config", str(config), "--host", "127.0.0.1", "--port", str(args.speech_port)],
            speech_base,
            "speech.log",
        )
        auto, health = start_2024141460331(
            "a4_auto", ["--port", str(args.automation_port)], auto_base, "automation.log"
        )
        assert health["mode"] == "http", health
        headers = {"X-Api-Key": env["A4_AUTO_API_KEY"], "X-Request-Id": "http-" + uuid4().hex}
        path = "/api/v1/a4/automation/audio-ready"
        with httpx.Client(base_url=auto_base, trust_env=False, timeout=120) as client:
            assert counts_2024141460331() == {"asrJobs": 0, "transcripts": 0}
            response = client.post(path, json=event)
            assert response.status_code == 401, response.text
            checks.append({"case": "missing_credential", "status": 401})
            for name, invalid in (
                ("invalid_uuid", {**event, "audioId": "invalid"}),
                ("invalid_utc", {**event, "startTime": "2026-10-30T00:00:00"}),
                ("reversed_interval", {**event, "endTime": event["startTime"]}),
            ):
                response = client.post(path, headers=headers, json=invalid)
                assert response.status_code == 400, response.text
                checks.append({"case": name, "status": 400})
            assert counts_2024141460331() == {"asrJobs": 0, "transcripts": 0}
            response = client.post(path, headers=headers, json=event)
            assert response.status_code == 200, response.text
            task = response.json()
            (run / "task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
            assert task["status"] == "succeeded" and task["saveConfirmed"], task
            assert task["transcriptIsMock"] is False and task["inputIsSynthetic"] is False
            assert task["result"]["recordingId"] == rid
            expected = {"asrJobs": 1, "transcripts": len(task["result"]["transcripts"])}
            assert expected["transcripts"] > 0 and counts_2024141460331() == expected
            checks.append({"case": "real_cpu_recognition_and_postgres_save", **expected})
            duplicate = client.post(path, headers=headers, json=event)
            assert duplicate.status_code == 200
            assert duplicate.json()["duplicate"] and duplicate.json()["taskId"] == task["taskId"]
            assert counts_2024141460331() == expected
            checks.append({"case": "duplicate_event_no_extra_job_or_transcript", "status": 200})
            conflict = client.post(
                path, headers=headers, json={**event, "source": "changed-source"}
            )
            assert conflict.status_code == 409 and counts_2024141460331() == expected
            checks.append({"case": "same_audio_conflicting_metadata", "status": 409})
            task_path = "/api/v1/a4/automation/tasks/" + task["taskId"]
            read = client.get(task_path, headers=headers)
            assert read.status_code == 200 and read.json()["saveConfirmed"]
            logs = client.get(task_path + "/logs", headers=headers).json()
            events = [item["eventType"] for item in logs["items"]]
            assert {
                "audio_received",
                "task_created",
                "asr_started",
                "save_started",
                "task_succeeded",
                "duplicate_ignored",
            } <= set(events), events
            (run / "logs.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
            with sqlite3.connect(run / "automation.sqlite3") as db:
                assert db.execute("SELECT count(*) FROM automation_task").fetchone()[0] == 1
            with psycopg.connect(runtime["appDsn"]) as db:
                rows = db.execute(
                    "SELECT t.id::text,t.job_id::text,t.text_raw,s.id::text,s.start_ms,s.end_ms "
                    "FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
                    "WHERE s.recording_id=%s ORDER BY s.start_ms",
                    (rid,),
                ).fetchall()
            with httpx.Client(base_url=speech_base, trust_env=False, timeout=20) as speech:
                for tid, jid, raw, sid, start_ms, end_ms in rows:
                    detail = speech.get(
                        "/api/v1/a4/transcripts/" + tid,
                        headers={"X-Api-Key": env["A4_SERVICE_API_KEY"]},
                    )
                    assert detail.status_code == 200, detail.text
                    value = detail.json()
                    assert (
                        value["recordingId"],
                        value["segmentId"],
                        value["jobId"],
                        value["textRaw"],
                        value["startMs"],
                        value["endMs"],
                    ) == (rid, sid, jid, raw, start_ms, end_ms)
            checks.append({"case": "database_rows_and_http_readback_match", "rows": len(rows)})
            auto.terminate()
            auto.wait(timeout=15)
            start_2024141460331(
                "a4_auto", ["--port", str(args.automation_port)], auto_base, "automation.log"
            )
            resumed = client.get(task_path, headers=headers)
            assert resumed.status_code == 200 and resumed.json()["saveConfirmed"]
            replay = client.post(path, headers=headers, json=event)
            assert replay.status_code == 200 and replay.json()["duplicate"]
            assert counts_2024141460331() == expected
            checks.append(
                {"case": "automation_service_restart_retains_task_and_dedup", "status": 200}
            )
        summary = {
            "status": "passed",
            "scope": "standalone automation HTTP -> speech HTTP -> PostgreSQL",
            "notificationSender": "local test script; not live A3 provider",
            "a3LiveProviderVerified": False,
            "teammateCommit": source_manifest["ref"],
            "recordingId": rid,
            "businessTaskId": task["taskId"],
            "asrJobId": task["result"]["jobId"],
            "transcriptIds": [row[0] for row in rows],
            "originalCaptureTimeKnown": False,
            "testTimeAnchor": True,
            "checks": checks,
            "executionSnapshot": fixture["executionSnapshot"],
        }
        (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"status": "passed", "summaryFile": str(run / "summary.json")}))
    finally:
        for child in reversed(processes):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main_2024141460331()
