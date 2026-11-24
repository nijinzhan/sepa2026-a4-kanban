"""Local real IAM login -> worker HTTP -> PostgreSQL revision acceptance.

Requires the read-only A-1 snapshot and isolated dependencies prepared for this audit.
Creates and removes only its own sample/account rows. No shared DDL or schema patches.
"""

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]


def port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def main():
    state = json.loads((ROOT / "outputs/postgres/runtime.local.json").read_text())
    output = ROOT / "outputs/own-fixes-20261107" / ("identity-http-" + uuid4().hex[:10])
    output.mkdir(parents=True)
    tag = "a4fix_" + uuid4().hex[:20]
    users = [str(uuid4()) for _ in range(3)]
    password, service_key = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    iam_port, worker_port = port(), port()
    iam_url = f"http://127.0.0.1:{iam_port}"
    worker_url = f"http://127.0.0.1:{worker_port}"
    processes, handles, checks = [], [], []
    recording = None
    role = None
    result = {
        "checks": checks,
        "identitySourceRef": "083d8cc1ea157297d46eeb61231e2461aa973c69",
        "realIamLogin": True,
        "sharedDdlChanged": False,
    }

    def check(name, condition):
        if not condition:
            raise RuntimeError("check failed: " + name)
        checks.append(name)

    def start(args, env, cwd, name, health):
        log = (output / (name + ".log")).open("wb")
        handles.append(log)
        proc = subprocess.Popen(args, env=env, cwd=cwd, stdout=log, stderr=log)
        processes.append(proc)
        with httpx.Client(trust_env=False) as client:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError(name + " process exited; see local log")
                try:
                    if client.get(health).status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
        raise RuntimeError(name + " startup timeout")

    try:
        audio = output / "sample.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 48000)
        with psycopg.connect(state["adminDsn"]) as db:
            source = db.execute(
                "SELECT channel_id,source_code FROM aud.recording LIMIT 1"
            ).fetchone()
            if not source:
                raise RuntimeError("initialized local audio baseline required")
            recording = db.execute(
                "INSERT INTO aud.recording(channel_id,source_code,started_at,duration_ms,"
                "file_path,storage_state,checksum_sha256) "
                "VALUES (%s,%s,now(),3000,%s,'verified',%s) RETURNING id",
                (*source, str(audio), hashlib.sha256(audio.read_bytes()).digest()),
            ).fetchone()[0]
            segment = db.execute(
                "INSERT INTO aud.segment(recording_id,sequence_no,start_ms,end_ms,is_speech) "
                "VALUES (%s,1,0,1000,true) RETURNING id",
                (recording,),
            ).fetchone()[0]
            tid = str(
                db.execute(
                    "INSERT INTO asr.transcript(segment_id,text_raw,language) "
                    "VALUES (%s,'original call','en') RETURNING id",
                    (segment,),
                ).fetchone()[0]
            )
            role = db.execute(
                "INSERT INTO iam.role(code,name) VALUES (%s,%s) RETURNING id", (tag, tag)
            ).fetchone()[0]
            db.execute(
                "INSERT INTO iam.role_permission SELECT %s,id,now() FROM iam.permission "
                "WHERE code='a4:transcript:edit'",
                (role,),
            )
            for i, uid in enumerate(users):
                db.execute(
                    "INSERT INTO iam.user_account(id,username,display_name,password_hash,status) "
                    "VALUES (%s,%s,%s,'temporary','active')",
                    (uid, tag + str(i), tag),
                )
                db.execute("SELECT iam.fn_set_password(%s,%s,false,NULL)", (uid, password))
                if i < 2:
                    db.execute(
                        "INSERT INTO iam.user_role(user_id,role_id) VALUES (%s,%s)", (uid, role)
                    )

        backend = ROOT / ".doc-sync-work/a1-audit-20261107/snapshot/朱炜亮/src/backend"
        env = dict(os.environ)
        env.update(
            DATABASE_URL=state["appDsn"],
            JWT_SECRET=secrets.token_urlsafe(48),
            PYTHONPATH=os.pathsep.join(
                [str(ROOT / ".doc-sync-work/iam-deps-20261107"), str(backend)]
            ),
        )
        start(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(iam_port),
            ],
            env,
            backend,
            "iam",
            iam_url + "/openapi.json",
        )
        config = json.loads((ROOT / "config.local.json").read_text(encoding="utf-8-sig"))
        for item in config["engine"].values():
            for key in ("modelPath", "tokensPath"):
                if key in item:
                    item[key] = str((ROOT / item[key]).resolve())
        config.update(
            storage={"mode": "postgres", "audioRoot": str(output), "dsnEnv": "A4_DATABASE_URL"},
            a5Url=iam_url + "/api/v1/a5",
            maxSegments=501,
            receiptDirectory=str(output / "receipts"),
        )
        config_path = output / "worker-config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        worker_env = dict(os.environ)
        worker_env.update(
            A4_DATABASE_URL=state["appDsn"],
            A4_SERVICE_API_KEY=service_key,
            PYTHONPATH=str(ROOT / "src"),
        )
        start(
            [
                sys.executable,
                "-m",
                "a4_speech.api_2024141460331",
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(worker_port),
            ],
            worker_env,
            ROOT,
            "worker",
            worker_url + "/health",
        )
        with httpx.Client(trust_env=False, timeout=30) as client:
            tokens = []
            for i in range(3):
                response = client.post(
                    iam_url + "/api/v1/a5/auth/login",
                    json={"username": tag + str(i), "password": password},
                )
                check("real-login-" + str(i), response.status_code == 200)
                tokens.append(response.json()["accessToken"])
            url = worker_url + "/api/v1/a4/transcripts/" + tid

            def edit(index, revision, **changes):
                return client.patch(
                    url,
                    headers={"X-Api-Key": service_key, "Authorization": "Bearer " + tokens[index]},
                    json={"expectedRevision": revision, "reason": "integration check", **changes},
                )

            first = edit(0, 1, text="corrected call")
            check(
                "authorized-text-save", first.status_code == 200 and first.json()["revision"] == 2
            )
            second = edit(1, 2, reviewStatus="accepted")
            check(
                "review-only-increments-revision",
                second.status_code == 200 and second.json()["revision"] == 3,
            )
            noop = edit(1, 3, text="corrected call", reviewStatus="accepted")
            check(
                "no-change-does-not-add-revision",
                noop.status_code == 200
                and noop.json()["revision"] == 3
                and not noop.json()["changed"],
            )
            check("viewer-forbidden", edit(2, 3, text="forbidden").status_code == 403)
            check("stale-version-conflict", edit(0, 2, text="stale").status_code == 409)
            invalid = client.patch(
                url,
                headers={"X-Api-Key": service_key, "Authorization": "Bearer invalid"},
                json={"expectedRevision": 3, "reason": "check", "text": "invalid"},
            )
            check("invalid-token-denied", invalid.status_code == 401)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(
                    pool.map(lambda i: edit(i, 3, text="concurrent " + str(i)).status_code, [0, 1])
                )
            check("concurrent-edit-one-winner", sorted(results) == [200, 409])
            with psycopg.connect(state["adminDsn"]) as db:
                db.execute("UPDATE iam.user_account SET status='disabled' WHERE id=%s", (users[0],))
            check("disabled-user-denied", edit(0, 4, text="disabled").status_code == 403)
            detail = client.get(url, headers={"X-Api-Key": service_key}).json()
            check(
                "original-and-history-protected",
                detail["textRaw"] == "original call"
                and detail["revision"] == 4
                and [e["revision"] for e in detail["edits"]] == [2, 3, 4],
            )
            check(
                "actor-from-real-identity",
                detail["edits"][0]["editorUserId"] == users[0]
                and detail["edits"][1]["editorUserId"] == users[1],
            )
            # A new connection proves storage survives request/process-local state.
            with psycopg.connect(state["appDsn"]) as db:
                row = db.execute(
                    "SELECT text_raw,revision FROM asr.transcript WHERE id=%s", (tid,)
                ).fetchone()
                check("fresh-db-connection-readback", row == ("original call", 4))
            processes[-1].terminate()
            processes[-1].wait(timeout=10)
            start(
                [
                    sys.executable,
                    "-m",
                    "a4_speech.api_2024141460331",
                    "--config",
                    str(config_path),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(worker_port),
                ],
                worker_env,
                ROOT,
                "worker-restarted",
                worker_url + "/health",
            )
            restarted = client.get(url, headers={"X-Api-Key": service_key})
            check(
                "worker-restart-preserves-revision-history",
                restarted.status_code == 200
                and restarted.json()["revision"] == 4
                and restarted.json()["edits"] == detail["edits"],
            )
        result.update(status="passed", checkCount=len(checks))
    except Exception as error:
        result.update(
            status="failed",
            errorType=type(error).__name__,
            detail="See local service logs; credentials omitted",
        )
        raise
    finally:
        for proc in reversed(processes):
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for handle in handles:
            handle.close()
        with psycopg.connect(state["adminDsn"]) as db:
            if recording:
                db.execute("DELETE FROM aud.recording WHERE id=%s", (recording,))
            db.execute("DELETE FROM iam.user_account WHERE id=ANY(%s::uuid[])", (users,))
            if role:
                db.execute("DELETE FROM iam.role WHERE id=%s", (role,))
            db.execute(
                "DELETE FROM iam.audit_log WHERE actor_user_id=ANY(%s::uuid[]) "
                "OR resource_id=ANY(%s::text[])",
                (users, users),
            )
        result["temporaryProcessesStopped"] = True
        result["testRowsRemoved"] = True
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "summaryFile": str(output / "summary.json"),
                    "status": result["status"],
                    "checks": len(checks),
                }
            )
        )


if __name__ == "__main__":
    main()
