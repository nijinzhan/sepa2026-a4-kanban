"""Start or verify the personal local A-3 -> A-4 -> A-5 integration stack."""

import argparse
import hashlib
import io
import json
import os
import secrets
import socket
import subprocess
import sys
import tarfile
import time
import wave
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a4_speech.cli_2024141460331 import (  # noqa: E402
    _engine_2024141460331,
    load_config_2024141460331,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def snapshot_sources(repo, state):
    ref = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=repo, text=True).strip()
    archive = subprocess.check_output(
        [
            "git",
            "archive",
            ref,
            "--",
            "刘睿/src/a4_auto",
            "于天名/app",
            "刘亦含/a3-ingest-路径修复/a3/storage.py",
            "吴美娴/a3-ingest_2024141460482/a3/models.py",
            "吴美娴/VHHH_AppDepDirZone_20261031_0000Z.mp3",
        ],
        cwd=repo,
    )
    target = state / "team-source" / ref
    files = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            path = (target / member.name).resolve()
            if not path.is_relative_to(target.resolve()):
                raise RuntimeError("source archive path escapes snapshot")
            data = bundle.extractfile(member).read()
            if path.exists() and path.read_bytes() != data:
                raise RuntimeError("pinned teammate source was modified: " + member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            files.append({"path": member.name, "sha256": hashlib.sha256(data).hexdigest()})
    if not all(
        (target / p).is_file()
        for p in [
            "刘睿/src/a4_auto/api.py",
            "于天名/app/main.py",
            "刘亦含/a3-ingest-路径修复/a3/storage.py",
            "吴美娴/a3-ingest_2024141460482/a3/models.py",
        ]
    ):
        raise RuntimeError("team repository does not contain the required current sources")
    write_json(state / "team-source-manifest.json", {"ref": ref, "files": files})
    return target, ref


def register_engines(dsn, snapshot):
    from psycopg.types.json import Jsonb

    with psycopg.connect(dsn) as db:
        for kind in ("vad", "asr"):
            item = snapshot[kind]
            db.execute(
                "INSERT INTO asr.engine(code,name,kind,engine_family,version,model_name,runtime,"
                "device,deploy_mode,config,notes) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,'cpu','embedded',%s,%s) "
                "ON CONFLICT(code) DO NOTHING",
                (
                    item["engineCode"],
                    item["modelName"],
                    kind,
                    "webrtc" if kind == "vad" else "sensevoice",
                    item["version"],
                    item["modelName"],
                    "webrtcvad-wheels" if kind == "vad" else "sherpa-onnx",
                    Jsonb(item),
                    "Personal local integration registry from actual model files",
                ),
            )
            row = db.execute(
                "SELECT version,model_name,is_enabled,config FROM asr.engine WHERE code=%s",
                (item["engineCode"],),
            ).fetchone()
            if row[:3] != (item["version"], item["modelName"], True):
                raise RuntimeError("existing model registry differs from configured model")
            for key in ("modelSha256", "tokensSha256"):
                if row[3].get(key) and row[3][key] != item.get(key):
                    raise RuntimeError("existing model registry has different model hashes")


class LocalStack:
    def __init__(self, args):
        self.args = args
        self.state = args.state_dir.resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.team, self.ref = snapshot_sources(args.team_repo.resolve(), self.state)
        self.runtime = json.loads(args.credentials.read_text(encoding="utf-8-sig"))
        self.dsn = self.runtime["appDsn"]
        key_file = self.state / "keys.local.json"
        if not key_file.exists():
            write_json(
                key_file,
                {name: secrets.token_hex(24) for name in ("a3", "speech", "auto", "query")},
            )
        self.keys = json.loads(key_file.read_text(encoding="utf-8"))
        self.ports = {
            "a3": args.port,
            "speech": args.port + 1,
            "auto": args.port + 2,
            "query": args.port + 3,
        }
        if not 1024 <= args.port <= 65532:
            raise ValueError("base port must be in 1024..65532")
        self.urls = {name: f"http://127.0.0.1:{port}" for name, port in self.ports.items()}
        self.processes, self.handles = [], []
        self.config = load_config_2024141460331(args.model_config)
        self.engine = _engine_2024141460331(self.config)
        self.model = self.engine.snapshot()
        register_engines(self.dsn, self.model)
        self._configure()

    def _configure(self):
        from sqlalchemy.engine import URL

        config = json.loads(json.dumps(self.config))
        config.pop("_directory", None)
        config.pop("a5Url", None)  # This stack intentionally does not supply a mock IAM service.
        for item in config["engine"].values():
            for name in ("modelPath", "tokensPath"):
                if name in item:
                    item[name] = str((Path(self.config["_directory"]) / item[name]).resolve())
        config["storage"] = {
            "mode": "postgres",
            "dsnEnv": "A4_DATABASE_URL",
            "audioRoot": str(self.state / "audio"),
        }
        config["receiptDirectory"] = str(self.state / "speech-receipts")
        write_json(self.state / "speech.local.json", config)
        write_json(
            self.state / "a3.local.json",
            {
                "audioRoot": str(self.state / "audio"),
                "inputRoot": str(self.state / "inbox"),
                "teamRepo": str(self.team),
                "sourceCode": "lihong-local-a3",
            },
        )
        (self.state / "inbox").mkdir(exist_ok=True)
        connection = psycopg.conninfo.conninfo_to_dict(self.dsn)
        database_url = URL.create(
            "postgresql+psycopg",
            username=connection.get("user"),
            password=connection.get("password"),
            host=connection.get("host"),
            port=int(connection.get("port", "5432")),
            database=connection.get("dbname"),
        ).render_as_string(hide_password=False)
        self.env = dict(os.environ)
        self.env.update(
            A4_DATABASE_URL=self.dsn,
            A3_LOCAL_API_KEY=self.keys["a3"],
            A4_SERVICE_API_KEY=self.keys["speech"],
            A4_AUTO_API_KEY=self.keys["auto"],
            A4_AUTO_URL=self.urls["auto"],
            A4_AUTO_MODE="http",
            A4_AUTO_DB=str(self.state / "automation.sqlite3"),
            A4_AUTO_TIMEOUT_SECONDS="600",
            A4_AUTO_MAX_ATTEMPTS="1",
            A4_SPEECH_URL=self.urls["speech"] + "/api/v1/a4",
            A4_SPEECH_API_KEY=self.keys["speech"],
            DATABASE_URL=database_url,
            ATC_API_KEYS=self.keys["query"],
            ATC_EXPORT_DIR=str(self.state / "exports"),
            ATC_A3_URL=self.urls["a3"] + "/api/v1/a3",
            ATC_A3_API_KEY=self.keys["a3"],
            PYTHONUTF8="1",
            PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1",
        )
        self.env.pop("ATC_A5_URL", None)
        write_json(
            self.state / "endpoints.local.json",
            {
                "urls": self.urls,
                "teamCommit": self.ref,
                "keysFile": str(self.state / "keys.local.json"),
                "scope": "local archive replay; no online capture or IAM permission acceptance",
            },
        )

    def start(self):
        for port in self.ports.values():
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"local port {port} is already in use")
        services = [
            (
                "speech",
                "a4_speech.api_2024141460331",
                [
                    "--config",
                    str(self.state / "speech.local.json"),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.ports["speech"]),
                ],
                ROOT / "src",
                "/health",
            ),
            (
                "auto",
                "a4_auto",
                ["--port", str(self.ports["auto"])],
                self.team / "刘睿/src",
                "/health",
            ),
            (
                "a3",
                "a4_speech.local_a3_2024141460331",
                ["--config", str(self.state / "a3.local.json"), "--port", str(self.ports["a3"])],
                ROOT / "src",
                "/health",
            ),
            (
                "query",
                "uvicorn",
                [
                    "app.main:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(self.ports["query"]),
                ],
                self.team / "于天名",
                "/docs",
            ),
        ]
        for name, module, arguments, source, health in services:
            env = {**self.env, "PYTHONPATH": str(source)}
            log = (self.state / (name + ".log")).open("ab")
            self.handles.append(log)
            child = subprocess.Popen(
                [sys.executable, "-m", module, *arguments],
                env=env,
                cwd=self.state,
                stdout=log,
                stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self.processes.append(child)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError(name + " exited; inspect its local log")
                try:
                    if (
                        httpx.get(self.urls[name] + health, timeout=1, trust_env=False).status_code
                        == 200
                    ):
                        break
                except httpx.RequestError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError(name + " did not become ready")

    def stop(self):
        had_children = bool(self.processes)
        for child in reversed(self.processes):
            if child.poll() is None:
                child.terminate()
        for child in self.processes:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in self.handles:
            log.close()
        self.processes, self.handles = [], []
        if had_children:
            # Windows can briefly retain a listening socket after the child exits.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                listening = False
                for port in self.ports.values():
                    with socket.socket() as probe:
                        probe.settimeout(0.1)
                        listening |= probe.connect_ex(("127.0.0.1", port)) == 0
                if not listening:
                    break
                time.sleep(0.1)

    def prepare_sample(self):
        import imageio_ffmpeg
        import soundfile as sf

        from a4_speech.audio_2024141460331 import decode_audio

        info_path = self.state / "sample.local.json"
        source = (
            self.args.audio or self.team / "吴美娴/VHHH_AppDepDirZone_20261031_0000Z.mp3"
        ).resolve()
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if (
                info["provenance"]["sourceSha256"] != source_hash
                or hashlib.sha256((self.state / "inbox/sample.mp3").read_bytes()).hexdigest()
                != info["clipSha256"]
            ):
                raise RuntimeError("sample source changed; choose a new state directory")
            return info
        with sf.SoundFile(source) as stream:
            head = stream.read(frames=stream.samplerate * 120, dtype="float32", always_2d=True)
            rate = stream.samplerate
        encoded = io.BytesIO()
        sf.write(encoded, head, rate, format="WAV", subtype="PCM_16")
        spans = self.engine.detect(decode_audio(encoded.getvalue()))
        if not spans:
            raise RuntimeError("no VAD speech detected in first 120 seconds of source")
        offset = max(0, spans[0][0] / 1000 - 1)
        target = self.state / "inbox/sample.mp3"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-y",
                "-ss",
                str(offset),
                "-i",
                str(source),
                "-t",
                "20",
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "32k",
                str(target),
            ],
            check=True,
            capture_output=True,
        )
        audio = decode_audio(target.read_bytes())
        info = {
            "startedAt": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "clipSha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "durationMs": round(audio.duration_ms),
            "provenance": {
                "sourceFile": source.name,
                "sourceSha256": source_hash,
                "clipOffsetMs": round(offset * 1000),
                "clipDurationMs": round(audio.duration_ms),
                "captureUtcVerified": False,
                "timeIsTestAnchor": True,
            },
        }
        write_json(info_path, info)
        return info

    def counts(self, rid):
        with psycopg.connect(self.dsn) as db:
            return db.execute(
                "SELECT (SELECT count(*) FROM aud.recording WHERE id=%s),"
                "(SELECT count(*) FROM asr.job WHERE recording_id=%s),"
                "(SELECT count(*) FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id "
                "WHERE s.recording_id=%s)",
                (rid, rid, rid),
            ).fetchone()

    def verify(self):
        sample = self.prepare_sample()
        payload = {
            "inputFile": "sample.mp3",
            "startedAt": sample["startedAt"],
            "provenance": sample["provenance"],
        }
        result_dir = self.state / "verification" / uuid4().hex[:12]
        result_dir.mkdir(parents=True)
        with httpx.Client(trust_env=False, timeout=600) as client:
            a3_headers = {"X-Api-Key": self.keys["a3"]}
            response = client.post(
                self.urls["a3"] + "/api/v1/a3/local/ingest", headers=a3_headers, json=payload
            )
            write_json(result_dir / "ingest.json", response.json())
            response.raise_for_status()
            result = response.json()
            row, task = result["recording"], result["automation"]
            rid = row["id"]
            if task["status"] not in {"succeeded", "partial"} or not task["saveConfirmed"]:
                raise RuntimeError(
                    "ASR/save did not finish successfully; inspect local ingest.json"
                )
            assert not task["transcriptIsMock"] and not task["inputIsSynthetic"]
            assert task["result"]["recordingId"] == rid
            expected_ids = {t["id"] for t in task["saveReceipt"]["transcripts"]}
            assert expected_ids and self.counts(rid) == (1, 1, len(expected_ids))
            content = client.get(
                self.urls["a3"] + f"/api/v1/a3/recordings/{rid}/content", headers=a3_headers
            )
            content.raise_for_status()
            assert (
                hashlib.sha256(content.content).hexdigest()
                == row["checksumSha256"]
                == sample["clipSha256"]
            )
            again = client.post(
                self.urls["a3"] + "/api/v1/a3/local/ingest", headers=a3_headers, json=payload
            )
            again.raise_for_status()
            assert again.json()["recording"]["duplicate"]
            assert again.json()["automation"]["duplicate"]
            assert again.json()["automation"]["taskId"] == task["taskId"]
            assert self.counts(rid) == (1, 1, len(expected_ids))
            logs = client.get(
                self.urls["auto"] + f"/api/v1/a4/automation/tasks/{task['taskId']}/logs",
                headers={"X-Api-Key": self.keys["auto"]},
                params={"pageSize": 100},
            )
            logs.raise_for_status()
            assert (
                logs.json()["items"][-1]["details"]["modelVersion"] == self.model["asr"]["version"]
            )
            query_headers = {"X-Api-Key": self.keys["query"]}
            query = client.get(
                self.urls["query"] + "/api/v1/a4/transcripts",
                headers=query_headers,
                params={"recordingId": rid, "pageSize": 500},
            )
            query.raise_for_status()
            assert {item["id"] for item in query.json()["items"]} == expected_ids
            exports = []
            for format_name in ("jsonl", "bundle_zip"):
                created = client.post(
                    self.urls["query"] + "/api/v1/a4/exports",
                    headers=query_headers,
                    json={
                        "name": "local-a3-a4-a5",
                        "format": format_name,
                        "includeAudio": format_name == "bundle_zip",
                        "audioFormat": "wav",
                        "strategy": {"recordingId": rid},
                    },
                )
                assert created.status_code == 202, created.text
                eid = created.json()["id"]
                for _ in range(120):
                    detail = client.get(
                        self.urls["query"] + f"/api/v1/a4/exports/{eid}", headers=query_headers
                    )
                    detail.raise_for_status()
                    job = detail.json()
                    if job["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.25)
                assert job["status"] == "succeeded", job
                artifact = client.get(
                    self.urls["query"] + f"/api/v1/a4/exports/{eid}/download", headers=query_headers
                )
                artifact.raise_for_status()
                assert hashlib.sha256(artifact.content).hexdigest() == job["checksumSha256"]
                audio_count = 0
                if format_name == "bundle_zip":
                    with zipfile.ZipFile(io.BytesIO(artifact.content)) as bundle:
                        manifest = json.loads(bundle.read("manifest.json"))
                        write_json(result_dir / "export-manifest.json", manifest)
                        files = [
                            n
                            for n in bundle.namelist()
                            if n.startswith("audio/") and n.endswith(".wav")
                        ]
                        assert len(files) == len(expected_ids)
                        for name in files:
                            with wave.open(io.BytesIO(bundle.read(name))) as audio:
                                assert audio.getnframes() > 0 and audio.getframerate() == 16000
                        audio_count = len(files)
                suffix = ".zip" if format_name == "bundle_zip" else ".jsonl"
                (result_dir / (eid + suffix)).write_bytes(artifact.content)
                exports.append(
                    {
                        "id": eid,
                        "format": format_name,
                        "status": job["status"],
                        "segments": job["segmentCount"],
                        "audioFiles": audio_count,
                    }
                )
            summary = {
                "status": "passed",
                "teamCommit": self.ref,
                "recordingId": rid,
                "taskId": task["taskId"],
                "transcriptIds": sorted(expected_ids),
                "asrStatus": task["status"],
                "durationMs": row["durationMs"],
                "sha256": row["checksumSha256"],
                "duplicateIngestAndNotification": True,
                "modelVersion": self.model["asr"]["version"],
                "exports": exports,
                "evidenceDirectory": str(result_dir),
                "captureUtcVerified": False,
            }
            write_json(result_dir / "summary.json", summary)
            return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "verify"))
    parser.add_argument("--team-repo", type=Path, default=ROOT.parent)
    parser.add_argument("--model-config", type=Path, default=ROOT / "config.postgres.local.json")
    parser.add_argument(
        "--credentials", type=Path, default=ROOT / "outputs/postgres/runtime.local.json"
    )
    parser.add_argument("--state-dir", type=Path, default=ROOT / "outputs/local-stack")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--port", type=int, default=18050)
    args = parser.parse_args()
    stack = LocalStack(args)
    try:
        stack.start()
        if args.mode == "verify":
            first = stack.verify()
            stack.stop()
            stack.start()
            after = stack.verify()
            assert (
                first["recordingId"] == after["recordingId"] and first["taskId"] == after["taskId"]
            )
            summary = {
                "status": "passed",
                "first": first,
                "afterServiceRestart": after,
                "persistentIdentityAndIdempotency": True,
            }
            write_json(stack.state / "verification-summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print(
                json.dumps(
                    {
                        "status": "running",
                        "urls": stack.urls,
                        "config": str(stack.state / "endpoints.local.json"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            while all(child.poll() is None for child in stack.processes):
                time.sleep(0.5)
            raise RuntimeError("a local service exited; inspect its log")
    except KeyboardInterrupt:
        pass
    finally:
        stack.stop()


if __name__ == "__main__":
    main()
