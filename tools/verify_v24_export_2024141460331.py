"""Verify main-repository V24 in a disposable PostgreSQL, including real audio export."""

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
import traceback
import wave
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
from psycopg.types.json import Jsonb
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from a4_speech.local_a3_2024141460331 import LocalAudioStore  # noqa: E402


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def command(arguments, *, data=None, env=None):
    result = subprocess.run(arguments, input=data, capture_output=True, env=env, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-3000:])
    return result.stdout


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class Verification:
    def __init__(self, args):
        self.args = args
        self.state = args.output.resolve()
        self.state.mkdir(parents=True, exist_ok=False)
        self.ref = (
            command(["git", "-C", str(args.team_repo), "rev-parse", args.ref]).decode().strip()
        )
        self.container = "a4-lihong-v24-20261121-" + uuid4().hex[:8]
        self.created = False
        self.children = []
        self.handles = []
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"}
        self.password = secrets.token_hex(24)
        self.env["PGPASSWORD"] = self.password
        self.result = {"status": "running", "teamCommit": self.ref, "checks": []}

    def check(self, name, condition):
        if not condition:
            raise AssertionError(name)
        self.result["checks"].append(name)

    def snapshot(self):
        paths = [
            "尹凯臣/migrations",
            "尹凯臣/infra/postgres/init/00-bootstrap.sql",
            "尹凯臣/infra/postgres/migrate.sh",
            "于天名/app",
            "刘亦含/a3-ingest-路径修复/a3/storage.py",
            "吴美娴/a3-ingest_2024141460482/a3/models.py",
            "吴美娴/VHHH_AppDepDirZone_20261031_0000Z.mp3",
        ]
        archive = command(
            ["git", "-C", str(self.args.team_repo), "archive", self.ref, "--", *paths]
        )
        self.team = self.state / "team-source"
        manifest = []
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for entry in bundle.getmembers():
                if not entry.isfile():
                    continue
                destination = (self.team / entry.name).resolve()
                if not destination.is_relative_to(self.team):
                    raise ValueError("snapshot escapes its root")
                payload = bundle.extractfile(entry).read()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                manifest.append({"path": entry.name, "sha256": sha(payload)})
        self.migrations = self.team / "尹凯臣/migrations"
        self.v24 = self.migrations / "V24__export_audio_bytes_bigint.sql"
        save(self.state / "source-manifest.json", {"ref": self.ref, "files": manifest})
        self.result["v24Sha256"] = sha(self.v24.read_bytes())

    def docker(self, *arguments, data=None):
        return command(["docker", *arguments], data=data, env=self.env)

    def sql(self, text):
        return self.docker(
            "exec",
            "-i",
            self.container,
            "psql",
            "-U",
            "alpha_dba",
            "-d",
            "alpha_atc",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            "-",
            data=text,
        )

    def migrate(self, phase, *, expect_failure=False):
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                "PGPASSWORD",
                "-e",
                "PGHOST=127.0.0.1",
                self.container,
                "bash",
                "/tmp/migrate.sh",
            ],
            env=self.env,
            capture_output=True,
        )
        (self.state / (phase + ".log")).write_bytes(result.stdout + result.stderr)
        if (result.returncode == 0) == expect_failure:
            raise RuntimeError("migration failed; see " + phase + ".log")
        return result.returncode != 0

    def database(self):
        env_file = self.state / "database.local.env"
        env_file.write_text(
            "POSTGRES_USER=alpha_dba\nPOSTGRES_DB=alpha_atc\nPOSTGRES_PASSWORD="
            + self.password
            + "\n",
            encoding="ascii",
        )
        image = self.args.image
        self.result["imageId"] = (
            self.docker("image", "inspect", image, "--format", "{{.Id}}").decode().strip()
        )
        self.docker(
            "run",
            "-d",
            "--name",
            self.container,
            "--label",
            "a4.owner=2024141460331",
            "--label",
            "a4.purpose=v24-regression",
            "--memory",
            "512m",
            "--memory-swap",
            "512m",
            "--cpus",
            "1",
            "--env-file",
            str(env_file),
            "--tmpfs",
            "/var/lib/postgresql/data:rw,size=256m",
            "-p",
            "127.0.0.1::5432",
            image,
            "-c",
            "shared_preload_libraries=pg_stat_statements",
            "-c",
            "timezone=UTC",
        )
        self.created = True
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    self.container,
                    "pg_isready",
                    "-U",
                    "alpha_dba",
                    "-d",
                    "alpha_atc",
                ],
                capture_output=True,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("temporary database did not become ready")
        self.port = int(
            self.docker("port", self.container, "5432/tcp").decode().strip().rsplit(":", 1)[1]
        )
        self.dsn = f"postgresql://alpha_app:{self.password}@127.0.0.1:{self.port}/alpha_atc"
        self.admin_dsn = f"postgresql://alpha_dba:{self.password}@127.0.0.1:{self.port}/alpha_atc"
        bootstrap = (self.team / "尹凯臣/infra/postgres/init/00-bootstrap.sql").read_bytes()
        prefix = (
            f"\\set app_password '{self.password}'\n\\set viewer_password '{self.password}'\n"
        ).encode()
        (self.state / "bootstrap.log").write_bytes(self.sql(prefix + bootstrap))
        baseline = self.state / "baseline-migrations"
        baseline.mkdir()
        for path in self.migrations.glob("*.sql"):
            if path != self.v24:
                (baseline / path.name).write_bytes(path.read_bytes())
        self.docker("exec", self.container, "mkdir", "-p", "/migrations")
        for path in sorted(baseline.iterdir()):
            self.docker("cp", str(path), self.container + ":/migrations/" + path.name)
        migration_runner = self.state / "migrate.sh"
        migration_runner.write_bytes(
            (self.team / "尹凯臣/infra/postgres/migrate.sh").read_bytes().replace(b"\r\n", b"\n")
        )
        self.docker("cp", str(migration_runner), self.container + ":/tmp/migrate.sh")
        failed = self.migrate("migration-before-v24", expect_failure=True)
        self.check(
            "ordered V1-V23 runner reproduces V19 VHHH seed dependency failure",
            failed
            and b"V19__local_mock_audio_channels.sql"
            in (self.state / "migration-before-v24.log").read_bytes()
            and b"airport_icao" in (self.state / "migration-before-v24.log").read_bytes(),
        )
        prerequisite = self.migrations / "R__01_core_reference.sql"
        self.result["migrationOrderIssue"] = {
            "failedVersion": "V19__local_mock_audio_channels.sql",
            "missingPrerequisite": "core.airport(VHHH), inserted by R__01_core_reference.sql",
            "mitigation": (
                "apply the upstream idempotent R__01 seed before retrying the unchanged runner"
            ),
            "prerequisiteSha256": sha(prerequisite.read_bytes()),
            "scope": "isolated test database only; no migration source changes",
        }
        self.sql(prerequisite.read_bytes())
        self.migrate("migration-baseline-after-seed-prerequisite")
        self.result["database"] = {
            "container": self.container,
            "host": "127.0.0.1",
            "port": self.port,
            "mainRepositoryOnly": True,
            "existingVolumesMounted": False,
            "temporaryDataDirectory": True,
            "memoryMiB": 512,
            "cpus": 1,
        }
        print("Fresh main-repository baseline V1-V23 ready", flush=True)

    def boundary_case(self, duration, bitrate, include_audio):
        with psycopg.connect(self.dsn) as db:
            try:
                with db.transaction(force_rollback=True):
                    channel = db.execute(
                        "INSERT INTO aud.channel(source_code,facility,name,mount_point,"
                        "bit_rate_kbps) "
                        "VALUES('local_v24_probe','APP','SQL fixture',%s,%s) RETURNING id",
                        (uuid4().hex, bitrate),
                    ).fetchone()[0]
                    recording = db.execute(
                        "INSERT INTO aud.recording(channel_id,source_code,started_at,duration_ms) "
                        "VALUES(%s,'local_v24_probe',%s,%s) RETURNING id",
                        (channel, datetime(2026, 11, 21, tzinfo=UTC), duration),
                    ).fetchone()[0]
                    db.execute(
                        "INSERT INTO aud.segment(recording_id,sequence_no,start_ms,end_ms) "
                        "VALUES(%s,1,0,%s)",
                        (recording, duration),
                    )
                    job = db.execute(
                        "INSERT INTO anno.export_job(code,name,format,strategy,include_audio) "
                        "VALUES(%s,'V24 boundary fixture','bundle_zip',%s,%s) RETURNING id",
                        (uuid4().hex, Jsonb({"recording_id": str(recording)}), include_audio),
                    ).fetchone()[0]
                    result = db.execute("SELECT anno.fn_build_export_items(%s)", (job,)).fetchone()[
                        0
                    ]
                    return {"status": "ok", "audioBytes": result["audio_bytes"]}
            except psycopg.Error as error:
                return {"status": "error", "sqlstate": error.sqlstate}

    def boundaries(self, fixed):
        inputs = [
            (33000, 64, True),
            (34000, 64, True),
            (34000, 64, False),
            (67000, 32, True),
            (68000, 32, True),
            (33554, 64, True),
            (33555, 64, True),
            (67108, 32, True),
            (67109, 32, True),
            (86400000, 320, True),
        ]
        rows = []
        for duration, bitrate, audio in inputs:
            actual = self.boundary_case(duration, bitrate, audio)
            overflow = audio and duration * bitrate * 1000 > 2147483647
            expected = (
                {"status": "error", "sqlstate": "22003"}
                if overflow and not fixed
                else {"status": "ok", "audioBytes": duration * bitrate // 8 if audio else 0}
            )
            self.check(
                f"{'after' if fixed else 'before'}: {duration}ms {bitrate}kbps audio={audio}",
                actual == expected,
            )
            rows.append(
                {"durationMs": duration, "bitRateKbps": bitrate, "includeAudio": audio, **actual}
            )
        self.result["afterSql" if fixed else "beforeSql"] = rows

    def sample(self):
        import imageio_ffmpeg
        import soundfile as sf

        self.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        source = self.team / "吴美娴/VHHH_AppDepDirZone_20261031_0000Z.mp3"
        inbox = self.state / "inbox"
        inbox.mkdir()
        clip = inbox / "source120.mp3"
        command(
            [
                self.ffmpeg,
                "-v",
                "error",
                "-i",
                str(source),
                "-t",
                "120",
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
                str(clip),
            ]
        )
        info = sf.info(str(clip))
        self.check("real MP3 decodes to exactly 120 seconds", info.frames == 120 * info.samplerate)
        store = LocalAudioStore(
            self.state / "audio", inbox, self.team, self.dsn, source_code="local_v24_probe"
        )
        recording = store.ingest(
            "source120.mp3",
            "2026-11-21T00:00:00Z",
            provenance={
                "scope": "V24 export fixture, no ASR",
                "sourceSha256": sha(source.read_bytes()),
                "captureUtcVerified": False,
            },
        )
        self.rid = recording["id"]
        self.segment_ids = []
        with psycopg.connect(self.dsn) as db:
            db.execute(
                "UPDATE aud.channel SET bit_rate_kbps=64 WHERE id=%s", (recording["channelId"],)
            )
            for i in range(2):
                segment = db.execute(
                    "INSERT INTO aud.segment(recording_id,sequence_no,start_ms,end_ms,notes) "
                    "VALUES(%s,%s,%s,%s,'V24 fixture, not VAD output') RETURNING id",
                    (self.rid, i + 1, i * 60000, (i + 1) * 60000),
                ).fetchone()[0]
                self.segment_ids.append(str(segment))
                db.execute(
                    "INSERT INTO asr.transcript(segment_id,text_raw,notes) VALUES(%s,%s,%s)",
                    (
                        segment,
                        f"V24 export test fixture {i + 1}; not ASR output",
                        "Not an ASR quality sample",
                    ),
                )
        self.clip = clip
        self.result["audioFixture"] = {
            "sourceFile": source.name,
            "sourceSha256": sha(source.read_bytes()),
            "clipSha256": recording["checksumSha256"],
            "durationMs": recording["durationMs"],
            "recordingId": self.rid,
            "segmentIds": self.segment_ids,
            "segmentDurationsMs": [60000, 60000],
            "transcriptsAreFixtures": True,
            "captureUtcVerified": False,
        }

    def start(self, name, arguments, source, extra_env, health):
        log = (self.state / (name + ".log")).open("ab")
        self.handles.append(log)
        env = {**self.env, "PYTHONPATH": str(source), "PYTHONIOENCODING": "utf-8", **extra_env}
        process = subprocess.Popen(
            [sys.executable, *arguments],
            cwd=self.state,
            env=env,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.children.append(process)
        with httpx.Client(trust_env=False, timeout=1) as client:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(name + " exited; see its log")
                try:
                    if client.get(health).status_code == 200:
                        return process
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
        raise RuntimeError(name + " readiness timeout")

    def services(self):
        a3_port, query_port = free_port(), free_port()
        self.a3_url = f"http://127.0.0.1:{a3_port}"
        self.query_url = f"http://127.0.0.1:{query_port}"
        self.a3_key, self.query_key = secrets.token_hex(24), secrets.token_hex(24)
        config = self.state / "a3.local.json"
        save(
            config,
            {
                "audioRoot": str(self.state / "audio"),
                "inputRoot": str(self.state / "inbox"),
                "teamRepo": str(self.team),
                "sourceCode": "local_v24_probe",
            },
        )
        self.start(
            "a3",
            [
                "-m",
                "a4_speech.local_a3_2024141460331",
                "--config",
                str(config),
                "--port",
                str(a3_port),
            ],
            ROOT / "src",
            {"A4_DATABASE_URL": self.dsn, "A3_LOCAL_API_KEY": self.a3_key},
            self.a3_url + "/health",
        )
        database_url = URL.create(
            "postgresql+psycopg",
            username="alpha_app",
            password=self.password,
            host="127.0.0.1",
            port=self.port,
            database="alpha_atc",
        ).render_as_string(hide_password=False)
        self.query_process = self.start(
            "query",
            [
                "-m",
                "uvicorn",
                "app.main:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(query_port),
            ],
            self.team / "于天名",
            {
                "DATABASE_URL": database_url,
                "ATC_API_KEYS": self.query_key,
                "ATC_A5_URL": "",
                "ATC_EXPORT_DIR": str(self.state / "exports"),
                "ATC_A3_URL": self.a3_url + "/api/v1/a3",
                "ATC_A3_API_KEY": self.a3_key,
            },
            self.query_url + "/docs",
        )

    def http_export(self, fixed):
        with httpx.Client(
            base_url=self.query_url,
            headers={"X-Api-Key": self.query_key},
            trust_env=False,
            timeout=60,
        ) as client:
            response = client.post(
                "/api/v1/a4/exports",
                json={
                    "name": "V24 60-second slices",
                    "format": "bundle_zip",
                    "includeAudio": True,
                    "audioFormat": "wav",
                    "strategy": {"recordingId": self.rid},
                },
            )
            try:
                response_body = response.json()
            except ValueError:
                response_body = {"body": response.text[:2000]}
            save(
                self.state / ("http-after-create.json" if fixed else "http-before-create.json"),
                response_body,
            )
            if not fixed:
                self.result["beforeHttp"] = {
                    "status": response.status_code,
                    "response": response_body,
                }
                self.check(
                    "before V24 HTTP audio export rejected with numeric overflow",
                    response.status_code in {422, 500},
                )
                return
            self.check("after V24 HTTP creation accepted", response.status_code == 202)
            initial_estimated_bytes = response_body["audioBytes"]
            job_id = response.json()["id"]
            for _ in range(120):
                job = client.get("/api/v1/a4/exports/" + job_id).json()
                if job["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.25)
            self.check("after V24 HTTP export succeeded", job["status"] == "succeeded")
            artifact = client.get(f"/api/v1/a4/exports/{job_id}/download")
            self.check(
                "ZIP download checksum matches job",
                artifact.status_code == 200 and sha(artifact.content) == job["checksumSha256"],
            )
            (self.state / "long-slices.zip").write_bytes(artifact.content)
            audio_results = []
            with zipfile.ZipFile(io.BytesIO(artifact.content)) as bundle:
                manifest = json.loads(bundle.read("manifest.json"))
                save(self.state / "export-manifest.json", manifest)
                self.check(
                    "ZIP manifest contains exactly both requested segments",
                    {i["segmentId"] for i in manifest["items"]} == set(self.segment_ids),
                )
                for i, segment in enumerate(self.segment_ids):
                    entry = next(item for item in manifest["items"] if item["segmentId"] == segment)
                    self.check("no audio error for " + segment, entry["audioError"] is None)
                    wav_data = bundle.read(entry["audioFile"])
                    with wave.open(io.BytesIO(wav_data)) as audio:
                        frames, rate, channels = (
                            audio.getnframes(),
                            audio.getframerate(),
                            audio.getnchannels(),
                        )
                        pcm = audio.readframes(frames)
                    expected = command(
                        [
                            self.ffmpeg,
                            "-v",
                            "error",
                            "-i",
                            str(self.clip),
                            "-ss",
                            str(i * 60),
                            "-t",
                            "60",
                            "-f",
                            "s16le",
                            "-c:a",
                            "pcm_s16le",
                            "pipe:1",
                        ]
                    )
                    self.check(
                        "60-second WAV matches source PCM for " + segment,
                        rate == 16000 and channels == 1 and frames == 960000 and pcm == expected,
                    )
                    audio_results.append(
                        {
                            "segmentId": segment,
                            "durationMs": frames * 1000 // rate,
                            "sampleRateHz": rate,
                            "fileSizeBytes": len(wav_data),
                            "pcmSha256": sha(pcm),
                        }
                    )
            self.result["afterHttp"] = {
                "status": "succeeded",
                "jobId": job_id,
                "segmentCount": job["segmentCount"],
                "audioBytesEstimatedBeforeExport": initial_estimated_bytes,
                "audioBytesActualWavAfterExport": job["audioBytes"],
                "zipSha256": sha(artifact.content),
                "audioFiles": audio_results,
            }
            with psycopg.connect(self.dsn) as db:
                byte_rows = db.execute(
                    "SELECT s.sequence_no,s.start_ms,s.end_ms,s.duration_ms,"
                    "c.bit_rate_kbps,i.audio_bytes "
                    "FROM aud.segment s JOIN aud.recording r ON r.id=s.recording_id "
                    "JOIN aud.channel c ON c.id=r.channel_id "
                    "JOIN anno.export_item i ON i.segment_id=s.id "
                    "WHERE i.export_job_id=%s ORDER BY s.sequence_no",
                    (job_id,),
                ).fetchall()
            self.result["afterHttp"]["byteEstimateRows"] = [
                {
                    "sequence": row[0],
                    "startMs": row[1],
                    "endMs": row[2],
                    "durationMs": row[3],
                    "bitRateKbps": row[4],
                    "audioBytes": row[5],
                }
                for row in byte_rows
            ]
            self.check(
                "two 60-second/64kbps A5 estimates match the V24 formula",
                initial_estimated_bytes == 960000
                and sum(row[5] for row in byte_rows) == 960000
                and all(row[3] == 60000 and row[4] == 64 for row in byte_rows)
                and len(byte_rows) == 2,
            )
            self.check(
                "completed export records the actual sizes of both WAV files",
                job["audioBytes"] == sum(item["fileSizeBytes"] for item in audio_results),
            )

    def filter_and_repeat(self):
        with psycopg.connect(self.dsn) as db:
            with db.transaction(force_rollback=True):
                job = db.execute(
                    "INSERT INTO anno.export_job(code,name,format,strategy,include_audio) "
                    "VALUES(%s,'segment_ids regression','bundle_zip',%s,true) RETURNING id",
                    (
                        uuid4().hex,
                        Jsonb({"recording_id": self.rid, "segment_ids": self.segment_ids[:1]}),
                    ),
                ).fetchone()[0]
                first = db.execute("SELECT anno.fn_build_export_items(%s)", (job,)).fetchone()[0]
                second = db.execute("SELECT anno.fn_build_export_items(%s)", (job,)).fetchone()[0]
                items = db.execute(
                    "SELECT segment_id FROM anno.export_item WHERE export_job_id=%s", (job,)
                ).fetchall()
                manifest = db.execute(
                    "SELECT item FROM ui.fn_export_manifest(%s)", (job,)
                ).fetchall()
                self.check(
                    "segment_ids and manifest still select one slice",
                    len(items) == len(manifest) == 1 and str(items[0][0]) == self.segment_ids[0],
                )
                self.check(
                    "repeated build is idempotent",
                    first == second and first["audio_bytes"] == 480000,
                )

    def run(self):
        self.snapshot()
        self.database()
        self.sample()
        self.boundaries(False)
        self.services()
        self.http_export(False)
        print("Old SQL and real 60-second HTTP export reproduce overflow", flush=True)
        self.docker("cp", str(self.v24), self.container + ":/migrations/" + self.v24.name)
        self.migrate("migration-v24")
        self.boundaries(True)
        self.filter_and_repeat()
        self.http_export(True)
        self.migrate("migration-repeat")
        with psycopg.connect(self.admin_dsn) as db:
            history = db.execute(
                "SELECT filename,checksum FROM public.alpha_migration_history "
                "ORDER BY applied_at,filename"
            ).fetchall()
            self.result["migrationCount"] = len(history)
            self.check(
                "V24 applied once and recorded original Git checksum",
                sum(n == self.v24.name and h == self.result["v24Sha256"] for n, h in history) == 1,
            )
        self.result["status"] = "passed"
        print("V24 SQL boundaries, real audio ZIP and repeat migration passed", flush=True)

    def cleanup(self):
        for child in reversed(self.children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for handle in self.handles:
            handle.close()
        if self.created:
            (self.state / "postgres.log").write_bytes(self.docker("logs", self.container))
            self.docker("stop", "--time", "10", self.container)
            self.result["temporaryContainerStopped"] = True
            self.result["temporaryContainer"] = self.container
        self.result["evidenceDirectory"] = str(self.state)
        save(self.state / "summary.json", self.result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-repo", type=Path, default=ROOT.parent)
    parser.add_argument("--ref", default="origin/main")
    parser.add_argument("--image", default="public.ecr.aws/docker/library/postgres:16")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / ("v24-export-20261121-" + uuid4().hex[:8])
    )
    run = Verification(parser.parse_args())
    try:
        run.run()
    except Exception:
        run.result["status"] = "failed"
        (run.state / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        run.cleanup()
    print(json.dumps(run.result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
