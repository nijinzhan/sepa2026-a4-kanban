"""Personal local A-3 bridge, reusing the team's original storage and metadata code."""

import argparse
import hashlib
import hmac
import importlib
import io
import json
import math
import os
import re
import sys
import tempfile
import threading
import types
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .audio_2024141460331 import decode_audio

MAX_AUDIO_BYTES = 64 * 1024 * 1024


class IngestError(RuntimeError):
    def __init__(self, status, code):
        super().__init__(code)
        self.status, self.code = status, code


def _storage_class(team_repo):
    root = Path(team_repo).resolve()
    paths = [root / "刘亦含/a3-ingest-路径修复/a3", root / "吴美娴/a3-ingest_2024141460482/a3"]
    if not (paths[0] / "storage.py").is_file() or not (paths[1] / "models.py").is_file():
        raise ValueError("A-3 storage patch and metadata sources are required")
    name = "_local_a3_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(p) for p in paths]
        sys.modules[name] = package
    return importlib.import_module(name + ".storage").AudioStorage


def _utc(value):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timezone required")
        return dt.astimezone(UTC)
    except (ValueError, TypeError, AttributeError):
        raise IngestError(422, "INVALID_UTC") from None


def _iso(dt):
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LocalAudioStore:
    def __init__(self, audio_root, input_root, team_repo, dsn, *, source_code="lihong-local-a3"):
        self.audio_root = Path(audio_root).resolve()
        self.input_root = Path(input_root).resolve()
        self.storage = _storage_class(team_repo)(str(self.audio_root))
        package = (
            "_local_a3_" + hashlib.sha256(str(Path(team_repo).resolve()).encode()).hexdigest()[:12]
        )
        self.audio_metadata = importlib.import_module(package + ".models").AudioMetadata
        self.dsn, self.source_code = dsn, source_code
        self.lock = threading.Lock()

    @contextmanager
    def _connection(self):
        import psycopg
        from psycopg.rows import dict_row

        try:
            with psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5) as db:
                yield db
        except psycopg.Error:
            raise IngestError(503, "DATABASE_UNAVAILABLE") from None

    def ingest(
        self,
        input_file,
        started_at,
        *,
        station_code="VHHH_APP",
        frequency=119.1,
        source_url="local-archive-replay",
        provenance=None,
    ):
        started = _utc(_iso(_utc(started_at)))
        if (
            not re.fullmatch(r"VHHH_(TWR|GND|APP|DEP|DEL|ATIS)", station_code)
            or not math.isfinite(frequency)
            or not 118 <= frequency <= 137
        ):
            raise IngestError(422, "INVALID_CHANNEL")
        path = (self.input_root / input_file).resolve()
        if not path.is_relative_to(self.input_root):
            raise IngestError(403, "INPUT_PATH_FORBIDDEN")
        try:
            if path.stat().st_size > MAX_AUDIO_BYTES:
                raise IngestError(413, "INPUT_TOO_LARGE")
            with path.open("rb") as source:
                payload = source.read(MAX_AUDIO_BYTES + 1)
        except OSError:
            raise IngestError(404, "INPUT_NOT_FOUND") from None
        if len(payload) > MAX_AUDIO_BYTES:
            raise IngestError(413, "INPUT_TOO_LARGE")
        try:
            import soundfile as sf

            audio = decode_audio(payload)
            info = sf.info(io.BytesIO(payload))
            if info.format != "MP3" or audio.duration_ms < 1000:
                raise ValueError("A-3 archive storage requires MP3 of at least one second")
        except ValueError:
            raise IngestError(422, "INVALID_AUDIO") from None
        duration = round(audio.duration_ms)
        digest = hashlib.sha256(payload).hexdigest()
        stamp = round(started.timestamp() * 1000)
        with self.lock:
            existing = [
                r
                for r in self.storage.iter_index()
                if r["station_code"] == station_code and _utc(r["collect_start"]) == started
            ]
            duplicate = bool(existing)
            if existing:
                if len(existing) != 1 or any(
                    (
                        existing[0]["sha256"] != digest,
                        existing[0]["frequency"] != round(frequency, 3),
                        existing[0]["source_url"] != source_url,
                    )
                ):
                    raise IngestError(409, "RECORDING_CONFLICT")
                meta = existing[0]
            else:
                registered = self._find_registered(
                    station_code, started, duration, frequency, source_url, digest, len(payload)
                )
                if registered:
                    duplicate = True
                    relative = self._restore_registered_audio(registered, payload, digest)
                    rebuilt = self._rebuild_index(
                        registered["id"],
                        station_code,
                        frequency,
                        source_url,
                        stamp,
                        duration,
                        relative,
                        len(payload),
                        digest,
                    )
                    self.storage.append_index(rebuilt)
                    meta = rebuilt.to_contract()
                else:
                    relative = self.storage.target_rel_path(
                        station_code, stamp, duration // 1000, "mp3"
                    )
                    if (self.audio_root / relative).exists():
                        raise IngestError(409, "STORAGE_PATH_CONFLICT")
                    meta = self.storage.write_segment(
                        payload, station_code, stamp, duration // 1000, "mp3", source_url, frequency
                    ).to_contract()
            target = (self.audio_root / meta["file_path"]).resolve()
            if not target.is_relative_to(self.audio_root) or (
                target.exists()
                and (not target.is_file() or self.storage.sha256_file(target) != digest)
            ):
                raise IngestError(409, "STORAGE_CONTENT_MISMATCH")
            if not target.exists():
                self._write_audio_file(target, payload)
            record = {
                "id": meta["audio_id"],
                "startedAt": _iso(started),
                "endedAt": _iso(started + timedelta(milliseconds=duration)),
                "durationMs": duration,
                "filePath": meta["file_path"],
                "fileSizeBytes": len(payload),
                "checksumSha256": digest,
                "sampleRateHz": info.samplerate,
                "audioChannels": info.channels,
                "stationCode": station_code,
                "frequencyMhz": frequency,
                "codec": "mp3",
                "quality": {
                    "localIntegrationFixture": True,
                    "originalCaptureTimeKnown": False,
                    "testTimeAnchor": True,
                    "preciseDurationMs": duration,
                    "sourceUrl": source_url,
                    "provenance": provenance or {},
                },
            }
            self._register(record)
            return {**self.lookup(record["id"]), "duplicate": duplicate}

    def _register(self, record):
        from psycopg.types.json import Jsonb

        with self._connection() as db:
            db.execute(
                "INSERT INTO core.data_source(code,name,category,notes) "
                "VALUES(%s,'Local A-3 archive replay','audio',%s) ON CONFLICT(code) DO NOTHING",
                (self.source_code, "Personal integration; capture UTC is a test anchor"),
            )
            channel = db.execute(
                "INSERT INTO aud.channel(source_code,facility,name,mount_point,"
                "capture_mode,codec,frequency_mhz) "
                "VALUES(%s,%s,%s,%s,'manual_upload','mp3',%s) ON CONFLICT(source_code,mount_point) "
                "DO UPDATE SET name=excluded.name,frequency_mhz=excluded.frequency_mhz "
                "RETURNING id",
                (
                    self.source_code,
                    record["stationCode"].split("_")[1],
                    "Local replay " + record["stationCode"],
                    record["stationCode"],
                    round(record["frequencyMhz"], 3),
                ),
            ).fetchone()["id"]
            db.execute(
                "INSERT INTO aud.recording(id,channel_id,source_code,capture_mode,"
                "started_at,ended_at,"
                "duration_ms,sample_rate_hz,audio_channels,codec,file_path,file_name,file_size_bytes,"
                "checksum_sha256,storage_volume,storage_state,quality,captured_by) "
                "VALUES(%s,%s,%s,'manual_upload',%s,%s,%s,%s,%s,'mp3',%s,%s,%s,%s,'local',"
                "'verified',%s,'2024141460331 local A-3 adapter') ON CONFLICT(id) DO NOTHING",
                (
                    record["id"],
                    channel,
                    self.source_code,
                    _utc(record["startedAt"]),
                    _utc(record["endedAt"]),
                    record["durationMs"],
                    record["sampleRateHz"],
                    record["audioChannels"],
                    record["filePath"],
                    Path(record["filePath"]).name,
                    record["fileSizeBytes"],
                    bytes.fromhex(record["checksumSha256"]),
                    Jsonb(record["quality"]),
                ),
            )
            row = db.execute("SELECT * FROM aud.recording WHERE id=%s", (record["id"],)).fetchone()
            if (
                row["source_code"] != self.source_code
                or row["channel_id"] != channel
                or row["duration_ms"] != record["durationMs"]
                or row["started_at"] != _utc(record["startedAt"])
                or row["file_path"] != record["filePath"]
                or bytes(row["checksum_sha256"]).hex() != record["checksumSha256"]
            ):
                raise IngestError(409, "DATABASE_RECORDING_CONFLICT")

    def _find_registered(
        self, station_code, started, duration, frequency, source_url, digest, size
    ):
        with self._connection() as db:
            rows = db.execute(
                "SELECT r.*,c.mount_point,c.frequency_mhz AS channel_frequency_mhz "
                "FROM aud.recording r JOIN aud.channel c ON c.id=r.channel_id "
                "WHERE r.source_code=%s AND c.mount_point=%s AND r.started_at=%s",
                (self.source_code, station_code, started),
            ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise IngestError(409, "RECORDING_CONFLICT")
        row = rows[0]
        quality = row["quality"] or {}
        if (
            row["duration_ms"] != duration
            or row["file_size_bytes"] != size
            or bytes(row["checksum_sha256"]).hex() != digest
            or quality.get("sourceUrl") != source_url
            or (
                row["channel_frequency_mhz"] is not None
                and float(row["channel_frequency_mhz"]) != round(frequency, 3)
            )
        ):
            raise IngestError(409, "RECORDING_CONFLICT")
        return row

    def _restore_registered_audio(self, row, payload, digest):
        relative = Path(row["file_path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise IngestError(409, "STORAGE_PATH_CONFLICT")
        target = (self.audio_root / relative).resolve()
        if not target.is_relative_to(self.audio_root):
            raise IngestError(409, "STORAGE_PATH_CONFLICT")
        if target.exists():
            if target.is_file() and self.storage.sha256_file(target) == digest:
                return relative.as_posix()
            raise IngestError(409, "STORAGE_CONTENT_MISMATCH")
        self._write_audio_file(target, payload)
        return relative.as_posix()

    def _write_audio_file(self, target, payload):
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".part_", dir=str(target.parent))
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _rebuild_index(
        self,
        recording_id,
        station_code,
        frequency,
        source_url,
        start_ms,
        duration,
        relative_path,
        size,
        digest,
    ):
        return self.audio_metadata(
            audio_id=str(recording_id),
            station_code=station_code,
            frequency_mhz=round(frequency, 3),
            source_url=source_url,
            collect_start_ms=start_ms,
            collect_end_ms=start_ms + duration,
            duration_s=duration // 1000,
            file_path=relative_path,
            file_size=size,
            file_format="mp3",
            sha256=digest,
        )

    def lookup(self, recording_id):
        try:
            recording_id = str(UUID(str(recording_id)))
        except ValueError:
            raise IngestError(400, "INVALID_RECORDING_ID") from None
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM aud.recording WHERE id=%s AND source_code=%s",
                (recording_id, self.source_code),
            ).fetchone()
        if not row:
            raise IngestError(404, "RECORDING_NOT_FOUND")
        return {
            "id": str(row["id"]),
            "channelId": str(row["channel_id"]),
            "startedAt": _iso(row["started_at"]),
            "endedAt": _iso(row["ended_at"]),
            "durationMs": row["duration_ms"],
            "sampleRateHz": row["sample_rate_hz"],
            "audioChannels": row["audio_channels"],
            "codec": row["codec"],
            "filePath": row["file_path"],
            "fileSizeBytes": row["file_size_bytes"],
            "checksumSha256": bytes(row["checksum_sha256"]).hex(),
            "storageState": row["storage_state"],
            "quality": row["quality"],
        }

    def content(self, recording_id):
        row = self.lookup(recording_id)
        path = (self.audio_root / row["filePath"]).resolve()
        if not path.is_relative_to(self.audio_root):
            raise IngestError(403, "AUDIO_PATH_FORBIDDEN")
        try:
            data = path.read_bytes()
        except OSError:
            raise IngestError(404, "AUDIO_UNAVAILABLE") from None
        if hashlib.sha256(data).hexdigest() != row["checksumSha256"]:
            raise IngestError(409, "AUDIO_CHECKSUM_MISMATCH")
        return data


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputFile: str
    startedAt: str
    stationCode: str = "VHHH_APP"
    frequencyMhz: float = 119.1
    sourceUrl: str = "local-archive-replay"
    provenance: dict = Field(default_factory=dict)
    notify: bool = True


def create_local_app(store, api_key, *, notifier=None):
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("local A-3 service key is required")
    app = FastAPI(title="Personal local A-3 archive adapter")

    def authorize(x_api_key: str | None = Header(None)):
        if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), api_key.encode()):
            raise IngestError(401, "UNAUTHORIZED")

    @app.exception_handler(IngestError)
    def error(_request: Request, exc: IngestError):
        return JSONResponse(
            {"status": exc.status, "code": exc.code},
            status_code=exc.status,
            media_type="application/problem+json",
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "mode": "local-archive-adapter"}

    @app.post("/api/v1/a3/local/ingest", dependencies=[Depends(authorize)])
    def ingest(body: IngestRequest):
        row = store.ingest(
            body.inputFile,
            body.startedAt,
            station_code=body.stationCode,
            frequency=body.frequencyMhz,
            source_url=body.sourceUrl,
            provenance=body.provenance,
        )
        result = {"recording": row, "notificationStatus": "not_requested"}
        if body.notify:
            if notifier is None:
                return JSONResponse(
                    {**result, "notificationStatus": "unconfigured"}, status_code=503
                )
            try:
                result.update(automation=notifier(row), notificationStatus="confirmed")
            except (httpx.HTTPError, ValueError):
                return JSONResponse(
                    {**result, "notificationStatus": "unconfirmed"}, status_code=502
                )
        return result

    @app.get("/api/v1/a3/recordings/{recording_id}", dependencies=[Depends(authorize)])
    def metadata(recording_id: UUID):
        return store.lookup(str(recording_id))

    @app.get("/api/v1/a3/recordings/{recording_id}/content", dependencies=[Depends(authorize)])
    def content(recording_id: UUID):
        data = store.content(str(recording_id))
        return Response(
            data,
            media_type="audio/mpeg",
            headers={"ETag": '"' + hashlib.sha256(data).hexdigest() + '"'},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18050)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    store = LocalAudioStore(
        config["audioRoot"],
        config["inputRoot"],
        config["teamRepo"],
        os.environ["A4_DATABASE_URL"],
        source_code=config["sourceCode"],
    )

    def notify(row):
        event = {
            "audioId": row["id"],
            "startTime": row["startedAt"],
            "endTime": row["endedAt"],
            "audioFormat": row["codec"],
            "source": "local-a3-archive-replay",
            "isSynthetic": False,
        }
        with httpx.Client(trust_env=False, timeout=600) as client:
            response = client.post(
                os.environ["A4_AUTO_URL"] + "/api/v1/a4/automation/audio-ready",
                headers={"X-Api-Key": os.environ["A4_AUTO_API_KEY"]},
                json=event,
            )
            response.raise_for_status()
            return response.json()

    import uvicorn

    uvicorn.run(
        create_local_app(store, os.environ["A3_LOCAL_API_KEY"], notifier=notify),
        host="127.0.0.1",
        port=args.port,
    )


if __name__ == "__main__":
    main()
