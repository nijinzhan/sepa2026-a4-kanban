"""Register a local integration recording and the actual configured engine snapshot."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import soundfile as sf
from psycopg.types.json import Jsonb

from a4_speech.audio_2024141460331 import decode_audio
from a4_speech.cli_2024141460331 import _engine_2024141460331, load_config_2024141460331


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.local.json"))
    parser.add_argument("--started-at", required=True, help="ISO timestamp, timezone required")
    parser.add_argument(
        "--test-anchor", action="store_true", help="UTC is a test anchor, not known capture time"
    )
    args = parser.parse_args()
    started = datetime.fromisoformat(args.started_at.replace("Z", "+00:00"))
    if started.tzinfo is None or started.utcoffset() is None:
        parser.error("started-at must include timezone")
    path = args.audio.resolve(strict=True)
    info = sf.info(path)
    if info.samplerate != 16000 or info.channels != 1 or info.format != "WAV":
        parser.error("this local importer requires 16 kHz mono WAV")
    config = load_config_2024141460331(args.config)
    storage = config.get("storage", {})
    audio_root = Path(config["_directory"]) / storage.get("audioRoot", "outputs")
    if not path.is_relative_to(audio_root.resolve()):
        parser.error("audio must be inside configured storage.audioRoot")
    data = path.read_bytes()
    audio = decode_audio(data)
    snapshot = _engine_2024141460331(config).snapshot()
    rid = str(uuid4())
    quality = {
        "localIntegrationFixture": True,
        "originalCaptureTimeKnown": not args.test_anchor,
        "testTimeAnchor": args.test_anchor,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    with psycopg.connect(os.environ["A4_DATABASE_ADMIN_URL"]) as conn:
        conn.execute(
            "INSERT INTO core.data_source(code,name,category,notes) VALUES(%s,%s,'audio',%s) "
            "ON CONFLICT(code) DO NOTHING",
            (
                "lihong-local-integration",
                "Local A4 integration fixture",
                "Manual import; not an A3 live capture",
            ),
        )
        channel = conn.execute(
            "INSERT INTO aud.channel(source_code,facility,name,mount_point,capture_mode,codec) "
            "VALUES('lihong-local-integration','OTHER','Local integration',"
            "'local-a4','manual_upload','wav') "
            "ON CONFLICT(source_code,mount_point) DO UPDATE SET name=excluded.name RETURNING id"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO aud.recording(id,channel_id,source_code,capture_mode,started_at,ended_at,"
            "duration_ms,sample_rate_hz,audio_channels,codec,file_path,file_name,file_size_bytes,"
            "checksum_sha256,storage_volume,storage_state,quality,captured_by) "
            "VALUES(%s,%s,'lihong-local-integration','manual_upload',%s,%s,%s,16000,1,'wav',%s,%s,%s,%s,"
            "'local','verified',%s,'2024141460331 local import')",
            (
                rid,
                channel,
                started,
                started + timedelta(milliseconds=audio.duration_ms),
                round(audio.duration_ms),
                str(path),
                path.name,
                len(data),
                hashlib.sha256(data).digest(),
                Jsonb(quality),
            ),
        )
        for kind in ("vad", "asr"):
            item = snapshot[kind]
            conn.execute(
                "INSERT INTO asr.engine(code,name,kind,engine_family,version,model_name,runtime,"
                "device,deploy_mode,config,notes) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,'cpu','embedded',%s,%s) "
                "ON CONFLICT(code) DO UPDATE SET version=excluded.version,"
                "model_name=excluded.model_name,"
                "config=excluded.config,is_enabled=true,device='cpu',deploy_mode='embedded'",
                (
                    item["engineCode"],
                    item["modelName"],
                    kind,
                    "webrtc" if kind == "vad" else "sensevoice",
                    item["version"],
                    item["modelName"],
                    "webrtcvad-wheels" if kind == "vad" else "sherpa-onnx",
                    Jsonb(item),
                    "Local integration registry from actual engine snapshot",
                ),
            )
    print(
        json.dumps(
            {
                "recordingId": rid,
                "startedAt": started.isoformat(),
                "durationMs": round(audio.duration_ms),
                "quality": quality,
                "executionSnapshot": snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main_2024141460331()
