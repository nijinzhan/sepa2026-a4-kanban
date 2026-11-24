"""Exercise the real A-3 storage code and optional isolated PostgreSQL rows."""

import hashlib
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from a4_speech import local_a3_2024141460331 as local_a3
from a4_speech.local_a3_2024141460331 import (
    IngestError,
    LocalAudioStore,
    create_local_app,
)

TEAM = Path(os.environ.get("A3_TEAM_REPO", Path(__file__).resolve().parents[2]))


@pytest.fixture
def mp3(tmp_path):
    import imageio_ffmpeg

    inbox = tmp_path / "inbox"
    inbox.mkdir()
    path = inbox / "sample.mp3"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-b:a",
            "32k",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def store(tmp_path, mp3):
    dsn = os.environ.get("A4_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("A4_TEST_DATABASE_URL is required for real registration tests")
    import psycopg

    source = "local_a3_test_" + uuid4().hex
    instance = LocalAudioStore(tmp_path / "audio", mp3.parent, TEAM, dsn, source_code=source)
    try:
        yield instance
    finally:
        with psycopg.connect(dsn) as db:
            db.execute("DELETE FROM aud.recording WHERE source_code=%s", (source,))
            db.execute("DELETE FROM aud.channel WHERE source_code=%s", (source,))
            db.execute("DELETE FROM core.data_source WHERE code=%s", (source,))


def test_invalid_audio_is_rejected_before_storage_or_registration(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "fake.mp3").write_bytes("这是模拟的ATC塔台音频数据_1".encode())
    store = LocalAudioStore(tmp_path / "audio", inbox, TEAM, "unused")
    with pytest.raises(IngestError) as error:
        store.ingest("fake.mp3", "2026-11-17T00:00:00Z")
    assert error.value.code == "INVALID_AUDIO"
    assert not list((tmp_path / "audio").rglob("*.mp3"))
    assert not store.storage.index_path.exists()


def test_oversized_input_is_rejected_before_read_or_decode(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "oversized.mp3").write_bytes(b"x" * 33)
    monkeypatch.setattr(local_a3, "MAX_AUDIO_BYTES", 32, raising=False)
    store = LocalAudioStore(tmp_path / "audio", inbox, TEAM, "unused")
    with pytest.raises(IngestError) as error:
        store.ingest("oversized.mp3", "2026-11-17T00:00:00Z")
    assert error.value.status == 413
    assert error.value.code == "INPUT_TOO_LARGE"
    assert not store.storage.index_path.exists()


def test_repeated_ingest_preserves_storage_and_database_uuid(store):
    first = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    second = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    assert first["id"] == second["id"]
    assert second["duplicate"] is True
    index = list(store.storage.iter_index())
    assert len(index) == 1 and index[0]["audio_id"] == first["id"]
    assert index[0]["file_path"].startswith("2026/11/17/VHHH_APP/")
    assert store.lookup(first["id"])["checksumSha256"] == index[0]["sha256"]
    assert store.content(first["id"]) == (store.input_root / "sample.mp3").read_bytes()


def test_database_recovers_uuid_audio_and_index_when_local_files_are_lost(store):
    first = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    content = (store.input_root / "sample.mp3").read_bytes()
    audio_path = store.audio_root / first["filePath"]
    audio_path.unlink()
    store.storage.index_path.unlink()

    recovered = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")

    assert recovered["id"] == first["id"]
    assert recovered["duplicate"] is True
    assert audio_path.read_bytes() == content
    index = list(store.storage.iter_index())
    assert len(index) == 1 and index[0]["audio_id"] == first["id"]
    assert len(list(store.audio_root.rglob("*.mp3"))) == 1


def test_same_time_different_file_is_conflict_without_overwriting(store):
    first = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    original = store.content(first["id"])
    (store.input_root / "other.mp3").write_bytes(original + b"different-tail")
    with pytest.raises(IngestError) as error:
        store.ingest("other.mp3", "2026-11-17T00:00:00Z")
    assert error.value.code == "RECORDING_CONFLICT"
    assert store.content(first["id"]) == original
    assert len(list(store.storage.iter_index())) == 1


def test_registration_failure_can_resume_without_another_storage_uuid(store, monkeypatch):
    register = store._register

    def unavailable(_record):
        raise IngestError(503, "DATABASE_UNAVAILABLE")

    monkeypatch.setattr(store, "_register", unavailable)
    with pytest.raises(IngestError):
        store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    original_id = list(store.storage.iter_index())[0]["audio_id"]
    monkeypatch.setattr(store, "_register", register)
    result = store.ingest("sample.mp3", "2026-11-17T00:00:00Z")
    assert result["id"] == original_id
    assert len(list(store.storage.iter_index())) == 1


def test_http_authentication_download_and_notification_after_commit(store):
    notified = []

    def notify(record):
        # Read from a new DB transaction: the row must already be committed.
        assert store.lookup(record["id"])["id"] == record["id"]
        notified.append(record["id"])
        return {"status": "accepted"}

    with TestClient(create_local_app(store, "local-test-key", notifier=notify)) as client:
        body = {"inputFile": "sample.mp3", "startedAt": "2026-11-17T00:00:00Z"}
        assert client.post("/api/v1/a3/local/ingest", json=body).status_code == 401
        headers = {"X-Api-Key": "local-test-key"}
        response = client.post("/api/v1/a3/local/ingest", headers=headers, json=body)
        assert response.status_code == 200, response.text
        record = response.json()["recording"]
        assert notified == [record["id"]]
        path = "/api/v1/a3/recordings/" + record["id"] + "/content"
        assert client.get(path).status_code == 401
        content = client.get(path, headers=headers)
        assert content.status_code == 200
        assert hashlib.sha256(content.content).hexdigest() == record["checksumSha256"]
        assert (
            client.get(
                "/api/v1/a3/recordings/" + str(uuid4()) + "/content", headers=headers
            ).status_code
            == 404
        )


def test_input_path_and_invalid_metadata_never_write_audio(store):
    for file, start, frequency in [
        ("../outside.mp3", "2026-11-17T00:00:00Z", 119.1),
        ("sample.mp3", "2026-11-17T00:00:00", 119.1),
        ("sample.mp3", "2026-11-17T00:00:00Z", 900.0),
    ]:
        with pytest.raises(IngestError):
            store.ingest(file, start, frequency=frequency)
    assert not list(store.audio_root.rglob("*.mp3"))
