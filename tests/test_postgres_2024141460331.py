"""Real PostgreSQL contract tests; run only against an initialized isolated database."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from a4_speech.client_2024141460331 import ApiError


class SecretDsn_2024141460331(str):
    def __repr__(self):
        return "<redacted PostgreSQL DSN>"


@pytest.fixture
def pg_2024141460331(tmp_path):
    dsn = os.environ.get("A4_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("A4_TEST_DATABASE_URL is not configured")
    dsn = SecretDsn_2024141460331(dsn)
    import psycopg

    from a4_speech.postgres_2024141460331 import PostgresTeamApi_2024141460331

    tag = uuid4().hex
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"local audio fixture")
    with psycopg.connect(dsn) as db:
        db.execute(
            "INSERT INTO core.data_source(code,name,category) VALUES (%s,%s,'audio')", (tag, tag)
        )
        channel = db.execute(
            "INSERT INTO aud.channel(source_code,facility,name,mount_point) "
            "VALUES (%s,'OTHER',%s,%s) RETURNING id",
            (tag, tag, tag),
        ).fetchone()[0]
        recording = db.execute(
            "INSERT INTO aud.recording(channel_id,source_code,started_at,duration_ms,"
            "file_path,storage_state,checksum_sha256) "
            "VALUES (%s,%s,now(),5000,%s,'verified',%s) RETURNING id",
            (channel, tag, str(audio), hashlib.sha256(audio.read_bytes()).digest()),
        ).fetchone()[0]
        for kind in ("vad", "asr"):
            db.execute(
                "INSERT INTO asr.engine(code,name,kind) VALUES (%s,%s,%s)", (tag + kind, tag, kind)
            )
    api = PostgresTeamApi_2024141460331(dsn, tmp_path)
    try:
        yield api, str(recording), tag, dsn
    finally:
        with psycopg.connect(dsn) as db:
            db.execute("DELETE FROM aud.recording WHERE id=%s", (recording,))
            db.execute("DELETE FROM aud.channel WHERE id=%s", (channel,))
            db.execute("DELETE FROM asr.engine WHERE code IN (%s,%s)", (tag + "vad", tag + "asr"))
            db.execute("DELETE FROM core.data_source WHERE code=%s", (tag,))


def job_2024141460331(pg, count=2, key="request"):
    api, rid, tag, _ = pg
    api.add_segments(
        rid,
        [
            {
                "startMs": n * 1000,
                "endMs": (n + 1) * 1000,
                "isSpeech": True,
                "speakerRole": "unknown",
            }
            for n in range(count)
        ],
        tag + "vad",
    )
    ids = [s["id"] for s in api.list_segments(rid)]
    params = {
        "a4_10_20": {
            "recordingId": rid,
            "targetSegmentIds": ids,
            "businessTaskId": "local-business",
            "executionSnapshot": {"asr": {"engineCode": tag + "asr"}},
        }
    }
    return api.create_job(ids, tag + "asr", params, key), ids, params


def payload_2024141460331(ids):
    return {
        "status": "succeeded",
        "processedSegments": len(ids),
        "failedSegments": 0,
        "stats": {"a4_10_20": {"businessTaskId": "local-business", "successfulSegmentIds": ids}},
        "transcripts": [
            {"segmentId": sid, "text": "cleared " + str(i), "language": "en"}
            for i, sid in enumerate(ids)
        ],
    }


def test_real_save_readback_and_retry_2024141460331(pg_2024141460331):
    api, rid, _, _ = pg_2024141460331
    job, ids, params = job_2024141460331(pg_2024141460331)
    payload = payload_2024141460331(ids)
    metadata = {k: v for k, v in payload.items() if k != "transcripts"}
    api.complete_job(job["id"], metadata)
    assert all(api.get_transcript(sid) is None for sid in ids)
    api.complete_job(job["id"], payload)
    original = [api.get_transcript(sid) for sid in ids]
    api.complete_job(job["id"], payload)
    assert original == [api.get_transcript(sid) for sid in ids]
    assert original[0]["jobId"] == job["id"]
    assert api.get_transcript_by_id(original[0]["id"])["textRaw"] == "cleared 0"
    assert api.list_transcript_edits(original[0]["id"]) == []
    assert api.get_job(job["id"])["params"]["a4_10_20"] == params["a4_10_20"]
    assert all(
        s["hasTranscript"] and s["transcriptStatus"] == "done" for s in api.list_segments(rid)
    )
    assert job["id"] in [j["id"] for j in api.list_jobs(rid)]
    assert api.get_audio(rid) == b"local audio fixture"
    assert (
        api.get_recording(rid)["checksumSha256"]
        == hashlib.sha256(b"local audio fixture").hexdigest()
    )


def test_tampered_or_foreign_batch_rolls_back_2024141460331(pg_2024141460331):
    api, _, _, dsn = pg_2024141460331
    job, ids, _ = job_2024141460331(pg_2024141460331)
    import psycopg

    with psycopg.connect(dsn) as db:
        foreign = db.execute(
            "INSERT INTO asr.job(kind,recording_id) SELECT 'asr',recording_id "
            "FROM asr.job WHERE id=%s RETURNING id",
            (job["id"],),
        ).fetchone()[0]
        db.execute(
            "INSERT INTO asr.transcript(segment_id,job_id,text_raw) VALUES (%s,%s,'other source')",
            (ids[1], foreign),
        )
    with pytest.raises(ApiError) as error:
        api.complete_job(job["id"], payload_2024141460331(ids))
    assert error.value.status == 409
    assert api.get_transcript(ids[0]) is None
    with psycopg.connect(dsn) as db:
        db.execute("DELETE FROM asr.transcript WHERE segment_id=%s", (ids[1],))
    api.complete_job(job["id"], payload_2024141460331(ids))
    tampered = payload_2024141460331(ids)
    tampered["transcripts"][0]["text"] = "tampered"
    with pytest.raises(ApiError):
        api.complete_job(job["id"], tampered)
    assert api.get_transcript(ids[0])["textRaw"] == "cleared 0"


def test_idempotency_and_concurrent_reservations_2024141460331(pg_2024141460331):
    api, _, tag, _ = pg_2024141460331
    job, ids, params = job_2024141460331(pg_2024141460331)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = list(
            pool.map(lambda _: api.create_job(ids, tag + "asr", params, "request"), range(2))
        )
    assert all(j["id"] == job["id"] for j in jobs)
    with pytest.raises(ApiError) as error:
        api.create_job(ids, tag + "asr", {**params, "changed": True}, "request")
    assert error.value.status == 409
    with pytest.raises(ApiError):
        api.create_job(ids, tag + "asr", params, "different")


def test_segment_conflicts_and_invalid_targets_2024141460331(pg_2024141460331):
    api, rid, tag, _ = pg_2024141460331
    job, ids, params = job_2024141460331(pg_2024141460331)
    assert (
        api.add_segments(
            rid,
            [{"startMs": 0, "endMs": 1000, "isSpeech": True, "speakerRole": "unknown"}],
            tag + "vad",
        )["skipped"]
        == 1
    )
    with pytest.raises(ApiError):
        api.add_segments(
            rid, [{"startMs": 3000, "endMs": 4000}, {"startMs": 500, "endMs": 1500}], tag + "vad"
        )
    assert len(api.list_segments(rid)) == 2
    with pytest.raises(ApiError):
        api.complete_job(job["id"], payload_2024141460331([str(uuid4())]))
    with pytest.raises(ApiError):
        api.complete_job(job["id"], payload_2024141460331([ids[0], ids[0]]))


def test_audio_escape_missing_and_revision_denied_2024141460331(pg_2024141460331):
    api, rid, _, dsn = pg_2024141460331
    import psycopg

    with psycopg.connect(dsn) as db:
        db.execute("UPDATE aud.recording SET file_path='../outside.wav' WHERE id=%s", (rid,))
    with pytest.raises(ApiError) as error:
        api.get_audio(rid)
    assert error.value.status == 403
    with pytest.raises(ApiError) as error:
        api.get_recording(str(uuid4()))
    assert error.value.status == 404
    with pytest.raises(ApiError) as error:
        api.revise_transcript(
            str(uuid4()), text="edit", expected_revision=1, reason="test", bearer_token="unverified"
        )
    assert (error.value.status, error.value.code) == (503, "REVISION_AUTH_UNAVAILABLE")


def test_concurrent_segment_and_save_retry_2024141460331(pg_2024141460331):
    api, rid, tag, _ = pg_2024141460331
    spans = [{"startMs": 0, "endMs": 1000, "isSpeech": True, "speakerRole": "unknown"}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: api.add_segments(rid, spans, tag + "vad"), range(2)))
    assert sum(r["inserted"] for r in results) == 1
    assert len(api.list_segments(rid)) == 1
    job, ids, _ = job_2024141460331(pg_2024141460331, count=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: api.complete_job(job["id"], payload_2024141460331(ids)), range(2))
        )
    assert all(r["status"] == "succeeded" for r in results)
    assert api.get_transcript(ids[0])["revision"] == 1


def test_terminal_metadata_cannot_be_overwritten_2024141460331(pg_2024141460331):
    api, _, _, _ = pg_2024141460331
    job, ids, _ = job_2024141460331(pg_2024141460331)
    payload = payload_2024141460331(ids)
    api.complete_job(job["id"], payload)
    changed = {**payload, "stats": {"a4_10_20": {"businessTaskId": "another-business"}}}
    with pytest.raises(ApiError) as error:
        api.complete_job(job["id"], changed)
    assert error.value.status == 409
    with pytest.raises(ApiError):
        api.complete_job(
            job["id"], {"status": "failed", "processedSegments": 2, "failedSegments": 2}
        )
    assert api.get_job(job["id"])["status"] == "succeeded"


def test_unsaved_digest_rejects_first_save_tampering_2024141460331(pg_2024141460331):
    from a4_speech.results_2024141460331 import result_digest

    api, rid, _, _ = pg_2024141460331
    job, ids, _ = job_2024141460331(pg_2024141460331)
    payload = payload_2024141460331(ids)
    digests, result_digests = {}, {}
    for i, segment in enumerate(api.list_segments(rid)):
        text = {
            **segment,
            "segmentId": segment["id"],
            "textRaw": "cleared " + str(i),
            "language": "en",
        }
        fields = {
            key: text[key] for key in ("segmentId", "recordingId", "startMs", "endMs", "textRaw")
        }
        digests[segment["id"]] = hashlib.sha256(
            json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        result_digests[segment["id"]] = result_digest(text)
    payload["stats"]["a4_10_20"].update(transcriptDigests=digests, resultDigests=result_digests)
    api.complete_job(job["id"], {k: v for k, v in payload.items() if k != "transcripts"})
    payload["transcripts"][1]["language"] = "zh"
    with pytest.raises(ApiError) as error:
        api.complete_job(job["id"], payload)
    assert error.value.code == "TRANSCRIPT_DIGEST_MISMATCH"
    assert all(api.get_transcript(sid) is None for sid in ids)
    payload["transcripts"][1]["language"] = "en"
    api.complete_job(job["id"], payload)
    assert api.get_transcript(ids[1])["textRaw"] == "cleared 1"


def test_engine_and_recording_binding_2024141460331(pg_2024141460331):
    api, rid, tag, dsn = pg_2024141460331
    api.add_segments(rid, [{"startMs": 0, "endMs": 1000}], tag + "vad")
    ids = [s["id"] for s in api.list_segments(rid)]
    params = {"a4_10_20": {"recordingId": rid, "targetSegmentIds": ids}}
    import psycopg

    with psycopg.connect(dsn) as db:
        db.execute("UPDATE asr.engine SET is_enabled=false WHERE code=%s", (tag + "asr",))
    assert tag + "asr" not in [e["code"] for e in api.get_engines("asr")]
    with pytest.raises(ApiError) as error:
        api.create_job(ids, tag + "asr", params, "disabled")
    assert error.value.code == "ENGINE_UNAVAILABLE"
    with psycopg.connect(dsn) as db:
        db.execute("UPDATE asr.engine SET is_enabled=true WHERE code=%s", (tag + "asr",))
    with pytest.raises(ApiError) as error:
        api.create_job([str(uuid4())], tag + "asr", params, "mismatch")
    assert error.value.code == "JOB_CONTEXT_MISMATCH"
    with pytest.raises(ApiError):
        api.create_job(ids, tag + "vad", params, "wrong-kind")


def test_partial_releases_failed_and_unprocessed_targets_2024141460331(pg_2024141460331):
    api, rid, tag, _ = pg_2024141460331
    job, ids, params = job_2024141460331(pg_2024141460331, count=3)
    payload = {
        "status": "partial",
        "processedSegments": 2,
        "failedSegments": 1,
        "stats": {
            "a4_10_20": {
                "processingStatus": "partial",
                "totalSegments": 3,
                "successfulSegmentIds": ids[:1],
                "failedSegmentsDetails": [{"segmentId": ids[1], "code": "ASR_FAILED"}],
                "unprocessedSegmentIds": ids[2:],
            }
        },
    }
    api.complete_job(job["id"], payload)
    states = {s["id"]: s["transcriptStatus"] for s in api.list_segments(rid)}
    assert states == {ids[0]: "queued", ids[1]: "failed", ids[2]: "failed"}
    retry_params = {"a4_10_20": {**params["a4_10_20"], "targetSegmentIds": ids[1:]}}
    retry = api.create_job(ids[1:], tag + "asr", retry_params, "partial-retry")
    api.complete_job(job["id"], payload)
    assert all(s["transcriptStatus"] == "queued" for s in api.list_segments(rid))
    # Original successful result may still be explicitly saved after the retry starts.
    api.complete_job(
        job["id"],
        {
            **payload,
            "transcripts": [{"segmentId": ids[0], "text": "original success", "language": "en"}],
        },
    )
    assert api.get_transcript(ids[0])["jobId"] == job["id"]
    assert api.get_job(retry["id"])["status"] == "queued"


@pytest.mark.parametrize("status", ["failed", "canceled"])
def test_old_terminal_retry_cannot_release_new_owner_2024141460331(pg_2024141460331, status):
    api, rid, tag, _ = pg_2024141460331
    old, ids, params = job_2024141460331(pg_2024141460331)
    failed = {"status": status, "processedSegments": 0, "failedSegments": 0}
    api.complete_job(old["id"], failed)
    new = api.create_job(ids, tag + "asr", params, "new-owner")
    api.complete_job(old["id"], failed)
    assert all(s["transcriptStatus"] == "queued" for s in api.list_segments(rid))
    with pytest.raises(ApiError) as error:
        api.create_job(ids, tag + "asr", params, "third-owner")
    assert error.value.code == "SEGMENT_BUSY"
    api.complete_job(new["id"], payload_2024141460331(ids))
    assert all(api.get_transcript(sid)["jobId"] == new["id"] for sid in ids)
