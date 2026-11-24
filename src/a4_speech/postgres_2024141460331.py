"""Opt-in TeamApi over the frozen A-5 schema; no DDL or identity impersonation.

Connections belong to individual operations, so instances can be shared by workers.
All writers take the recording lock first (then job and ordered segment locks).
The private params namespace stores request identity without adding shared tables.
"""

import hashlib
import json
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx

from .client_2024141460331 import ApiError, revision_payload, segment_limit_2024141460331
from .results_2024141460331 import result_digest, validate_outcome

PRIVATE_2024141460331 = "_a4_postgres_2024141460331"
TERMINAL_2024141460331 = {"succeeded", "partial", "failed", "canceled"}


def identifier_2024141460331(value):
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise ApiError(422, "INVALID_IDENTIFIER") from None


def external_2024141460331(value):
    """Convert database values at the contract boundary without losing UTC."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, bytes | memoryview):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {key: external_2024141460331(item) for key, item in value.items()}
    if isinstance(value, list):
        return [external_2024141460331(item) for item in value]
    return value


def row_2024141460331(row):
    result = {}
    for key, value in row.items():
        first, *rest = key.split("_")
        result[first + "".join(word.title() for word in rest)] = external_2024141460331(value)
    return result


def merge_2024141460331(old, new):
    """Enrich immutable outcome evidence; never erase or replace prior evidence."""
    result = deepcopy(old)
    for key, value in new.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_2024141460331(result[key], value)
            elif result[key] != value:
                raise ApiError(409, "JOB_CONTEXT_CONFLICT")
        else:
            result[key] = deepcopy(value)
    return result


class PostgresTeamApi_2024141460331:
    """Read/write adapter; callers supply a DSN obtained from an environment variable."""

    def __init__(self, dsn, audio_root, *, max_segments=500, a5_url=None, identity_client=None):
        self._dsn = dsn
        self.audio_root = Path(audio_root).resolve()
        self.max_segments = segment_limit_2024141460331(max_segments)
        self.a5_url = a5_url.rstrip("/") if a5_url else None
        self.identity_client = identity_client

    @contextmanager
    def _transaction_2024141460331(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise ApiError(503, "POSTGRES_DRIVER_UNAVAILABLE") from None
        try:
            with psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=10) as db:
                db.execute("SET LOCAL lock_timeout = '15s'")
                db.execute("SET LOCAL statement_timeout = '30s'")
                yield db
        except ApiError as exc:
            # Reached only after the connection context completed its rollback.
            # A rollback/connection failure instead enters the psycopg.Error branch.
            if exc.code == "REVISION_HISTORY_INVALID":
                exc.write_outcome = "rolled_back"
            elif exc.code in {
                "REVISION_AUTH_DENIED",
                "REVISION_CONFLICT",
                "TRANSCRIPT_NOT_FOUND",
            }:
                exc.write_outcome = "not_written"
            raise
        except psycopg.Error as exc:
            # Database errors can contain credentials, paths and SQL parameters.
            state = exc.sqlstate or ""
            status = 409 if state.startswith("23") else (422 if state.startswith("22") else 503)
            raise ApiError(
                status,
                "DATABASE_CONFLICT"
                if status == 409
                else "POSTGRES_UNAVAILABLE"
                if status == 503
                else "DATABASE_INPUT_INVALID",
            ) from None

    def _recording_2024141460331(self, db, recording_id, lock=False):
        row = db.execute(
            "SELECT * FROM aud.recording WHERE id=%s" + (" FOR UPDATE" if lock else ""),
            (identifier_2024141460331(recording_id),),
        ).fetchone()
        if row is None:
            raise ApiError(404, "RECORDING_NOT_FOUND")
        return row

    def _audio_path_2024141460331(self, row):
        if row["storage_volume"] != "local" or not row["file_path"]:
            raise ApiError(404, "AUDIO_UNAVAILABLE")
        try:
            path = Path(row["file_path"])
            path = (path if path.is_absolute() else self.audio_root / path).resolve()
            path.relative_to(self.audio_root)
        except (ValueError, OSError, RuntimeError):
            raise ApiError(403, "AUDIO_PATH_FORBIDDEN") from None
        return path

    def get_recording(self, recording_id):
        with self._transaction_2024141460331() as db:
            row = self._recording_2024141460331(db, recording_id)
            if row["file_path"]:
                self._audio_path_2024141460331(row)
            return row_2024141460331(row)

    def get_audio(self, recording_id):
        with self._transaction_2024141460331() as db:
            row = self._recording_2024141460331(db, recording_id)
            if row["storage_state"] not in {"stored", "verified"}:
                raise ApiError(404, "AUDIO_UNAVAILABLE")
            path = self._audio_path_2024141460331(row)
            try:
                content = path.read_bytes()
            except OSError:
                raise ApiError(404, "AUDIO_UNAVAILABLE") from None
            if not content:
                raise ApiError(404, "AUDIO_UNAVAILABLE")
            return content

    def get_engines(self, kind):
        if kind not in {"asr", "vad"}:
            raise ApiError(422, "INVALID_ENGINE_KIND")
        with self._transaction_2024141460331() as db:
            return [
                row_2024141460331(row)
                for row in db.execute(
                    "SELECT * FROM asr.engine WHERE kind=%s AND is_enabled ORDER BY code", (kind,)
                ).fetchall()
            ]

    def _engine_2024141460331(self, db, code, kind):
        if not db.execute(
            "SELECT code FROM asr.engine WHERE code=%s AND kind=%s AND is_enabled FOR SHARE",
            (code, kind),
        ).fetchone():
            raise ApiError(422, "ENGINE_UNAVAILABLE")

    def list_segments(self, recording_id):
        with self._transaction_2024141460331() as db:
            self._recording_2024141460331(db, recording_id)
            return [
                row_2024141460331(row)
                for row in db.execute(
                    "SELECT * FROM aud.segment WHERE recording_id=%s ORDER BY start_ms",
                    (recording_id,),
                ).fetchall()
            ]

    def add_segments(self, recording_id, segments, engine_code):
        if not isinstance(segments, list) or not 1 <= len(segments) <= 2000:
            raise ApiError(422, "INVALID_SEGMENTS")
        with self._transaction_2024141460331() as db:
            recording = self._recording_2024141460331(db, recording_id, True)
            self._engine_2024141460331(db, engine_code, "vad")
            existing = db.execute(
                "SELECT * FROM aud.segment WHERE recording_id=%s ORDER BY start_ms FOR UPDATE",
                (recording_id,),
            ).fetchall()
            sequence = max((row["sequence_no"] for row in existing), default=0)
            inserted = skipped = 0
            for segment in segments:
                if not isinstance(segment, dict):
                    raise ApiError(422, "INVALID_SEGMENT")
                start, end = segment.get("startMs"), segment.get("endMs")
                speech, role = segment.get("isSpeech", True), segment.get("speakerRole", "unknown")
                score = segment.get("vadScore")
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end
                    or (recording["duration_ms"] is not None and end > recording["duration_ms"])
                    or type(speech) is not bool
                    or role not in {"atc", "pilot", "unknown", "mixed"}
                ):
                    raise ApiError(422, "INVALID_SEGMENT")
                overlaps = [
                    row for row in existing if start < row["end_ms"] and row["start_ms"] < end
                ]
                if overlaps:
                    if len(overlaps) == 1 and all(
                        overlaps[0][key] == value
                        for key, value in {
                            "start_ms": start,
                            "end_ms": end,
                            "is_speech": speech,
                            "speaker_role": role,
                            "vad_score": score,
                        }.items()
                    ):
                        skipped += 1
                        continue
                    raise ApiError(409, "SEGMENT_CONFLICT")
                sequence += 1
                row = db.execute(
                    "INSERT INTO aud.segment(recording_id,sequence_no,start_ms,end_ms,"
                    "is_speech,speaker_role,vad_score) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                    (recording_id, sequence, start, end, speech, role, score),
                ).fetchone()
                existing.append(row)
                inserted += 1
            db.execute(
                "UPDATE aud.recording SET segment_count=(SELECT count(*) FROM aud.segment "
                "WHERE recording_id=%s), speech_count=(SELECT count(*) FROM aud.segment "
                "WHERE recording_id=%s AND is_speech), speech_duration_ms=(SELECT "
                "coalesce(sum(duration_ms),0) FROM aud.segment "
                "WHERE recording_id=%s AND is_speech) WHERE id=%s",
                (recording_id,) * 4,
            )
            return {"recordingId": str(recording["id"]), "inserted": inserted, "skipped": skipped}

    def _job_result_2024141460331(self, row):
        result = row_2024141460331(row)
        context = row["params"].get("a4_10_20", {})
        result["totalSegments"] = len(context.get("targetSegmentIds", []))
        return result

    def _reservation_owners_2024141460331(self, db, recording_id):
        """Caller holds recording lock; private ownership outlives status metadata.

        A succeeded result awaiting explicit save still owns its successful targets.
        Release is monotonic, so replaying old outcomes cannot reclaim new ownership.
        The fallback supports jobs created before the explicit ownership field existed.
        """
        owners = {}
        for job in db.execute(
            "SELECT id,params,stats,status FROM asr.job WHERE recording_id=%s "
            "AND kind='asr' ORDER BY submitted_at,id",
            (recording_id,),
        ).fetchall():
            private = job["params"].get(PRIVATE_2024141460331)
            if not isinstance(private, dict):
                continue
            reserved = private.get("reservedSegmentIds")
            if reserved is None:
                targets = job["params"].get("a4_10_20", {}).get("targetSegmentIds", [])
                if job["status"] in {"failed", "canceled"}:
                    reserved = []
                elif job["status"] == "partial":
                    reserved = job["stats"].get("a4_10_20", {}).get("successfulSegmentIds", [])
                else:
                    reserved = targets
            for sid in reserved:
                owners[sid] = str(job["id"])
        return owners

    def create_job(self, segment_ids, engine_code, params, idempotency_key):
        if not isinstance(segment_ids, list) or not 1 <= len(segment_ids) <= self.max_segments:
            raise ApiError(422, "INVALID_TARGETS")
        ids = [identifier_2024141460331(sid) for sid in segment_ids]
        if (
            len(set(ids)) != len(ids)
            or not isinstance(params, dict)
            or PRIVATE_2024141460331 in params
        ):
            raise ApiError(422, "INVALID_TARGETS")
        if (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 64
            or not idempotency_key.isascii()
            or not idempotency_key.isprintable()
        ):
            raise ApiError(422, "INVALID_IDEMPOTENCY_KEY")
        context = params.get("a4_10_20")
        if not isinstance(context, dict) or context.get("targetSegmentIds") != ids:
            raise ApiError(422, "JOB_CONTEXT_MISMATCH")
        recording_id = identifier_2024141460331(context.get("recordingId"))
        snapshot = context.get("executionSnapshot", {})
        if snapshot and (
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("asr"), dict)
            or snapshot.get("asr", {}).get("engineCode") != engine_code
        ):
            raise ApiError(422, "JOB_CONTEXT_MISMATCH")
        request = {"segmentIds": ids, "engineCode": engine_code, "params": params}
        with self._transaction_2024141460331() as db:
            from psycopg.types.json import Jsonb

            # Cross-recording reuse of a key must conflict as well.
            db.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                ("a4-2024141460331:" + idempotency_key,),
            )
            self._recording_2024141460331(db, recording_id, True)
            old = db.execute(
                "SELECT * FROM asr.job WHERE params -> %s ->> 'idempotencyKey' = %s",
                (PRIVATE_2024141460331, idempotency_key),
            ).fetchall()
            if old:
                if (
                    len(old) != 1
                    or old[0]["params"][PRIVATE_2024141460331].get("request") != request
                ):
                    raise ApiError(409, "IDEMPOTENCY_CONFLICT")
                return self._job_result_2024141460331(old[0])
            self._engine_2024141460331(db, engine_code, "asr")
            owners = self._reservation_owners_2024141460331(db, recording_id)
            if any(sid in owners for sid in ids):
                raise ApiError(409, "SEGMENT_BUSY")
            rows = db.execute(
                "SELECT * FROM aud.segment WHERE id=ANY(%s::uuid[]) ORDER BY id FOR UPDATE", (ids,)
            ).fetchall()
            if len(rows) != len(ids) or any(
                str(row["recording_id"]) != recording_id or not row["is_speech"] for row in rows
            ):
                raise ApiError(422, "JOB_TARGET_MISMATCH")
            if any(
                row["has_transcript"]
                or row["transcript_status"] in {"queued", "processing", "done", "edited"}
                for row in rows
            ):
                raise ApiError(409, "SEGMENT_BUSY")
            if db.execute(
                "SELECT id FROM asr.transcript WHERE segment_id=ANY(%s::uuid[]) LIMIT 1", (ids,)
            ).fetchone():
                raise ApiError(409, "RESULT_EXISTS")
            stored = {
                **deepcopy(params),
                PRIVATE_2024141460331: {
                    "idempotencyKey": idempotency_key,
                    "request": deepcopy(request),
                    "reservedSegmentIds": ids,
                },
            }
            row = db.execute(
                "INSERT INTO asr.job(kind,engine_code,recording_id,params) "
                "VALUES ('asr',%s,%s,%s) RETURNING *",
                (engine_code, recording_id, Jsonb(stored)),
            ).fetchone()
            db.execute(
                "UPDATE aud.segment SET transcript_status='queued' WHERE id=ANY(%s::uuid[])", (ids,)
            )
            return self._job_result_2024141460331(row)

    def get_job(self, job_id):
        with self._transaction_2024141460331() as db:
            row = db.execute(
                "SELECT * FROM asr.job WHERE id=%s", (identifier_2024141460331(job_id),)
            ).fetchone()
            if row is None:
                raise ApiError(404, "JOB_NOT_FOUND")
            return self._job_result_2024141460331(row)

    def list_jobs(self, recording_id):
        with self._transaction_2024141460331() as db:
            self._recording_2024141460331(db, recording_id)
            return [
                self._job_result_2024141460331(row)
                for row in db.execute(
                    "SELECT * FROM asr.job WHERE recording_id=%s AND kind='asr' "
                    "ORDER BY submitted_at,id",
                    (recording_id,),
                ).fetchall()
            ]

    def complete_job(self, job_id, payload):
        if not isinstance(payload, dict) or payload.get("status") not in TERMINAL_2024141460331:
            raise ApiError(422, "INVALID_JOB_OUTCOME")
        texts = payload.get("transcripts", [])
        stats = payload.get("stats", {})
        if not isinstance(texts, list) or not isinstance(stats, dict):
            raise ApiError(422, "INVALID_JOB_OUTCOME")
        with self._transaction_2024141460331() as db:
            from psycopg.types.json import Jsonb

            jid = identifier_2024141460331(job_id)
            initial = db.execute("SELECT recording_id FROM asr.job WHERE id=%s", (jid,)).fetchone()
            if not initial:
                raise ApiError(404, "JOB_NOT_FOUND")
            self._recording_2024141460331(db, initial["recording_id"], True)
            job = db.execute("SELECT * FROM asr.job WHERE id=%s FOR UPDATE", (jid,)).fetchone()
            context = job["params"].get("a4_10_20", {})
            ids = context.get("targetSegmentIds", [])
            if (
                job["kind"] != "asr"
                or not ids
                or str(job["recording_id"]) != context.get("recordingId")
            ):
                raise ApiError(409, "JOB_CONTEXT_MISMATCH")
            rows = db.execute(
                "SELECT * FROM aud.segment WHERE id=ANY(%s::uuid[]) ORDER BY id FOR UPDATE", (ids,)
            ).fetchall()
            if len(rows) != len(ids) or any(
                row["recording_id"] != job["recording_id"] for row in rows
            ):
                raise ApiError(409, "JOB_TARGET_MISMATCH")
            processed = payload.get("processedSegments", job["processed_segments"])
            failed = payload.get("failedSegments", job["failed_segments"])
            if (
                type(processed) is not int
                or type(failed) is not int
                or not 0 <= failed <= processed <= len(ids)
                or (payload["status"] == "succeeded" and (processed != len(ids) or failed))
            ):
                raise ApiError(422, "INVALID_JOB_COUNTERS")
            error = payload.get("errorMessage", job["error_message"])
            if job["status"] in TERMINAL_2024141460331 and (
                job["status"],
                job["processed_segments"],
                job["failed_segments"],
                job["error_message"],
            ) != (payload["status"], processed, failed, error):
                raise ApiError(409, "JOB_OUTCOME_CONFLICT")
            merged = merge_2024141460331(job["stats"], stats)
            details = merged.get("a4_10_20", {})
            if not isinstance(details, dict):
                raise ApiError(422, "JOB_CONTEXT_MISMATCH")
            for key in ("recordingId", "targetSegmentIds", "executionSnapshot", "businessTaskId"):
                if key in context and key in details and context[key] != details[key]:
                    raise ApiError(409, "JOB_CONTEXT_CONFLICT")
            if payload["status"] == "partial" or "processingStatus" in details:
                try:
                    validate_outcome(context, details, payload["status"], processed, failed)
                except ValueError:
                    raise ApiError(422, "INVALID_JOB_OUTCOME") from None
            owners = self._reservation_owners_2024141460331(db, job["recording_id"])
            owned = {sid for sid in ids if owners.get(sid) == jid}
            seen = set()
            for text in texts:
                if not isinstance(text, dict):
                    raise ApiError(422, "INVALID_TRANSCRIPT")
                sid = identifier_2024141460331(text.get("segmentId"))
                raw, language = text.get("text"), text.get("language", "en")
                if (
                    sid not in ids
                    or sid in seen
                    or not isinstance(raw, str)
                    or not raw.strip()
                    or not isinstance(language, str)
                    or not language.strip()
                ):
                    raise ApiError(409, "TRANSCRIPT_ASSOCIATION")
                seen.add(sid)
                segment = next(row for row in rows if str(row["id"]) == sid)
                observed = {
                    **row_2024141460331(segment),
                    "segmentId": sid,
                    "textRaw": raw,
                    "language": language,
                }
                fields = {
                    key: observed[key]
                    for key in ("segmentId", "recordingId", "startMs", "endMs", "textRaw")
                }
                digest = hashlib.sha256(
                    json.dumps(
                        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
                for name, expected in (
                    ("transcriptDigests", digest),
                    ("resultDigests", result_digest(observed)),
                ):
                    if name in details and (
                        not isinstance(details[name], dict) or details[name].get(sid) != expected
                    ):
                        raise ApiError(409, "TRANSCRIPT_DIGEST_MISMATCH")
                existing = db.execute(
                    "SELECT * FROM asr.transcript WHERE segment_id=%s FOR UPDATE", (sid,)
                ).fetchone()
                if existing:
                    if (
                        existing["job_id"] != job["id"]
                        or existing["text_raw"] != raw
                        or existing["language"] != language
                    ):
                        raise ApiError(409, "TRANSCRIPT_CONFLICT")
                else:
                    if sid not in owned:
                        raise ApiError(409, "RESERVATION_CONFLICT")
                    db.execute(
                        "INSERT INTO asr.transcript("
                        "segment_id,job_id,engine_code,text_raw,language) "
                        "VALUES (%s,%s,%s,%s,%s)",
                        (sid, jid, job["engine_code"], raw, language),
                    )
            if texts and (
                payload["status"] not in {"succeeded", "partial"}
                or len(texts) != processed - failed
                or (
                    "successfulSegmentIds" in details
                    and set(details["successfulSegmentIds"]) != seen
                )
            ):
                raise ApiError(409, "TRANSCRIPT_OUTCOME_MISMATCH")
            # Release only this job's current reservations. An old outcome replay must
            # not change segments already reserved by a newer recognition attempt.
            if payload["status"] in {"failed", "canceled"}:
                release = owned
            elif payload["status"] == "partial":
                release = owned - set(details["successfulSegmentIds"])
            else:
                release = set()
            if release:
                db.execute(
                    "UPDATE aud.segment SET transcript_status='failed' "
                    "WHERE id=ANY(%s::uuid[]) AND NOT has_transcript",
                    (sorted(release),),
                )
            updated_params = deepcopy(job["params"])
            private = updated_params.setdefault(PRIVATE_2024141460331, {})
            private["reservedSegmentIds"] = [sid for sid in ids if sid in owned - release - seen]
            row = db.execute(
                "UPDATE asr.job SET status=%s,processed_segments=%s,failed_segments=%s,"
                "error_message=%s,stats=%s,params=%s,finished_at=coalesce(finished_at,now()) "
                "WHERE id=%s RETURNING *",
                (
                    payload["status"],
                    processed,
                    failed,
                    error,
                    Jsonb(merged),
                    Jsonb(updated_params),
                    jid,
                ),
            ).fetchone()
            return self._job_result_2024141460331(row)

    def _transcript_2024141460331(self, db, key, value):
        # key is an internal fixed SQL identifier, never caller input.
        row = db.execute(
            "SELECT t.*,s.recording_id,s.start_ms,s.end_ms,s.start_at,s.end_at "
            "FROM asr.transcript t JOIN aud.segment s ON s.id=t.segment_id WHERE t." + key + "=%s",
            (identifier_2024141460331(value),),
        ).fetchone()
        if row is None:
            return None
        result = row_2024141460331(row)
        result["text"] = row["text_norm"] if row["text_norm"] is not None else row["text_raw"]
        result["edits"] = [
            row_2024141460331(edit)
            for edit in db.execute(
                "SELECT * FROM asr.transcript_edit WHERE transcript_id=%s ORDER BY revision",
                (row["id"],),
            ).fetchall()
        ]
        return result

    def get_transcript(self, segment_id):
        with self._transaction_2024141460331() as db:
            return self._transcript_2024141460331(db, "segment_id", segment_id)

    def get_transcript_by_id(self, transcript_id):
        with self._transaction_2024141460331() as db:
            result = self._transcript_2024141460331(db, "id", transcript_id)
            if result is None:
                raise ApiError(404, "TRANSCRIPT_NOT_FOUND")
            return result

    def list_transcript_edits(self, transcript_id):
        return self.get_transcript_by_id(transcript_id)["edits"]

    def revise_transcript(
        self,
        transcript_id,
        *,
        text=None,
        review_status=None,
        expected_revision,
        reason,
        bearer_token=None,
    ):
        try:
            revision_payload(
                text=text,
                review_status=review_status,
                expected_revision=expected_revision,
                reason=reason,
            )
        except ValueError:
            raise ApiError(422, "INVALID_REVISION", write_outcome="not_written") from None
        if (
            not isinstance(bearer_token, str)
            or not bearer_token
            or any(ord(c) < 33 or ord(c) > 126 for c in bearer_token)
        ):
            raise ApiError(401, "REVISION_AUTH_REQUIRED", write_outcome="not_written")
        if not self.a5_url or self.identity_client is None:
            raise ApiError(503, "REVISION_AUTH_UNAVAILABLE", write_outcome="not_written")
        try:
            response = self.identity_client.get(
                self.a5_url + "/auth/me",
                headers={"Authorization": "Bearer " + bearer_token},
                follow_redirects=False,
            )
        except httpx.HTTPError:
            raise ApiError(503, "REVISION_AUTH_UNAVAILABLE", write_outcome="not_written") from None
        if response.status_code in {401, 403}:
            raise ApiError(
                response.status_code,
                "REVISION_AUTH_DENIED",
                write_outcome="not_written",
            )
        if response.status_code != 200:
            raise ApiError(503, "REVISION_AUTH_UNAVAILABLE", write_outcome="not_written")
        try:
            principal = response.json()
            actor = str(UUID(principal["id"]))
            if principal.get("status") != "active":
                raise ValueError("inactive identity")
        except (ValueError, TypeError, KeyError, AttributeError):
            raise ApiError(503, "REVISION_AUTH_UNAVAILABLE", write_outcome="not_written") from None
        with self._transaction_2024141460331() as db:
            # Token identity comes only from IAM; permissions are re-read from this DB.
            user = db.execute(
                "SELECT status,iam.fn_has_permission(id,'a4:transcript:edit') AS allowed "
                "FROM iam.user_account WHERE id=%s FOR SHARE",
                (actor,),
            ).fetchone()
            if not user or user["status"] != "active" or not user["allowed"]:
                raise ApiError(403, "REVISION_AUTH_DENIED")
            tid = identifier_2024141460331(transcript_id)
            before = db.execute(
                "SELECT * FROM asr.transcript WHERE id=%s FOR UPDATE", (tid,)
            ).fetchone()
            if before is None:
                raise ApiError(404, "TRANSCRIPT_NOT_FOUND")
            if before["revision"] != expected_revision:
                raise ApiError(409, "REVISION_CONFLICT")
            old_text = (
                before["text_norm"] if before["text_norm"] is not None else before["text_raw"]
            )
            new_text = old_text if text is None else text
            new_status = before["review_status"] if review_status is None else review_status
            changed = new_text != old_text or new_status != before["review_status"]
            if changed:
                db.execute(
                    "SELECT set_config('alpha.actor_user_id',%s,true), "
                    "set_config('alpha.edit_reason',%s,true)",
                    (actor, reason),
                )
                # Explicit revision also covers review-only edits under the original V7
                # trigger; its text-change branch assigns the same OLD.revision + 1.
                after = db.execute(
                    "UPDATE asr.transcript SET text_norm=%s,review_status=%s,"
                    "revision=revision+1,is_edited=true,edited_by=%s,edited_at=clock_timestamp(),"
                    "token_count=%s WHERE id=%s AND revision=%s RETURNING *",
                    (
                        before["text_norm"] if new_text == old_text else new_text,
                        new_status,
                        actor,
                        len(new_text.split()),
                        tid,
                        expected_revision,
                    ),
                ).fetchone()
                if after is None:
                    raise ApiError(409, "REVISION_CONFLICT")
                history = db.execute(
                    "SELECT * FROM asr.transcript_edit WHERE transcript_id=%s AND revision=%s",
                    (tid, expected_revision + 1),
                ).fetchall()
                if (
                    len(history) != 1
                    or str(history[0]["editor_user_id"]) != actor
                    or history[0]["reason"] != reason
                ):
                    raise ApiError(503, "REVISION_HISTORY_INVALID")
                from psycopg.types.json import Jsonb

                db.execute(
                    "SELECT iam.fn_write_audit(%s,'asr.transcript.edit','A-4','transcript',"
                    "%s,%s,%s,'success',%s,NULL,NULL,NULL)",
                    (
                        actor,
                        tid,
                        Jsonb(
                            {
                                "revision": expected_revision,
                                "review_status": before["review_status"],
                            }
                        ),
                        Jsonb({"revision": after["revision"], "review_status": new_status}),
                        reason,
                    ),
                )
            return {**self._transcript_2024141460331(db, "id", tid), "changed": changed}
