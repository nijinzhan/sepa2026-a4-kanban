"""Observe Chinese feedback, correlation and audio provenance at existing boundaries."""

import io
import json
import logging
import re
import wave
from datetime import datetime

import httpx
import numpy as np
import pytest
from support_2024141460331 import RID, Engine, SharedServer

from a4_speech.audio_2024141460331 import decode_audio
from a4_speech.cli_2024141460331 import main_2024141460331
from a4_speech.client_2024141460331 import ApiError, TeamApi
from a4_speech.service_2024141460331 import (
    ProcessingError_2024141460331,
    SpeechService_2024141460331,
)


def encoded_wav(rate=48000, channels=2, seconds=0.1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros((round(rate * seconds), channels), dtype="<i2").tobytes())
    return stream.getvalue()


@pytest.mark.parametrize(
    "rate,channels,actions", [(48000, 2, ["downmix_to_mono", "resample"]), (16000, 1, [])]
)
def test_decoded_audio_exposes_actual_source_and_conversion(rate, channels, actions):
    audio = decode_audio(encoded_wav(rate, channels))
    info = getattr(audio, "processing_info", None)
    assert isinstance(info, dict), "source/conversion provenance must accompany decoded audio"
    assert info["sourceSampleRateHz"] == rate
    assert info["sourceChannels"] == channels
    assert info["sourceFormat"] == "WAV"
    assert info["sourceSubtype"] == "PCM_16"
    assert info["sourceSamples"] == round(rate / 10)
    assert info["outputSampleRateHz"] == 16000
    assert info["outputChannels"] == 1
    assert info["outputSamples"] == 1600
    assert info["actions"] == actions
    assert audio.duration_ms == 100


def test_processing_error_provides_chinese_message_without_changing_code():
    result = ProcessingError_2024141460331(
        "SAVE_UNCONFIRMED", job_id="job-1"
    ).as_dict_2024141460331()
    assert result["code"] == "SAVE_UNCONFIRMED"
    assert result["saveConfirmed"] is False
    assert re.search(r"[\u4e00-\u9fff]", result.get("message", ""))
    assert result.get("traceId")


def test_cli_config_error_has_trace_and_chinese_feedback(tmp_path, capsys):
    assert (
        main_2024141460331(["--config", str(tmp_path / "missing-secret.json"), "check-model"]) == 2
    )
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["code"] == "CONFIG_ERROR"
    assert re.search(r"[\u4e00-\u9fff]", output.get("message", ""))
    assert output.get("traceId")
    entries = [json.loads(line) for line in captured.err.splitlines()]
    assert entries and all(e["traceId"] == output["traceId"] for e in entries)
    assert all(
        datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).utcoffset().total_seconds()
        == 0
        for e in entries
    )
    assert all(e.get("logCode") for e in entries)
    assert str(tmp_path) not in captured.out + captured.err


def test_service_trace_reaches_http_results_and_logs_and_provenance_roundtrip(tmp_path, caplog):
    server = SharedServer()
    server.audio = encoded_wav(48000, 2, 3)
    server.recording.update(sampleRateHz=48000, audioChannels=2)
    headers = []

    def boundary(request):
        headers.append(request.headers.get("X-Request-Id"))
        return server(request)

    caplog.set_level(logging.INFO, logger="a4_speech")
    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        api = TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, Engine(), receipt_dir=tmp_path)
        result = service.run_2024141460331(RID, "observability", "business-1")
        assert result.get("traceId"), "each entry point needs one correlation identifier"
        assert set(headers) == {result["traceId"]}
        assert result["audioProcessing"]["sourceSampleRateHz"] == 48000
        job = server.jobs[result["jobId"]]
        assert job["stats"]["a4_10_20"]["audioProcessing"] == result["audioProcessing"]
        history = service.read_transcript_2024141460331(
            result["transcripts"][0]["id"], result["jobId"]
        )
        assert history["audioProcessing"] == result["audioProcessing"]
        assert history["traceId"] != result["traceId"]
    messages = [r.getMessage() for r in caplog.records if r.name.startswith("a4_speech")]
    entries = [json.loads(m) for m in messages]
    own = [e for e in entries if e["traceId"] == result["traceId"]]
    assert any(e.get("jobId") == result["jobId"] for e in own)
    assert any(e.get("logCode") == "A4.AUDIO.DECODED" for e in own)
    assert all(e.get("timestamp") and e.get("logCode") for e in entries)
    assert not any("flight 1 cleared" in m for m in messages)


def test_no_speech_retains_conversion_record_and_chinese_explanation(tmp_path):
    server, engine = SharedServer(), Engine()
    engine.spans = []
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path,
        )
        result = service.transcribe_2024141460331(RID, "no-speech-trace")
    assert result["status"] == "no_speech" and not server.writes
    assert result.get("traceId") and result["audioProcessing"]["actions"] == []
    assert re.search(r"[\u4e00-\u9fff]", result.get("message", ""))


def test_upstream_error_keeps_its_request_id_and_has_local_trace(tmp_path):
    def unavailable(request):
        return httpx.Response(
            503,
            json={"code": "DATABASE_UNAVAILABLE", "message": "private-token-123"},
            headers={"X-Request-Id": "server-request-1"},
        )

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(),
            receipt_dir=tmp_path,
        )
        with pytest.raises(ProcessingError_2024141460331) as caught:
            service.transcribe_2024141460331(RID, "unavailable")
    error = caught.value.as_dict_2024141460331()
    assert error["requestId"] == "server-request-1"
    assert error.get("traceId") and error["traceId"] != error["requestId"]
    assert re.search(r"[\u4e00-\u9fff]", error.get("message", ""))
    assert "private-token-123" not in json.dumps(error)


def test_read_failure_exposes_trace_and_does_not_reuse_it_for_next_call(tmp_path):
    server = SharedServer()
    headers = []

    def boundary(request):
        headers.append(request.headers.get("X-Request-Id"))
        return server(request)

    with httpx.Client(transport=httpx.MockTransport(boundary)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(spans=[]),
            receipt_dir=tmp_path,
        )
        with pytest.raises(ApiError) as caught:
            service.read_transcript_2024141460331("missing")
        failed_trace = getattr(caught.value, "trace_id", None)
        assert failed_trace and headers == [failed_trace]
        headers.clear()
        result = service.transcribe_2024141460331(RID, "after-read-error")
        assert result["traceId"] != failed_trace
        assert set(headers) == {result["traceId"]}


def test_partial_result_provenance_survives_independent_save_and_read(tmp_path, caplog):
    server = SharedServer()
    caplog.set_level(logging.INFO, logger="a4_speech")
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(fail_at=2),
            receipt_dir=tmp_path,
        )
        result = service.transcribe_2024141460331(RID, "partial-trace", "business-1")
        assert result["status"] == "partial" and result.get("message")
        # Saving trusts the previously recorded decoder facts, not caller-edited diagnostics.
        result["audioProcessing"]["sourceSampleRateHz"] = 999
        saved = service.save_result_2024141460331(result)
        history = service.read_transcript_2024141460331(saved["transcripts"][0]["id"])
        assert saved["status"] == history["processingStatus"] == "partial"
        assert saved["audioProcessing"]["sourceSampleRateHz"] == 16000
        assert history["audioProcessing"] == saved["audioProcessing"]
        assert len({result["traceId"], saved["traceId"], history["traceId"]}) == 3
        assert history["processingTraceId"] == saved["processingTraceId"] == result["traceId"]
    assert any(
        record.levelno == logging.WARNING
        and json.loads(record.getMessage()).get("status") == "partial"
        for record in caplog.records
        if record.name.startswith("a4_speech")
    )


def test_unconfirmed_save_retains_trace_provenance_and_effective_text(tmp_path):
    server = SharedServer()

    def reject_save(request):
        if request.url.path.endswith("/complete") and "transcripts" in json.loads(request.content):
            return httpx.Response(503, json={"code": "DATABASE_UNAVAILABLE"})
        return server(request)

    with httpx.Client(transport=httpx.MockTransport(reject_save)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(),
            receipt_dir=tmp_path,
        )
        with pytest.raises(ProcessingError_2024141460331) as caught:
            service.run_2024141460331(RID, "unconfirmed-trace", "business-1")
    result = caught.value.as_dict_2024141460331()
    assert result["code"] == "SAVE_UNCONFIRMED" and result["saveStatus"] == "unconfirmed"
    assert len(result["transcripts"]) == 3 and not result["saveConfirmed"]
    assert result["traceId"] == result["processingTraceId"]
    assert result["audioProcessing"]["sourceSampleRateHz"] == 16000


def test_legacy_history_does_not_invent_audio_conversion_or_origin_trace(tmp_path):
    server = SharedServer()
    with httpx.Client(transport=httpx.MockTransport(server)) as http:
        service = SpeechService_2024141460331(
            TeamApi(http, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            Engine(),
            receipt_dir=tmp_path,
        )
        result = service.run_2024141460331(RID, "legacy-trace", "business-1")
        stats = server.jobs[result["jobId"]]["stats"]["a4_10_20"]
        stats.pop("audioProcessing")
        stats.pop("processingTraceId")
        history = service.read_transcript_2024141460331(
            result["transcripts"][0]["id"], result["jobId"]
        )
    assert "audioProcessing" not in history and "processingTraceId" not in history
    assert history["traceId"] != result["traceId"]
