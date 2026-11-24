"""WebRTC boundaries with unchanged CPU SenseVoice and the existing service protocol."""

import json
import sys
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from types import ModuleType

import httpx
import numpy as np
import pytest
from support_2024141460331 import RID, SharedServer
from test_models_2024141460331 import fake_sherpa

from a4_speech.audio_2024141460331 import Audio
from a4_speech.cli_2024141460331 import _engine_2024141460331
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.models_2024141460331 import SherpaEngine
from a4_speech.service_2024141460331 import SpeechService_2024141460331


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    for name in ("sensevoice.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(name.encode())
    original_version = metadata.version
    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: "2.0.14.post1" if name == "webrtcvad-wheels" else original_version(name),
    )
    return {
        "vad": {
            "engineCode": "webrtc-vad",
            "modelName": "webrtcvad-wheels",
            "version": "2.0.14.post1",
            "aggressiveness": 2,
            "frameMs": 20,
            "mergeSilenceMs": 500,
            "minSpeechMs": 250,
            "paddingMs": 0,
        },
        "asr": {
            "engineCode": "sensevoice",
            "modelName": "SenseVoiceSmall",
            "version": "2024-07-17",
            "modelPath": str(tmp_path / "sensevoice.onnx"),
            "tokensPath": str(tmp_path / "tokens.txt"),
            "language": "en",
            "useItn": True,
            "numThreads": 2,
        },
    }


def native_vad_boundary(monkeypatch, active_frames):
    """Replace only native classification; keep PCM framing and boundary logic real."""
    active_frames = set(active_frames)

    class Vad:
        def __init__(self, mode):
            assert mode == 2
            self.index = 0

        def is_speech(self, pcm, sample_rate):
            assert sample_rate == 16000 and len(pcm) == 640
            active = self.index in active_frames
            self.index += 1
            return active

    native = ModuleType("webrtcvad")
    native.Vad = Vad
    monkeypatch.setitem(sys.modules, "webrtcvad", native)


def test_cli_executes_webrtc_sensevoice_without_a_silero_model(manifest, tmp_path):
    engine = _engine_2024141460331({"engine": manifest, "_directory": str(tmp_path)})
    snapshot = engine.snapshot()

    assert isinstance(engine, SherpaEngine)
    assert snapshot["vad"]["engineCode"] == "webrtc-vad"
    assert snapshot["vad"]["distributionVersion"] == "2.0.14.post1"
    assert snapshot["asr"]["engineCode"] == "sensevoice"
    assert snapshot["asr"]["modelSha256"] == sha256(b"sensevoice.onnx").hexdigest()
    assert snapshot["asr"]["parameters"]["provider"] == "cpu"
    assert "modelPath" not in json.dumps(snapshot)


@pytest.mark.parametrize(
    ("active", "samples", "expected"),
    [
        ([*range(10), *range(14, 29)], 16000, [(0, 580)]),
        (range(26), 8005, [(0, 501)]),
        (range(5), 16000, []),
        ([], 16000, []),
    ],
)
def test_webrtc_source_times_tail_short_speech_and_silence(
    manifest, monkeypatch, active, samples, expected
):
    native_vad_boundary(monkeypatch, active)
    engine = SherpaEngine(manifest)

    assert engine.detect(Audio(np.full(samples, 0.2, dtype=np.float32), 16000)) == expected


def test_same_sensevoice_model_recognizes_exact_requested_samples(manifest, monkeypatch):
    native = fake_sherpa(monkeypatch, text="cleared for takeoff")
    samples = np.arange(16000, dtype=np.float32) / 16000
    engine = SherpaEngine(manifest)

    text = engine.recognize(Audio(samples, 16000), 100, 300)

    assert text == "cleared for takeoff"
    np.testing.assert_array_equal(native.input[1], samples[1600:4800])
    assert native.kwargs["model"] == manifest["asr"]["modelPath"]
    assert native.kwargs["tokens"] == manifest["asr"]["tokensPath"]
    assert native.kwargs["provider"] == "cpu"
    assert native.kwargs["language"] == "en" and native.kwargs["use_itn"] is True


@pytest.mark.parametrize("asset", ["modelPath", "tokensPath"])
def test_changed_sensevoice_assets_are_rejected_before_vad(manifest, monkeypatch, asset):
    native_vad_boundary(monkeypatch, range(20))
    engine = SherpaEngine(manifest)
    Path(manifest["asr"][asset]).write_bytes(b"replacement")

    with pytest.raises(ValueError, match="changed|snapshot"):
        engine.detect(Audio(np.ones(16000, dtype=np.float32), 16000))


def test_changed_webrtc_runtime_is_rejected_before_execution(manifest, monkeypatch):
    engine = SherpaEngine(manifest)
    monkeypatch.setattr(metadata, "version", lambda name: "unregistered-version")

    with pytest.raises(ValueError, match="version|runtime|WebRTC"):
        engine.snapshot()


@pytest.mark.parametrize(
    ("key", "value"), [("version", "other"), ("frameMs", 15), ("paddingMs", 100)]
)
def test_unrecorded_webrtc_configuration_is_not_silently_accepted(manifest, key, value):
    manifest["vad"][key] = value

    with pytest.raises(ValueError, match="WebRTC|VAD"):
        SherpaEngine(manifest)


def test_webrtc_sensevoice_saves_and_reads_with_correct_model_history(
    manifest, monkeypatch, tmp_path
):
    native_vad_boundary(monkeypatch, range(35))
    fake_sherpa(monkeypatch, text="cleared for takeoff")
    engine = SherpaEngine(manifest)
    snapshot = engine.snapshot()
    server = SharedServer()
    server.engines = {
        kind: {
            "code": snapshot[kind]["engineCode"],
            "name": snapshot[kind]["modelName"],
            "modelName": snapshot[kind]["modelName"],
            "version": snapshot[kind]["version"],
            "kind": kind,
            "isEnabled": True,
            "device": "cpu",
            "deployMode": "embedded",
        }
        for kind in ("vad", "asr")
    }
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        service = SpeechService_2024141460331(
            TeamApi(client, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"),
            engine,
            receipt_dir=tmp_path / "receipts",
        )
        result = service.run_2024141460331(RID, "webrtc-sensevoice", business_task_id="business-1")
        assert result["status"] == "succeeded" and result["saveConfirmed"] is True
        transcript = result["transcripts"][0]
        history = service.read_transcript_2024141460331(transcript["id"], result["jobId"])

    assert (transcript["startMs"], transcript["endMs"]) == (0, 700)
    assert history["transcript"]["textRaw"] == "cleared for takeoff"
    job = server.jobs[result["jobId"]]
    assert job["engineCode"] == "sensevoice"
    assert job["stats"]["a4_10_20"]["executionSnapshot"] == snapshot
