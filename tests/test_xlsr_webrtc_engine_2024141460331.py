"""Behavioral tests for the frozen WebRTC + XLS-R engine and A-4 service contract."""

import os
import sys
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np
import pytest
from support_2024141460331 import RID, SharedServer

import a4_speech.models_xlsr_webrtc_2024141460331 as xlsr_module
from a4_speech.audio_2024141460331 import Audio
from a4_speech.client_2024141460331 import TeamApi
from a4_speech.service_2024141460331 import SpeechService_2024141460331


@pytest.fixture
def engine_config(tmp_path, monkeypatch):
    model_dir = tmp_path / "xlsr"
    model_dir.mkdir()
    assets = {
        "model.safetensors": b"verified test checkpoint",
        "config.json": b"{}",
        "preprocessor_config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "vocab.json": b"{}",
        "added_tokens.json": b"{}",
    }
    for name, content in assets.items():
        (model_dir / name).write_bytes(content)
    monkeypatch.setattr(
        xlsr_module,
        "XLSR_MODEL_SHA256",
        sha256(assets["model.safetensors"]).hexdigest(),
    )
    monkeypatch.setattr(
        xlsr_module.metadata,
        "version",
        lambda distribution: {
            "webrtcvad-wheels": "2.0.14.post1",
            "transformers": "5.20.2",
            "torch": "2.14.1+cu130",
        }[distribution],
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
            "engineCode": "xlsr-atc",
            "modelName": xlsr_module.XLSR_MODEL_NAME,
            "version": xlsr_module.XLSR_MODEL_REVISION,
            "modelPath": str(model_dir),
            "device": "cuda:0",
            "language": "en",
        },
    }


def install_vad_double(monkeypatch, speech_frame_indexes):
    indexes = set(speech_frame_indexes)

    class Vad:
        def __init__(self, mode):
            assert mode == 2
            self.frame_index = 0

        def is_speech(self, frame, sample_rate):
            assert sample_rate == 16000
            assert len(frame) == 640
            current = self.frame_index
            self.frame_index += 1
            return current in indexes

    module = ModuleType("webrtcvad")
    module.Vad = Vad
    monkeypatch.setitem(sys.modules, "webrtcvad", module)


def install_transformer_doubles(monkeypatch, text="cleared for takeoff"):
    state = SimpleNamespace(samples=None, processor_kwargs=None, load_kwargs=[])

    class Tensor:
        def to(self, *args, **kwargs):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.load_kwargs.append(("processor", path, kwargs))
            return cls()

        def __call__(self, samples, **kwargs):
            state.samples = np.asarray(samples).copy()
            state.processor_kwargs = kwargs
            return SimpleNamespace(input_values=Tensor())

        def batch_decode(self, predicted_ids):
            return [text]

    class Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.load_kwargs.append(("model", path, kwargs))
            return cls()

        def to(self, device):
            state.device = device
            return self

        def eval(self):
            state.eval_called = True

        def __call__(self, input_values):
            return SimpleNamespace(logits=Tensor())

    @contextmanager
    def inference_mode():
        yield

    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        Tensor=Tensor,
        float32="float32",
        inference_mode=inference_mode,
        argmax=lambda logits, dim: Tensor(),
    )
    transformers = ModuleType("transformers")
    transformers.AutoModelForCTC = Model
    transformers.Wav2Vec2Processor = Processor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return state


def test_webrtc_frames_merge_short_gaps_and_keep_source_relative_milliseconds(
    engine_config, monkeypatch
):
    install_vad_double(monkeypatch, [*range(10), *range(14, 29)])
    engine = xlsr_module.WebRtcXlsrEngine(engine_config)

    spans = engine.detect(Audio(np.full(16000, 0.25, dtype=np.float32), 16000))

    assert spans == [(0, 580)]


def test_webrtc_pads_partial_frame_but_clamps_segment_to_source_duration(
    engine_config, monkeypatch
):
    install_vad_double(monkeypatch, range(26))
    engine = xlsr_module.WebRtcXlsrEngine(engine_config)

    spans = engine.detect(Audio(np.full(8005, 0.25, dtype=np.float32), 16000))

    assert spans == [(0, 501)]


def test_xlsr_loads_only_local_checkpoint_and_recognizes_the_requested_time_slice(
    engine_config, monkeypatch
):
    state = install_transformer_doubles(monkeypatch)
    engine = xlsr_module.WebRtcXlsrEngine(engine_config)
    samples = np.arange(16000, dtype=np.float32) / 16000

    text = engine.recognize(Audio(samples, 16000), 100, 300)

    assert text == "cleared for takeoff"
    assert np.array_equal(state.samples, samples[1600:4800])
    assert state.processor_kwargs["sampling_rate"] == 16000
    assert all(call[2]["local_files_only"] is True for call in state.load_kwargs)
    assert state.load_kwargs[1][2]["trust_remote_code"] is False
    assert state.load_kwargs[1][2]["use_safetensors"] is True
    assert state.device == "cuda:0" and state.eval_called


def test_cli_selects_xlsr_adapter_for_the_frozen_engine_pair(engine_config, monkeypatch, tmp_path):
    import a4_speech.cli_2024141460331 as cli

    selected = object()

    class EngineFactory:
        def __init__(self, manifest):
            assert manifest["vad"]["engineCode"] == "webrtc-vad"
            assert manifest["asr"]["engineCode"] == "xlsr-atc"
            self.instance = selected

    monkeypatch.setattr(cli, "WebRtcXlsrEngine", EngineFactory, raising=False)

    engine = cli._engine_2024141460331({"engine": engine_config, "_directory": str(tmp_path)})

    assert engine.instance is selected


def test_snapshot_rejects_checkpoint_changed_after_engine_creation(engine_config):
    engine = xlsr_module.WebRtcXlsrEngine(engine_config)
    frozen = engine.snapshot()
    (Path(engine_config["asr"]["modelPath"]) / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint changed"):
        engine.verify_snapshot(frozen)


def test_xlsr_manifest_rejects_unrecorded_runtime_parameter(engine_config):
    engine_config["asr"]["beamSize"] = 20

    with pytest.raises(ValueError, match="unsupported XLS-R parameters"):
        xlsr_module.WebRtcXlsrEngine(engine_config)


def test_selected_engine_runs_through_a4_job_and_saves_transcript(
    engine_config, monkeypatch, tmp_path
):
    install_vad_double(monkeypatch, range(35))
    install_transformer_doubles(monkeypatch, "cleared for takeoff")
    server = SharedServer()
    server.engines = {
        "vad": {
            "code": "webrtc-vad",
            "name": "WebRTC VAD",
            "modelName": "webrtcvad-wheels",
            "kind": "vad",
            "isEnabled": True,
            "device": "cpu",
            "deployMode": "embedded",
            "version": "2.0.14.post1",
        },
        "asr": {
            "code": "xlsr-atc",
            "name": xlsr_module.XLSR_MODEL_NAME,
            "modelName": xlsr_module.XLSR_MODEL_NAME,
            "kind": "asr",
            "isEnabled": True,
            "device": "cuda",
            "deployMode": "embedded",
            "version": xlsr_module.XLSR_MODEL_REVISION,
        },
    }
    engine = xlsr_module.WebRtcXlsrEngine(engine_config)

    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        api = TeamApi(client, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path)
        result = service.run_2024141460331(RID, "xlsr-webrtc", business_task_id="business-1")

    assert result["status"] == "succeeded"
    assert result["saveConfirmed"] is True
    assert result["transcripts"][0]["textRaw"] == "cleared for takeoff"
    assert result["transcripts"][0]["startMs"] == 0
    assert result["transcripts"][0]["endMs"] == 700
    job = server.jobs[result["jobId"]]
    assert job["engineCode"] == "xlsr-atc"
    assert job["stats"]["a4_10_20"]["executionSnapshot"]["vad"]["engineCode"] == "webrtc-vad"


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("A4_XLSR_REAL_MODEL_TESTS") != "1",
    reason="set A4_XLSR_REAL_MODEL_TESTS=1 to run offline CUDA inference",
)
def test_real_xlsr_engine_saves_and_reads_a_frozen_atco2_recording(tmp_path):
    from a4_speech.cli_2024141460331 import _engine_2024141460331, load_config_2024141460331

    root = Path.cwd()
    config_path = Path(
        os.environ.get("A4_XLSR_CONFIG", root / "config.experimental-xlsr-webrtc.example.json")
    )
    audio_path = Path(
        os.environ.get(
            "A4_XLSR_REAL_AUDIO",
            root / "outputs/atco2-1h-20261021/original/ATCO2-ASRdataset-v1_beta/DATA/"
            "LKPR_RUZYNE_Radar_120_520MHz_20201029_075414.wav",
        )
    )
    if not config_path.is_file() or not audio_path.is_file():
        pytest.skip("local XLS-R config or frozen ATCO2 sample is unavailable")
    import soundfile as sf

    raw = audio_path.read_bytes()
    info = sf.info(audio_path)
    server = SharedServer()
    server.audio = raw
    server.recording.update(
        durationMs=round(info.duration * 1000),
        sampleRateHz=info.samplerate,
        audioChannels=info.channels,
        fileSizeBytes=len(raw),
        checksumSha256=sha256(raw).hexdigest(),
    )
    server.engines = {
        "vad": {
            "code": "webrtc-vad",
            "name": "WebRTC VAD",
            "modelName": "webrtcvad-wheels",
            "kind": "vad",
            "isEnabled": True,
            "device": "cpu",
            "deployMode": "embedded",
            "version": "2.0.14.post1",
        },
        "asr": {
            "code": "xlsr-atc",
            "name": xlsr_module.XLSR_MODEL_NAME,
            "modelName": xlsr_module.XLSR_MODEL_NAME,
            "kind": "asr",
            "isEnabled": True,
            "device": "cuda",
            "deployMode": "embedded",
            "version": xlsr_module.XLSR_MODEL_REVISION,
        },
    }
    config = load_config_2024141460331(config_path)
    engine = _engine_2024141460331(config)
    with httpx.Client(transport=httpx.MockTransport(server)) as client:
        api = TeamApi(client, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4")
        service = SpeechService_2024141460331(api, engine, receipt_dir=tmp_path)
        result = service.run_2024141460331(
            RID, "real-xlsr-webrtc", business_task_id="test-business"
        )
        assert result["status"] == "succeeded" and result["saveConfirmed"] is True
        transcript = result["transcripts"][0]
        history = service.read_transcript_2024141460331(transcript["id"], result["jobId"])

    assert history["transcript"]["textRaw"] == transcript["textRaw"]
