"""Real file hashes and adapter behavior; inference doubles never claim model accuracy."""

import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from a4_speech.audio import Audio, decode_audio
from a4_speech.models import SherpaEngine


@pytest.fixture
def config(tmp_path):
    for filename in ("vad.onnx", "sensevoice.onnx", "tokens.txt"):
        (tmp_path / filename).write_bytes(filename.encode())
    return {
        "vad": {
            "engineCode": "silero-vad",
            "modelName": "silero",
            "version": "trusted-manifest-1",
            "modelPath": str(tmp_path / "vad.onnx"),
        },
        "asr": {
            "engineCode": "sensevoice",
            "modelName": "SenseVoiceSmall",
            "version": "trusted-manifest-1",
            "modelPath": str(tmp_path / "sensevoice.onnx"),
            "tokensPath": str(tmp_path / "tokens.txt"),
            "language": "en",
            "useItn": False,
        },
    }


def test_snapshot_hashes_actual_files_without_disclosing_paths(config):
    snapshot = SherpaEngine(config).snapshot()
    assert snapshot["vad"]["modelSha256"] == hashlib.sha256(b"vad.onnx").hexdigest()
    assert snapshot["asr"]["tokensSha256"] == hashlib.sha256(b"tokens.txt").hexdigest()
    assert snapshot["asr"]["parameters"]["language"] == "en"
    assert snapshot["asr"]["parameters"]["useItn"] is False
    assert "modelPath" not in json.dumps(snapshot)
    assert "tokensPath" not in json.dumps(snapshot)


def test_modified_model_rejects_frozen_snapshot(config):
    engine = SherpaEngine(config)
    frozen = engine.snapshot()
    Path(config["asr"]["modelPath"]).write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="changed|snapshot"):
        engine.verify_snapshot(frozen)


def test_external_version_claim_cannot_replace_trusted_manifest(config):
    engine = SherpaEngine(config)
    frozen = engine.snapshot()
    frozen["asr"]["version"] = "someone-else-version"
    with pytest.raises(ValueError, match="snapshot"):
        engine.verify_snapshot(frozen)


def test_snapshot_detects_tokens_changes(config):
    engine = SherpaEngine(config)
    frozen = engine.snapshot()
    Path(config["asr"]["tokensPath"]).write_bytes(b"different token mapping")
    with pytest.raises(ValueError, match="changed|snapshot"):
        engine.verify_snapshot(frozen)


def test_config_is_copied_before_execution(config):
    engine = SherpaEngine(config)
    config["asr"]["language"] = "zh"
    assert engine.snapshot()["asr"]["parameters"]["language"] == "en"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("asr", "modelPath", "missing.onnx"),
        ("asr", "language", "invalid-language"),
        ("asr", "language", []),
        ("asr", "useItn", "false"),
        ("vad", "threshold", 1.5),
        ("vad", "numThreads", 0),
        ("vad", "provider", "cuda"),
        ("vad", "windowSize", 1),
    ],
)
def test_invalid_manifest_fails_before_native_model_load(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        SherpaEngine(config)


def fake_sherpa(monkeypatch, spans=(), text="Recognized text", load_error=None):
    """Double only the native inference boundary, keeping adapter/time logic real."""
    state = SimpleNamespace(windows=[], input=None, kwargs=None)

    class Detector:
        def __init__(self, cfg, buffer_size_in_seconds):
            state.config = cfg
            self.queue = deque()

        def accept_waveform(self, samples):
            state.windows.append(samples.copy())

        def flush(self):
            self.queue.extend(
                SimpleNamespace(start=start, samples=np.zeros(length, dtype=np.float32))
                for start, length in spans
            )

        def empty(self):
            return not self.queue

        @property
        def front(self):
            return self.queue[0]

        def pop(self):
            self.queue.popleft()

    class Stream:
        result = SimpleNamespace(text=text)

        def accept_waveform(self, sample_rate, samples):
            state.input = (sample_rate, samples.copy())

    class Recognizer:
        @classmethod
        def from_sense_voice(cls, **kwargs):
            state.kwargs = kwargs
            if load_error:
                raise load_error
            return cls()

        def create_stream(self):
            return Stream()

        def decode_stream(self, stream):
            pass

    module = SimpleNamespace(
        VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()),
        VoiceActivityDetector=Detector,
        OfflineRecognizer=Recognizer,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return state


def test_vad_flushes_tail_and_clips_padding_to_source_time(config, monkeypatch):
    state = fake_sherpa(monkeypatch, spans=[(1600, 3200), (15040, 1400)])
    audio = Audio(np.ones(16000, dtype=np.float32), 16000)
    segments = SherpaEngine(config).detect(audio)
    assert segments == [(100, 300), (940, 1000)]
    assert sum(len(window) for window in state.windows) >= 16000
    np.testing.assert_array_equal(state.windows[-1][:128], np.ones(128))
    np.testing.assert_array_equal(state.windows[-1][128:], np.zeros(384))


def test_no_vad_segments_produces_empty_list(config, monkeypatch):
    fake_sherpa(monkeypatch)
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    assert SherpaEngine(config).detect(audio) == []


def test_actual_vad_overlap_is_rejected_instead_of_altering_boundaries(config, monkeypatch):
    fake_sherpa(monkeypatch, spans=[(1600, 3200), (3200, 3200)])
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="overlap"):
        SherpaEngine(config).detect(audio)


def test_adjacent_sample_spans_do_not_create_millisecond_overlap(config, monkeypatch):
    fake_sherpa(monkeypatch, spans=[(1601, 3200), (4801, 3200)])
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    assert SherpaEngine(config).detect(audio) == [(100, 301), (301, 501)]


def test_recognition_crops_by_original_recording_offsets(config, monkeypatch):
    state = fake_sherpa(monkeypatch, text="  Cleared to land.  ")
    audio = Audio(np.arange(16000, dtype=np.float32) / 16000, 16000)
    text = SherpaEngine(config).recognize(audio, 100, 300)
    assert text == "  Cleared to land.  "
    assert state.input[0] == 16000
    np.testing.assert_array_equal(state.input[1], audio.samples[1600:4800])
    assert state.kwargs["language"] == "en"
    assert state.kwargs["use_itn"] is False
    assert "version" not in state.kwargs


def test_prepare_loads_actual_config_without_running_segment_asr(config, monkeypatch):
    state = fake_sherpa(monkeypatch)
    engine = SherpaEngine(config)
    engine.prepare()
    assert state.kwargs["model"] == config["asr"]["modelPath"]
    assert state.kwargs["tokens"] == config["asr"]["tokensPath"]
    assert state.kwargs["language"] == "en"
    assert state.input is None


def test_prepare_reports_bad_model_loading_before_any_asr_attempt(config, monkeypatch):
    state = fake_sherpa(monkeypatch, load_error=RuntimeError("invalid ONNX model"))
    with pytest.raises(ValueError, match="initialization"):
        SherpaEngine(config).prepare()
    assert state.input is None


def test_prepare_revalidates_frozen_model_before_loading(config, monkeypatch):
    state = fake_sherpa(monkeypatch)
    engine = SherpaEngine(config)
    Path(config["asr"]["modelPath"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed|snapshot"):
        engine.prepare()
    assert state.kwargs is None


@pytest.mark.parametrize(("start", "end"), [(-1, 100), (100, 100), (0, 1001), (0.5, 100)])
def test_invalid_recognition_intervals_are_rejected(config, start, end):
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError):
        SherpaEngine(config).recognize(audio, start, end)


def test_blank_model_result_is_explicit_failure(config, monkeypatch):
    fake_sherpa(monkeypatch, text=" ")
    audio = Audio(np.zeros(16000, dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="empty|blank"):
        SherpaEngine(config).recognize(audio, 0, 1000)


def test_native_inference_never_uses_changed_files(config, monkeypatch):
    fake_sherpa(monkeypatch)
    engine = SherpaEngine(config)
    Path(config["asr"]["modelPath"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed|snapshot"):
        engine.recognize(Audio(np.zeros(16000, dtype=np.float32), 16000), 0, 1000)


@pytest.mark.real_model
def test_actual_silero_silence_and_sensevoice_audio():
    """Opt-in local assets; does not download models or call a team service."""
    config_path = os.environ.get("A4_REAL_MODEL_CONFIG")
    audio_path = os.environ.get("A4_REAL_TEST_AUDIO")
    if not config_path or not audio_path:
        pytest.skip("set A4_REAL_MODEL_CONFIG and A4_REAL_TEST_AUDIO to local files")
    configuration_path = Path(config_path).resolve()
    configuration = json.loads(configuration_path.read_text(encoding="utf-8-sig"))
    manifest = configuration.get("engine", configuration)
    # Match the CLI: model assets are relative to the manifest file rather
    # than the shell's working directory. Both complete and bare manifests work.
    for kind in ("vad", "asr"):
        item = manifest[kind]
        for key in ("modelPath", "tokensPath"):
            if key in item and isinstance(item[key], str):
                asset = Path(item[key])
                if not asset.is_absolute():
                    item[key] = str(configuration_path.parent / asset)
    engine = SherpaEngine(manifest)
    assert engine.detect(Audio(np.zeros(32000, dtype=np.float32), 16000)) == []
    audio = decode_audio(Path(audio_path).read_bytes())
    segments = engine.detect(audio)
    assert segments, "The provided real test recording must contain speech"
    texts = [engine.recognize(audio, start, end) for start, end in segments]
    assert all(text.strip() for text in texts)
