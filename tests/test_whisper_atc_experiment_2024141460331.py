"""Guard against reference loss and accidental in-training test claims."""

import importlib.util
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).resolve().parents[1] / "tools/whisper_atc_experiment_2024141460331.py"
    spec = importlib.util.spec_from_file_location("whisper_atc_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_annotation_removal_keeps_callsign_and_instruction():
    tool = load_tool()
    raw = "[#callsign]speedbird one two[/#callsign] climb [#altitude]one hundred[/#altitude]"
    assert tool.strip_annotations(raw).split() == [
        "speedbird",
        "one",
        "two",
        "climb",
        "one",
        "hundred",
    ]


def test_published_split_requires_unique_recording_membership():
    tool = load_tool()
    with pytest.raises(ValueError, match="both"):
        tool.build_split_index({"train": [{"id": "a"}], "validation": [{"id": "a"}]})


def test_missing_recording_is_unknown_not_held_out():
    tool = load_tool()
    index = tool.build_split_index({"train": [{"id": "a"}], "validation": [{"id": "b"}]})
    assert tool.split_label(index, "a") == "train"
    assert tool.split_label(index, "b") == "validation"
    assert tool.split_label(index, "c") == "unknown"


def test_short_form_cannot_silently_truncate_input():
    tool = load_tool()
    with pytest.raises(ValueError, match="30"):
        tool.validate_audio_span(480001, 16000)
    with pytest.raises(ValueError, match="16000"):
        tool.validate_audio_span(8000, 8000)
    tool.validate_audio_span(480000, 16000)


def test_pcm_verification_checks_samples_not_wav_header():
    import io

    import numpy as np
    import soundfile as sf

    tool = load_tool()
    data = np.array([0, 1000, -2000, 32767], dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, data, 16000, format="WAV", subtype="PCM_16")
    assert tool.pcm_digest(buf.getvalue()) == tool.pcm_digest(buf.getvalue())
    changed = io.BytesIO()
    sf.write(changed, data + np.int16(1), 16000, format="WAV", subtype="PCM_16")
    assert tool.pcm_digest(buf.getvalue()) != tool.pcm_digest(changed.getvalue())
