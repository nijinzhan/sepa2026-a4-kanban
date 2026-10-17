"""Audio behavior tests use actual WAV bytes rather than a decoder double."""

import io
import wave

import numpy as np
import pytest

from a4_speech.audio import Audio, decode_audio


def wav_bytes(samples, sample_rate):
    output = io.BytesIO()
    values = np.asarray(samples, dtype="<i2")
    channels = values.shape[1] if values.ndim == 2 else 1
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(values.tobytes())
    return output.getvalue()


def test_stereo_is_averaged_and_timing_preserved():
    samples = np.tile([8192, 24576], (16000, 1))
    audio = decode_audio(wav_bytes(samples, 16000))
    assert audio.samples.dtype == np.float32
    assert audio.samples.shape == (16000,)
    assert audio.sample_rate == 16000
    assert audio.duration_ms == 1000
    np.testing.assert_allclose(audio.samples, 0.5)


def test_resampling_preserves_duration_and_end_of_recording():
    samples = np.full(8000, 8192, dtype=np.int16)
    audio = decode_audio(wav_bytes(samples, 8000))
    assert audio.samples.shape == (16000,)
    assert audio.duration_ms == 1000
    assert np.mean(audio.samples[-100:]) > 0.24


@pytest.mark.parametrize("data", [b"", b"not an audio file"])
def test_decode_rejects_invalid_data(data):
    with pytest.raises(ValueError, match="audio"):
        decode_audio(data)


def test_decode_rejects_empty_waveform():
    with pytest.raises(ValueError, match="empty"):
        decode_audio(wav_bytes([], 16000))


@pytest.mark.parametrize(
    ("samples", "sample_rate"),
    [
        (np.array([np.nan], dtype=np.float32), 16000),
        (np.array([np.inf], dtype=np.float32), 16000),
        (np.zeros((3, 2), dtype=np.float32), 16000),
        (np.zeros(10, dtype=np.float32), 0),
        (np.zeros(0, dtype=np.float32), 16000),
    ],
)
def test_audio_rejects_invalid_waveforms(samples, sample_rate):
    with pytest.raises(ValueError):
        Audio(samples, sample_rate)


def test_submillisecond_duration_uses_sample_clock():
    audio = Audio(np.zeros(8, dtype=np.float32), 16000)
    assert audio.duration_ms == 0.5
