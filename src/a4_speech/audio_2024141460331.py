"""Decode source audio without changing its relative time reference."""

from dataclasses import dataclass
from io import BytesIO
from math import gcd

import numpy as np


@dataclass(frozen=True)
class Audio:
    """A finite, nonempty, mono waveform; timing follows its sample clock."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("audio sample_rate must be a positive integer")
        try:
            values = np.asarray(self.samples, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("audio samples must be numeric") from exc
        if values.ndim != 1 or values.size == 0:
            raise ValueError("audio must be nonempty mono samples")
        if not np.isfinite(values).all():
            raise ValueError("audio contains non-finite samples")
        object.__setattr__(self, "samples", np.ascontiguousarray(values))

    @property
    def duration_ms(self) -> float:
        return len(self.samples) * 1000 / self.sample_rate


def decode_audio(data: bytes) -> Audio:
    """Decode libsndfile-supported formats and resample to 16 kHz mono float32.

    MP3 availability follows the installed libsndfile. Unsupported encodings
    fail explicitly rather than guessing a raw PCM interpretation.
    """
    if not isinstance(data, bytes | bytearray) or not data:
        raise ValueError("audio input must contain encoded audio bytes")
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ValueError("audio decoding requires soundfile and scipy") from exc
    try:
        samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError("audio format is unsupported or the audio file is invalid") from exc
    if samples.shape[0] == 0:
        raise ValueError("audio waveform is empty")
    if not np.isfinite(samples).all():
        raise ValueError("audio contains non-finite samples")
    mono = np.mean(samples, axis=1, dtype=np.float32)
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        mono = resample_poly(
            mono,
            16000 // divisor,
            sample_rate // divisor,
            padtype="line" if len(mono) > 1 else "constant",
        )
    return Audio(mono, 16000)
