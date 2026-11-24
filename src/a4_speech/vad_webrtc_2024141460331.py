"""Frozen CPU WebRTC segmentation, independent of the downstream ASR model."""

from copy import deepcopy
from importlib import metadata
from math import ceil

import numpy as np

from .audio_2024141460331 import Audio

WEBRTC_CONFIG = {
    "engineCode": "webrtc-vad",
    "modelName": "webrtcvad-wheels",
    "version": "2.0.14.post1",
    "aggressiveness": 2,
    "frameMs": 20,
    "mergeSilenceMs": 500,
    "minSpeechMs": 250,
    "paddingMs": 0,
}


class WebRtcVad:
    """Keep the previously evaluated frame and postprocessing parameters unchanged."""

    def __init__(self, config: dict) -> None:
        if config != WEBRTC_CONFIG:
            raise ValueError("VAD manifest must match the frozen WebRTC configuration")
        self._config = deepcopy(config)
        self._verify_runtime()

    def _verify_runtime(self) -> None:
        try:
            installed = metadata.version("webrtcvad-wheels")
        except metadata.PackageNotFoundError as exc:
            raise ValueError("WebRTC VAD requires webrtcvad-wheels==2.0.14.post1") from exc
        if installed != self._config["version"]:
            raise ValueError("WebRTC runtime version does not match the frozen manifest")

    def snapshot(self) -> dict:
        self._verify_runtime()
        return {
            "engineCode": self._config["engineCode"],
            "modelName": self._config["modelName"],
            "version": self._config["version"],
            "distributionVersion": self._config["version"],
            "parameters": {
                "sampleRate": 16000,
                "provider": "cpu",
                **{
                    key: self._config[key]
                    for key in (
                        "aggressiveness",
                        "frameMs",
                        "mergeSilenceMs",
                        "minSpeechMs",
                        "paddingMs",
                    )
                },
            },
        }

    def detect(self, audio: Audio) -> list[tuple[int, int]]:
        if not isinstance(audio, Audio) or audio.sample_rate != 16000:
            raise ValueError("WebRTC VAD input must be decoded 16 kHz mono audio")
        if not np.isfinite(audio.samples).all():
            raise ValueError("WebRTC VAD input contains non-finite audio samples")
        self._verify_runtime()
        try:
            import webrtcvad

            detector = webrtcvad.Vad(self._config["aggressiveness"])
            frame_samples = 16000 * self._config["frameMs"] // 1000
            scaled = np.clip(audio.samples, -1.0, 1.0)
            pcm = np.clip(np.rint(scaled * 32768.0), -32768, 32767).astype("<i2")
            flags = []
            for offset in range(0, len(pcm), frame_samples):
                frame = pcm[offset : offset + frame_samples]
                if len(frame) < frame_samples:
                    frame = np.pad(frame, (0, frame_samples - len(frame)))
                flags.append(detector.is_speech(frame.tobytes(), 16000))
        except Exception as exc:
            raise ValueError(f"WebRTC VAD inference failed ({type(exc).__name__})") from exc
        self._verify_runtime()
        total_ms = ceil(audio.duration_ms)
        raw = []
        start = None
        for index, active in enumerate(flags + [False]):
            if active and start is None:
                start = index * self._config["frameMs"]
            elif not active and start is not None:
                end = min(index * self._config["frameMs"], total_ms)
                if end > start:
                    raw.append((start, end))
                start = None
        merged = []
        for start_ms, end_ms in raw:
            if merged and start_ms - merged[-1][1] <= self._config["mergeSilenceMs"]:
                merged[-1] = (merged[-1][0], end_ms)
            else:
                merged.append((start_ms, end_ms))
        return [span for span in merged if span[1] - span[0] >= self._config["minSpeechMs"]]
