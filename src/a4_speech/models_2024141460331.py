"""Trusted local Silero VAD and SenseVoice adapter using sherpa-onnx APIs.

Version labels come from the operator's local model manifest. Actual model and
token bytes are SHA256 checked; a caller-provided version cannot replace them.
No model download, energy-threshold fallback, or synthetic transcript is used.
"""

from copy import deepcopy
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .audio_2024141460331 import Audio


class SherpaEngine:
    """CPU Silero VAD and SenseVoice, constructed from a trusted local manifest."""

    def __init__(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("model config must be an object with vad and asr")
        self._config = deepcopy(config)
        self._parameters: dict[str, dict[str, Any]] = {}
        common = {"engineCode", "modelName", "version", "modelPath", "numThreads", "provider"}
        for kind in ("vad", "asr"):
            item = self._config.get(kind)
            if not isinstance(item, dict):
                raise ValueError(f"{kind} model configuration is missing")
            extras = (
                {
                    "threshold",
                    "minSilenceDuration",
                    "minSpeechDuration",
                    "maxSpeechDuration",
                    "windowSize",
                }
                if kind == "vad"
                else {"tokensPath", "language", "useItn"}
            )
            if set(item) - common - extras:
                raise ValueError(f"{kind} config contains unsupported model parameters")
            for key in ("engineCode", "modelName", "version", "modelPath"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{kind}.{key} must be a nonempty string")
            threads = item.get("numThreads", 1)
            if type(threads) is not int or threads <= 0:
                raise ValueError(f"{kind}.numThreads must be a positive integer")
            if item.get("provider", "cpu") != "cpu":
                raise ValueError("only the CPU execution provider is supported")
            params = {"sampleRate": 16000, "numThreads": threads, "provider": "cpu"}
            if kind == "vad":
                for name, default in (
                    ("threshold", 0.5),
                    ("minSilenceDuration", 0.5),
                    ("minSpeechDuration", 0.25),
                    ("maxSpeechDuration", 20.0),
                ):
                    value = item.get(name, default)
                    if type(value) not in (int, float) or not isfinite(value):
                        raise ValueError(f"vad.{name} must be a finite number")
                    if value <= 0 or (name == "threshold" and value > 1):
                        raise ValueError(f"vad.{name} is outside its valid range")
                    params[name] = float(value)
                if item.get("windowSize", 512) != 512:
                    raise ValueError("16 kHz Silero VAD requires windowSize=512")
                params["windowSize"] = 512
                if params["minSpeechDuration"] >= params["maxSpeechDuration"]:
                    raise ValueError("vad.maxSpeechDuration must exceed minSpeechDuration")
            else:
                if not isinstance(item.get("tokensPath"), str) or not item["tokensPath"]:
                    raise ValueError("asr.tokensPath is required")
                language = item.get("language", "auto")
                if not isinstance(language, str) or language not in {
                    "auto",
                    "zh",
                    "en",
                    "ja",
                    "ko",
                    "yue",
                }:
                    raise ValueError("unsupported SenseVoice language")
                use_itn = item.get("useItn", True)
                if type(use_itn) is not bool:
                    raise ValueError("asr.useItn must be a boolean")
                params.update(language=language, useItn=use_itn)
            self._parameters[kind] = params
        self._frozen = self._current_snapshot()
        self._recognizer = None

    @staticmethod
    def _hash_file(filename: str, label: str) -> str:
        try:
            with Path(filename).open("rb") as source:
                digest = sha256()
                size = 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                if not size:
                    raise ValueError(f"{label} model file is empty")
                return digest.hexdigest()
        except OSError as exc:
            raise ValueError(f"{label} model file is missing or unreadable") from exc

    def _current_snapshot(self) -> dict:
        snapshot = {"schemaVersion": 1}
        for kind in ("vad", "asr"):
            item = self._config[kind]
            entry = {key: item[key] for key in ("engineCode", "modelName", "version")}
            entry["modelSha256"] = self._hash_file(item["modelPath"], kind)
            if kind == "asr":
                entry["tokensSha256"] = self._hash_file(item["tokensPath"], "asr tokens")
            entry["parameters"] = deepcopy(self._parameters[kind])
            snapshot[kind] = entry
        return snapshot

    def snapshot(self) -> dict:
        """Return this engine's frozen identity only while its files still match."""
        self.verify_snapshot(self._frozen)
        return deepcopy(self._frozen)

    def verify_snapshot(self, snapshot: dict) -> None:
        if not isinstance(snapshot, dict) or snapshot != self._frozen:
            raise ValueError("execution snapshot does not match the trusted model manifest")
        if self._current_snapshot() != self._frozen:
            raise ValueError("model files changed after the execution snapshot was fixed")

    @staticmethod
    def _sherpa():
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ValueError("actual inference requires the sherpa-onnx package") from exc
        return sherpa_onnx

    @staticmethod
    def _validate_audio(audio: Audio) -> None:
        if not isinstance(audio, Audio) or audio.sample_rate != 16000:
            raise ValueError("model input must be decoded 16 kHz mono audio")
        if not np.isfinite(audio.samples).all():
            raise ValueError("model input contains non-finite audio samples")

    def detect(self, audio: Audio) -> list[tuple[int, int]]:
        """Flush the final VAD window; return source-relative millisecond spans."""
        self._validate_audio(audio)
        self.verify_snapshot(self._frozen)
        sherpa = self._sherpa()
        params = self._parameters["vad"]
        spans = []
        try:
            config = sherpa.VadModelConfig()
            config.silero_vad.model = self._config["vad"]["modelPath"]
            config.silero_vad.threshold = params["threshold"]
            config.silero_vad.min_silence_duration = params["minSilenceDuration"]
            config.silero_vad.min_speech_duration = params["minSpeechDuration"]
            config.silero_vad.max_speech_duration = params["maxSpeechDuration"]
            config.silero_vad.window_size = params["windowSize"]
            config.sample_rate = 16000
            config.num_threads = params["numThreads"]
            config.provider = params["provider"]
            vad = sherpa.VoiceActivityDetector(
                config, buffer_size_in_seconds=max(params["maxSpeechDuration"] + 2, 30)
            )

            def drain() -> None:
                while not vad.empty():
                    segment = vad.front
                    start = max(0, int(segment.start))
                    end = min(len(audio.samples), int(segment.start) + len(segment.samples))
                    if end > start:
                        spans.append((start, end))
                    vad.pop()

            for offset in range(0, len(audio.samples), params["windowSize"]):
                window = audio.samples[offset : offset + params["windowSize"]]
                if len(window) < params["windowSize"]:
                    window = np.pad(window, (0, params["windowSize"] - len(window)))
                vad.accept_waveform(window)
                drain()
            vad.flush()
            drain()
        except Exception as exc:
            raise ValueError(f"VAD model inference failed ({type(exc).__name__})") from exc
        self.verify_snapshot(self._frozen)
        # Reject actual overlap; adjust only the millisecond rounding of
        # source-disjoint intervals so the public non-overlap contract holds.
        result = []
        previous_sample_end = 0
        for start, end in spans:
            if start < previous_sample_end:
                raise ValueError("VAD returned overlapping or out-of-order speech segments")
            previous_sample_end = end
            start_ms = start * 1000 // 16000
            end_ms = ceil(end * 1000 / 16000)
            if result and start_ms < result[-1][1]:
                start_ms = result[-1][1]
            if end_ms > start_ms:
                result.append((start_ms, end_ms))
        return result

    def prepare(self) -> None:
        """Load and verify SenseVoice without starting a segment ASR attempt.

        The application calls this before creating its ASR job so an invalid
        model does not get counted as a failed recognition segment.
        """
        self.verify_snapshot(self._frozen)
        params = self._parameters["asr"]
        try:
            if self._recognizer is None:
                sherpa = self._sherpa()
                self._recognizer = sherpa.OfflineRecognizer.from_sense_voice(
                    model=self._config["asr"]["modelPath"],
                    tokens=self._config["asr"]["tokensPath"],
                    language=params["language"],
                    use_itn=params["useItn"],
                    num_threads=params["numThreads"],
                    sample_rate=16000,
                    provider="cpu",
                    decoding_method="greedy_search",
                    debug=False,
                )
        except Exception as exc:
            raise ValueError(f"ASR model initialization failed ({type(exc).__name__})") from exc
        self.verify_snapshot(self._frozen)

    def recognize(self, audio: Audio, start_ms: int, end_ms: int) -> str:
        """Return original SenseVoice text for a segment of the source waveform."""
        self._validate_audio(audio)
        if (
            type(start_ms) is not int
            or type(end_ms) is not int
            or start_ms < 0
            or end_ms <= start_ms
            or end_ms > ceil(audio.duration_ms)
        ):
            raise ValueError("ASR segment must satisfy 0 <= start_ms < end_ms <= audio duration")
        self.prepare()
        try:
            start = start_ms * 16
            end = min(len(audio.samples), end_ms * 16)
            if start >= end:
                raise ValueError("ASR segment contains no audio samples")
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio.samples[start:end])
            self._recognizer.decode_stream(stream)
            text = stream.result.text
        except Exception as exc:
            raise ValueError(f"ASR model inference failed ({type(exc).__name__})") from exc
        self.verify_snapshot(self._frozen)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ASR model returned empty or blank text")
        return text
