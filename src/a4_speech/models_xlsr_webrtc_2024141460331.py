"""Offline WebRTC VAD and ATC XLS-R engine for the existing A-4 service protocol."""

from copy import deepcopy
from hashlib import sha256
from importlib import metadata
from math import ceil
from pathlib import Path

import numpy as np

from .audio_2024141460331 import Audio

XLSR_MODEL_NAME = "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim"
XLSR_MODEL_REVISION = "01a4853599abf22090b8963e6167b08a890b1e90"
XLSR_MODEL_SHA256 = "8967523e4d6f79daa431dbc4a36374a2c3e67697f70e5860e4614f23fea498b8"
WEBRTC_DISTRIBUTION = "webrtcvad-wheels"
WEBRTC_VERSION = "2.0.14.post1"
TRANSFORMERS_VERSION = "5.20.2"
TORCH_VERSION = "2.14.1+cu130"
SAMPLE_RATE = 16000
ASSET_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "added_tokens.json",
)


class WebRtcXlsrEngine:
    """Use the frozen local WebRTC + XLS-R stack behind the A-4 engine contract."""

    def __init__(self, config: dict) -> None:
        if not isinstance(config, dict) or set(config) != {"vad", "asr"}:
            raise ValueError("model config must contain only vad and asr")
        self._config = deepcopy(config)
        vad, asr = self._config["vad"], self._config["asr"]
        if not isinstance(vad, dict) or not isinstance(asr, dict):
            raise ValueError("vad and asr model configurations are required")
        self._validate_vad(vad)
        self._validate_asr(asr)

        try:
            self._runtime_versions = {
                "webrtcvad": metadata.version(WEBRTC_DISTRIBUTION),
                "transformers": metadata.version("transformers"),
                "torch": metadata.version("torch"),
            }
        except metadata.PackageNotFoundError as exc:
            raise ValueError("the locked WebRTC/XLS-R runtime dependencies are missing") from exc
        expected_versions = {
            "webrtcvad": WEBRTC_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "torch": TORCH_VERSION,
        }
        if self._runtime_versions != expected_versions:
            raise ValueError(
                "the installed WebRTC/XLS-R runtime does not match the frozen versions"
            )

        self._vad_parameters = {
            "sampleRate": SAMPLE_RATE,
            "aggressiveness": 2,
            "frameMs": 20,
            "mergeSilenceMs": 500,
            "minSpeechMs": 250,
            "paddingMs": 0,
        }
        self._asr_parameters = {
            "sampleRate": SAMPLE_RATE,
            "language": "en",
            "device": "cuda:0",
            "precision": "float32",
            "decodingMethod": "ctc_greedy",
            "externalLanguageModel": False,
            "transformersVersion": TRANSFORMERS_VERSION,
            "torchVersion": TORCH_VERSION,
        }

        self._weight_path = Path(asr["modelPath"]) / "model.safetensors"
        identity_before = self._file_identity(self._weight_path)
        if self._hash_file(self._weight_path, "XLS-R weights") != XLSR_MODEL_SHA256:
            raise ValueError("XLS-R checkpoint hash does not match the frozen revision")
        self._weight_identity = self._file_identity(self._weight_path)
        if identity_before != self._weight_identity:
            raise ValueError("XLS-R weights changed while the checkpoint was being verified")
        self._assets = {
            name: self._hash_file(Path(asr["modelPath"]) / name, "XLS-R processor asset")
            for name in ASSET_FILES
        }
        self._frozen = self._current_snapshot()
        self._processor = None
        self._model = None

    @staticmethod
    def _validate_vad(vad: dict) -> None:
        expected = {
            "engineCode": "webrtc-vad",
            "modelName": WEBRTC_DISTRIBUTION,
            "version": WEBRTC_VERSION,
            "aggressiveness": 2,
            "frameMs": 20,
            "mergeSilenceMs": 500,
            "minSpeechMs": 250,
            "paddingMs": 0,
        }
        if vad != expected:
            raise ValueError("VAD manifest must match the frozen WebRTC configuration")

    @staticmethod
    def _validate_asr(asr: dict) -> None:
        expected = {
            "engineCode": "xlsr-atc",
            "modelName": XLSR_MODEL_NAME,
            "version": XLSR_MODEL_REVISION,
            "language": "en",
            "device": "cuda:0",
        }
        if set(asr) != set(expected) | {"modelPath"}:
            raise ValueError("ASR manifest contains unsupported XLS-R parameters")
        if any(asr.get(key) != value for key, value in expected.items()):
            raise ValueError("ASR manifest must match the frozen local XLS-R configuration")
        model_path = asr.get("modelPath")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("asr.modelPath must name the local XLS-R checkpoint directory")

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int, int]:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ValueError("XLS-R checkpoint file is missing or unreadable") from exc
        if not path.is_file() or stat.st_size <= 0:
            raise ValueError("XLS-R checkpoint file is missing or empty")
        return stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", 0)

    @staticmethod
    def _hash_file(path: Path, label: str) -> str:
        try:
            digest = sha256()
            size = 0
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if not size:
                raise ValueError(f"{label} file is empty")
            return digest.hexdigest()
        except OSError as exc:
            raise ValueError(f"{label} file is missing or unreadable") from exc

    def _current_snapshot(self) -> dict:
        vad, asr = self._config["vad"], self._config["asr"]
        return {
            "schemaVersion": 1,
            "vad": {
                "engineCode": vad["engineCode"],
                "modelName": vad["modelName"],
                "version": vad["version"],
                "distributionVersion": self._runtime_versions["webrtcvad"],
                "parameters": deepcopy(self._vad_parameters),
            },
            "asr": {
                "engineCode": asr["engineCode"],
                "modelName": asr["modelName"],
                "version": asr["version"],
                "revision": XLSR_MODEL_REVISION,
                "modelSha256": XLSR_MODEL_SHA256,
                "processorAssetsSha256": deepcopy(self._assets),
                "parameters": deepcopy(self._asr_parameters),
            },
        }

    def snapshot(self) -> dict:
        """Return the fixed engine identity while the local files still match."""
        self.verify_snapshot(self._frozen)
        return deepcopy(self._frozen)

    def verify_snapshot(self, snapshot: dict) -> None:
        if not isinstance(snapshot, dict) or snapshot != self._frozen:
            raise ValueError("execution snapshot does not match the frozen XLS-R manifest")
        if self._file_identity(self._weight_path) != self._weight_identity:
            raise ValueError("XLS-R checkpoint changed after the execution snapshot was fixed")
        for name, expected in self._assets.items():
            actual = self._hash_file(
                Path(self._config["asr"]["modelPath"]) / name, "XLS-R processor asset"
            )
            if actual != expected:
                raise ValueError(
                    "XLS-R processor assets changed after the execution snapshot was fixed"
                )

    @staticmethod
    def _validate_audio(audio: Audio) -> None:
        if not isinstance(audio, Audio) or audio.sample_rate != SAMPLE_RATE:
            raise ValueError("model input must be decoded 16 kHz mono audio")
        if not np.isfinite(audio.samples).all():
            raise ValueError("model input contains non-finite audio samples")

    def detect(self, audio: Audio) -> list[tuple[int, int]]:
        """Run WebRTC on 20 ms PCM frames and return source-relative spans."""
        self._validate_audio(audio)
        self.verify_snapshot(self._frozen)
        try:
            import webrtcvad

            detector = webrtcvad.Vad(self._vad_parameters["aggressiveness"])
            frame_samples = SAMPLE_RATE * self._vad_parameters["frameMs"] // 1000
            scaled = np.clip(audio.samples, -1.0, 1.0)
            pcm = np.clip(np.rint(scaled * 32768.0), -32768, 32767).astype("<i2")
            flags = []
            for offset in range(0, len(pcm), frame_samples):
                frame = pcm[offset : offset + frame_samples]
                if len(frame) < frame_samples:
                    frame = np.pad(frame, (0, frame_samples - len(frame)))
                flags.append(detector.is_speech(frame.tobytes(), SAMPLE_RATE))
        except Exception as exc:
            raise ValueError(f"WebRTC VAD inference failed ({type(exc).__name__})") from exc
        self.verify_snapshot(self._frozen)

        total_ms = ceil(audio.duration_ms)
        raw = []
        start = None
        for index, active in enumerate(flags + [False]):
            if active and start is None:
                start = index * self._vad_parameters["frameMs"]
            elif not active and start is not None:
                end = min(index * self._vad_parameters["frameMs"], total_ms)
                if end > start:
                    raw.append((start, end))
                start = None
        merged = []
        for start_ms, end_ms in raw:
            if merged and start_ms - merged[-1][1] <= self._vad_parameters["mergeSilenceMs"]:
                merged[-1] = (merged[-1][0], end_ms)
            else:
                merged.append((start_ms, end_ms))
        return [span for span in merged if span[1] - span[0] >= self._vad_parameters["minSpeechMs"]]

    def prepare(self) -> None:
        """Load the local FP32 checkpoint on CUDA, with remote loading disabled."""
        self.verify_snapshot(self._frozen)
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCTC, Wav2Vec2Processor

            if not torch.cuda.is_available():
                raise ValueError("the frozen XLS-R engine requires an available CUDA device")
            model_path = self._config["asr"]["modelPath"]
            processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCTC.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
            ).to(self._asr_parameters["device"])
            model.eval()
        except Exception as exc:
            if isinstance(exc, ValueError) and "CUDA" in str(exc):
                raise
            raise ValueError(f"XLS-R model initialization failed ({type(exc).__name__})") from exc
        self._processor, self._model = processor, model
        self.verify_snapshot(self._frozen)

    def recognize(self, audio: Audio, start_ms: int, end_ms: int) -> str:
        """Return greedy CTC text for the exact source-relative segment interval."""
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
        self.verify_snapshot(self._frozen)
        start, end = start_ms * 16, min(len(audio.samples), end_ms * 16)
        if start >= end:
            raise ValueError("ASR segment contains no audio samples")
        try:
            import torch

            inputs = self._processor(
                audio.samples[start:end], sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            with torch.inference_mode():
                logits = self._model(inputs.input_values.to(self._asr_parameters["device"])).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(predicted_ids.detach().cpu())[0]
        except Exception as exc:
            raise ValueError(f"XLS-R model inference failed ({type(exc).__name__})") from exc
        self.verify_snapshot(self._frozen)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("XLS-R model returned empty or blank text")
        return text
