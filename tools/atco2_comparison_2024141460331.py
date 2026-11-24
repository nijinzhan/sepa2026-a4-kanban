"""Frozen local ATCO2 ASR/VAD comparison; raw speech data stays under ignored outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent.parent
DEFAULT_ROOT = HERE / "outputs" / "atco2-1h-20261021"
DEFAULT_MODEL = HERE / "models" / "wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim"
MODEL_SHA256 = "8967523e4d6f79daa431dbc4a36374a2c3e67697f70e5860e4614f23fea498b8"
MODEL_REVISION = "01a4853599abf22090b8963e6167b08a890b1e90"
VAD_PARAMETERS = {
    "engine": "WebRTC VAD",
    "package": "webrtcvad-wheels==2.0.14.post1",
    "sampleRate": 16000,
    "frameMs": 20,
    "aggressiveness": 2,
    "mergeSilenceMs": 500,
    "minSpeechMs": 250,
    "paddingMs": 0,
}
CONDITIONS = ("xml", "silero", "webrtc")
ASR_NAMES = ("SenseVoiceSmall-2024-07-17-int8", "XLS-R-300M-ATC-CTC-greedy-FP32")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_evaluator():
    path = HERE / "tools" / "atco2_eval_2024141460331.py"
    spec = importlib.util.spec_from_file_location("atco2_eval_2024141460331", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen baseline scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def consecutive_spans(flags: list[bool], frame_ms: int, total_ms: int) -> list[list[int]]:
    spans = []
    start = None
    for index, active in enumerate(flags + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append([start * frame_ms, min(index * frame_ms, total_ms)])
            start = None
    return [span for span in spans if span[1] > span[0]]


def postprocess_vad_flags(
    flags: list[bool],
    total_ms: int,
    frame_ms: int = 20,
    merge_silence_ms: int = 500,
    min_speech_ms: int = 250,
) -> list[list[int]]:
    """Merge WebRTC frame runs with the frozen Silero silence/speech durations."""
    if total_ms < 0 or frame_ms <= 0 or merge_silence_ms < 0 or min_speech_ms < 0:
        raise ValueError("invalid WebRTC post-processing parameters")
    raw = consecutive_spans(flags, frame_ms, total_ms)
    merged: list[list[int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] <= merge_silence_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [span for span in merged if span[1] - span[0] >= min_speech_ms]


def experiment_paths(root: Path, output: Path) -> dict[str, Path]:
    return {
        "baseline": root / "evaluation-49",
        "controlled": root / "diagnosis-20261021" / "controlled-boundary-comparison.json",
        "output": output,
        "plan": output / "experiment-plan.json",
        "vad": output / "webrtc-vad.jsonl",
        "vadRuntime": output / "webrtc-runtime.json",
        "xlsr": output / "xlsr-results.jsonl",
        "xlsrRuntime": output / "xlsr-runtime.json",
        "sensevoice": output / "sensevoice-webrtc-results.jsonl",
        "sensevoiceRuntime": output / "sensevoice-webrtc-runtime.json",
    }


def freeze(args) -> None:
    root = args.root.resolve()
    output = args.output.resolve()
    paths = experiment_paths(root, output)
    if paths["plan"].exists() and any(path != paths["plan"] for path in output.iterdir()):
        raise ValueError("experiment output already contains run results; use a new directory")
    baseline = paths["baseline"]
    baseline_plan = json.loads((baseline / "plan.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    manifest_path = baseline / "manifest.json"
    references_path = baseline / "references.json"
    results_path = baseline / "results.jsonl"
    model_path = args.model.resolve()
    for name, expected in (
        (manifest_path, baseline_plan["manifestSha256"]),
        (references_path, baseline_plan["referencesSha256"]),
        (results_path, baseline_summary["files"]["results.jsonl"]),
    ):
        if sha256(name) != expected:
            raise ValueError(f"frozen baseline input checksum mismatch: {name.name}")
    if sha256(model_path / "model.safetensors") != MODEL_SHA256:
        raise ValueError("local XLS-R weights do not match the verified checkpoint")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != 49 or {item["airport"] for item in manifest} != {
        "LKPR",
        "LKTB",
        "LSGS",
        "LSZB",
        "LSZH",
        "LZIB",
        "YSSY",
    }:
        raise ValueError("the experiment requires the frozen 49-recording, seven-airport sample")
    controlled = json.loads(paths["controlled"].read_text(encoding="utf-8"))
    controlled_ids = {row["recordingId"] for row in controlled["recordings"]}
    if controlled_ids != {item["recordingId"] for item in manifest}:
        raise ValueError("controlled XML-boundary outputs do not match the frozen sample")
    output.mkdir(parents=True, exist_ok=True)
    if any(path != paths["plan"] for path in output.iterdir()):
        raise ValueError("comparison output directory is not empty")
    git_root = HERE.parent
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(git_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    plan = {
        "createdUtc": datetime.now(UTC).isoformat(),
        "purpose": "public ATC cross-corpus comparison; not VHHH performance",
        "sampleCount": len(manifest),
        "sampleSeconds": sum(row["durationMs"] for row in manifest) / 1000,
        "airports": sorted({row["airport"] for row in manifest}),
        "manifestSha256": sha256(manifest_path),
        "referencesSha256": sha256(references_path),
        "baselineResultsSha256": sha256(results_path),
        "baselinePlanSha256": sha256(baseline / "plan.json"),
        "controlledOutputsSha256": sha256(paths["controlled"]),
        "evaluationScriptSha256": sha256(HERE / "tools" / "atco2_eval_2024141460331.py"),
        "experimentScriptSha256": sha256(SCRIPT),
        "sourceGitHead": git_head,
        "baselineExecutionSnapshot": baseline_plan["executionSnapshot"],
        "model": {
            "id": "Jzuluaga/wav2vec2-xls-r-300m-en-atc-uwb-atcc-and-atcosim",
            "revision": MODEL_REVISION,
            "localPath": str(model_path),
            "modelSha256": MODEL_SHA256,
            "class": "AutoModelForCTC + Wav2Vec2Processor",
            "device": "cuda:0",
            "dtype": "float32",
            "batchSize": 1,
            "decoder": "CTC greedy; no external language model",
            "trustRemoteCode": False,
            "offlineOnly": True,
        },
        "vad": VAD_PARAMETERS,
        "conditions": list(CONDITIONS),
        "asr": list(ASR_NAMES),
        "metrics": [
            "micro strict WER using frozen baseline scorer",
            "micro custom numeric-normalized WER using frozen baseline scorer",
            "per-airport and per-recording paired outputs",
            "10,000 paired recording bootstrap replicates for WER differences",
            "WebRTC/Silero boundary matching against XML text intervals",
            "elapsed inference time and RTF as device-specific descriptive metrics",
        ],
        "baselineRoot": str(root),
    }
    write_json(paths["plan"], plan)
    print(
        json.dumps({"frozen": str(paths["plan"]), "recordings": len(manifest)}, ensure_ascii=False)
    )


def load_plan(args) -> tuple[dict, dict[str, Path]]:
    root = args.root.resolve()
    output = args.output.resolve()
    paths = experiment_paths(root, output)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    if sha256(SCRIPT) != plan["experimentScriptSha256"]:
        raise ValueError("experiment script changed after the protocol was frozen")
    if sha256(HERE / "tools" / "atco2_eval_2024141460331.py") != plan["evaluationScriptSha256"]:
        raise ValueError("the frozen scorer changed after the protocol was recorded")
    checks = (
        (paths["baseline"] / "manifest.json", plan["manifestSha256"]),
        (paths["baseline"] / "references.json", plan["referencesSha256"]),
        (paths["baseline"] / "results.jsonl", plan["baselineResultsSha256"]),
        (paths["baseline"] / "plan.json", plan["baselinePlanSha256"]),
        (paths["controlled"], plan["controlledOutputsSha256"]),
    )
    for path, expected in checks:
        if sha256(path) != expected:
            raise ValueError(f"frozen input checksum mismatch: {path.name}")
    model_file = Path(plan["model"]["localPath"]) / "model.safetensors"
    if sha256(model_file) != plan["model"]["modelSha256"]:
        raise ValueError("verified XLS-R model weights changed after freeze")
    return plan, paths


def run_vad(args) -> None:
    plan, paths = load_plan(args)
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError("install the locked webrtcvad-wheels dependency first") from exc
    manifest = json.loads((paths["baseline"] / "manifest.json").read_text(encoding="utf-8"))
    existing = read_jsonl(paths["vad"]) if paths["vad"].exists() else []
    done = {row["recordingId"] for row in existing}
    expected_ids = {row["recordingId"] for row in manifest}
    if len(done) != len(existing) or done - expected_ids:
        raise ValueError("existing WebRTC output contains duplicate or unexpected recordings")
    frame_ms = VAD_PARAMETERS["frameMs"]
    samples_per_frame = 16000 * frame_ms // 1000
    started = time.perf_counter()
    mode = "a" if paths["vad"].exists() else "w"
    with paths["vad"].open(mode, encoding="utf-8") as output:
        for item in manifest:
            recording_id = item["recordingId"]
            if recording_id in done:
                continue
            audio_path = paths["baseline"].parent / item["audioRelative"]
            if sha256(audio_path) != item["audioSha256"]:
                raise ValueError(f"frozen audio checksum mismatch: {recording_id}")
            samples, sample_rate = sf.read(audio_path, dtype="int16", always_2d=False)
            if sample_rate != 16000 or samples.ndim != 1:
                raise ValueError(f"expected frozen 16 kHz mono PCM: {recording_id}")
            vad = webrtcvad.Vad(VAD_PARAMETERS["aggressiveness"])
            pcm = np.asarray(samples, dtype="<i2")
            flags = []
            for start in range(0, len(pcm), samples_per_frame):
                frame = pcm[start : start + samples_per_frame]
                if len(frame) < samples_per_frame:
                    frame = np.pad(frame, (0, samples_per_frame - len(frame)))
                flags.append(vad.is_speech(frame.tobytes(), sample_rate))
            total_ms = math.ceil(len(pcm) * 1000 / sample_rate)
            spans = postprocess_vad_flags(
                flags,
                total_ms,
                frame_ms,
                VAD_PARAMETERS["mergeSilenceMs"],
                VAD_PARAMETERS["minSpeechMs"],
            )
            row = {
                "recordingId": recording_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "vadSegments": [{"startMs": start, "endMs": end} for start, end in spans],
                "speechFrames": sum(flags),
                "totalFrames": len(flags),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(recording_id)
            print(
                json.dumps(
                    {
                        "stage": "webrtc-vad",
                        "progress": f"{len(done)}/{len(manifest)}",
                        "recordingId": recording_id,
                    }
                ),
                flush=True,
            )
    if done == expected_ids:
        results_hash = sha256(paths["vad"])
        write_json(
            paths["vadRuntime"],
            {
                "finishedUtc": datetime.now(UTC).isoformat(),
                "elapsedSecondsThisInvocation": time.perf_counter() - started,
                "python": sys.version,
                "platform": platform.platform(),
                "webrtcvadDistribution": importlib.metadata.version("webrtcvad-wheels"),
                "parameters": VAD_PARAMETERS,
                "resultsSha256": results_hash,
                "recordings": len(done),
                "sourceManifestSha256": plan["manifestSha256"],
            },
        )


def run_xlsr(args) -> None:
    plan, paths = load_plan(args)
    if (
        not paths["vadRuntime"].exists()
        or sha256(paths["vad"])
        != json.loads(paths["vadRuntime"].read_text(encoding="utf-8"))["resultsSha256"]
    ):
        raise ValueError("complete, checksum-verified WebRTC VAD output is required")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen XLS-R experiment requires CUDA")
    model_path = plan["model"]["localPath"]
    device = torch.device("cuda:0")
    load_started = time.perf_counter()
    processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to(device)
    model.eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats(device)

    baseline = paths["baseline"]
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    references = json.loads((baseline / "references.json").read_text(encoding="utf-8"))
    baseline_results = {row["recordingId"]: row for row in read_jsonl(baseline / "results.jsonl")}
    web_results = {row["recordingId"]: row for row in read_jsonl(paths["vad"])}
    existing = read_jsonl(paths["xlsr"]) if paths["xlsr"].exists() else []
    done = {row["recordingId"] for row in existing}
    expected_ids = {row["recordingId"] for row in manifest}
    if len(done) != len(existing) or done - expected_ids:
        raise ValueError("existing XLS-R results contain duplicate or unexpected recordings")
    if set(baseline_results) != expected_ids or set(web_results) != expected_ids:
        raise ValueError("baseline or WebRTC outputs do not cover the frozen sample")
    mode = "a" if paths["xlsr"].exists() else "w"
    elapsed = 0.0
    with paths["xlsr"].open(mode, encoding="utf-8") as output:
        for item in manifest:
            recording_id = item["recordingId"]
            if recording_id in done:
                continue
            audio_path = baseline.parent / item["audioRelative"]
            if sha256(audio_path) != item["audioSha256"]:
                raise ValueError(f"frozen audio checksum mismatch: {recording_id}")
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(f"expected frozen 16 kHz mono PCM: {recording_id}")
            condition_spans = {
                "xml": [
                    [segment["startMs"], segment["endMs"]]
                    for segment in references[recording_id]
                    if segment.get("text", "").strip()
                ],
                "silero": [
                    [segment["startMs"], segment["endMs"]]
                    for segment in baseline_results[recording_id]["vadSegments"]
                ],
                "webrtc": [
                    [segment["startMs"], segment["endMs"]]
                    for segment in web_results[recording_id]["vadSegments"]
                ],
            }
            conditions = {}
            record_started = time.perf_counter()
            for condition in CONDITIONS:
                segments = []
                errors = []
                asr_seconds = 0.0
                for start_ms, end_ms in condition_spans[condition]:
                    start = max(0, start_ms * 16)
                    end = min(len(audio), end_ms * 16)
                    if end <= start:
                        errors.append({"startMs": start_ms, "endMs": end_ms, "type": "empty_audio"})
                        segments.append(
                            {
                                "startMs": start_ms,
                                "endMs": end_ms,
                                "textRaw": "",
                                "elapsedSeconds": 0,
                            }
                        )
                        continue
                    try:
                        inputs = processor(
                            audio[start:end], sampling_rate=16000, return_tensors="pt"
                        )
                        inputs = {key: value.to(device) for key, value in inputs.items()}
                        torch.cuda.synchronize()
                        call_started = time.perf_counter()
                        with torch.inference_mode():
                            logits = model(**inputs).logits
                            predicted_ids = torch.argmax(logits, dim=-1)
                        torch.cuda.synchronize()
                        text = processor.batch_decode(predicted_ids.detach().cpu())[0]
                        call_seconds = time.perf_counter() - call_started
                        asr_seconds += call_seconds
                        segments.append(
                            {
                                "startMs": start_ms,
                                "endMs": end_ms,
                                "textRaw": text,
                                "elapsedSeconds": call_seconds,
                            }
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        errors.append({"startMs": start_ms, "endMs": end_ms, "message": message})
                        segments.append(
                            {
                                "startMs": start_ms,
                                "endMs": end_ms,
                                "textRaw": "",
                                "elapsedSeconds": 0,
                            }
                        )
                conditions[condition] = {
                    "segments": segments,
                    "hypothesis": " ".join(x["textRaw"] for x in segments if x["textRaw"]),
                    "asrSeconds": asr_seconds,
                    "errors": errors,
                }
            torch.cuda.synchronize()
            record_seconds = time.perf_counter() - record_started
            elapsed += record_seconds
            row = {
                "recordingId": recording_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "conditions": conditions,
                "recordProcessingSeconds": record_seconds,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(recording_id)
            print(
                json.dumps(
                    {
                        "stage": "xlsr",
                        "progress": f"{len(done)}/{len(manifest)}",
                        "recordingId": recording_id,
                        "seconds": round(record_seconds, 3),
                    }
                ),
                flush=True,
            )
    if done == expected_ids:
        torch.cuda.synchronize()
        write_json(
            paths["xlsrRuntime"],
            {
                "startedUtc": datetime.now(UTC).isoformat(),
                "modelLoadSeconds": load_seconds,
                "recordProcessingSecondsThisInvocation": elapsed,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "peakAllocatedMiB": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peakReservedMiB": torch.cuda.max_memory_reserved(device) / 1024**2,
                "transformers": importlib.metadata.version("transformers"),
                "numpy": np.__version__,
                "soundfile": sf.__version__,
                "model": plan["model"],
                "sourceVADSha256": sha256(paths["vad"]),
                "resultsSha256": sha256(paths["xlsr"]),
                "recordings": len(done),
            },
        )


def run_sensevoice_webrtc(args) -> None:
    plan, paths = load_plan(args)
    if (
        not paths["vadRuntime"].exists()
        or sha256(paths["vad"])
        != json.loads(paths["vadRuntime"].read_text(encoding="utf-8"))["resultsSha256"]
    ):
        raise ValueError("complete, checksum-verified WebRTC VAD output is required")
    sys.path.insert(0, str(HERE / "src"))
    from a4_speech.audio_2024141460331 import decode_audio
    from a4_speech.cli_2024141460331 import (
        _engine_2024141460331,
        load_config_2024141460331,
    )

    baseline = paths["baseline"]
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    vad_rows = {row["recordingId"]: row for row in read_jsonl(paths["vad"])}
    config = load_config_2024141460331(args.config.resolve())
    engine = _engine_2024141460331(config)
    if engine.snapshot() != plan["baselineExecutionSnapshot"]:
        raise ValueError("SenseVoice no longer matches the frozen baseline model snapshot")
    engine.prepare()
    existing = read_jsonl(paths["sensevoice"]) if paths["sensevoice"].exists() else []
    done = {row["recordingId"] for row in existing}
    expected_ids = {row["recordingId"] for row in manifest}
    if len(done) != len(existing) or done - expected_ids:
        raise ValueError(
            "existing SenseVoice WebRTC results contain duplicate or unexpected recordings"
        )
    elapsed = 0.0
    mode = "a" if paths["sensevoice"].exists() else "w"
    with paths["sensevoice"].open(mode, encoding="utf-8") as output:
        for item in manifest:
            recording_id = item["recordingId"]
            if recording_id in done:
                continue
            audio_path = baseline.parent / item["audioRelative"]
            if sha256(audio_path) != item["audioSha256"]:
                raise ValueError(f"frozen audio checksum mismatch: {recording_id}")
            audio = decode_audio(audio_path.read_bytes())
            segments = []
            errors = []
            asr_seconds = 0.0
            record_started = time.perf_counter()
            for span in vad_rows[recording_id]["vadSegments"]:
                start_ms, end_ms = span["startMs"], span["endMs"]
                call_started = time.perf_counter()
                try:
                    text = engine.recognize(audio, start_ms, end_ms)
                except Exception as exc:
                    text = ""
                    errors.append(
                        {
                            "startMs": start_ms,
                            "endMs": end_ms,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                call_seconds = time.perf_counter() - call_started
                asr_seconds += call_seconds
                segments.append(
                    {
                        "startMs": start_ms,
                        "endMs": end_ms,
                        "textRaw": text,
                        "elapsedSeconds": call_seconds,
                    }
                )
            record_seconds = time.perf_counter() - record_started
            elapsed += record_seconds
            row = {
                "recordingId": recording_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "segments": segments,
                "hypothesis": " ".join(x["textRaw"] for x in segments if x["textRaw"]),
                "asrSeconds": asr_seconds,
                "recordProcessingSeconds": record_seconds,
                "errors": errors,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(recording_id)
            print(
                json.dumps(
                    {
                        "stage": "sensevoice-webrtc",
                        "progress": f"{len(done)}/{len(manifest)}",
                        "recordingId": recording_id,
                        "seconds": round(record_seconds, 3),
                    }
                ),
                flush=True,
            )
    if done == expected_ids:
        write_json(
            paths["sensevoiceRuntime"],
            {
                "finishedUtc": datetime.now(UTC).isoformat(),
                "elapsedSecondsThisInvocation": elapsed,
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("sherpa-onnx", "numpy", "scipy", "soundfile")
                },
                "executionSnapshot": engine.snapshot(),
                "sourceVADSha256": sha256(paths["vad"]),
                "resultsSha256": sha256(paths["sensevoice"]),
                "recordings": len(done),
            },
        )


def _small_score(evaluator, reference: list[str], hypothesis: str, numeric: bool) -> dict:
    ref_tokens = evaluator.tokenize_2024141460331(" ".join(reference), numeric=numeric)
    hyp_tokens = evaluator.tokenize_2024141460331(hypothesis, numeric=numeric)
    score = evaluator.score_words_2024141460331(ref_tokens, hyp_tokens)
    return {key: score[key] for key in ("N", "S", "D", "I", "C", "wer")}


def _condition_scores(evaluator, references: dict, predictions: dict[str, str]) -> dict:
    result = {}
    for condition in CONDITIONS:
        result[condition] = {
            "strict": _small_score(evaluator, references, predictions.get(condition, ""), False),
            "numericNormalized": _small_score(
                evaluator, references, predictions.get(condition, ""), True
            ),
        }
    return result


def aggregate_condition(records: list[dict], model: str, condition: str, key: str) -> dict:
    counts = {
        name: sum(row["scores"][model][condition][key][name] for row in records)
        for name in ("N", "S", "D", "I", "C")
    }
    counts["wer"] = (counts["S"] + counts["D"] + counts["I"]) / counts["N"] if counts["N"] else None
    return counts


def bootstrap_difference(left: list[dict], right: list[dict], replicates: int, seed: int) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires the same nonempty recording set")
    rng = np.random.default_rng(seed)
    count = len(left)
    indices = rng.integers(0, count, size=(replicates, count))
    left_errors = np.array([r["S"] + r["D"] + r["I"] for r in left])
    right_errors = np.array([r["S"] + r["D"] + r["I"] for r in right])
    denominators = np.array([r["N"] for r in left])
    sampled_n = denominators[indices].sum(axis=1)
    diffs = (right_errors[indices].sum(axis=1) / sampled_n) - (
        left_errors[indices].sum(axis=1) / sampled_n
    )
    return {
        "differenceRightMinusLeft": float(
            (right_errors.sum() - left_errors.sum()) / denominators.sum()
        ),
        "bootstrap95PercentileInterval": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
        "replicates": replicates,
        "unit": "recording; micro-WER ratio per paired resample",
    }


def _aggregate_boundary(rows: list[dict]) -> dict:
    boundary_rows = [row["boundary"] for row in rows]
    keys = (
        "reference_regions",
        "predicted_regions",
        "matched_regions",
        "missed_regions",
        "extra_regions",
        "reference_speech_ms",
        "predicted_speech_ms",
        "overlap_ms",
    )
    result = {key: sum(row[key] for row in boundary_rows) for key in keys}
    for side in ("start", "end"):
        values = [value for row in boundary_rows for value in row[f"{side}_abs_errors_ms"]]
        result[f"{side}AbsErrorMs"] = {
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
        }
    result["durationRecall"] = (
        result["overlap_ms"] / result["reference_speech_ms"]
        if result["reference_speech_ms"]
        else None
    )
    result["durationPrecision"] = (
        result["overlap_ms"] / result["predicted_speech_ms"] if result["predicted_speech_ms"] else 0
    )
    return result


def summarize(args) -> None:
    plan, paths = load_plan(args)
    for path, runtime_path in (
        (paths["vad"], paths["vadRuntime"]),
        (paths["xlsr"], paths["xlsrRuntime"]),
        (paths["sensevoice"], paths["sensevoiceRuntime"]),
    ):
        if not runtime_path.exists():
            raise ValueError(f"completed stage output is required: {runtime_path.name}")
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if sha256(path) != runtime["resultsSha256"]:
            raise ValueError(f"stage output checksum mismatch: {path.name}")
    baseline = paths["baseline"]
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    references = json.loads((baseline / "references.json").read_text(encoding="utf-8"))
    baseline_rows = {row["recordingId"]: row for row in read_jsonl(baseline / "results.jsonl")}
    control = json.loads(paths["controlled"].read_text(encoding="utf-8"))
    control_rows = {row["recordingId"]: row for row in control["recordings"]}
    vad_rows = {row["recordingId"]: row for row in read_jsonl(paths["vad"])}
    xlsr_rows = {row["recordingId"]: row for row in read_jsonl(paths["xlsr"])}
    sense_webrtc_rows = {row["recordingId"]: row for row in read_jsonl(paths["sensevoice"])}
    expected = {row["recordingId"] for row in manifest}
    if any(
        set(rows) != expected
        for rows in (baseline_rows, control_rows, vad_rows, xlsr_rows, sense_webrtc_rows)
    ):
        raise ValueError("one or more result stages do not cover exactly the frozen 49 recordings")
    evaluator = load_evaluator()
    results = []
    webrtc_boundary_rows = []
    for item in manifest:
        record_id = item["recordingId"]
        reference_segments = references[record_id]
        reference_regions = [
            (segment["startMs"], segment["endMs"])
            for segment in reference_segments
            if evaluator.tokenize_2024141460331(segment["text"])
        ]
        webrtc_spans = [
            (segment["startMs"], segment["endMs"]) for segment in vad_rows[record_id]["vadSegments"]
        ]
        webrtc_boundary_rows.append(
            {"boundary": evaluator.boundary_score_2024141460331(reference_regions, webrtc_spans)}
        )
        reference_texts = [segment["text"] for segment in reference_segments]
        predictions = {
            ("sensevoice", "xml"): " ".join(
                row["text"] for row in control_rows[record_id]["oracleSegments"]
            ),
            ("sensevoice", "silero"): " ".join(
                row["textRaw"] for row in baseline_rows[record_id]["vadSegments"]
            ),
            ("sensevoice", "webrtc"): sense_webrtc_rows[record_id]["hypothesis"],
            **{
                ("xlsr", condition): xlsr_rows[record_id]["conditions"][condition]["hypothesis"]
                for condition in CONDITIONS
            },
        }
        model_scores = {}
        for asr in ("sensevoice", "xlsr"):
            model_scores[asr] = {}
            for condition in CONDITIONS:
                model_scores[asr][condition] = {
                    "strict": _small_score(
                        evaluator, reference_texts, predictions[(asr, condition)], False
                    ),
                    "numericNormalized": _small_score(
                        evaluator, reference_texts, predictions[(asr, condition)], True
                    ),
                }
        condition_errors = {
            "sensevoice": {
                "xml": len(control_rows[record_id].get("errors", [])),
                "silero": len(baseline_rows[record_id].get("errors", [])),
                "webrtc": len(sense_webrtc_rows[record_id].get("errors", [])),
            },
            "xlsr": {
                condition: len(xlsr_rows[record_id]["conditions"][condition]["errors"])
                for condition in CONDITIONS
            },
        }
        results.append(
            {
                "recordingId": record_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "scores": model_scores,
                "segments": {
                    "xml": len(control_rows[record_id]["oracleSegments"]),
                    "silero": len(baseline_rows[record_id]["vadSegments"]),
                    "webrtc": len(vad_rows[record_id]["vadSegments"]),
                },
                "errors": condition_errors,
            }
        )

    summary = {
        "recordings": len(results),
        "audioSeconds": sum(row["durationMs"] for row in results) / 1000,
        "sampleAirports": plan["airports"],
        "protocol": {
            "manifestSha256": plan["manifestSha256"],
            "referencesSha256": plan["referencesSha256"],
            "XLSRModel": plan["model"],
            "WebRTCVAD": plan["vad"],
        },
        "arms": {},
        "pairwiseWER": {},
        "vadBoundary": {
            "silero": _aggregate_boundary([baseline_rows[row["recordingId"]] for row in manifest]),
            "webrtc": _aggregate_boundary(webrtc_boundary_rows),
            "referenceCaveat": (
                "XML transcription intervals are a proxy for VAD boundaries, not "
                "independently verified frame-level VAD ground truth."
            ),
        },
        "runtime": {
            "xlsr": json.loads(paths["xlsrRuntime"].read_text(encoding="utf-8")),
            "webrtcVad": json.loads(paths["vadRuntime"].read_text(encoding="utf-8")),
            "sensevoiceWebrtc": json.loads(paths["sensevoiceRuntime"].read_text(encoding="utf-8")),
            "speedComparisonCaveat": (
                "SenseVoice baseline and XML outputs were cached from earlier runs; "
                "elapsed times are not a paired same-device speed ranking."
            ),
        },
    }
    for model in ("sensevoice", "xlsr"):
        for condition in CONDITIONS:
            for metric in ("strict", "numericNormalized"):
                summary["arms"].setdefault(model, {}).setdefault(condition, {})[metric] = (
                    aggregate_condition(results, model, condition, metric)
                )
            summary["arms"][model][condition]["recordingsWithInferenceErrors"] = sum(
                row["errors"][model][condition] > 0 for row in results
            )
            summary["arms"][model][condition]["segmentCount"] = sum(
                row["segments"][condition] for row in results
            )
    rng_seed = 20261021
    for condition in CONDITIONS:
        summary["pairwiseWER"][f"XLSR_minus_SenseVoice__{condition}"] = {}
        for metric in ("strict", "numericNormalized"):
            paired = [row["scores"]["sensevoice"][condition][metric] for row in results]
            paired_right = [row["scores"]["xlsr"][condition][metric] for row in results]
            summary["pairwiseWER"][f"XLSR_minus_SenseVoice__{condition}"][metric] = (
                bootstrap_difference(paired, paired_right, 10000, rng_seed)
            )
        rng_seed += 1
    for model in ("sensevoice", "xlsr"):
        for metric in ("strict", "numericNormalized"):
            paired_vad = {}
            for compare in ("webrtc",):
                left = [row["scores"][model]["silero"][metric] for row in results]
                right = [row["scores"][model][compare][metric] for row in results]
                paired_vad[compare] = bootstrap_difference(left, right, 10000, rng_seed)
            summary["pairwiseWER"][f"{model}_VAD__webrtc_minus_silero__{metric}"] = paired_vad
            rng_seed += 1
    per_airport = {}
    for airport in sorted({row["airport"] for row in results}):
        group = [row for row in results if row["airport"] == airport]
        per_airport[airport] = {
            model: {
                condition: {
                    metric: aggregate_condition(group, model, condition, metric)
                    for metric in ("strict", "numericNormalized")
                }
                for condition in CONDITIONS
            }
            for model in ("sensevoice", "xlsr")
        }
    summary["airports"] = per_airport
    summary["files"] = {
        "experimentPlan": sha256(paths["plan"]),
        "webrtcVadResults": sha256(paths["vad"]),
        "xlsrResults": sha256(paths["xlsr"]),
        "sensevoiceWebrtcResults": sha256(paths["sensevoice"]),
    }
    write_json(paths["output"] / "comparison-summary.json", summary)
    write_comparison_csv(paths["output"] / "comparison-summary.csv", summary)
    write_record_csv(paths["output"] / "per-recording-comparison.csv", results)
    write_vad_csv(paths["output"] / "vad-boundary-comparison.csv", summary["vadBoundary"])
    write_report(paths["output"] / "对照实验报告.md", summary)
    print(
        json.dumps(
            {"summary": str(paths["output"] / "comparison-summary.json"), "arms": summary["arms"]},
            ensure_ascii=False,
            indent=2,
        )
    )


def write_comparison_csv(path: Path, summary: dict) -> None:
    fields = [
        "asr",
        "vad",
        "strictWER",
        "strictN",
        "strictS",
        "strictD",
        "strictI",
        "numericWER",
        "numericN",
        "numericS",
        "numericD",
        "numericI",
        "segments",
        "recordingsWithInferenceErrors",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for asr, conditions in summary["arms"].items():
            for vad, metrics in conditions.items():
                strict, numeric = metrics["strict"], metrics["numericNormalized"]
                writer.writerow(
                    {
                        "asr": asr,
                        "vad": vad,
                        "strictWER": strict["wer"],
                        **{f"strict{name}": strict[name] for name in ("N", "S", "D", "I")},
                        "numericWER": numeric["wer"],
                        **{f"numeric{name}": numeric[name] for name in ("N", "S", "D", "I")},
                        "segments": metrics["segmentCount"],
                        "recordingsWithInferenceErrors": metrics["recordingsWithInferenceErrors"],
                    }
                )


def write_record_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "recordingId",
        "airport",
        "asr",
        "vad",
        "strictWER",
        "strictN",
        "strictS",
        "strictD",
        "strictI",
        "numericWER",
        "numericN",
        "numericS",
        "numericD",
        "numericI",
        "segments",
        "errors",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for asr, conditions in row["scores"].items():
                for vad, metrics in conditions.items():
                    strict, numeric = metrics["strict"], metrics["numericNormalized"]
                    writer.writerow(
                        {
                            "recordingId": row["recordingId"],
                            "airport": row["airport"],
                            "asr": asr,
                            "vad": vad,
                            "strictWER": strict["wer"],
                            **{f"strict{name}": strict[name] for name in ("N", "S", "D", "I")},
                            "numericWER": numeric["wer"],
                            **{f"numeric{name}": numeric[name] for name in ("N", "S", "D", "I")},
                            "segments": row["segments"][vad],
                            "errors": row["errors"][asr][vad],
                        }
                    )


def write_vad_csv(path: Path, boundary: dict) -> None:
    fields = [
        "vad",
        "referenceRegions",
        "predictedRegions",
        "matchedRegions",
        "missedRegions",
        "extraRegions",
        "durationRecall",
        "durationPrecision",
        "startErrorMeanMs",
        "startErrorMedianMs",
        "startErrorP95Ms",
        "endErrorMeanMs",
        "endErrorMedianMs",
        "endErrorP95Ms",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for vad in ("silero", "webrtc"):
            row = boundary[vad]
            writer.writerow(
                {
                    "vad": vad,
                    "referenceRegions": row["reference_regions"],
                    "predictedRegions": row["predicted_regions"],
                    "matchedRegions": row["matched_regions"],
                    "missedRegions": row["missed_regions"],
                    "extraRegions": row["extra_regions"],
                    "durationRecall": row["durationRecall"],
                    "durationPrecision": row["durationPrecision"],
                    "startErrorMeanMs": row["startAbsErrorMs"]["mean"],
                    "startErrorMedianMs": row["startAbsErrorMs"]["median"],
                    "startErrorP95Ms": row["startAbsErrorMs"]["p95"],
                    "endErrorMeanMs": row["endAbsErrorMs"]["mean"],
                    "endErrorMedianMs": row["endAbsErrorMs"]["median"],
                    "endErrorP95Ms": row["endAbsErrorMs"]["p95"],
                }
            )


def write_report(path: Path, summary: dict) -> None:
    rows = []
    for model, conditions in summary["arms"].items():
        for vad, metrics in conditions.items():
            strict, numeric = metrics["strict"], metrics["numericNormalized"]
            rows.append(
                f"| {model} | {vad} | {strict['wer']:.2%} | "
                f"{numeric['wer']:.2%} | {strict['N']} | "
                f"{strict['S']}/{strict['D']}/{strict['I']} | "
                f"{metrics['segmentCount']} | {metrics['recordingsWithInferenceErrors']} |"
            )
    b = summary["vadBoundary"]
    rb = summary["runtime"]["xlsr"]
    lines = [
        "# ATCO2 ASR 与 VAD 对照实验报告",
        "",
        (
            f"实验范围：冻结的 {summary['recordings']} 条公开 ATCO2 英语录音、"
            f"{summary['audioSeconds']:.3f} 秒、7 个机场。该结果是跨语料模型诊断，"
            "不代表 VHHH 业务性能。"
        ),
        "",
        "## 六组转写结果",
        "",
        (
            "严格 WER 是主指标；数字归一化 WER 是既有自定义敏感性指标。"
            "S/D/I 分别为替换、删除、插入。呼号标注在这批样本中未通过"
            "质量标记核验，不据此发布正式呼号准确率。"
        ),
        "",
        "| ASR | 边界输入 | 严格 WER | 数字归一化 WER | N | S/D/I | 片段数 | 有推理错误的录音数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "XLS-R 与 SenseVoice 的配对差值及按录音重抽样的 95% 区间：",
        "",
    ]
    for condition in CONDITIONS:
        for metric, label in (("strict", "严格WER"), ("numericNormalized", "数字归一化WER")):
            item = summary["pairwiseWER"][f"XLSR_minus_SenseVoice__{condition}"][metric]
            lo, hi = item["bootstrap95PercentileInterval"]
            lines.append(
                f"- {condition} / {label}: XLS-R 减 SenseVoice = "
                f"{item['differenceRightMinusLeft']:+.2%}，"
                f"95%区间 [{lo:+.2%}, {hi:+.2%}]。"
            )
    lines.extend(
        [
            "",
            "## VAD 边界诊断",
            "",
            (
                "| VAD | 参考区间 | 预测区间 | 一对一匹配 | 未匹配参考/预测 | "
                "语音时长覆盖率 | 起点绝对误差中位数 | 终点绝对误差中位数 |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for vad in ("silero", "webrtc"):
        item = b[vad]
        lines.append(
            f"| {vad} | {item['reference_regions']} | {item['predicted_regions']} | "
            f"{item['matched_regions']} | {item['missed_regions']}/{item['extra_regions']} | "
            f"{item['durationRecall']:.2%} | "
            f"{item['startAbsErrorMs']['median']:.1f} ms | "
            f"{item['endAbsErrorMs']['median']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            (
                f"限制：{b['referenceCaveat']} 匹配使用首轮固定口径 "
                "（参考间隔≤500 ms 合并，IoU≥0.5 一对一匹配）；指标只作诊断。"
                "XML 人工边界是控制组，不是可部署 VAD。"
                "WebRTC 参数在推理前固定，不基于本批得分调参。"
            ),
            "",
            "## 运行资源与复现",
            "",
            (
                f"XLS-R GPU：{rb['gpu']}；PyTorch {rb['torch']} / CUDA {rb['cuda']}；FP32；"
                f"模型加载 {rb['modelLoadSeconds']:.2f} 秒；"
                f"峰值已分配显存 {rb['peakAllocatedMiB']:.0f} MiB。"
            ),
            "",
            (
                f"模型权重 SHA-256：`{summary['protocol']['XLSRModel']['modelSha256']}`；"
                f"样本清单 SHA-256：`{summary['protocol']['manifestSha256']}`；"
                f"参考文本 SHA-256：`{summary['protocol']['referencesSha256']}`。"
                "全部输入为本地文件，未调用外部推理 API。SenseVoice 既有结果来自首轮缓存，"
                "WebRTC 新臂由同一冻结 SenseVoice 模型和本地 CPU 运行。"
            ),
            "",
            (
                "原始识别文本只保存在本机忽略目录中。交付目录应仅包含本报告、"
                "汇总表、逐录音指标表及可复现实验脚本；"
                "ATCO2 原始音频和完整转写不得公开分发。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "run-vad", "run-xlsr", "run-sensevoice-webrtc", "summarize")
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=HERE / "config.local.json")
    args = parser.parse_args()
    if args.output is None:
        args.output = args.root / "model-vad-comparison-20261021"
    if args.command == "freeze":
        freeze(args)
    elif args.command == "run-vad":
        run_vad(args)
    elif args.command == "run-xlsr":
        run_xlsr(args)
    elif args.command == "run-sensevoice-webrtc":
        run_sensevoice_webrtc(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
