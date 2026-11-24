from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent.parent
ROOT = HERE / "outputs" / "atco2-1h-20261021"
EXISTING = ROOT / "model-vad-comparison-20261021"
OUTPUT = ROOT / "silero-v6.2.3-xlsr-eval"
BASELINE = ROOT / "evaluation-49"
EXPECTED_MODEL_SHA256 = "E1122837F4154C511485FE0B9C64455F7B929C96FBB8D79FBDB336383EBD3720"
VAD_PARAMETERS = {
    "sampleRate": 16000,
    "threshold": 0.5,
    "minSpeechDurationMs": 250,
    "minSilenceDurationMs": 500,
    "maxSpeechDurationSeconds": 20,
    "speechPadMs": 0,
    "windowSizeSamples": 512,
    "negativeThreshold": "official API default (threshold - 0.15)",
    "torchThreads": 1,
}
CONDITIONS = ("xml", "silero_v5", "webrtc", "silero_v6.2.3")
FROZEN_XLSR_ERROR_COUNTS = {
    "xml": {"strict": (267, 76, 12), "numericNormalized": (266, 76, 14)},
    "silero_v5": {"strict": (274, 97, 13), "numericNormalized": (274, 94, 13)},
    "webrtc": {"strict": (276, 68, 20), "numericNormalized": (275, 69, 20)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_comparison_module():
    sys.path.insert(0, str(HERE / "tools"))
    import atco2_comparison_2024141460331 as comparison

    return comparison


def validate_inputs():
    comparison = load_comparison_module()
    plan, paths = comparison.load_plan(SimpleNamespace(root=ROOT, output=EXISTING))
    model_file = Path(plan["model"]["localPath"]) / "model.safetensors"
    if sha256(model_file) != plan["model"]["modelSha256"]:
        raise ValueError("frozen XLS-R checkpoint checksum mismatch")
    xlsr_runtime = read_json(paths["xlsrRuntime"])
    if sha256(paths["xlsr"]) != xlsr_runtime["resultsSha256"]:
        raise ValueError("frozen XLS-R results checksum mismatch")
    manifest = read_json(BASELINE / "manifest.json")
    references = read_json(BASELINE / "references.json")
    if len(manifest) != 49 or len(references) != 49:
        raise ValueError("expected the frozen 49-recording sample")
    return comparison, plan, paths, manifest, references


def load_vad_model():
    import torch
    import silero_vad
    from silero_vad.utils_vad import init_jit_model

    package_version = importlib.metadata.version("silero-vad")
    if package_version != "6.2.3":
        raise ValueError(f"expected silero-vad==6.2.3, found {package_version}")
    source = Path(silero_vad.__file__).parent / "data" / "silero_vad.jit"
    if sha256(source).upper() != EXPECTED_MODEL_SHA256:
        raise ValueError("official Silero v6 JIT weight checksum mismatch")
    temp_path = Path(tempfile.gettempdir()) / f"silero-v6-{EXPECTED_MODEL_SHA256[:12]}.jit"
    if any(ord(char) > 127 for char in str(temp_path)):
        temp_path = Path("D:/course_project") / f"silero-v6-{EXPECTED_MODEL_SHA256[:12]}.jit"
    if not temp_path.exists() or sha256(temp_path).upper() != EXPECTED_MODEL_SHA256:
        shutil.copyfile(source, temp_path)
    if sha256(temp_path).upper() != EXPECTED_MODEL_SHA256:
        raise ValueError("temporary Silero model copy failed checksum verification")
    torch.set_num_threads(VAD_PARAMETERS["torchThreads"])
    model = init_jit_model(str(temp_path))
    model.eval()
    return torch, model, source, temp_path


def run_vad(comparison, plan, manifest):
    import torch
    from silero_vad import get_speech_timestamps

    output_path = OUTPUT / "silero-v6-vad.jsonl"
    runtime_path = OUTPUT / "silero-v6-vad-runtime.json"
    model_sha = EXPECTED_MODEL_SHA256.lower()
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {row["recordingId"] for row in existing}
    expected = {item["recordingId"] for item in manifest}
    if len(done) != len(existing) or done - expected:
        raise ValueError("existing v6 VAD output contains duplicate or unexpected recordings")
    if runtime_path.exists() and done == expected:
        runtime = read_json(runtime_path)
        if sha256(output_path) == runtime["resultsSha256"]:
            return
        raise ValueError("completed v6 VAD output checksum mismatch")

    torch, model, model_source, model_path = load_vad_model()
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        for item in manifest:
            recording_id = item["recordingId"]
            if recording_id in done:
                continue
            audio_path = BASELINE.parent / item["audioRelative"]
            if sha256(audio_path) != item["audioSha256"]:
                raise ValueError(f"frozen audio checksum mismatch: {recording_id}")
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != VAD_PARAMETERS["sampleRate"] or audio.ndim != 1:
                raise ValueError(f"expected frozen 16 kHz mono audio: {recording_id}")
            model.reset_states()
            with torch.inference_mode():
                timestamps = get_speech_timestamps(
                    torch.from_numpy(np.ascontiguousarray(audio)),
                    model,
                    threshold=VAD_PARAMETERS["threshold"],
                    sampling_rate=sample_rate,
                    min_speech_duration_ms=VAD_PARAMETERS["minSpeechDurationMs"],
                    max_speech_duration_s=VAD_PARAMETERS["maxSpeechDurationSeconds"],
                    min_silence_duration_ms=VAD_PARAMETERS["minSilenceDurationMs"],
                    speech_pad_ms=VAD_PARAMETERS["speechPadMs"],
                    window_size_samples=VAD_PARAMETERS["windowSizeSamples"],
                    return_seconds=False,
                )
            spans = [
                {
                    "startMs": round(segment["start"] * 1000 / sample_rate),
                    "endMs": round(segment["end"] * 1000 / sample_rate),
                }
                for segment in timestamps
            ]
            row = {
                "recordingId": recording_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "vadSegments": spans,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(recording_id)
            print(json.dumps({"stage": "silero-v6-vad", "progress": f"{len(done)}/49", "recordingId": recording_id}), flush=True)

    write_json(
        runtime_path,
        {
            "finishedUtc": datetime.now(UTC).isoformat(),
            "elapsedSeconds": time.perf_counter() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sileroVadPackage": importlib.metadata.version("silero-vad"),
            "modelFile": model_source.name,
            "modelSha256": model_sha,
            "parameters": VAD_PARAMETERS,
            "sourceManifestSha256": plan["manifestSha256"],
            "resultsSha256": sha256(output_path),
            "recordings": len(done),
        },
    )


def run_xlsr(comparison, plan, manifest):
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2Processor

    vad_path = OUTPUT / "silero-v6-vad.jsonl"
    vad_runtime_path = OUTPUT / "silero-v6-vad-runtime.json"
    output_path = OUTPUT / "xlsr-silero-v6-results.jsonl"
    runtime_path = OUTPUT / "xlsr-silero-v6-runtime.json"
    if not vad_runtime_path.exists() or sha256(vad_path) != read_json(vad_runtime_path)["resultsSha256"]:
        raise ValueError("complete, checksum-verified v6 VAD output is required")
    if not torch.cuda.is_available():
        raise RuntimeError("the XLS-R comparison requires CUDA")
    vad_rows = {row["recordingId"]: row for row in read_jsonl(vad_path)}
    expected = {item["recordingId"] for item in manifest}
    if set(vad_rows) != expected:
        raise ValueError("v6 VAD output does not cover the frozen sample")

    model_path = plan["model"]["localPath"]
    processor = Wav2Vec2Processor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    existing = read_jsonl(output_path) if output_path.exists() else []
    done = {row["recordingId"] for row in existing}
    if len(done) != len(existing) or done - expected:
        raise ValueError("existing v6 XLS-R output contains duplicate or unexpected recordings")

    started = time.perf_counter()
    elapsed = 0.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        for item in manifest:
            recording_id = item["recordingId"]
            if recording_id in done:
                continue
            audio_path = BASELINE.parent / item["audioRelative"]
            if sha256(audio_path) != item["audioSha256"]:
                raise ValueError(f"frozen audio checksum mismatch: {recording_id}")
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if sample_rate != 16000 or audio.ndim != 1:
                raise ValueError(f"expected frozen 16 kHz mono audio: {recording_id}")
            segments = []
            record_started = time.perf_counter()
            for vad_segment in vad_rows[recording_id]["vadSegments"]:
                start = max(0, round(vad_segment["startMs"] * sample_rate / 1000))
                end = min(len(audio), round(vad_segment["endMs"] * sample_rate / 1000))
                if end <= start:
                    raise ValueError(f"empty v6 VAD segment: {recording_id} {vad_segment}")
                inputs = processor(audio[start:end], sampling_rate=16000, return_tensors="pt")
                inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
                with torch.inference_mode():
                    predicted_ids = torch.argmax(model(**inputs).logits, dim=-1)
                text = processor.batch_decode(predicted_ids.detach().cpu())[0]
                segments.append({**vad_segment, "textRaw": text})
            torch.cuda.synchronize()
            record_seconds = time.perf_counter() - record_started
            elapsed += record_seconds
            row = {
                "recordingId": recording_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "segments": segments,
                "hypothesis": " ".join(segment["textRaw"] for segment in segments),
                "recordProcessingSeconds": record_seconds,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            done.add(recording_id)
            print(json.dumps({"stage": "xlsr-silero-v6", "progress": f"{len(done)}/49", "recordingId": recording_id, "seconds": round(record_seconds, 3)}), flush=True)

    if done == expected:
        torch.cuda.synchronize()
        write_json(
            runtime_path,
            {
                "finishedUtc": datetime.now(UTC).isoformat(),
                "recordProcessingSeconds": elapsed,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "peakAllocatedMiB": torch.cuda.max_memory_allocated(0) / 1024**2,
                "transformers": importlib.metadata.version("transformers"),
                "model": plan["model"],
                "sourceVADSha256": sha256(vad_path),
                "resultsSha256": sha256(output_path),
                "recordings": len(done),
            },
        )


def score_conditions(comparison, evaluator, references, prediction_rows):
    scores = {}
    source_conditions = {"xml": "xml", "silero_v5": "silero", "webrtc": "webrtc"}
    for condition in CONDITIONS:
        if condition == "silero_v6.2.3":
            prediction = prediction_rows["hypothesis"]
        else:
            prediction = prediction_rows["conditions"][source_conditions[condition]]["hypothesis"]
        ref_text = " ".join(segment["text"] for segment in references)
        scores[condition] = {
            "strict": comparison._small_score(evaluator, [ref_text], prediction, False),
            "numericNormalized": comparison._small_score(evaluator, [ref_text], prediction, True),
        }
    return scores


def summarize(comparison, plan, manifest, references):
    vad_path = OUTPUT / "silero-v6-vad.jsonl"
    asr_path = OUTPUT / "xlsr-silero-v6-results.jsonl"
    for result_path, runtime_path in (
        (vad_path, OUTPUT / "silero-v6-vad-runtime.json"),
        (asr_path, OUTPUT / "xlsr-silero-v6-runtime.json"),
    ):
        if not runtime_path.exists() or sha256(result_path) != read_json(runtime_path)["resultsSha256"]:
            raise ValueError(f"missing or checksum-invalid output: {result_path.name}")

    evaluator = comparison.load_evaluator()
    xlsr_rows = {row["recordingId"]: row for row in read_jsonl(EXISTING / "xlsr-results.jsonl")}
    v6_rows = {row["recordingId"]: row for row in read_jsonl(asr_path)}
    v6_vad_rows = {row["recordingId"]: row for row in read_jsonl(vad_path)}
    expected = {item["recordingId"] for item in manifest}
    if any(set(rows) != expected for rows in (xlsr_rows, v6_rows, v6_vad_rows)):
        raise ValueError("one or more conditions do not cover exactly the frozen sample")

    records = []
    boundary_rows = {condition: [] for condition in ("silero_v5", "webrtc", "silero_v6.2.3")}
    for item in manifest:
        record_id = item["recordingId"]
        reference = references[record_id]
        reference_regions = [
            (segment["startMs"], segment["endMs"])
            for segment in reference
            if evaluator.tokenize_2024141460331(segment["text"])
        ]
        predictions = {
            "conditions": xlsr_rows[record_id]["conditions"],
            "hypothesis": v6_rows[record_id]["hypothesis"],
        }
        condition_scores = score_conditions(
            comparison,
            evaluator,
            reference,
            predictions,
        )
        for condition, spans in (
            ("silero_v5", predictions["conditions"]["silero"]["segments"]),
            ("webrtc", predictions["conditions"]["webrtc"]["segments"]),
            ("silero_v6.2.3", v6_vad_rows[record_id]["vadSegments"]),
        ):
            boundary_rows[condition].append(
                {
                    "boundary": evaluator.boundary_score_2024141460331(
                        reference_regions,
                        [(span["startMs"], span["endMs"]) for span in spans],
                    )
                }
            )
        records.append(
            {
                "recordingId": record_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "scores": {"xlsr": condition_scores},
                "segments": {
                    "xml": len(predictions["conditions"]["xml"]["segments"]),
                    "silero_v5": len(predictions["conditions"]["silero"]["segments"]),
                    "webrtc": len(predictions["conditions"]["webrtc"]["segments"]),
                    "silero_v6.2.3": len(v6_vad_rows[record_id]["vadSegments"]),
                },
            }
        )

    metrics = ("strict", "numericNormalized")
    summary = {
        "status": "exploratory; reused the previously evaluated 49-recording set",
        "recordings": len(records),
        "audioSeconds": sum(row["durationMs"] for row in records) / 1000,
        "airports": plan["airports"],
        "model": plan["model"],
        "sileroV6": {
            "package": "silero-vad==6.2.3",
            "modelSha256": EXPECTED_MODEL_SHA256.lower(),
            "parameters": VAD_PARAMETERS,
            "claimCaveat": "Upstream quality claims are not ATC-specific and v6 requires its own operating point calibration.",
        },
        "arms": {},
        "pairedWER": {},
        "vadBoundary": {
            condition: comparison._aggregate_boundary(rows)
            for condition, rows in boundary_rows.items()
        },
        "boundaryReferenceCaveat": "XML transcript intervals are boundary proxies, not independently verified frame-level VAD labels.",
        "sampleCaveat": "This same 49-recording sample was used in earlier model/VAD comparisons; results do not constitute independent confirmation or VHHH performance.",
    }
    for condition in CONDITIONS:
        summary["arms"][condition] = {
            metric: comparison.aggregate_condition(records, "xlsr", condition, metric)
            for metric in metrics
        }
        summary["arms"][condition]["segmentCount"] = sum(
            row["segments"][condition] for row in records
        )
    for condition, expected_counts in FROZEN_XLSR_ERROR_COUNTS.items():
        for metric, expected in expected_counts.items():
            actual = summary["arms"][condition][metric]
            if (actual["S"], actual["D"], actual["I"]) != expected:
                raise ValueError(
                    f"frozen XLS-R baseline mismatch for {condition}/{metric}: "
                    f"expected S/D/I={expected}, got {(actual['S'], actual['D'], actual['I'])}"
                )
    for condition in ("silero_v5", "webrtc"):
        summary["pairedWER"][f"silero_v6.2.3_minus_{condition}"] = {
            metric: comparison.bootstrap_difference(
                [row["scores"]["xlsr"][condition][metric] for row in records],
                [row["scores"]["xlsr"]["silero_v6.2.3"][metric] for row in records],
                10000,
                20261021 + len(summary["pairedWER"]),
            )
            for metric in metrics
        }
    summary["files"] = {
        "experimentPlanSha256": sha256(EXISTING / "experiment-plan.json"),
        "vadResultsSha256": sha256(vad_path),
        "xlsrResultsSha256": sha256(asr_path),
    }
    write_json(OUTPUT / "summary.json", summary)

    with (OUTPUT / "per-recording.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["recordingId", "airport", "segments", "xmlWER", "sileroV5WER", "webrtcWER", "sileroV6WER"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "recordingId": row["recordingId"],
                    "airport": row["airport"],
                    "segments": row["segments"]["silero_v6.2.3"],
                    "xmlWER": row["scores"]["xlsr"]["xml"]["strict"]["wer"],
                    "sileroV5WER": row["scores"]["xlsr"]["silero_v5"]["strict"]["wer"],
                    "webrtcWER": row["scores"]["xlsr"]["webrtc"]["strict"]["wer"],
                    "sileroV6WER": row["scores"]["xlsr"]["silero_v6.2.3"]["strict"]["wer"],
                }
            )
    write_report(summary)


def write_report(summary):
    arms = summary["arms"]
    lines = [
        "# Silero v6.2.3 and XLS-R Exploratory Comparison",
        "",
        f"Status: exploratory on {summary['recordings']} previously evaluated ATCO2 recordings; not independent validation and not VHHH performance.",
        "",
        "## XLS-R WER",
        "",
        "| Input boundary | Strict WER | Numeric-normalized WER | Segments |",
        "|---|---:|---:|---:|",
    ]
    for condition, label in (
        ("xml", "XML reference"),
        ("silero_v5", "Silero v5.0"),
        ("webrtc", "WebRTC"),
        ("silero_v6.2.3", "Silero v6.2.3"),
    ):
        lines.append(
            f"| {label} | {arms[condition]['strict']['wer']:.2%} | "
            f"{arms[condition]['numericNormalized']['wer']:.2%} | {arms[condition]['segmentCount']} |"
        )
    lines.extend(["", "## Paired WER Differences", "", "Positive means v6.2.3 has higher WER than the comparator.", ""])
    for comparator in ("silero_v5", "webrtc"):
        result = summary["pairedWER"][f"silero_v6.2.3_minus_{comparator}"]
        for metric, label in (("strict", "strict"), ("numericNormalized", "numeric-normalized")):
            interval = result[metric]["bootstrap95PercentileInterval"]
            lines.append(
                f"- v6.2.3 minus {comparator}, {label}: {result[metric]['differenceRightMinusLeft']:+.2%}; "
                f"paired 95% bootstrap interval [{interval[0]:+.2%}, {interval[1]:+.2%}]."
            )
    lines.extend(["", "## Boundary Diagnostics", "", "XML timing is a proxy, not independently verified frame-level VAD ground truth.", ""])
    lines.extend(
        [
            "| VAD | Duration recall | Duration precision | Median start error | Median end error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, label in (("silero_v5", "Silero v5.0"), ("webrtc", "WebRTC"), ("silero_v6.2.3", "Silero v6.2.3")):
        metric = summary["vadBoundary"][condition]
        lines.append(
            f"| {label} | {metric['durationRecall']:.2%} | {metric['durationPrecision']:.2%} | "
            f"{metric['startAbsErrorMs']['median']:.1f} ms | {metric['endAbsErrorMs']['median']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- This sample has already been used for prior comparisons; the new arm is exploratory and must not be presented as an independent test.",
            "- The old v5 arm used sherpa-onnx; v6.2.3 uses the official Silero Python API. The comparison therefore covers the deployable model-plus-wrapper path, not weights alone.",
            "- Threshold 0.5 was retained for a drop-in comparison. Official v6 release notes report improved general/noisy-domain performance, but provide no ATC-specific guarantee; a separately held-out development set is needed to calibrate v6's operating point.",
            "- All ASR outputs use the same frozen XLS-R checkpoint, 16 kHz audio, CTC greedy decoding, and scorer.",
            "",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "vad", "asr", "summarize"), nargs="?", default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison, plan, paths, manifest, references = validate_inputs()
    controlled = read_json(paths["controlled"])
    write_json(OUTPUT / "controlled-boundary-comparison.json", controlled)
    if args.stage in ("all", "vad"):
        run_vad(comparison, plan, manifest)
    if args.stage in ("all", "asr"):
        run_xlsr(comparison, plan, manifest)
    if args.stage in ("all", "summarize"):
        summarize(comparison, plan, manifest, references)


if __name__ == "__main__":
    main()
