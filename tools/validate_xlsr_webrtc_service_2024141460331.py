"""Replay frozen ATCO2 through real models and a clearly labelled HTTP test boundary."""

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE / "tools"))

from atco2_eval_2024141460331 import (  # noqa: E402
    score_words_2024141460331,
    tokenize_2024141460331,
)
from support_2024141460331 import RID, SharedServer  # noqa: E402

from a4_speech.cli_2024141460331 import (  # noqa: E402
    _engine_2024141460331,
    load_config_2024141460331,
)
from a4_speech.client_2024141460331 import TeamApi  # noqa: E402
from a4_speech.service_2024141460331 import SpeechService_2024141460331  # noqa: E402


def run(config_path: Path, root: Path, output: Path, baseline_engine: str = "xlsr") -> dict:
    baseline = root / "evaluation-49"
    previous = root / "model-vad-comparison-20261021"
    manifest_file = baseline / "manifest.json"
    references_file = baseline / "references.json"
    if baseline_engine not in {"xlsr", "sensevoice"}:
        raise ValueError("expected a frozen XLS-R or SenseVoice comparison")
    prefix = "sensevoice-webrtc" if baseline_engine == "sensevoice" else "xlsr"
    previous_file = previous / f"{prefix}-results.jsonl"
    previous_runtime = json.loads((previous / f"{prefix}-runtime.json").read_text(encoding="utf-8"))
    plan = json.loads((previous / "experiment-plan.json").read_text(encoding="utf-8"))
    for name, path in (("manifest", manifest_file), ("references", references_file)):
        if sha256(path.read_bytes()).hexdigest() != plan[f"{name}Sha256"]:
            raise ValueError(f"frozen {name} checksum mismatch")
    if sha256(previous_file.read_bytes()).hexdigest() != previous_runtime["resultsSha256"]:
        raise ValueError("previous experiment outputs do not match their recorded checksum")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    references = json.loads(references_file.read_text(encoding="utf-8"))
    old = {
        row["recordingId"]: row
        for row in (
            json.loads(line) for line in previous_file.read_text(encoding="utf-8").splitlines()
        )
    }
    if len(manifest) != 49 or set(old) != {item["recordingId"] for item in manifest}:
        raise ValueError("expected the existing frozen set of exactly 49 recordings")
    engine = _engine_2024141460331(load_config_2024141460331(config_path))
    snapshot = engine.snapshot()
    if snapshot["vad"]["engineCode"] != "webrtc-vad":
        raise ValueError("this replay requires the frozen WebRTC segmentation")
    if baseline_engine == "sensevoice":
        if snapshot["asr"] != previous_runtime["executionSnapshot"]["asr"]:
            raise ValueError("SenseVoice model, tokens or decoding parameters changed")
        source_vad = previous / "webrtc-vad.jsonl"
        if sha256(source_vad.read_bytes()).hexdigest() != previous_runtime["sourceVADSha256"]:
            raise ValueError("previous WebRTC source boundaries checksum mismatch")
    registry = {
        kind: {
            "code": snapshot[kind]["engineCode"],
            "name": snapshot[kind]["modelName"],
            "modelName": snapshot[kind]["modelName"],
            "version": snapshot[kind]["version"],
            "kind": kind,
            "device": "cuda" if baseline_engine == "xlsr" and kind == "asr" else "cpu",
            "deployMode": "embedded",
            "isEnabled": True,
        }
        for kind in ("vad", "asr")
    }
    started = time.perf_counter()
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for index, item in enumerate(manifest):
            raw = (root / item["audioRelative"]).read_bytes()
            if sha256(raw).hexdigest() != item["audioSha256"]:
                raise ValueError("frozen sample checksum mismatch")
            server = SharedServer()
            server.audio = raw
            server.engines = registry
            server.recording.update(
                durationMs=item["durationMs"],
                sampleRateHz=item["sampleRate"],
                audioChannels=item["channels"],
                fileSizeBytes=len(raw),
                checksumSha256=item["audioSha256"],
            )
            with tempfile.TemporaryDirectory(prefix="a4-service-validation-") as receipt:
                with httpx.Client(transport=httpx.MockTransport(server)) as client:
                    api = TeamApi(
                        client, "https://team.test/api/v1/a3", "https://team.test/api/v1/a4"
                    )
                    service = SpeechService_2024141460331(api, engine, receipt_dir=Path(receipt))
                    result = service.run_2024141460331(
                        RID, f"frozen-{index}", business_task_id="validation-only"
                    )
                    if result["status"] != "succeeded" or result["saveConfirmed"] is not True:
                        raise ValueError(f"service flow failed for {item['recordingId']}")
                    for transcript in result["transcripts"]:
                        history = service.read_transcript_2024141460331(
                            transcript["id"], result["jobId"]
                        )
                        if history["transcript"]["textRaw"] != transcript["textRaw"]:
                            raise ValueError("saved transcript readback mismatch")
            segments = [
                {key: transcript[key] for key in ("startMs", "endMs", "textRaw")}
                for transcript in result["transcripts"]
            ]
            prior = old[item["recordingId"]]
            if baseline_engine == "xlsr":
                prior = prior["conditions"]["webrtc"]
            prior_segments = [
                {key: segment[key] for key in ("startMs", "endMs", "textRaw")}
                for segment in prior["segments"]
            ]
            reference = " ".join(segment["text"] for segment in references[item["recordingId"]])
            hypothesis = " ".join(segment["textRaw"] for segment in segments)
            score = score_words_2024141460331(
                tokenize_2024141460331(reference), tokenize_2024141460331(hypothesis)
            )
            row = {
                "recordingId": item["recordingId"],
                "airport": item["airport"],
                "audioSha256": item["audioSha256"],
                "durationMs": item["durationMs"],
                "segments": segments,
                "strict": {key: score[key] for key in ("S", "D", "I", "N", "wer")},
                "matchesPreviousSegmentsAndText": segments == prior_segments,
                "saveConfirmed": True,
                "readbackMatched": True,
                "httpBoundary": "httpx.MockTransport; in-memory test server",
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if (index + 1) % 10 == 0:
                print(json.dumps({"progress": index + 1, "total": len(manifest)}), flush=True)
    totals = {key: sum(row["strict"][key] for row in rows) for key in ("S", "D", "I", "N")}
    summary = {
        "finishedUtc": datetime.now(UTC).isoformat(),
        "recordings": len(rows),
        "savedAndReadBack": sum(row["readbackMatched"] for row in rows),
        "matchesPreviousSegmentsAndText": sum(
            row["matchesPreviousSegmentsAndText"] for row in rows
        ),
        "totalSegments": sum(len(row["segments"]) for row in rows),
        "strict": {**totals, "wer": (totals["S"] + totals["D"] + totals["I"]) / totals["N"]},
        "elapsedSeconds": time.perf_counter() - started,
        "executionSnapshot": snapshot,
        "manifestSha256": sha256(manifest_file.read_bytes()).hexdigest(),
        "referencesSha256": sha256(references_file.read_bytes()).hexdigest(),
        "previousOutputsSha256": sha256(previous_file.read_bytes()).hexdigest(),
        "resultsSha256": sha256((output / "results.jsonl").read_bytes()).hexdigest(),
        "dataset": "ATCO2; reused frozen set; non-VHHH; not an independent holdout",
        "httpBoundary": "simulated A3/A4 HTTP transport; no real team service was contacted",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=HERE / "config.experimental-xlsr-webrtc.example.json"
    )
    parser.add_argument("--root", type=Path, default=HERE / "outputs/atco2-1h-20261021")
    parser.add_argument(
        "--output", type=Path, default=HERE / "outputs/xlsr-webrtc-service-20261021"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.root, args.output), ensure_ascii=False, indent=2))
