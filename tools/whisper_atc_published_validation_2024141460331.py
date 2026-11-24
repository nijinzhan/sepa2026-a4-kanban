"""Reproduce no-prompt / metadata-prompt evaluation on all 113 published validation files.

This is an author's model-selection validation set, not a fresh independent test.
Prompt vocabulary is read as literals from the pinned LGPL-3.0 author script.
The downloaded script itself is never executed.
"""

import argparse
import ast
import io
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
import whisper_atc_experiment_2024141460331 as common
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.utils import logging


def summarize_existing(output):
    """Correct labels/diagnostics without modifying saved model predictions."""
    source = output / "results.jsonl"
    results = common.read_jsonl(source)
    keys = {(row["recordingId"], row["condition"]) for row in results}
    if len(results) != 226 or len(keys) != 226:
        raise ValueError("expected 113 unique recordings in both conditions")
    evaluator = common.load_module("eval_atco2", common.HERE / "tools/atco2_eval_2024141460331.py")
    normalizers = common.load_normalizers()
    for row in results:
        row["scores"] = common.all_scores(
            row["reference"], row["hypothesis"], evaluator, normalizers
        )
    # common.aggregate's all49 label belongs only to the earlier frozen sample.
    summary = [row for row in common.aggregate(results) if row["subset"] == "validation"]
    common.write_json(output / "summary-corrected.json", summary)
    common.write_json(
        output / "postprocessing-audit.json",
        {
            "inputResultsSha256": common.sha256(source),
            "postprocessingScriptSha256": common.sha256(Path(__file__)),
            "inferencePlan": common.read_json(output / "plan.json"),
            "corrections": [
                "omit redundant common.aggregate all49 label for the 113-file set",
                "HF output is generated-only; flag >=220 tokens, not >=224",
            ],
            "predictionsModified": False,
            "possibleTokenLimitCounts": {
                condition: sum(
                    row["outputTokens"] >= 220 for row in results if row["condition"] == condition
                )
                for condition in ("no_prompt", "metadata_prompt")
            },
            "limitRows": [
                {k: row[k] for k in ("recordingId", "condition", "outputTokens")}
                for row in results
                if row["outputTokens"] >= 220
            ],
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    logging.set_verbosity_error()
    output = common.OUT / "published-validation-113"
    output.mkdir(exist_ok=True)
    if args.summarize_only:
        print(json.dumps(summarize_existing(output)), flush=True)
        return
    parquet = next((common.RESEARCH / "published-atco2-dataset/data").glob("validation-*.parquet"))
    expected = "9c345ebf0ff60c6b7b6f79063dd615e7d0e051343f4c04465590ed24f29e159c"
    if common.sha256(parquet) != expected:
        raise ValueError("published validation parquet changed")
    data = pq.read_table(parquet).to_pylist()
    if len(data) != 113 or len({row["audio"]["path"] for row in data}) != 113:
        raise ValueError("expected 113 unique validation recordings")
    author_file = common.RESEARCH / "Evaluate/Evaluate-FineTuned.py"
    literals = {}
    for node in ast.walk(ast.parse(author_file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("nato", "terminology"):
                    literals[target.id] = ast.literal_eval(node.value)
    if set(literals) != {"nato", "terminology"}:
        raise ValueError("author prompt literals not found")
    info = common.MODELS["mixed"]
    local = common.HERE / "models" / info["id"].split("/")[-1]
    if common.sha256(local / "model.safetensors") != info["sha256"]:
        raise ValueError("model checksum mismatch")
    plan = {
        "model": info,
        "validationCount": 113,
        "dataSha256": expected,
        "scriptSha256": common.sha256(Path(__file__)),
        "commonScriptSha256": common.sha256(Path(common.__file__)),
        "authorScriptSha256": common.sha256(author_file),
        "conditions": ["no_prompt", "metadata_prompt"],
        "dtype": "float16",
        "device": "cuda",
        "numBeams": 1,
        "language": "en",
        "task": "transcribe",
        "maxNewTokens": 220,
        "promptTruncation": "startofprev + last 223 prompt tokens; output allowance 220",
        "purpose": "published-validation replication, not a new held-out project test",
        "referenceInPrompt": False,
    }
    if (output / "plan.json").exists():
        if common.read_json(output / "plan.json") != plan:
            raise ValueError("existing protocol changed")
    else:
        common.write_json(output / "plan.json", plan)
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            local, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    model.generation_config.forced_decoder_ids = None
    result_path = output / "results.jsonl"
    previous = common.read_jsonl(result_path) if result_path.exists() else []
    done = {(row["recordingId"], row["condition"]) for row in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate result rows")
    started = time.perf_counter()
    for index, row in enumerate(data):
        rid = Path(row["audio"]["path"]).stem
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        common.validate_audio_span(len(audio), rate)
        features = processor.feature_extractor(
            audio, sampling_rate=rate, return_tensors="pt", return_attention_mask=True
        )
        prompt = (
            "Air Traffic Control Communications "
            + row["info"].replace("\n", " ")
            + " "
            + literals["nato"].replace(",", " ")
            + " "
            + literals["terminology"].replace(",", " ")
        )
        full_prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt")
        prompt_ids = torch.cat((full_prompt_ids[:1], full_prompt_ids[1:][-223:])).to("cuda")
        for condition in plan["conditions"]:
            if (rid, condition) in done:
                continue
            start = time.perf_counter()
            kwargs = {"prompt_ids": prompt_ids} if condition == "metadata_prompt" else {}
            with torch.inference_mode():
                ids = model.generate(
                    features.input_features.to("cuda", torch.float16),
                    attention_mask=features.attention_mask.to("cuda"),
                    language="en",
                    task="transcribe",
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=220,
                    return_timestamps=False,
                    **kwargs,
                )
            torch.cuda.synchronize()
            result = {
                "recordingId": rid,
                "model": "mixed",
                "condition": condition,
                "publishedSplit": "validation",
                "reference": row["text"],
                "hypothesis": processor.batch_decode(ids, skip_special_tokens=True)[0].strip(),
                "asrSeconds": time.perf_counter() - start,
                "errors": [],
                "promptTokens": len(prompt_ids) if kwargs else 0,
                "outputTokens": int(ids.shape[-1]),
            }
            if ids.shape[-1] >= 220:
                result["errors"].append({"type": "possibleTokenLimit"})
            with result_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        if (index + 1) % 10 == 0 or index == 112:
            print(f"published validation {index + 1}/113", flush=True)
    summary = summarize_existing(output)
    common.write_json(
        output / "runtime.json", {"secondsThisInvocation": time.perf_counter() - started}
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
