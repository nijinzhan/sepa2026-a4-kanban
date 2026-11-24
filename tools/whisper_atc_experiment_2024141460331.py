"""Local WhisperATC replication and independent audit of the frozen ATCO2 sample.

Raw references/audio remain under ignored outputs/. This does not change the
previous six-arm experiment or its primary scoring rules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / "outputs/atco2-1h-20261021"
OUT = ROOT / "whisper-atc-20261021"
RESEARCH = OUT / "research"
CONDITIONS = ("whole", "xml", "silero", "webrtc")
MODELS = {
    "atcosim": {
        "id": "jlvdoorn/whisper-large-v3-atcosim",
        "revision": "99a8cc0071e479a494372925650737582241070a",
        "sha256": "7f1f75dea64ed0f483a62edfcbbfcec40c10ee13b387fa9bfa1466343a86b10b",
        "interpretation": "ATCOSIM-only declared fine-tuning; ATCO2 cross-corpus test",
    },
    "mixed": {
        "id": "jlvdoorn/whisper-large-v3-atco2-asr-atcosim",
        "revision": "efd96bd7862709d2d6aeb55621934a9965db8ed7",
        "sha256": "fdd401356e0e8eb5a6d6688a5f2df9f3c52215973151b8eff05b4cf7a152ea29",
        "interpretation": "ATCO2 training overlap; full sample is diagnostic only",
    },
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_annotations(text):
    """Remove marker tokens, preserving the words between paired markers."""
    return re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)


def build_split_index(splits):
    result = {}
    for split, rows in splits.items():
        for row in rows:
            if row["id"] in result:
                raise ValueError(f"recording appears in both/duplicate splits: {row['id']}")
            result[row["id"]] = {**row, "split": split}
    return result


def split_label(index, recording_id):
    return index.get(recording_id, {}).get("split", "unknown")


def validate_audio_span(samples, sample_rate):
    if sample_rate != 16000:
        raise ValueError("Whisper experiment requires 16000 Hz without implicit resampling")
    if samples <= 0 or samples > 30 * sample_rate:
        raise ValueError("short-form input must be nonempty and at most 30 seconds")


def pcm_digest(wav):
    source = io.BytesIO(wav) if isinstance(wav, bytes) else wav
    audio, rate = sf.read(source, dtype="int16", always_2d=True)
    digest = hashlib.sha256()
    digest.update(f"{rate}:{audio.shape}".encode())
    digest.update(audio.astype("<i2").tobytes())
    return digest.hexdigest()


def load_inputs():
    baseline = ROOT / "evaluation-49"
    manifest = read_json(baseline / "manifest.json")
    references = read_json(baseline / "references.json")
    plan = read_json(baseline / "plan.json")
    if sha256(baseline / "manifest.json") != plan["manifestSha256"]:
        raise ValueError("frozen manifest changed")
    if sha256(baseline / "references.json") != plan["referencesSha256"]:
        raise ValueError("frozen references changed")
    if len(manifest) != 49 or len({row["recordingId"] for row in manifest}) != 49:
        raise ValueError("expected 49 unique frozen recordings")
    for row in manifest:
        for relative, checksum in (("audioRelative", "audioSha256"), ("xmlRelative", "xmlSha256")):
            if sha256(ROOT / row[relative]) != row[checksum]:
                raise ValueError(f"input changed: {row['recordingId']} {relative}")
    return manifest, references


def published_splits():
    import pyarrow.parquet as pq

    splits = {}
    for split in ("train", "validation"):
        paths = list((RESEARCH / "published-atco2-dataset/data").glob(f"{split}-*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"expected pinned published {split} parquet")
        rows = []
        for row in pq.read_table(paths[0]).to_pylist():
            rows.append(
                {
                    "id": Path(row["audio"]["path"]).stem,
                    "text": row["text"],
                    "pcmSha256": pcm_digest(row["audio"]["bytes"]),
                }
            )
        splits[split] = rows
    return build_split_index(splits)


def load_normalizers():
    from whisper.normalizers import EnglishTextNormalizer

    author = load_module("pinned_whisper_atc_normalizer", RESEARCH / "Evaluate/Normalizer.py")
    return EnglishTextNormalizer(), author.filterAndNormalize


def all_scores(reference, hypothesis, evaluator, normalizers):
    import jiwer

    openai, author = normalizers
    strict_r = evaluator.tokenize_2024141460331(reference, numeric=False)
    strict_h = evaluator.tokenize_2024141460331(hypothesis, numeric=False)
    numeric_r = evaluator.tokenize_2024141460331(reference, numeric=True)
    numeric_h = evaluator.tokenize_2024141460331(hypothesis, numeric=True)
    pairs = {
        "strict": (strict_r, strict_h),
        "numeric": (numeric_r, numeric_h),
        "openai": (
            openai(strip_annotations(reference)).split(),
            openai(strip_annotations(hypothesis)).split(),
        ),
        "authorSafe": (
            author(strip_annotations(reference)).split(),
            author(strip_annotations(hypothesis)).split(),
        ),
    }
    scores = {}
    for name, (ref, hyp) in pairs.items():
        result = jiwer.process_words(" ".join(ref), " ".join(hyp))
        scores[name] = {
            "N": len(ref),
            "S": result.substitutions,
            "D": result.deletions,
            "I": result.insertions,
            "C": result.hits,
            "wer": result.wer,
        }
        if name in ("strict", "numeric"):
            own = evaluator.score_words_2024141460331(ref, hyp)
            # Alignment tie-breaking may change S/D/I individually, but not edit distance.
            if sum(own[k] for k in ("S", "D", "I")) != sum(
                scores[name][k] for k in ("S", "D", "I")
            ):
                raise AssertionError(f"independent scorer disagreement: {name}")
    return scores


def existing_hypotheses():
    previous = ROOT / "model-vad-comparison-20261021"
    records = []
    for row in read_jsonl(previous / "xlsr-results.jsonl"):
        for condition, arm in row["conditions"].items():
            records.append(
                {
                    "recordingId": row["recordingId"],
                    "model": "xlsr",
                    "condition": condition,
                    "hypothesis": arm["hypothesis"],
                    "errors": arm["errors"],
                }
            )
    controlled = read_json(ROOT / "diagnosis-20261021/controlled-boundary-comparison.json")
    for row in controlled["recordings"]:
        records.append(
            {
                "recordingId": row["recordingId"],
                "model": "sensevoice",
                "condition": "xml",
                "hypothesis": " ".join(x["text"] for x in row["oracleSegments"]),
            }
        )
    for row in read_jsonl(ROOT / "evaluation-49/results.jsonl"):
        records.append(
            {
                "recordingId": row["recordingId"],
                "model": "sensevoice",
                "condition": "silero",
                "hypothesis": " ".join(x["textRaw"] for x in row["vadSegments"]),
            }
        )
    for row in read_jsonl(previous / "sensevoice-webrtc-results.jsonl"):
        records.append(
            {
                "recordingId": row["recordingId"],
                "model": "sensevoice",
                "condition": "webrtc",
                "hypothesis": row["hypothesis"],
            }
        )
    return records


def aggregate(rows):
    groups = {}
    for row in rows:
        for subset in ("all49", row["publishedSplit"]):
            key = (row["model"], row["condition"], subset)
            group = groups.setdefault(
                key,
                {
                    "model": key[0],
                    "condition": key[1],
                    "subset": key[2],
                    "count": 0,
                    "errors": 0,
                    "scores": {},
                },
            )
            group["count"] += 1
            group["errors"] += len(row.get("errors", []))
            for rule, score in row["scores"].items():
                total = group["scores"].setdefault(rule, {k: 0 for k in ("N", "S", "D", "I", "C")})
                for field in total:
                    total[field] += score[field]
    for group in groups.values():
        for total in group["scores"].values():
            total["wer"] = (
                (total["S"] + total["D"] + total["I"]) / total["N"] if total["N"] else None
            )
    return list(groups.values())


def audit():
    manifest, references = load_inputs()
    index = published_splits()
    evaluator = load_module("frozen_atco2_evaluator", HERE / "tools/atco2_eval_2024141460331.py")
    normalizers = load_normalizers()
    ref_text = {key: " ".join(x["text"] for x in rows) for key, rows in references.items()}
    joins = []
    for row in manifest:
        rid = row["recordingId"]
        published = index.get(rid)
        if published is None:
            joins.append(
                {
                    "recordingId": rid,
                    "publishedSplit": "unknown",
                    "pcmExactMatch": None,
                    "referenceTokensEqual": None,
                    "rawAuthorNormalizedWords": len(normalizers[1](ref_text[rid]).split()),
                    "safeAuthorNormalizedWords": len(
                        normalizers[1](strip_annotations(ref_text[rid])).split()
                    ),
                }
            )
            continue
        if published["pcmSha256"] != pcm_digest(ROOT / row["audioRelative"]):
            raise ValueError(f"published/local PCM mismatch: {rid}")
        local_tokens = evaluator.tokenize_2024141460331(ref_text[rid], numeric=False)
        old_tokens = evaluator.tokenize_2024141460331(published["text"], numeric=False)
        joins.append(
            {
                "recordingId": rid,
                "publishedSplit": published["split"],
                "pcmExactMatch": True,
                "referenceTokensEqual": local_tokens == old_tokens,
                "localWords": len(local_tokens),
                "publishedWords": len(old_tokens),
                "rawAuthorNormalizedWords": len(normalizers[1](ref_text[rid]).split()),
                "safeAuthorNormalizedWords": len(
                    normalizers[1](strip_annotations(ref_text[rid])).split()
                ),
            }
        )
    rescored = []
    for row in existing_hypotheses():
        row["publishedSplit"] = split_label(index, row["recordingId"])
        row["scores"] = all_scores(
            ref_text[row["recordingId"]], row["hypothesis"], evaluator, normalizers
        )
        rescored.append(row)
    if len(rescored) != 294:
        raise ValueError("expected six complete arms of 49 recordings")
    # Confirm the new independent micro scores reproduce the previously saved six arms.
    old_summary = read_json(ROOT / "model-vad-comparison-20261021/comparison-summary.json")
    totals = aggregate(rescored)
    for group in totals:
        if group["subset"] != "all49":
            continue
        old = old_summary["arms"][group["model"]][group["condition"]]
        for new_rule, old_rule in (("strict", "strict"), ("numeric", "numericNormalized")):
            if abs(group["scores"][new_rule]["wer"] - old[old_rule]["wer"]) > 1e-12:
                raise AssertionError("saved baseline metrics could not be independently reproduced")
    audit_result = {
        "createdUtc": datetime.now(UTC).isoformat(),
        "publishedDatasetRevision": "3b96f5791582cf6f6f348ca46fd502a981918d85",
        "publishedCounts": {
            split: sum(x["split"] == split for x in index.values())
            for split in ("train", "validation")
        },
        "sampleSplitCounts": {
            split: sum(x["publishedSplit"] == split for x in joins)
            for split in ("train", "validation", "unknown")
        },
        "pcmExactMatches": sum(x["pcmExactMatch"] is True for x in joins),
        "referenceDifferences": sum(x["referenceTokensEqual"] is False for x in joins),
        "independentScorerComparisons": len(rescored) * 2,
        "scorerMismatches": 0,
        "normalizationWarning": (
            "authorSafe first removes marker tokens only; original greedy bracket removal "
            "can erase entity content; decimal separators and leading zeros can be lost "
            "in author normalization"
        ),
        "recordings": joins,
    }
    write_json(OUT / "audit.json", audit_result)
    (OUT / "existing-rescored.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rescored), encoding="utf-8"
    )
    write_json(OUT / "existing-rescored-summary.json", totals)
    print(
        json.dumps(
            {k: v for k, v in audit_result.items() if k != "recordings"}, ensure_ascii=False
        ),
        flush=True,
    )


def freeze():
    manifest, _ = load_inputs()
    files = [
        HERE / "tools/atco2_eval_2024141460331.py",
        Path(__file__),
        RESEARCH / "Evaluate/Normalizer.py",
        ROOT / "evaluation-49/manifest.json",
        ROOT / "evaluation-49/references.json",
        ROOT / "evaluation-49/results.jsonl",
        ROOT / "model-vad-comparison-20261021/webrtc-vad.jsonl",
        OUT / "audit.json",
        OUT / "existing-rescored.jsonl",
        OUT / "preprocessing-audit.json",
    ]
    models = {}
    for key, info in MODELS.items():
        local = HERE / "models" / info["id"].split("/")[-1]
        actual = sha256(local / "model.safetensors")
        if actual != info["sha256"]:
            raise ValueError(f"model checksum mismatch: {key}")
        models[key] = {**info, "localPath": str(local)}
    plan = {
        "createdUtc": datetime.now(UTC).isoformat(),
        "models": models,
        "samples": len(manifest),
        "durationSeconds": sum(row["durationMs"] for row in manifest) / 1000,
        "conditions": CONDITIONS,
        "language": "en",
        "task": "transcribe",
        "prompt": None,
        "device": "cuda:0",
        "dtype": "float16",
        "batchSize": 1,
        "numBeams": 1,
        "doSample": False,
        "maxNewTokens": 440,
        "returnTimestamps": False,
        "preprocessing": (
            "soundfile float32, native 16k mono; HF Whisper 128-mel 30s padded features; "
            "explicit no truncation of valid audio"
        ),
        "scoring": (
            "strict primary; conservative numeric; OpenAI; authorSafe sensitivity. "
            "No test-set tuning."
        ),
        "files": {str(path.relative_to(HERE)): sha256(path) for path in files},
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "soundfile",
                "jiwer",
                "openai-whisper",
                "scipy",
                "pyarrow",
            )
        },
    }
    path = OUT / "experiment-plan.json"
    if any(OUT.glob("whisper-*-results.jsonl")):
        raise ValueError("results already exist; cannot replace experiment plan")
    write_json(path, plan)
    print("FROZEN", path, flush=True)


def condition_spans(manifest, references):
    silero = {
        row["recordingId"]: row["vadSegments"]
        for row in read_jsonl(ROOT / "evaluation-49/results.jsonl")
    }
    webrtc = {
        row["recordingId"]: row["vadSegments"]
        for row in read_jsonl(ROOT / "model-vad-comparison-20261021/webrtc-vad.jsonl")
    }
    return {
        row["recordingId"]: {
            "whole": [{"startMs": 0, "endMs": row["durationMs"]}],
            "xml": references[row["recordingId"]],
            "silero": silero[row["recordingId"]],
            "webrtc": webrtc[row["recordingId"]],
        }
        for row in manifest
    }


def run(model_key):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    plan = read_json(OUT / "experiment-plan.json")
    for relative, checksum in plan["files"].items():
        if sha256(HERE / relative) != checksum:
            raise ValueError(f"frozen dependency changed: {relative}")
    model_info = plan["models"][model_key]
    local = Path(model_info["localPath"])
    if sha256(local / "model.safetensors") != model_info["sha256"]:
        raise ValueError("model changed since freeze")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this frozen experiment")
    manifest, references = load_inputs()
    spans = condition_spans(manifest, references)
    splits = {
        row["recordingId"]: row["publishedSplit"]
        for row in read_json(OUT / "audit.json")["recordings"]
    }
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    started = time.perf_counter()
    model = (
        WhisperForConditionalGeneration.from_pretrained(
            local, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    model.generation_config.forced_decoder_ids = None
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    output = OUT / f"whisper-{model_key}-results.jsonl"
    existing = read_jsonl(output) if output.exists() else []
    done = {(row["recordingId"], row["condition"]) for row in existing}
    if len(done) != len(existing):
        raise ValueError("duplicate completed inference rows")
    total_start = time.perf_counter()
    for item in manifest:
        rid = item["recordingId"]
        audio, rate = sf.read(ROOT / item["audioRelative"], dtype="float32")
        if audio.ndim != 1:
            raise ValueError("expected mono")
        validate_audio_span(len(audio), rate)
        for condition in CONDITIONS:
            if (rid, condition) in done:
                continue
            segments, errors = [], []
            for span in spans[rid][condition]:
                a, b = (
                    int(span["startMs"] * rate / 1000),
                    min(len(audio), int(span["endMs"] * rate / 1000)),
                )
                validate_audio_span(b - a, rate)
                start = time.perf_counter()
                try:
                    inputs = processor.feature_extractor(
                        audio[a:b],
                        sampling_rate=rate,
                        return_tensors="pt",
                        return_attention_mask=True,
                        padding="max_length",
                    )
                    with torch.inference_mode():
                        ids = model.generate(
                            inputs.input_features.to("cuda", torch.float16),
                            attention_mask=inputs.attention_mask.to("cuda"),
                            language="en",
                            task="transcribe",
                            num_beams=1,
                            do_sample=False,
                            max_new_tokens=440,
                            return_timestamps=False,
                        )
                    torch.cuda.synchronize()
                    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                    if ids.shape[-1] >= 444:
                        errors.append({"type": "tokenLimitReached", "startMs": span["startMs"]})
                except Exception as exc:
                    text = ""
                    errors.append(
                        {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "startMs": span["startMs"],
                        }
                    )
                segments.append(
                    {
                        "startMs": span["startMs"],
                        "endMs": span["endMs"],
                        "textRaw": text,
                        "elapsedSeconds": time.perf_counter() - start,
                    }
                )
            row = {
                "recordingId": rid,
                "airport": item["airport"],
                "model": model_key,
                "condition": condition,
                "publishedSplit": splits[rid],
                "segments": segments,
                "hypothesis": " ".join(x["textRaw"] for x in segments),
                "errors": errors,
                "asrSeconds": sum(x["elapsedSeconds"] for x in segments),
            }
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{model_key} {rid} completed", flush=True)
    write_json(
        OUT / f"whisper-{model_key}-runtime.json",
        {
            "model": model_info,
            "loadSeconds": load_seconds,
            "runSecondsThisInvocation": time.perf_counter() - total_start,
            "previousRows": len(existing),
            "gpu": torch.cuda.get_device_name(0),
            "peakAllocatedMiB": torch.cuda.max_memory_allocated() / 1024**2,
            "peakReservedMiB": torch.cuda.max_memory_reserved() / 1024**2,
            "planSha256": sha256(OUT / "experiment-plan.json"),
            "completedUtc": datetime.now(UTC).isoformat(),
        },
    )


def summarize():
    _, references = load_inputs()
    evaluator = load_module("frozen_atco2_evaluator", HERE / "tools/atco2_eval_2024141460331.py")
    normalizers = load_normalizers()
    rows = read_jsonl(OUT / "existing-rescored.jsonl")
    for model_key in MODELS:
        current = read_jsonl(OUT / f"whisper-{model_key}-results.jsonl")
        expected = {(rid, condition) for rid in references for condition in CONDITIONS}
        if len(current) != 196 or {(x["recordingId"], x["condition"]) for x in current} != expected:
            raise ValueError(f"incomplete/duplicate Whisper inference: {model_key}")
        for row in current:
            ref = " ".join(x["text"] for x in references[row["recordingId"]])
            row["scores"] = all_scores(ref, row["hypothesis"], evaluator, normalizers)
        rows.extend(current)
    totals = aggregate(rows)
    write_json(OUT / "comparison-summary.json", totals)
    with (OUT / "comparison-summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "condition",
                "subset",
                "count",
                "errors",
                "rule",
                "N",
                "S",
                "D",
                "I",
                "C",
                "wer",
            ],
        )
        writer.writeheader()
        for group in totals:
            for rule, score in group["scores"].items():
                writer.writerow(
                    {**{k: v for k, v in group.items() if k != "scores"}, "rule": rule, **score}
                )
    print(json.dumps([x for x in totals if x["subset"] == "all49"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "freeze", "run", "summarize"))
    parser.add_argument("--model", choices=tuple(MODELS), default="atcosim")
    args = parser.parse_args()
    if args.stage == "run":
        run(args.model)
    else:
        {"audit": audit, "freeze": freeze, "summarize": summarize}[args.stage]()
