"""Offline ATCO2 research baseline; source recordings stay in ignored outputs.

The primary WER only strips annotation markers, case and punctuation. A second,
explicitly custom normalization expands numeric writing and spoken cardinals.
Neither score is claimed comparable to a paper using a different normalizer.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

DIGITS = "zero one two three four five six seven eight nine".split()
SMALL = dict(zip(DIGITS, range(10), strict=True))
SMALL.update(
    dict(
        zip(
            (
                "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
            ).split(),
            range(10, 20),
            strict=True,
        )
    )
)
TENS = dict(
    zip(
        "twenty thirty forty fifty sixty seventy eighty ninety".split(),
        range(20, 100, 10),
        strict=True,
    )
)
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
NUMBER_WORDS = set(SMALL) | set(TENS) | set(SCALES)
UNIT_PATTERN = "(?:" + "|".join(DIGITS[1:]) + ")"
UNDER_100_PATTERN = (
    "(?:"
    + "|".join(
        [
            "zero",
            UNIT_PATTERN,
            *list(SMALL)[10:],
            "(?:" + "|".join(TENS) + ")(?: " + UNIT_PATTERN + ")?",
        ]
    )
    + ")"
)
UNDER_1000_PATTERN = (
    "(?:"
    + UNIT_PATTERN
    + " hundred(?: (?:and )?"
    + UNDER_100_PATTERN
    + ")?|"
    + UNDER_100_PATTERN
    + ")"
)
CARDINAL_PATTERN = re.compile(
    UNDER_1000_PATTERN + "(?: thousand(?: (?:and )?" + UNDER_1000_PATTERN + ")?)?"
)


def tokenize_2024141460331(text, numeric=False):
    text = re.sub(r"\[[^\]]*\]", " ", unicodedata.normalize("NFKC", text).lower())
    # Keep decimals and split an alphanumeric callsign into letters and digits.
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?", text)
    if not numeric:
        return words
    words = ["nine" if x == "niner" else "point" if x == "decimal" else x for x in words]
    result = []
    i = 0
    while i < len(words):
        word = words[i]
        if word[0].isdigit():
            result.extend("point" if c == "." else DIGITS[int(c)] for c in word)
            i += 1
        elif word in NUMBER_WORDS:
            run = [word]
            j = i + 1
            while j < len(words):
                if words[j] in NUMBER_WORDS:
                    run.append(words[j])
                elif words[j] == "and" and j + 1 < len(words) and words[j + 1] in NUMBER_WORDS:
                    run.append("and")
                else:
                    break
                j += 1
            if all(x in DIGITS for x in run):
                result.extend(run)
            elif CARDINAL_PATTERN.fullmatch(" ".join(run)):
                # Conservative cardinal blocks; adjacent digit words remain separate.
                current = 0
                total = 0
                for x in run:
                    if x == "and":
                        continue
                    if x in SMALL or x in TENS:
                        current += SMALL.get(x, TENS.get(x, 0))
                    elif x == "hundred":
                        current = (current or 1) * 100
                    else:
                        total += (current or 1) * SCALES[x]
                        current = 0
                result.extend(DIGITS[int(c)] for c in str(total + current))
            else:
                # Ambiguous ATC sequences must not silently become a different number.
                result.extend(run)
            i = j
        else:
            result.append(word)
            i += 1
    return result


def score_words_2024141460331(reference, hypothesis):
    """Unit-cost Levenshtein; ties prefer equal/substitution, then deletion."""
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        costs[i][0] = i
    for j in range(m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i][j] = min(
                costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
            )
    operations = []
    i, j = n, m
    while i or j:
        if (
            i
            and j
            and costs[i][j] == (costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]))
        ):
            operations.append(
                {
                    "op": "C" if reference[i - 1] == hypothesis[j - 1] else "S",
                    "r": i - 1,
                    "h": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            operations.append({"op": "D", "r": i - 1, "h": None})
            i -= 1
        else:
            operations.append({"op": "I", "r": None, "h": j - 1})
            j -= 1
    operations.reverse()
    counts = Counter(x["op"] for x in operations)
    return {
        "N": n,
        **{k: counts[k] for k in ("S", "D", "I", "C")},
        "wer": costs[n][m] / n if n else None,
        "operations": operations,
    }


def span_retained_2024141460331(start, end, operations):
    matches = [x for x in operations if x["r"] is not None and start <= x["r"] < end]
    if not matches or len(matches) != end - start or any(x["op"] != "C" for x in matches):
        return False
    indices = [x["h"] for x in matches]
    return indices == list(range(indices[0], indices[0] + len(indices)))


def merge_regions_2024141460331(regions, gap_ms=0):
    result = []
    for start, end in sorted(regions):
        if end <= start:
            raise ValueError("empty or reversed time region")
        if result and start - result[-1][1] <= gap_ms:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def boundary_score_2024141460331(reference, hypothesis, gap_ms=500, min_iou=0.5):
    raw_ref = merge_regions_2024141460331(reference)
    raw_hyp = merge_regions_2024141460331(hypothesis)
    reference = merge_regions_2024141460331(reference, gap_ms)
    overlap = sum(max(0, min(b, d) - max(a, c)) for a, b in raw_ref for c, d in raw_hyp)
    weights = []
    for a, b in reference:
        row = []
        for c, d in hypothesis:
            intersection = max(0, min(b, d) - max(a, c))
            iou = intersection / (b - a + d - c - intersection)
            # Maximize valid match count first, then total IoU.
            row.append(10000 + iou if iou >= min_iou else 0)
        weights.append(row)
    matches = []
    if reference and hypothesis:
        rr, hh = linear_sum_assignment(weights, maximize=True)
        for r, h in zip(rr, hh, strict=True):
            if weights[r][h]:
                matches.append(
                    {
                        "reference": int(r),
                        "hypothesis": int(h),
                        "iou": weights[r][h] - 10000,
                        "start_error_ms": hypothesis[h][0] - reference[r][0],
                        "end_error_ms": hypothesis[h][1] - reference[r][1],
                    }
                )
    return {
        "reference_regions": len(reference),
        "predicted_regions": len(hypothesis),
        "matched_regions": len(matches),
        "missed_regions": len(reference) - len(matches),
        "extra_regions": len(hypothesis) - len(matches),
        "matches": matches,
        "start_abs_errors_ms": [abs(x["start_error_ms"]) for x in matches],
        "end_abs_errors_ms": [abs(x["end_error_ms"]) for x in matches],
        "reference_speech_ms": sum(b - a for a, b in raw_ref),
        "predicted_speech_ms": sum(b - a for a, b in raw_hyp),
        "overlap_ms": overlap,
    }


def reference_tokens_2024141460331(segments):
    tokens = []
    callsigns = []
    for segment in segments:
        text = segment["text"]
        offset = 0
        for match in re.finditer(r"\[#callsign\](.*?)\[/#callsign\]", text, re.I | re.S):
            tokens.extend(tokenize_2024141460331(text[offset : match.start()], numeric=True))
            start = len(tokens)
            tokens.extend(tokenize_2024141460331(match.group(1), numeric=True))
            if len(tokens) > start:
                callsigns.append(
                    {
                        "start": start,
                        "end": len(tokens),
                        "text": match.group(1),
                        "correct_tagging": segment["correct_tagging"],
                    }
                )
            offset = match.end()
        tokens.extend(tokenize_2024141460331(text[offset:], numeric=True))
    numbers = []
    i = 0
    while i < len(tokens):
        if tokens[i] not in DIGITS:
            i += 1
            continue
        start = i
        i += 1
        while i < len(tokens) and (
            tokens[i] in DIGITS
            or (tokens[i] == "point" and i + 1 < len(tokens) and tokens[i + 1] in DIGITS)
        ):
            i += 1
        numbers.append({"start": start, "end": i, "text": " ".join(tokens[start:i])})
    return {"tokens": tokens, "callsigns": callsigns, "numbers": numbers}


def sha256_2024141460331(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json_2024141460331(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_2024141460331(args):
    """Freeze the sample before any recognizer is invoked."""
    root = args.root.resolve()
    target = root / "evaluation-49"
    target.mkdir(exist_ok=True)
    if (target / "plan.json").exists():
        raise ValueError("plan already exists; use a separate data directory for a new selection")
    audio_dir = root / "original" / "ATCO2-ASRdataset-v1_beta" / "DATA"
    candidates = []
    excluded = []
    inventory = Counter()
    airports = Counter()
    eligible_airports = Counter()
    references = {}
    for xml in sorted((root / "verified-xml").glob("*.xml")):
        record_id = xml.stem
        airport = record_id.split("_")[0]
        audio = audio_dir / (record_id + ".wav")
        info = sf.info(audio)
        segments = []
        reasons = set()
        for s in ET.parse(xml).getroot().findall("segment"):
            segment = {
                "startMs": round(float(s.findtext("start")) * 1000),
                "endMs": round(float(s.findtext("end")) * 1000),
                "text": s.findtext("text") or "",
                "speaker": s.findtext("speaker"),
                **{
                    k: s.findtext("tags/" + k)
                    for k in ("correct_transcript", "correct_tagging", "non_english")
                },
            }
            inventory["segments"] += 1
            inventory["tagging_flag_" + str(segment["correct_tagging"])] += 1
            if segment["correct_transcript"] != "1":
                reasons.add("correct_transcript_not_1")
            if segment["non_english"] != "0" or re.search(
                r"\[\s*(?:ne(?:\b|-)|#nonenglish)", segment["text"], re.I
            ):
                reasons.add("mixed_or_non_english")
            if not 0 <= segment["startMs"] < segment["endMs"] <= math.ceil(info.duration * 1000):
                reasons.add("reference_boundary_outside_audio")
            segments.append(segment)
        if not tokenize_2024141460331(" ".join(s["text"] for s in segments)):
            reasons.add("no_reference_words")
        inventory["files"] += 1
        inventory["durationMs"] += round(info.duration * 1000)
        airports[airport] += 1
        if reasons:
            excluded.append({"recordingId": record_id, "reasons": sorted(reasons)})
            continue
        eligible_airports[airport] += 1
        candidate = {
            "recordingId": record_id,
            "airport": airport,
            "audioRelative": audio.relative_to(root).as_posix(),
            "xmlRelative": xml.relative_to(root).as_posix(),
            "audioSha256": sha256_2024141460331(audio),
            "xmlSha256": sha256_2024141460331(xml),
            "durationMs": round(info.duration * 1000),
            "sampleRate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "referenceSegments": len(segments),
            "selectionHash": hashlib.sha256(("a4-atco2-v1|" + record_id).encode()).hexdigest(),
        }
        candidates.append(candidate)
        references[record_id] = segments
    manifest = []
    for airport in sorted(eligible_airports):
        group = sorted(
            (x for x in candidates if x["airport"] == airport),
            key=lambda x: (x["selectionHash"], x["recordingId"]),
        )
        if len(group) < args.per_airport:
            raise ValueError("insufficient eligible recordings for equal airport sampling")
        manifest.extend(group[: args.per_airport])
    if not manifest:
        raise ValueError("no paired data found")
    write_json_2024141460331(target / "manifest.json", manifest)
    with (target / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    write_json_2024141460331(
        target / "references.json",
        {x["recordingId"]: references[x["recordingId"]] for x in manifest},
    )
    write_json_2024141460331(
        target / "inventory.json",
        {
            **inventory,
            "allAirports": airports,
            "eligibleAirports": eligible_airports,
            "eligibleFiles": len(candidates),
            "excluded": excluded,
        },
    )
    from a4_speech.cli_2024141460331 import (
        _engine_2024141460331,
        load_config_2024141460331,
    )

    config = load_config_2024141460331(args.config)
    engine = _engine_2024141460331(config)
    archives = []
    for filename, url in (
        (
            "ATCO2-ASRdataset-v1_beta.tgz",
            "https://www.replaywell.com/atco2/download/ATCO2-ASRdataset-v1_beta.tgz",
        ),
        ("atco2_1h_xml-tar.gz", "https://atco2.org/data/atco2_1h_xml-tar.gz"),
    ):
        path = root / filename
        archives.append(
            {
                "filename": filename,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256_2024141460331(path),
            }
        )
    plan = {
        "frozenUtc": datetime.now(UTC).isoformat(),
        "purpose": "public ATC baseline; NOT VHHH",
        "selection": "eligible English files; 7 airports; first k ascending SHA256(a4-atco2-v1|id)",
        "perAirport": args.per_airport,
        "recordings": len(manifest),
        "audioSeconds": sum(x["durationMs"] for x in manifest) / 1000,
        "manifestSha256": sha256_2024141460331(target / "manifest.json"),
        "referencesSha256": sha256_2024141460331(target / "references.json"),
        "executionSnapshot": engine.snapshot(),
        "archives": archives,
        "license": (
            "ATCO2 package End-User Agreement; internal language-engineering research; "
            "no public redistribution of substantial data"
        ),
        "sourcePage": "https://www.atco2.org/data",
        "primaryMetric": (
            "micro WER=(S+D+I)/N; lowercase; strip bracket markers and "
            "punctuation; keep numeric spelling"
        ),
        "secondaryMetric": (
            "custom conservative English-cardinal/digit normalization, "
            "niner=nine, decimal=point; not a standard ATCO2 score"
        ),
        "entities": (
            "aligned span exact retention after numeric normalization; XML callsign "
            "tags diagnostic if correct_tagging=0; numbers derived from reference words"
        ),
        "boundary": (
            "XML segments with reference words; union overlaps; merge gaps<=500ms "
            "for region matching; one-to-one IoU>=0.5; duration coverage uses "
            "unfilled original union"
        ),
        "failures": (
            "keep every frozen recording; missing output scores deletions; "
            "no exclusion after inference"
        ),
        "timing": (
            "one shared ASR recognizer; startup separated; each file includes decode, "
            "VAD, ASR and adapter model-hash checks; no API/DB calls"
        ),
        "evaluationScriptSha256AtPreparation": sha256_2024141460331(__file__),
    }
    write_json_2024141460331(target / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": len(manifest),
                "seconds": plan["audioSeconds"],
                "eligible": len(candidates),
                "excluded": len(excluded),
                "target": str(target),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def summarize_2024141460331(target):
    records = [
        json.loads(line)
        for line in (target / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if len({r["recordingId"] for r in records}) != len(records):
        raise ValueError("duplicate result IDs")
    expected = {x["recordingId"] for x in manifest}
    if {r["recordingId"] for r in records} != expected:
        raise ValueError("incomplete or unexpected results; refusing a complete report")
    summary = {
        "recordings": len(records),
        "successfulRecordings": sum(not r["errors"] for r in records),
        "failedRecordings": sum(bool(r["errors"]) for r in records),
        "noVadRecordings": sum(not r["vadSegments"] for r in records),
        "audioSeconds": sum(r["durationMs"] for r in records) / 1000,
        "processingSeconds": sum(r["processingSeconds"] for r in records),
        "vadSeconds": sum(r["vadSeconds"] for r in records),
        "asrSeconds": sum(r["asrSeconds"] for r in records),
        "referenceTurns": sum(r["referenceTurns"] for r in records),
        "vadSegments": sum(len(r["vadSegments"]) for r in records),
    }
    summary["rtf"] = summary["processingSeconds"] / summary["audioSeconds"]
    for kind in ("strict", "numericNormalized"):
        counts = {k: sum(r[kind][k] for r in records) for k in ("N", "S", "D", "I", "C")}
        summary[kind] = {**counts, "wer": (counts["S"] + counts["D"] + counts["I"]) / counts["N"]}
    for kind in ("callsigns", "numbers"):
        spans = [x for r in records for x in r[kind]]
        summary[kind] = {
            "total": len(spans),
            "retained": sum(x["retained"] for x in spans),
            "notRetained": sum(not x["retained"] for x in spans),
            "retentionRate": sum(x["retained"] for x in spans) / len(spans) if spans else None,
        }
        if kind == "callsigns":
            summary[kind]["taggingFlags"] = dict(Counter(x["correct_tagging"] for x in spans))
    boundary = {}
    for k in (
        "reference_regions",
        "predicted_regions",
        "matched_regions",
        "missed_regions",
        "extra_regions",
        "reference_speech_ms",
        "predicted_speech_ms",
        "overlap_ms",
    ):
        boundary[k] = sum(r["boundary"][k] for r in records)
    for kind in ("start", "end"):
        errors = [x for r in records for x in r["boundary"][kind + "_abs_errors_ms"]]
        boundary[kind + "AbsErrorMs"] = {
            "count": len(errors),
            "mean": float(np.mean(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
            "p95": float(np.percentile(errors, 95)) if errors else None,
        }
    boundary["durationRecall"] = boundary["overlap_ms"] / boundary["reference_speech_ms"]
    boundary["durationPrecision"] = (
        boundary["overlap_ms"] / boundary["predicted_speech_ms"]
        if boundary["predicted_speech_ms"]
        else 0
    )
    summary["boundary"] = boundary
    summary["airports"] = {}
    for airport in sorted({r["airport"] for r in records}):
        group = [r for r in records if r["airport"] == airport]
        item = {
            "recordings": len(group),
            "audioSeconds": sum(r["durationMs"] for r in group) / 1000,
        }
        for kind in ("strict", "numericNormalized"):
            count = {k: sum(r[kind][k] for r in group) for k in ("N", "S", "D", "I")}
            item[kind] = {**count, "wer": (count["S"] + count["D"] + count["I"]) / count["N"]}
        summary["airports"][airport] = item
    summary["files"] = {
        name: sha256_2024141460331(target / name)
        for name in (
            "plan.json",
            "manifest.json",
            "references.json",
            "inventory.json",
            "runtime.json",
            "results.jsonl",
        )
    }
    write_json_2024141460331(target / "summary.json", summary)
    fields = [
        "recordingId",
        "airport",
        "durationSeconds",
        "strictWER",
        "numericNormalizedWER",
        "S",
        "D",
        "I",
        "N",
        "callsignsTotal",
        "callsignsRetained",
        "numbersTotal",
        "numbersRetained",
        "vadSegments",
        "matchedRegions",
        "missedRegions",
        "extraRegions",
        "processingSeconds",
        "rtf",
        "errors",
    ]
    with (target / "per-recording.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "recordingId": r["recordingId"],
                    "airport": r["airport"],
                    "durationSeconds": r["durationMs"] / 1000,
                    "strictWER": r["strict"]["wer"],
                    "numericNormalizedWER": r["numericNormalized"]["wer"],
                    **{k: r["strict"][k] for k in ("S", "D", "I", "N")},
                    **{kind + "Total": len(r[kind]) for kind in ("callsigns", "numbers")},
                    **{
                        kind + "Retained": sum(x["retained"] for x in r[kind])
                        for kind in ("callsigns", "numbers")
                    },
                    "vadSegments": len(r["vadSegments"]),
                    "matchedRegions": r["boundary"]["matched_regions"],
                    "missedRegions": r["boundary"]["missed_regions"],
                    "extraRegions": r["boundary"]["extra_regions"],
                    "processingSeconds": r["processingSeconds"],
                    "rtf": r["processingSeconds"] / (r["durationMs"] / 1000),
                    "errors": json.dumps(r["errors"], ensure_ascii=False),
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run_2024141460331(args):
    from a4_speech.audio_2024141460331 import decode_audio
    from a4_speech.cli_2024141460331 import (
        _engine_2024141460331,
        load_config_2024141460331,
    )

    root = args.root.resolve()
    target = root / "evaluation-49"
    plan = json.loads((target / "plan.json").read_text(encoding="utf-8"))
    if (target / "results.jsonl").exists():
        raise ValueError("results already exist; summarize or use a separate frozen run")
    for name, key in (("manifest.json", "manifestSha256"), ("references.json", "referencesSha256")):
        if sha256_2024141460331(target / name) != plan[key]:
            raise ValueError("frozen evaluation data changed")
    if sha256_2024141460331(__file__) != plan["evaluationScriptSha256AtPreparation"]:
        raise ValueError("evaluation script changed after sample preparation")
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    references = json.loads((target / "references.json").read_text(encoding="utf-8"))
    for item in manifest:
        for kind in ("audio", "xml"):
            if sha256_2024141460331(root / item[kind + "Relative"]) != item[kind + "Sha256"]:
                raise ValueError("frozen source file changed: " + item["recordingId"])
    started = datetime.now(UTC).isoformat()
    clock = time.perf_counter()
    engine = _engine_2024141460331(load_config_2024141460331(args.config))
    if engine.snapshot() != plan["executionSnapshot"]:
        raise ValueError("engine differs from frozen plan")
    engine.prepare()
    runtime = {
        "startedUtc": started,
        "initializationSeconds": time.perf_counter() - clock,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            p: importlib.metadata.version(p) for p in ("sherpa-onnx", "numpy", "scipy", "soundfile")
        },
        "sourceGitHead": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "evaluationScriptSha256": sha256_2024141460331(__file__),
        "executionSnapshot": engine.snapshot(),
    }
    write_json_2024141460331(target / "runtime.json", runtime)
    with (target / "results.jsonl").open("w", encoding="utf-8") as output:
        for index, item in enumerate(manifest, 1):
            record_id = item["recordingId"]
            reference = references[record_id]
            errors = []
            spans = []
            recognized = []
            vad_seconds = asr_seconds = 0.0
            clock = time.perf_counter()
            try:
                audio = decode_audio((root / item["audioRelative"]).read_bytes())
                before_vad = time.perf_counter()
                spans = engine.detect(audio)
                vad_seconds = time.perf_counter() - before_vad
                for start, end in spans:
                    before_asr = time.perf_counter()
                    try:
                        text = engine.recognize(audio, start, end)
                    except Exception as exc:
                        text = ""
                        errors.append(
                            {
                                "stage": "asr",
                                "startMs": start,
                                "endMs": end,
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                    elapsed = time.perf_counter() - before_asr
                    asr_seconds += elapsed
                    recognized.append(
                        {"startMs": start, "endMs": end, "textRaw": text, "elapsedSeconds": elapsed}
                    )
            except Exception as exc:
                errors.append(
                    {"stage": "decode_or_vad", "type": type(exc).__name__, "message": str(exc)}
                )
            processing_seconds = time.perf_counter() - clock
            hypothesis = " ".join(x["textRaw"] for x in recognized)
            ref_text = " ".join(x["text"] for x in reference)
            strict = score_words_2024141460331(
                tokenize_2024141460331(ref_text), tokenize_2024141460331(hypothesis)
            )
            ref = reference_tokens_2024141460331(reference)
            hyp = tokenize_2024141460331(hypothesis, numeric=True)
            normalized = score_words_2024141460331(ref["tokens"], hyp)
            for kind in ("callsigns", "numbers"):
                for span in ref[kind]:
                    span["retained"] = span_retained_2024141460331(
                        span["start"], span["end"], normalized["operations"]
                    )
            ref_regions = [
                (s["startMs"], s["endMs"]) for s in reference if tokenize_2024141460331(s["text"])
            ]
            result = {
                "recordingId": record_id,
                "airport": item["airport"],
                "durationMs": item["durationMs"],
                "referenceTurns": len(reference),
                "vadSegments": recognized,
                "referenceTokens": ref["tokens"],
                "hypothesisTokens": hyp,
                "strict": strict,
                "numericNormalized": normalized,
                "callsigns": ref["callsigns"],
                "numbers": ref["numbers"],
                "boundary": boundary_score_2024141460331(ref_regions, spans),
                "vadSeconds": vad_seconds,
                "asrSeconds": asr_seconds,
                "processingSeconds": processing_seconds,
                "errors": errors,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(manifest)}",
                        "recordingId": record_id,
                        "seconds": round(processing_seconds, 3),
                        "WER": round(strict["wer"], 4),
                        "vadSegments": len(spans),
                        "errors": len(errors),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    runtime["finishedUtc"] = datetime.now(UTC).isoformat()
    write_json_2024141460331(target / "runtime.json", runtime)
    summarize_2024141460331(target)


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, default=Path("outputs/atco2-1h-20261021"))
    parser.add_argument("--config", type=Path, default=Path("config.local.json"))
    parser.add_argument("--per-airport", type=int, default=7)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_2024141460331(args)
    elif args.command == "run":
        run_2024141460331(args)
    else:
        summarize_2024141460331(args.root.resolve() / "evaluation-49")


if __name__ == "__main__":
    main_2024141460331()
