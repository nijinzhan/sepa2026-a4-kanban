"""Small offline course-project ASR check using pinned TeamBlue source only.

No database, HTTP services, training, accuracy threshold, or human reference required.
Human listening is explicitly separate from automatic execution checks.
Only the existing C15 ATC source is included. The user-provided song is excluded.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF = "82ef964905383322cf94816074873700e05dc5d5"


def digest_2024141460331(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def json_write_2024141460331(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def repetition_notes_2024141460331(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    seen, repeated = set(), []
    for pos in range(max(0, len(words) - 7)):
        phrase = " ".join(words[pos : pos + 8])
        if phrase in seen and phrase not in repeated:
            repeated.append(phrase)
        seen.add(phrase)
    # Diagnostic hint only: radio readbacks can legitimately repeat words.
    return repeated[:3]


def main_2024141460331():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/vhhh-light-20261107-82ef9649-c15-only"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError("Use a new output directory to preserve the previous run")
    out.mkdir(parents=True)
    source_paths = [
        "李洪/src/a4_speech/__init__.py",
        "李洪/src/a4_speech/audio_2024141460331.py",
        "李洪/src/a4_speech/models_2024141460331.py",
        "李洪/src/a4_speech/vad_webrtc_2024141460331.py",
        "李洪/config.teacher-sensevoice.example.json",
    ]
    snapshot = out / "team-source"
    source_entries = []
    for relative in source_paths:
        data = subprocess.check_output(["git", "show", f"{args.ref}:{relative}"], cwd=ROOT.parent)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_entries.append({"path": relative, "sha256": digest_2024141460331(target)})
    json_write_2024141460331(
        out / "source-manifest.json", {"ref": args.ref, "files": source_entries}
    )
    sys.path.insert(0, str(snapshot / "李洪/src"))
    import numpy as np
    import soundfile as sf

    from a4_speech.audio_2024141460331 import Audio, decode_audio
    from a4_speech.models_2024141460331 import SherpaEngine

    engine_module = sys.modules["a4_speech.models_2024141460331"]
    assert Path(engine_module.__file__).resolve().is_relative_to(snapshot)
    config = json.loads(
        (snapshot / "李洪/config.teacher-sensevoice.example.json").read_text(encoding="utf-8")
    )
    model_config = deepcopy(config["engine"])
    for kind in model_config.values():
        for key in ("modelPath", "tokensPath"):
            if key in kind:
                kind[key] = str((ROOT / kind[key]).resolve())
    engine = SherpaEngine(model_config)
    execution = engine.snapshot()
    assert (
        execution["asr"]["modelSha256"]
        == "625facec8fb54410d0e34543947cd80c2e0b1e568df00395bd9a31d3d00f23e8"
    )
    assert (
        execution["asr"]["tokensSha256"]
        == "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"
    )
    json_write_2024141460331(out / "execution-snapshot.json", execution)
    inputs = {
        "C15": ROOT / "outputs/vhhh-c15-20261023/C15.mp3",
    }
    decoded, input_records = {}, []
    for code, path in inputs.items():
        data = path.read_bytes()
        decoded[code] = decode_audio(data)
        input_records.append(
            {
                "id": code,
                "fileName": path.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "decodedDurationMs": decoded[code].duration_ms,
                "audioProcessing": decoded[code].processing_info,
                "provenance": "Existing C15 public VHHH material record",
                "originalCaptureUtcKnown": False,
            }
        )
    # Keep the original three C15 windows; exclusion is based on source type, not ASR output.
    windows = [
        ("C15-01", "C15", 0, 20000),
        ("C15-02", "C15", 240000, 260000),
        ("C15-03", "C15", 480000, 500000),
    ]
    selection = [
        {"id": sid, "source": code, "startMs": start, "endMs": end}
        for sid, code, start, end in windows
    ]
    json_write_2024141460331(
        out / "selection-before-inference.json", {"selection": selection, "inputs": input_records}
    )
    (out / "clips").mkdir()
    results = []
    load_started = time.perf_counter()
    engine.prepare()
    load_seconds = time.perf_counter() - load_started
    for sid, code, start, end in windows:
        whole = decoded[code]
        assert 0 <= start < end <= whole.duration_ms
        clip = Audio(whole.samples[start * 16 : end * 16], 16000)
        clip_path = out / "clips" / f"{sid}.wav"
        sf.write(clip_path, clip.samples, 16000, subtype="PCM_16")
        # Keep an optional small listening copy; never replace the original model input.
        try:
            import imageio_ffmpeg

            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            encoder = shutil.which("ffmpeg")
        mp3 = clip_path.with_suffix(".mp3") if encoder else None
        if encoder:
            subprocess.run(
                [
                    encoder,
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(clip_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-b:a",
                    "32k",
                    str(mp3),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        started = time.perf_counter()
        spans = engine.detect(clip)
        segments = []
        for begin, finish in spans:
            assert 0 <= begin < finish <= clip.duration_ms
            entry = {
                "startMsInClip": begin,
                "endMsInClip": finish,
                "startMsInSource": start + begin,
                "endMsInSource": start + finish,
            }
            try:
                entry["textRaw"] = engine.recognize(clip, begin, finish)
                entry["status"] = "succeeded"
            except ValueError as exc:
                entry.update(status="failed", textRaw=None, error=str(exc))
            segments.append(entry)
        elapsed = time.perf_counter() - started
        texts = [s["textRaw"] for s in segments if s["status"] == "succeeded"]
        combined = "\n".join(texts)
        item = {
            "id": sid,
            "source": code,
            "sourceStartMs": start,
            "sourceEndMs": end,
            "durationMs": clip.duration_ms,
            "wav": str(clip_path.relative_to(out)),
            "listeningMp3": str(mp3.relative_to(out)) if mp3 else None,
            "wavSha256": digest_2024141460331(clip_path),
            "status": "no_speech"
            if not spans
            else "succeeded"
            if len(texts) == len(spans)
            else "partial"
            if texts
            else "failed",
            "segmentCount": len(spans),
            "successfulSegments": len(texts),
            "textRaw": combined,
            "segments": segments,
            "elapsedSeconds": round(elapsed, 3),
            "relativeTimeBoundsValid": True,
            "utf8JsonRoundTrip": json.loads(json.dumps(combined, ensure_ascii=False)) == combined,
            "automaticRepetitionHints": repetition_notes_2024141460331(combined),
            "humanListeningStatus": "not_performed",
            "contentCorrespondence": "not_independently_confirmed",
            "callSignAndNumberCorrectness": "not_independently_confirmed",
        }
        results.append(item)
        json_write_2024141460331(out / f"{sid}.json", item)
        print(
            json.dumps(
                {"id": sid, "status": item["status"], "segments": len(spans), "textRaw": combined},
                ensure_ascii=False,
            ),
            flush=True,
        )
    for entry in source_entries:
        assert digest_2024141460331(snapshot / entry["path"]) == entry["sha256"]
    summary = {
        "recordedUtc": datetime.now(UTC).isoformat(),
        "teamRef": args.ref,
        "scope": "offline_small_sample_execution_and_review_materials",
        "databaseUsed": False,
        "httpServiceStarted": False,
        "pBlueUsed": False,
        "model": execution,
        "sourceFiles": source_entries,
        "inputs": input_records,
        "sampleCount": len(results),
        "totalSampleSeconds": sum(s[3] - s[2] for s in windows) / 1000,
        "selectionFixedBeforeInference": True,
        "sampleResults": results,
        "modelLoadSeconds": round(load_seconds, 3),
        "machineExecutionSucceededCount": sum(x["status"] == "succeeded" for x in results),
        "humanListeningPerformed": False,
        "accuracyThreshold": None,
        "werCerCalculated": False,
        "manualFullTranscriptRequired": False,
        "qualitativeAcceptance": "human_content_correspondence_unconfirmed",
        "shortWindowsDoNotValidateLongRecordingSegmentation": True,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "sherpaOnnx": importlib.metadata.version("sherpa-onnx"),
            "webrtcVad": importlib.metadata.version("webrtcvad-wheels"),
            "numpy": np.__version__,
        },
    }
    json_write_2024141460331(out / "summary.json", summary)
    with (out / "sample-records.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "id",
            "source",
            "sourceStartMs",
            "sourceEndMs",
            "status",
            "segmentCount",
            "textRaw",
            "relativeTimeBoundsValid",
            "elapsedSeconds",
            "humanListeningStatus",
            "contentCorrespondence",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(
        json.dumps(
            {
                "summary": str(out / "summary.json"),
                "sampleCount": len(results),
                "machineExecutionSucceededCount": summary["machineExecutionSucceededCount"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main_2024141460331()
