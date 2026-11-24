"""Profile real local CPU ASR on a long audio file without storing transcripts."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any

PERSONAL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PERSONAL_ROOT.parent
DEFAULT_AUDIO = Path("outputs/v24-export-20261121-confirmed/inbox/source120.mp3")
DEFAULT_CONFIG = Path("config.teacher-sensevoice.local.json")
DEFAULT_OUTPUT = Path("outputs/long-audio-asr-20261121/summary.json")
PROFILED_SOURCES = (
    "src/a4_speech/audio_2024141460331.py",
    "src/a4_speech/cli_2024141460331.py",
    "src/a4_speech/models_2024141460331.py",
    "src/a4_speech/vad_webrtc_2024141460331.py",
)


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
_KERNEL32.GetCurrentProcess.argtypes = []
_KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
_PSAPI.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ProcessMemoryCountersEx),
    wintypes.DWORD,
]
_PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def _memory_snapshot() -> dict[str, float]:
    if os.name != "nt":
        raise RuntimeError("this profiler records Windows process memory only")
    process = _KERNEL32.GetCurrentProcess()
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    ok = _PSAPI.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    mib = 1024 * 1024
    return {
        "workingSetMiB": counters.WorkingSetSize / mib,
        "peakWorkingSetMiB": counters.PeakWorkingSetSize / mib,
        "privateUsageMiB": counters.PrivateUsage / mib,
        "peakPagefileMiB": counters.PeakPagefileUsage / mib,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_personal_path(value: Path) -> Path:
    return value if value.is_absolute() else PERSONAL_ROOT / value


def _failure_reason(exc: Exception) -> str:
    message = str(exc)
    if message == "ASR model returned empty or blank text":
        return "empty_or_blank_text"
    if message.startswith("ASR model inference failed"):
        return "model_inference_error"
    return type(exc).__name__


def _git_fingerprint() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    sources = {}
    for relative in PROFILED_SOURCES:
        path = PERSONAL_ROOT / relative
        status = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain", "--", str(path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        sources[relative] = {"sha256": _sha256(path), "gitStatus": status or "clean"}
    return {"head": head, "sourceFiles": sources}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    total_started = time.perf_counter()
    memory = {"baseline": _memory_snapshot()}

    audio_path = _resolve_personal_path(args.audio).resolve()
    config_path = _resolve_personal_path(args.config).resolve()
    input_started = time.perf_counter()
    encoded_audio = audio_path.read_bytes()
    input_read_seconds = time.perf_counter() - input_started
    memory["afterInputRead"] = _memory_snapshot()

    sys.path.insert(0, str(PERSONAL_ROOT / "src"))
    from a4_speech.audio_2024141460331 import decode_audio
    from a4_speech.cli_2024141460331 import (
        _engine_2024141460331,
        load_config_2024141460331,
    )

    decode_started = time.perf_counter()
    audio = decode_audio(encoded_audio)
    decode_seconds = time.perf_counter() - decode_started
    memory["afterDecode"] = _memory_snapshot()

    config_started = time.perf_counter()
    config = load_config_2024141460331(config_path)
    engine = _engine_2024141460331(config)
    model_snapshot = engine.snapshot()
    engine_setup_seconds = time.perf_counter() - config_started
    memory["afterEngineSetup"] = _memory_snapshot()

    prepare_started = time.perf_counter()
    engine.prepare()
    model_prepare_seconds = time.perf_counter() - prepare_started
    memory["afterModelPrepare"] = _memory_snapshot()

    vad_started = time.perf_counter()
    spans = engine.detect(audio)
    vad_seconds = time.perf_counter() - vad_started
    memory["afterVad"] = _memory_snapshot()

    segment_results = []
    asr_seconds = 0.0
    for index, (start_ms, end_ms) in enumerate(spans, start=1):
        started = time.perf_counter()
        try:
            text = engine.recognize(audio, start_ms, end_ms)
            elapsed = time.perf_counter() - started
            asr_seconds += elapsed
            segment_results.append(
                {
                    "index": index,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "durationMs": end_ms - start_ms,
                    "status": "success",
                    "textCharacters": len(text.strip()),
                    "textSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "elapsedSeconds": elapsed,
                }
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            asr_seconds += elapsed
            segment_results.append(
                {
                    "index": index,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "durationMs": end_ms - start_ms,
                    "status": "failure",
                    "errorType": type(exc).__name__,
                    "failureReason": _failure_reason(exc),
                    "elapsedSeconds": elapsed,
                }
            )
        memory[f"afterSegment{index}"] = _memory_snapshot()

    final_memory = _memory_snapshot()
    memory["final"] = final_memory
    succeeded = sum(item["status"] == "success" for item in segment_results)
    failed = len(segment_results) - succeeded
    duration_seconds = audio.duration_ms / 1000
    wall_seconds = time.perf_counter() - total_started
    if not segment_results:
        result_status = "no_speech"
    elif failed:
        result_status = "partial_failure"
    else:
        result_status = "passed"
    summary = {
        "schemaVersion": 1,
        "createdAtLocal": datetime.now().astimezone().isoformat(),
        "scope": (
            "direct local A-4 decoder, WebRTC VAD, and SenseVoice CPU model; "
            "no A-3/A-5 service or database"
        ),
        "result": {
            "status": result_status,
            "vadSegments": len(spans),
            "asrSucceeded": succeeded,
            "asrFailed": failed,
            "allDetectedSegmentsSucceeded": bool(segment_results) and failed == 0,
        },
        "input": {
            "path": str(audio_path.relative_to(REPOSITORY_ROOT)),
            "bytes": len(encoded_audio),
            "sha256": hashlib.sha256(encoded_audio).hexdigest(),
            "durationSeconds": duration_seconds,
            "processingInfo": audio.processing_info,
        },
        "configuration": {
            "path": str(config_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _sha256(config_path),
            "modelSnapshot": model_snapshot,
        },
        "source": _git_fingerprint(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logicalCpuCount": os.cpu_count(),
            "provider": "cpu",
        },
        "timingSeconds": {
            "inputRead": input_read_seconds,
            "decode": decode_seconds,
            "engineSetupAndIdentity": engine_setup_seconds,
            "modelPrepare": model_prepare_seconds,
            "vad": vad_seconds,
            "asrSum": asr_seconds,
            "endToEnd": wall_seconds,
            "audioDuration": duration_seconds,
            "asrRealTimeFactor": asr_seconds / duration_seconds,
            "endToEndRealTimeFactor": wall_seconds / duration_seconds,
        },
        "memoryMiB": memory,
        "segments": segment_results,
    }

    output_path = _resolve_personal_path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summaryPath={output_path}")
    return 0 if summary["result"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
