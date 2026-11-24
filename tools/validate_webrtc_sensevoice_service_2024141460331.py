"""Replay frozen ATCO2 with CPU WebRTC/SenseVoice and simulated team HTTP services."""

import argparse
import json
from pathlib import Path

from validate_xlsr_webrtc_service_2024141460331 import HERE, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.example.json")
    parser.add_argument("--root", type=Path, default=HERE / "outputs/atco2-1h-20261021")
    parser.add_argument(
        "--output", type=Path, default=HERE / "outputs/webrtc-sensevoice-service-20261023"
    )
    args = parser.parse_args()
    summary = run(args.config, args.root, args.output, baseline_engine="sensevoice")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (
        summary["savedAndReadBack"] != summary["recordings"]
        or summary["matchesPreviousSegmentsAndText"] != summary["recordings"]
    ):
        raise SystemExit("frozen service replay differs from prior WebRTC/SenseVoice outputs")
