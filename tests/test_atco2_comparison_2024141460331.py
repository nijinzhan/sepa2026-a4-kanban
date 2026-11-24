"""Protocol-level tests for WebRTC frame-to-interval conversion."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "atco2_comparison_2024141460331.py"
SPEC = importlib.util.spec_from_file_location("atco2_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_vad_runs_merge_short_silence_and_drop_short_speech():
    flags = [True] * 15 + [False] * 10 + [True] * 15 + [False] * 30 + [True] * 5
    assert comparison.postprocess_vad_flags(flags, len(flags) * 20) == [[0, 800]]


def test_vad_clips_partial_final_frame_to_audio_duration():
    assert comparison.postprocess_vad_flags([False, True], 25, min_speech_ms=0) == [[20, 25]]


def test_vad_keeps_a_silence_gap_longer_than_the_frozen_limit():
    flags = [True] * 15 + [False] * 26 + [True] * 15
    assert comparison.postprocess_vad_flags(flags, len(flags) * 20) == [[0, 300], [820, 1120]]


def test_aggregate_and_paired_bootstrap_use_micro_wer_and_recording_resampling():
    rows = [
        {
            "scores": {
                "sensevoice": {"xml": {"strict": {"N": 10, "S": 1, "D": 0, "I": 0, "C": 9}}},
                "xlsr": {"xml": {"strict": {"N": 10, "S": 2, "D": 0, "I": 0, "C": 8}}},
            }
        },
        {
            "scores": {
                "sensevoice": {"xml": {"strict": {"N": 10, "S": 1, "D": 0, "I": 0, "C": 9}}},
                "xlsr": {"xml": {"strict": {"N": 10, "S": 2, "D": 0, "I": 0, "C": 8}}},
            }
        },
    ]
    summary = comparison.aggregate_condition(rows, "sensevoice", "xml", "strict")
    assert (summary["N"], summary["S"], summary["wer"]) == (20, 2, 0.1)
    paired = comparison.bootstrap_difference(
        [row["scores"]["sensevoice"]["xml"]["strict"] for row in rows],
        [row["scores"]["xlsr"]["xml"]["strict"] for row in rows],
        replicates=100,
        seed=7,
    )
    assert paired["differenceRightMinusLeft"] == 0.1
    assert paired["bootstrap95PercentileInterval"] == [0.1, 0.1]
