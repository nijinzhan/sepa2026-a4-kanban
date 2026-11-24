"""Hand-scored cases protect the baseline from misleading aggregate metrics."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "atco2_eval_2024141460331.py"
SPEC = importlib.util.spec_from_file_location("atco2_eval", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_markup_noise_and_punctuation_are_not_scored_as_spoken_words():
    assert evaluation.tokenize_2024141460331(
        "[#callsign]SWISS Two Two Four[/#callsign], [noise] [hes] descend!"
    ) == ["swiss", "two", "two", "four", "descend"]


@pytest.mark.parametrize(
    ("spoken", "written", "expected"),
    [
        (
            "flight level one hundred",
            "Flight level 100",
            ["flight", "level", "one", "zero", "zero"],
        ),
        ("Swiss two two four", "Swiss224", ["swiss", "two", "two", "four"]),
        (
            "one three four decimal five six zero",
            "134.560",
            ["one", "three", "four", "point", "five", "six", "zero"],
        ),
        ("five thousand feet", "5000 feet", ["five", "zero", "zero", "zero", "feet"]),
        ("one niner", "19", ["one", "nine"]),
    ],
)
def test_numeric_format_does_not_create_semantic_digit_errors(spoken, written, expected):
    assert evaluation.tokenize_2024141460331(spoken, numeric=True) == expected
    assert evaluation.tokenize_2024141460331(written, numeric=True) == expected


def test_wer_counts_substitutions_and_insertions_with_reference_denominator():
    result = evaluation.score_words_2024141460331(
        ["alpha", "bravo", "charlie"], ["alpha", "delta", "echo", "foxtrot"]
    )
    assert (result["N"], result["S"], result["D"], result["I"], result["C"]) == (3, 2, 0, 1, 1)
    assert result["wer"] == 1.0


def test_no_speech_output_is_scored_as_deletions_not_excluded():
    result = evaluation.score_words_2024141460331(["cleared", "to", "land"], [])
    assert (result["N"], result["D"], result["wer"]) == (3, 3, 1.0)
    assert evaluation.score_words_2024141460331([], ["hello"])["I"] == 1
    assert evaluation.score_words_2024141460331([], ["hello"])["wer"] is None


def test_entity_retention_rejects_a_missing_digit_and_internal_insertions():
    good = evaluation.score_words_2024141460331(
        ["swiss", "two", "four"], ["swiss", "two", "four", "roger"]
    )
    assert evaluation.span_retained_2024141460331(0, 3, good["operations"])
    bad = evaluation.score_words_2024141460331(
        ["swiss", "two", "four"], ["swiss", "two", "zero", "four"]
    )
    assert not evaluation.span_retained_2024141460331(0, 3, bad["operations"])
    missing = evaluation.score_words_2024141460331(["swiss", "two", "four"], ["swiss", "four"])
    assert not evaluation.span_retained_2024141460331(0, 3, missing["operations"])


def test_boundary_metrics_keep_misses_and_do_not_fill_silence_in_coverage():
    result = evaluation.boundary_score_2024141460331(
        [(0, 1000), (1300, 2000), (3000, 4000)], [(100, 2100), (5000, 6000)]
    )
    assert result["reference_regions"] == 2
    assert result["matched_regions"] == 1
    assert result["missed_regions"] == result["extra_regions"] == 1
    assert result["start_abs_errors_ms"] == [100]
    assert result["end_abs_errors_ms"] == [100]
    assert result["reference_speech_ms"] == 2700
    assert result["predicted_speech_ms"] == 3000
    assert result["overlap_ms"] == 1600


def test_ambiguous_number_word_sequence_is_not_reinterpreted_as_a_cardinal():
    assert evaluation.tokenize_2024141460331("one two hundred", numeric=True) == [
        "one",
        "two",
        "hundred",
    ]


def test_reference_entity_ranges_survive_markup_removal():
    result = evaluation.reference_tokens_2024141460331(
        [
            {
                "text": (
                    "[#callsign]Swiss two two four[/#callsign] climb [#value]one hundred[/#value]"
                ),
                "correct_tagging": "0",
            }
        ]
    )
    assert result["tokens"] == ["swiss", "two", "two", "four", "climb", "one", "zero", "zero"]
    assert result["callsigns"][0]["start"] == 0
    assert result["callsigns"][0]["end"] == 4
    assert result["callsigns"][0]["correct_tagging"] == "0"
    assert [(x["start"], x["end"]) for x in result["numbers"]] == [(1, 4), (5, 8)]
