import pytest

from bajeti.evaluation.gold import Evidence
from bajeti.evaluation.metrics import (
    extract_amounts,
    first_relevant_rank,
    matches_evidence,
    number_found,
    rate,
    score_answer,
    wilson,
)


@pytest.mark.parametrize("text, expected", [
    ("TZS 980,500 milioni", 980.5e9),
    ("shilingi bilioni 980.5", 980.5e9),          # Swahili: scale before the number
    ("980.5 billion shillings", 980.5e9),
    ("TZS 1.25 trillion", 1.25e12),
    ("Sh57.04 trilioni", 57.04e12),                # currency glued to the number
    ("TSh980.5bn", 980.5e9),
    ("TZS 5m", 5e6),
    ("1,250,000,000,000", 1.25e12),
])
def test_amounts_in_both_languages(text, expected):
    assert number_found(expected, extract_amounts(text))


def test_fiscal_years_and_citations_are_not_amounts():
    found = extract_amounts("Mwaka 2025/26 [S12] ulitengewa TZS 10 bilioni.")
    assert found == [10e9]


def test_lone_letter_before_a_number_is_not_a_scale():
    assert extract_amounts("Vote m 46") == [46.0]


def test_tolerance_is_half_a_percent():
    assert number_found(1000.0, [1004.0])
    assert not number_found(1000.0, [1010.0])


def test_score_answer_needs_all_numbers_and_keywords():
    text = "Wizara ya Afya ilitengewa shilingi bilioni 980.5 mwaka 2025/26 [S1]."
    assert score_answer(text, [980.5e9], ["wizara ya afya"])["correct"]
    missing_kw = score_answer(text, [980.5e9], ["Wizara ya Elimu"])
    assert missing_kw == {"numbers_ok": True, "keywords_ok": False, "correct": False}
    assert not score_answer(text, [1.25e12], [])["correct"]


def test_evidence_matching_uses_page_overlap():
    ev = [Evidence("vol4", [10, 11])]
    assert matches_evidence("vol4", 11, 12, ev)
    assert not matches_evidence("vol4", 12, 13, ev)
    assert not matches_evidence("other", 10, 10, ev)
    assert not matches_evidence("vol4", None, None, ev)
    assert first_relevant_rank([("x", 1, 1), ("vol4", 9, 10)], ev) == 2
    assert first_relevant_rank([("x", 1, 1)], ev) == 0


def test_wilson_interval_known_values():
    lo, hi = wilson(40, 80)
    assert lo == pytest.approx(0.3923, abs=1e-3) and hi == pytest.approx(0.6077, abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(10, 10)[1] == 1.0


def test_rate_summary():
    r = rate([True, False, True, True])
    assert r["n"] == 4 and r["k"] == 3 and r["rate"] == 0.75 and r["ci_low"] < 0.75 < r["ci_high"]
