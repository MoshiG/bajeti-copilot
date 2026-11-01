from bajeti.numbers import extract_amount_spans, unverified_figures

TABLE = "| Fungu | Taasisi | Kiasi (TZS milioni) |\n| 52 | Wizara ya Afya | 980,500 |"


def test_same_amount_in_another_unit_is_verified():
    assert unverified_figures("Afya ilipata shilingi bilioni 980.5 [S1].", [TABLE]) == []
    assert unverified_figures("Health received TZS 980,500 million [S1].", [TABLE]) == []


def test_misquoted_figure_is_flagged():
    assert unverified_figures("Afya ilipata shilingi bilioni 908.5 [S1].", [TABLE]) == [
        "bilioni 908.5"]


def test_amounts_not_in_the_evidence_are_flagged():
    flagged = unverified_figures("The increase was TZS 120.5 billion [S1].", [TABLE])
    assert flagged == ["120.5 billion"]


def test_percentages_and_labels_are_not_checked():
    text = ("This is a 12.5 percent increase, asilimia 15 ya bajeti, or 15% overall, under "
            "Section 117 and Fungu 52 on ukurasa 212 [S1].")
    assert unverified_figures(text, [TABLE]) == []


def test_fiscal_year_spellings_are_not_amounts():
    assert extract_amount_spans("FY2025/26, 2025-26 and 2025/2026") == []


def test_small_numbers_years_and_citation_ids_are_ignored():
    assert unverified_figures("In 2025/26, 3 votes [S12] were listed.", [TABLE]) == []


def test_figures_from_the_question_count_as_evidence():
    answer = "Within the TZS 500 billion ceiling, fund clinics [S1]."
    assert unverified_figures(answer, [TABLE]) == ["500 billion"]
    assert unverified_figures(answer, [TABLE, "budget ceiling of TZS 500 billion"]) == []


def test_spans_keep_the_written_form():
    assert extract_amount_spans("TSh980.5bn") == [("980.5bn", 980.5e9)]
