import sqlite3

import pytest

from bajeti.index.budget_db import (
    UnsafeQueryError,
    build_budget_db,
    detect_unit,
    map_header,
    parse_number,
    run_readonly,
    schema_description,
)

HEADER = ["Vote", "Project Code", "Project Name", "Local (Shs '000)", "Foreign (Shs '000)",
          "Total (Shs '000)"]
VOL4 = "volume_iv_2025_26"


def _t(table_id, doc_id, page, rows, fy="2025/26"):
    return {"table_id": table_id, "doc_id": doc_id, "fiscal_year": fy, "page_no": page,
            "doc_type": "estimates", "rows": rows}


TABLES = [
    _t("v4:p10:t0", VOL4, 10, [
        HEADER,
        ["Vote 46 - Ministry of Education, Science and Technology", "", "", "", "", ""],
        ["", "4390", "Secondary Education Quality Improvement", "120,000", "380,500", "500,500"],
        ["", "4391", "Higher Education for Economic Transformation", "50,000", "(1,000)", "49,000"],
        ["Total Vote 46", "", "", "170,000", "379,500", "549,500"],
    ]),
    _t("v4:p11:t0", VOL4, 11, [  # continuation: no header of its own
        ["Vote 52 - Ministry of Health", "", "", "", "", ""],
        ["", "5410", "Health Sector Support", "200,000", "800,000", "1,000,000"],
        ["Total Vote 52", "", "", "200,000", "800,000", "1,000,000"],
    ]),
    _t("v4:p40:t0", VOL4, 40, [  # same width, but far away: must NOT be read as a continuation
        ["Region", "District", "Facility", "2023", "2024", "2025"],
        ["Dodoma", "Bahi", "Kituo", "1", "2", "3"],
    ]),
    _t("sp:p2:t0", "speech_2025_26_sw", 2, [
        ["Fungu", "Taasisi", "Kiasi (TZS milioni)"],
        ["46", "Wizara ya Elimu", "1,250,000"],
        ["52", "Wizara ya Afya", "980,500"],
    ]),
    _t("sp:p5:t0", "speech_2025_26_sw", 5, [
        ["Indicator", "2023", "2024"],
        ["GDP growth", "5.1", "5.5"],
    ]),
]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "budget.sqlite"
    stats = build_budget_db(TABLES, path, [
        {"doc_id": VOL4, "title": "Volume IV", "doc_type": "estimates", "fiscal_year": "2025/26",
         "language": "en"},
    ])
    return path, stats


def _q(path, sql):
    return run_readonly(path, sql).rows


@pytest.mark.parametrize("raw, expected", [
    ("1,250,000", 1250000.0), ("(1,000)", -1000.0), ("-", None), ("", None),
    ("12.5", 12.5), ("Wizara", None), ("1 000", 1000.0), ("−5", -5.0),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


def test_detect_unit():
    assert detect_unit(["Kiasi (TZS milioni)"]) == 1e6
    assert detect_unit(["Total (Shs '000)"]) == 1e3
    assert detect_unit(["Amount (TZS bn)"]) == 1e9
    assert detect_unit(["Amount"]) == 1.0


def test_header_mapping_english_and_swahili():
    assert map_header(HEADER) == {0: "vote_code", 1: "project_code", 2: "project_name",
                                  3: "local", 4: "foreign", 5: "total"}
    assert map_header(["Fungu", "Taasisi", "Kiasi (TZS milioni)"]) == {
        0: "vote_code", 1: "vote_name", 2: "total"}


def test_every_cell_kept_in_raw_cells(db):
    path, stats = db
    assert stats["cells"] == sum(len(r) for t in TABLES for r in t["rows"])
    assert _q(path, "SELECT value FROM raw_cells WHERE table_id='sp:p5:t0' AND number=5.5")


def test_vote_heading_rows_carry_down_and_units_apply(db):
    path, _ = db
    rows = _q(path, "SELECT vote_code, vote_name, project_name, local_tzs, foreign_tzs, total_tzs "
                    "FROM allocations WHERE project_code='4391'")
    assert rows == [("46", "Ministry of Education, Science and Technology",
                     "Higher Education for Economic Transformation", 50e6, -1e6, 49e6)]


def test_continuation_table_reuses_header(db):
    path, _ = db
    rows = _q(path, "SELECT vote_code, total_tzs, page_no FROM allocations "
                    "WHERE project_code='5410'")
    assert rows == [("52", 1e9, 11)]


def test_distant_same_width_table_not_treated_as_continuation(db):
    path, _ = db
    assert not _q(path, "SELECT * FROM allocations WHERE table_id='v4:p40:t0'")


def test_total_rows_flagged_and_excluded_from_view(db):
    path, _ = db
    flagged = _q(path, "SELECT COUNT(*) FROM allocations WHERE is_total_row=1")[0][0]
    assert flagged == 2
    view = _q(path, f"SELECT vote_code, local_tzs, foreign_tzs, total_tzs FROM vote_totals "
                    f"WHERE doc_id='{VOL4}' ORDER BY vote_code")
    assert view == [("46", 170e6, 379.5e6, 549.5e6), ("52", 200e6, 800e6, 1e9)]
    # The document's own printed total agrees with our sum: a built-in consistency check.
    printed = _q(path, "SELECT total_tzs FROM allocations WHERE is_total_row=1 "
                       "AND vote_code='46'")[0][0]
    assert printed == view[0][3]


def test_documents_are_never_summed_together(db):
    path, _ = db
    rows = _q(path, "SELECT doc_id, total_tzs FROM vote_totals WHERE vote_code='46' "
                    "ORDER BY doc_id")
    assert rows == [("speech_2025_26_sw", 1.25e12), (VOL4, 549.5e6)]


def test_non_budget_table_only_in_raw_cells(db):
    path, _ = db
    assert not _q(path, "SELECT * FROM allocations WHERE table_id='sp:p5:t0'")


@pytest.mark.parametrize("sql", [
    "DELETE FROM allocations",
    "SELECT 1; DROP TABLE allocations",
    "WITH x AS (SELECT 1) DELETE FROM allocations",
    "ATTACH DATABASE 'other.db' AS o",
    "PRAGMA table_info(allocations)",
])
def test_unsafe_queries_rejected(db, sql):
    path, _ = db
    with pytest.raises(UnsafeQueryError):
        run_readonly(path, sql)


def test_connection_itself_is_read_only(db):
    path, _ = db
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM allocations")
    conn.close()


def test_replace_function_allowed(db):
    path, _ = db
    assert _q(path, "SELECT replace(vote_name, 'Ministry', 'Wizara') FROM allocations LIMIT 1")


def test_runaway_query_times_out(db):
    path, _ = db
    with pytest.raises(TimeoutError):
        run_readonly(path, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                           "SELECT max(x) FROM c", timeout_s=0.2)


def test_results_truncate_and_render(db):
    path, _ = db
    result = run_readonly(path, "SELECT * FROM raw_cells", max_rows=5)
    assert len(result.rows) == 5 and result.truncated
    md = run_readonly(path, "SELECT vote_code, total_tzs FROM vote_totals "
                            "WHERE vote_code='52' AND doc_id LIKE 'vol%'").to_markdown()
    assert "| 52 | 1,000,000,000 |" in md


def test_schema_description_lists_years_and_notes(db):
    path, _ = db
    text = schema_description(path)
    assert "CREATE TABLE allocations" in text and "2025/26" in text and "doc_id" in text


def test_repeated_header_on_next_page_keeps_the_vote(tmp_path):
    tables = [
        _t("v:p20:t0", VOL4, 20, [HEADER, ["Vote 46 - Ministry of Education", "", "", "", "", ""],
                                  ["", "4390", "Schools", "10", "20", "30"]]),
        _t("v:p21:t0", VOL4, 21, [HEADER, ["", "4392", "Teacher housing", "5", "5", "10"]]),
    ]
    path = tmp_path / "b.sqlite"
    build_budget_db(tables, path)
    assert _q(path, "SELECT vote_code FROM allocations WHERE project_code='4392'") == [("46",)]


def test_picks_the_column_for_the_documents_own_year():
    header = ["Vote", "Description", "Actual 2024/25", "Approved Estimates 2025/26",
              "Estimates 2026/27"]
    assert map_header(header, "2026/27")[4] == "total"
    assert map_header(header, "2025/26")[3] == "total"
    assert map_header(["Vote", "Name", "Estimate 2025-2026"], "2025/26")[2] == "total"
    assert map_header(header)[3] == "total"          # no year known: first amount column


def test_more_unit_spellings():
    assert detect_unit(["Amount (Shs ’000)"]) == 1e3
    assert detect_unit(["Amount (TZS Mill.)"]) == 1e6


def test_vote_totals_rows_cite_every_page(db):
    from bajeti.tools import SourceRegistry, make_query_budget

    path, _ = db
    tool, reg = make_query_budget(path), SourceRegistry()
    out = tool.run({"sql": f"SELECT doc_id, vote_code, total_tzs, pages FROM vote_totals "
                           f"WHERE doc_id='{VOL4}' ORDER BY vote_code"}, reg)
    assert out.source_ids == ["S1", "S2"]            # vote 46 on p.10, vote 52 on p.11
    assert "| S1 |" in out.content and "| S2 |" in out.content


@pytest.mark.parametrize("sql", [
    "SELECT vote_name FROM allocations WHERE vote_name LIKE '%Create%'",
    "SELECT 'a;b' AS x",
    "SELECT 'it''s; DROP TABLE allocations' AS x",
    "-- totals per vote\nSELECT vote_code FROM allocations",
])
def test_string_contents_and_comments_do_not_trip_the_guard(db, sql):
    path, _ = db
    run_readonly(path, sql)


@pytest.mark.parametrize("sql", [
    "SELECT 1 /* hide */; DELETE FROM allocations",
    "/* comment */ DELETE FROM allocations",
    "SELECT 'x'; DROP TABLE allocations",
])
def test_guard_still_blocks_real_statements(db, sql):
    path, _ = db
    with pytest.raises(UnsafeQueryError):
        run_readonly(path, sql)
