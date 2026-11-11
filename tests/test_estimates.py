"""Volume IV parser, tested on a synthetic document that reproduces the real layout's quirks:
summary page with blank cells, a wrapped vote name with its amounts on the next line and a
total-only 'Direct Transfer' row; a vote's first page opening with an introduction block; a
project continued across a page break; a total wrapped onto its own line; a bracketed
negative."""

from __future__ import annotations

import pytest

from bajeti.index.budget_db import build_budget_db, run_readonly
from bajeti.ingest.estimates import _amount, parse_estimates, reconcile

FS = 7.0
# Right edges of the six amount columns, estimate total, and summary columns (as in the real PDF)
P_RIGHT = [262, 320, 386, 468, 541, 604, 770]
S_HEAD = {"101": 233, "102": 284, "103": 336, "201": 386, "202": 429, "203": 468, "Total": 526}
S_RIGHT = [276, 328, 372, 419, 463, 505, 551]


def _right(page, text, right, y, fs=FS):
    import pymupdf

    page.insert_text((right - pymupdf.get_text_length(text, "helv", fs), y), text,
                     fontsize=fs, fontname="helv")


def _left(page, text, x, y, fs=FS):
    page.insert_text((x, y), text, fontsize=fs, fontname="helv")


SFS = 5.5   # the summary page's font is smaller, so adjacent columns never touch


def _amounts_row(page, y, values, codes=("L", "T", "0GT")):
    for v, r in zip(values[:6], P_RIGHT[:6], strict=True):
        _right(page, v, r, y)
    for code, x in zip(codes, (621, 651, 682), strict=True):
        _left(page, code, x, y)
    _right(page, values[6], P_RIGHT[6], y)


def _total_row(page, y, label, values, wrap_total=False):
    _left(page, label, 71, y)
    for v, r in zip(values[:6], P_RIGHT[:6], strict=True):
        _right(page, v, r, y)
    _right(page, values[6], P_RIGHT[6], y + (12 if wrap_total else 0))


def _table_header(page, y):
    _left(page, "Item", 71, y)
    _left(page, "Description", 100, y)
    _left(page, "Local", 221, y + 10)
    _left(page, "Forex", 311, y + 10)
    _left(page, "Shs", 262, y + 20)


def _build(path):
    import pymupdf

    doc = pymupdf.open()

    # Page 1: vote summary, empty cells left blank (as in 2025/26)
    s = doc.new_page(width=841, height=595)
    _left(s, "APPROVED BUDGET BY MAJOR CATEGORIES", 245, 40)
    _left(s, "Vote Code and Name", 111, 60)
    for code, x in S_HEAD.items():
        _left(s, code, x, 60)
    _left(s, "A: MINISTRIES, INDEPENDENT DEPARTMENTS AND AGENCIES (MDAs)", 50, 75)
    rows = [  # code, name, {column index: amount}, wrapped name tail, amounts on next line
        ("001", "Public Debt", {1: "14,219,350,702,000", 6: "14,219,350,702,000"}, None, False),
        ("002", "Teachers' Service Commission",
         {0: "11,792,001,000", 1: "8,110,307,000", 3: "264,215,000", 4: "1,678,508,000",
          6: "21,845,031,000"}, None, False),
        ("024", "The Tanzania Cooperative Development",
         {0: "13,846,358,000", 1: "8,130,236,000", 3: "697,176,000", 6: "22,673,770,000"},
         "Commission", True),
        ("043", "Direct Transfer to Tanzania Cashewnut Board",
         {0: "-", 1: "-", 2: "-", 3: "-", 4: "-", 5: "-", 6: "284,502,389,400"}, None, False),
        ("052-", "Direct Transfer to Universal Health Insurance Fund",   # dash against the code
         {0: "-", 1: "-", 2: "-", 3: "-", 4: "-", 5: "-", 6: "193,519,348,100"}, None, False),
    ]
    y = 90
    for code, name, cells, tail, next_line in rows:
        _left(s, f"{code} {name}" if code.endswith("-") else f"{code} - {name}", 50, y, SFS)
        ay = y + 12 if next_line else y
        for col, v in cells.items():
            _right(s, v, S_RIGHT[col], ay, SFS)
        y = ay + 12
        if tail:
            _left(s, tail, 50, y, SFS)
            y += 12
        y += 4

    # Page 2: vote 002's first project page: introduction block, then the table
    p = doc.new_page(width=841, height=595)
    _left(p, "Vote 002 Teachers' Service Commission", 328, 30)
    _left(p, "A. ESTIMATE of the amount required in the year ending 30th June, 2026", 71, 45)
    _left(p, "(Shs.1,942,723,000)", 386, 57)
    _left(p, "2023/2024", 225, 70)
    _left(p, "2024/2025", 372, 70)
    _left(p, "2025/2026", 513, 70)
    _table_header(p, 82)
    _left(p, "Sub Vote 1003 PLANNING, MONITORING AND EVALUATION UNIT", 71, 120)
    _left(p, "6212 Construction & Rehabilitation of Govt Buildings", 84, 132)
    _amounts_row(p, 144, ["0", "0", "144,215,000", "0", "264,215,000", "0", "264,215,000"])
    _total_row(p, 156, "Total of Subvote",
               ["0", "0", "144,215,000", "0", "264,215,000", "0", "264,215,000"])
    _left(p, "Sub Vote 2001 TEACHERS SERVICE", 71, 170)
    _left(p, "6531 Project Monitoring and Evaluation", 84, 182)   # amounts on the next page
    _left(p, "1", 417, 580)                                        # page number

    # Page 3: continuation, wrapped total, bracketed negative, vote total
    p = doc.new_page(width=841, height=595)
    _left(p, "Vote 002 Teachers' Service Commission", 328, 30)
    _table_header(p, 45)
    _amounts_row(p, 80, ["(300,000)", "0", "0", "0", "1,000,000,000", "0", "1,000,000,000"])
    _left(p, "6532 Teachers Data Management System Upgrade", 84, 92)
    _amounts_row(p, 104, ["0", "0", "0", "0", "0", "678,508,000", "678,508,000"],
                 codes=("G", "T", "WB"))
    _total_row(p, 116, "Total of Subvote",
               ["(300,000)", "0", "0", "0", "1,000,000,000", "678,508,000", "1,678,508,000"],
               wrap_total=True)
    _total_row(p, 140, "Total of Vote",
               ["(300,000)", "0", "144,215,000", "0", "1,264,215,000", "678,508,000",
                "1,942,723,000"])
    _left(p, "2", 417, 580)
    doc.save(path)


@pytest.fixture(scope="module")
def parsed(tmp_path_factory):
    path = tmp_path_factory.mktemp("vol4") / "volume_iv_test.pdf"
    _build(path)
    return parse_estimates(path, "volume_iv_test", "2025/26")


@pytest.mark.parametrize("token, value", [
    ("1,250,000", 1250000.0), ("-", 0.0), ("(300,000)", -300000.0), ("L", None),
    ("0GT", None), ("12.50", 12.5),
])
def test_amount_tokens(token, value):
    assert _amount(token) == value


def test_summary_rows_by_column_position(parsed):
    rows = {r.vote_code: r for r in parsed.summary}
    assert set(rows) == {"001", "002", "024", "043", "052"}
    debt = rows["001"]                     # long right-aligned total must land in 'total'
    assert debt.oc_tzs == debt.total_tzs == 14_219_350_702_000 and debt.lga_dev_tzs == 0
    tsc = rows["002"]                      # blank cells: values must not shift left
    assert (tsc.pe_tzs, tsc.oc_tzs, tsc.lga_oc_tzs) == (11_792_001_000, 8_110_307_000, 0)
    assert (tsc.dev_local_tzs, tsc.dev_foreign_tzs) == (264_215_000, 1_678_508_000)
    assert tsc.section.startswith("A: MINISTRIES")


def test_wrapped_name_and_amounts_on_next_line(parsed):
    coop = next(r for r in parsed.summary if r.vote_code == "024")
    assert coop.vote_name == "The Tanzania Cooperative Development Commission"
    assert coop.pe_tzs == 13_846_358_000 and coop.total_tzs == 22_673_770_000


def test_total_only_direct_transfer_row(parsed):
    row = next(r for r in parsed.summary if r.vote_code == "043")
    assert row.total_tzs == 284_502_389_400 and row.pe_tzs == row.dev_local_tzs == 0


def test_row_with_dash_printed_against_the_code(parsed):
    row = next(r for r in parsed.summary if r.vote_code == "052")
    assert row.vote_name == "Direct Transfer to Universal Health Insurance Fund"
    assert row.total_tzs == 193_519_348_100


def test_projects_with_first_page_intro_and_page_break(parsed):
    projects = {p.project_code: p for p in parsed.projects}
    assert set(projects) == {"6212", "6531", "6532"}
    first = projects["6212"]
    assert (first.vote_code, first.sub_vote_code, first.page_no) == ("002", "1003", 2)
    assert (first.actual_year, first.approved_year) == ("2023/2024", "2024/2025")
    assert first.approved_local_tzs == 144_215_000 and first.est_total_tzs == 264_215_000
    carried = projects["6531"]             # name on page 2, amounts on page 3
    assert carried.page_no == 3 and carried.project_name == "Project Monitoring and Evaluation"
    assert carried.actual_local_tzs == -300_000 and carried.est_local_tzs == 1_000_000_000
    grant = projects["6532"]
    assert (grant.est_forex_tzs, grant.funding, grant.donor) == (678_508_000, "G", "WB")


def test_every_printed_total_reconciles(parsed):
    rec = reconcile(parsed)
    assert rec["subvote_totals_matched"] == [2, 2]          # incl. the wrapped total
    assert rec["vote_totals_matched"] == [1, 1]
    assert rec["summary_rows_add_up"] == [5, 5]
    assert not rec["mismatches"]


def test_reconcile_reports_a_missing_project(parsed):
    from dataclasses import replace

    broken = replace(parsed, projects=[p for p in parsed.projects if p.project_code != "6532"])
    rec = reconcile(broken)
    assert rec["subvote_totals_matched"] == [1, 2] and rec["vote_totals_matched"] == [0, 1]
    assert rec["mismatches"][0]["printed"] == 1_678_508_000


def test_database_tables_and_citations(parsed, tmp_path):
    from bajeti.tools import SourceRegistry, make_query_budget

    db = tmp_path / "b.sqlite"
    stats = build_budget_db([], db, [{"doc_id": "volume_iv_test", "title": "Volume IV 2025/26",
                                      "doc_type": "estimates", "fiscal_year": "2025/26",
                                      "language": "en"}], parsed.to_records())
    assert stats["projects"] == 3 and stats["vote_summary"] == 5
    view = run_readonly(db, "SELECT est_total_tzs, pages FROM project_votes "
                            "WHERE vote_code='002'").rows
    assert view == [(1_942_723_000, "2,3")]
    tool, reg = make_query_budget(db), SourceRegistry()
    out = tool.run({"sql": "SELECT doc_id, vote_code, est_total_tzs, pages FROM project_votes"},
                   reg)
    assert out.source_ids == ["S1", "S2"]                    # both pages behind the total
    assert "[S2] Volume IV 2025/26, p. 3" in out.content
    assert "vote_summary" in tool.description                # the model is told what to use


def test_pipeline_runs_parser_and_reports(tmp_path, monkeypatch):
    import json
    import shutil

    from bajeti.config import config_from_dict
    from bajeti.corpus import Document, Manifest
    from bajeti.ingest.pipeline import ingest_corpus

    raw = tmp_path / "raw"
    raw.mkdir()
    _build(raw / "vol.pdf")
    shutil.move(raw / "vol.pdf", raw / "volume_iv_test.pdf")
    cfg = config_from_dict({"paths": {"raw_dir": str(raw), "processed_dir": str(tmp_path / "p")},
                            "ingest": {"ocr_fallback": False}})
    doc = Document(id="volume_iv_test", title="Volume IV", doc_type="estimates",
                   fiscal_year="2025/26", language="en", manual=True)
    report = ingest_corpus(cfg, Manifest(documents=[doc]))[0]
    assert report.estimates["projects"] == 3 and report.estimates["vote_totals_matched"] == [1, 1]
    saved = json.loads((tmp_path / "p" / "estimates.json").read_text())
    assert len(saved["projects"]) == 3 and len(saved["vote_summary"]) == 5
