import json

from bajeti.config import config_from_dict
from bajeti.tools import (
    SourceRegistry,
    build_tools,
    make_query_budget,
    make_rank_interventions,
    make_search_docs,
)


def test_search_docs_registers_citable_sources(retriever):
    tool, reg = make_search_docs(retriever, ["2025/26"]), SourceRegistry()
    out = tool.run({"query": "Wizara ya Afya"}, reg)
    assert not out.error and out.source_ids[0] == "S1"
    assert out.content.startswith("[S1] Hotuba ya Bajeti 2025/26, p. 2 — SEHEMU YA PILI")
    src = reg.sources["S1"]
    assert src.kind == "document" and src.page == 2 and src.chunk_id
    again = tool.run({"query": "Wizara ya Afya"}, reg)          # same chunk keeps its id
    assert again.source_ids[0] == "S1"
    assert "2025/26" in tool.description


def test_search_docs_filters_and_errors(retriever):
    tool, reg = make_search_docs(retriever, []), SourceRegistry()
    assert "without filters" in tool.run({"query": "kodi", "fiscal_year": "1999/00"}, reg).content
    assert tool.run({"query": " "}, reg).error
    assert not reg.sources


def test_query_budget_adds_source_column_and_legend(built_cfg):
    tool, reg = make_query_budget(built_cfg.paths.sqlite_path), SourceRegistry()
    out = tool.run({"sql": "SELECT vote_code, vote_name, total_tzs, doc_id, page_no "
                           "FROM allocations ORDER BY vote_code"}, reg)
    assert not out.error
    assert "| 46 | Wizara ya Elimu | 1,250,000,000,000 | speech_fixture | 2 | S1 |" in out.content
    assert "[S1] Hotuba ya Bajeti 2025/26, p. 2" in out.content   # title from documents table
    assert out.source_ids == ["S1"]                               # one page -> one source


def test_query_budget_uncitable_rows_are_flagged(built_cfg):
    tool, reg = make_query_budget(built_cfg.paths.sqlite_path), SourceRegistry()
    out = tool.run({"sql": "SELECT SUM(total_tzs) FROM allocations"}, reg)
    assert "cannot be cited" in out.content and not out.source_ids


def test_query_budget_errors_return_text_not_exceptions(built_cfg):
    tool, reg = make_query_budget(built_cfg.paths.sqlite_path), SourceRegistry()
    assert "rejected" in tool.run({"sql": "DROP TABLE allocations"}, reg).content
    bad = tool.run({"sql": "SELECT nope FROM allocations"}, reg)
    assert bad.error and "SQL error" in bad.content
    assert tool.run({}, reg).error
    assert "CREATE TABLE allocations" in tool.description        # schema shown to the model


def test_rank_interventions_tool_accepts_stringified_json():
    tool = make_rank_interventions()
    items = [{"name": "Clinics", "cost": 6, "benefit": 8}, {"name": "Roads", "cost": 5,
                                                              "benefit": 5}]
    out = tool.run({"interventions": json.dumps(items), "budget_ceiling": 6}, SourceRegistry())
    assert not out.error and "| Clinics | 6 | 8 |" in out.content
    assert "Cite the sources of the costs" in out.content
    assert tool.run({"interventions": items}, SourceRegistry()).error      # no ceiling
    assert tool.run({"interventions": [], "budget_ceiling": 5}, SourceRegistry()).error


def test_build_tools_follows_ablation_flags(built_cfg, retriever):
    names = [t.name for t in build_tools(built_cfg, retriever)]
    assert names == ["search_docs", "query_budget", "rank_interventions"]
    off = config_from_dict({**built_cfg.to_dict(),
                            "tools": {"enable_sql": False, "enable_optimizer": False}})
    assert [t.name for t in build_tools(off, retriever)] == ["search_docs"]
