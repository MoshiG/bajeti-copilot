import pytest

from bajeti.agent import REFUSAL_EN, REFUSAL_SW, Agent, build_agent, is_refusal
from bajeti.config import AgentConfig
from bajeti.llm import ChatResult, FakeLLM, ToolCall
from bajeti.tools import build_tools


def call(name, args, cid="c1"):
    return ChatResult(content=None, tool_calls=[ToolCall(cid, name, args)], prompt_tokens=10,
                      completion_tokens=5)


SEARCH = call("search_docs", {"query": "Wizara ya Afya"})
SQL = call("query_budget", {"sql": "SELECT vote_code, total_tzs, doc_id, page_no "
                                   "FROM allocations WHERE vote_code='52'"}, "c2")


@pytest.fixture
def make_agent(built_cfg, retriever):
    def _make(responses, **agent_cfg):
        llm = FakeLLM(responses)
        cfg = AgentConfig(**{"max_steps": 5, "require_citations": True, **agent_cfg})
        return Agent(llm, build_tools(built_cfg, retriever), cfg), llm
    return _make


def test_search_then_cited_answer(make_agent):
    agent, llm = make_agent([SEARCH, ChatResult("Wizara ya Afya imetengewa TZS 980,500 milioni "
                                                "kwa 2025/26 [S1].", prompt_tokens=20)])
    ans = agent.answer("Wizara ya Afya imetengewa kiasi gani?")
    assert ans.citations[0].id == "S1" and ans.citations[0].page == 2
    assert not ans.refused and not ans.uncited and not ans.invalid_citations
    assert [s.tool for s in ans.steps] == ["search_docs"]
    assert ans.llm_calls == 2 and ans.prompt_tokens == 30
    second = llm.calls[1]["messages"]
    assert second[2]["role"] == "assistant" and second[2]["tool_calls"][0]["id"] == "c1"
    assert second[3]["role"] == "tool" and second[3]["tool_call_id"] == "c1"
    assert second[3]["content"].startswith("[S1]")


def test_sql_answer_cites_table_rows(make_agent):
    agent, _ = make_agent([SQL, ChatResult("Fungu 52 ilipata TZS 980.5 bilioni [S1].")])
    ans = agent.answer("Fungu 52 ilipata kiasi gani?")
    assert ans.citations[0].kind == "table_row" and ans.citations[0].doc_id == "speech_fixture"


def test_invented_citations_are_removed_and_counted(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult("Elimu ilipata fedha zaidi [S1][S7].")])
    ans = agent.answer("q")
    assert ans.text == "Elimu ilipata fedha zaidi [S1]."
    assert ans.invalid_citations == ["S7"] and [c.id for c in ans.citations] == ["S1"]


def test_uncited_answer_gets_one_repair_turn(make_agent):
    agent, llm = make_agent([SEARCH, ChatResult("Afya ilipata 980,500."),
                             ChatResult("Afya ilipata TZS 980,500 milioni [S1].")])
    ans = agent.answer("q")
    assert ans.citations and not ans.uncited and ans.llm_calls == 3
    assert llm.calls[2]["tools"] is None
    assert "no [S#] citations" in llm.calls[2]["messages"][-1]["content"]


def test_still_uncited_after_repair_is_flagged(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult("Afya ilipata fedha."), ChatResult("Bado.")])
    ans = agent.answer("q")
    assert ans.uncited and not ans.citations


@pytest.mark.parametrize("reply", [f"{REFUSAL_EN}. I searched for X.",
                                   f"{REFUSAL_SW}. Nilitafuta X."])
def test_refusals_are_recognised_and_not_repaired(make_agent, reply):
    agent, _ = make_agent([SEARCH, ChatResult(reply)])
    ans = agent.answer("Bajeti ya mwezi ni ngapi?")
    assert ans.refused and not ans.uncited and ans.llm_calls == 2


def test_answer_without_any_evidence_is_flagged_uncited(make_agent):
    agent, _ = make_agent([ChatResult("Bajeti ni TZS 57 trilioni.")])
    ans = agent.answer("q")
    assert ans.uncited and not ans.steps and ans.llm_calls == 1   # nothing to repair from


def test_bad_tool_calls_become_errors_the_model_sees(make_agent):
    agent, llm = make_agent([
        call("delete_everything", {}),
        call("search_docs", {"_raw": "{bad", "_error": "invalid_json"}, "c2"),
        SEARCH,
        ChatResult("Jibu [S1]."),
    ])
    ans = agent.answer("q")
    assert [s.error for s in ans.steps] == [True, True, False]
    assert "unknown tool 'delete_everything'" in ans.steps[0].result
    assert "not valid JSON" in ans.steps[1].result
    assert ans.citations


def test_repeated_calls_are_cached(make_agent):
    agent, _ = make_agent([SEARCH, SEARCH, ChatResult("Jibu [S1].")])
    ans = agent.answer("q")
    assert [s.cached for s in ans.steps] == [False, True]


def test_step_limit_forces_a_final_answer(make_agent):
    agent, llm = make_agent([SEARCH, SQL, ChatResult("Kwa ushahidi uliopo [S1].")], max_steps=2)
    ans = agent.answer("q")
    assert ans.hit_step_limit and ans.citations
    assert llm.calls[-1]["tools"] is None
    assert "tool-call limit" in llm.calls[-1]["messages"][-1]["content"]


def test_only_enabled_tools_are_offered(built_cfg, retriever):
    from bajeti.config import config_from_dict

    cfg = config_from_dict({**built_cfg.to_dict(), "tools": {"enable_sql": False,
                                                              "enable_optimizer": False}})
    llm = FakeLLM([ChatResult(f"{REFUSAL_EN}.")])
    build_agent(cfg, llm=llm, retriever=retriever).answer("q")
    offered = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert offered == ["search_docs"]


def test_is_refusal_is_strict_about_position():
    assert is_refusal(f"  {REFUSAL_EN.upper()}.")
    assert not is_refusal(f"The minister said: {REFUSAL_EN}.")


def test_cli_ask_prints_answer_sources_and_trace(built_cfg, retriever, monkeypatch, capsys,
                                                 tmp_path):
    import yaml

    from bajeti import cli, llm

    monkeypatch.setattr(llm, "build_llm", lambda cfg: FakeLLM([SEARCH, ChatResult("Jibu [S1].")]))
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump(built_cfg.to_dict()))
    assert cli.main(["-c", str(cfg_file), "ask", "Afya?", "--trace", "--embedder", "hashing",
                     "--reranker", "overlap"]) == 0
    out = capsys.readouterr().out
    assert "--- step 1: search_docs" in out and "Jibu [S1]." in out
    assert "Sources:\n[S1] Hotuba ya Bajeti 2025/26, p. 2" in out


def test_misquoted_figure_is_flagged_on_the_answer(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult("Afya ilipata shilingi milioni 908,500 [S1].")])
    ans = agent.answer("Wizara ya Afya imetengewa kiasi gani?")
    assert ans.citations and ans.unverified_figures == ["milioni 908,500"]


def test_correct_figure_in_another_unit_passes(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult("Afya ilipata shilingi bilioni 980.5 [S1].")])
    assert agent.answer("Wizara ya Afya imetengewa kiasi gani?").unverified_figures == []


def test_refusals_skip_the_figure_check(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult(f"{REFUSAL_EN}. I searched 1975 budgets.")])
    assert agent.answer("q").unverified_figures == []


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_bad_questions_rejected_before_any_model_call(make_agent, question):
    agent, llm = make_agent([])
    ans = agent.answer(question)
    assert ans.rejected and not llm.calls and "1000" in ans.text


def test_grouped_citations_are_checked_individually(make_agent):
    agent, _ = make_agent([SEARCH, ChatResult("Afya ilipata fedha [S1, S9].")])
    ans = agent.answer("q")
    assert ans.text == "Afya ilipata fedha [S1]." and ans.invalid_citations == ["S9"]
