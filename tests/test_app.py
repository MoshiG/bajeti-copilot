import pytest

from bajeti.agent import REFUSAL_EN, Agent
from bajeti.app import Demo, RateLimiter, build_app, trace_markdown, warnings_markdown
from bajeti.config import AgentConfig
from bajeti.llm import ChatResult, FakeLLM, ToolCall
from bajeti.tools import build_tools

SEARCH = ChatResult(None, [ToolCall("c1", "search_docs", {"query": "Wizara ya Afya"})])


@pytest.fixture
def demo(built_cfg, retriever):
    def _make(responses, limit=10):
        agent = Agent(FakeLLM(responses), build_tools(built_cfg, retriever), AgentConfig())
        return Demo(agent, RateLimiter(limit=limit, window_s=60))
    return _make


def test_good_answer_shows_sources_and_trace_without_warnings(demo):
    body, sources, trace = demo([SEARCH, ChatResult("Afya: shilingi bilioni 980.5 [S1].")]) \
        .respond("Wizara ya Afya imetengewa kiasi gani?")
    assert body == "Afya: shilingi bilioni 980.5 [S1]." and "⚠️" not in body
    assert "**[S1]** Hotuba ya Bajeti 2025/26, p. 2" in sources
    assert "`search_docs`" in trace


def test_misquoted_figure_is_warned_about(demo):
    body, _, _ = demo([SEARCH, ChatResult("Afya: shilingi bilioni 908.5 [S1].")]).respond("q?")
    assert "Figures not found in the sources" in body and "`bilioni 908.5`" in body


def test_uncited_answer_is_warned_about(demo):
    body, sources, _ = demo([SEARCH, ChatResult("Afya ilipata fedha."),
                             ChatResult("Bado bila marejeo.")]).respond("q?")
    assert "No citations" in body and sources == ""


def test_refusal_has_no_warnings(demo):
    body, _, _ = demo([SEARCH, ChatResult(f"{REFUSAL_EN}.")]).respond("Kenya 1975?")
    assert "⚠️" not in body


def test_input_limits_and_rate_limit(demo):
    d = demo([SEARCH, ChatResult("Jibu [S1].")], limit=1)
    assert "type a question" in d.respond("  ")[0]
    assert "under 1000 characters" in d.respond("x" * 1001)[0]
    assert d.respond("q?", "1.2.3.4")[0] == "Jibu [S1]."
    assert "Too many questions" in d.respond("q?", "1.2.3.4")[0]


def test_model_outage_gives_a_friendly_message(built_cfg, retriever):
    class Down:
        def chat(self, *a, **k):
            raise ConnectionError("endpoint down")

    d = Demo(Agent(Down(), build_tools(built_cfg, retriever), AgentConfig()), RateLimiter())
    body, sources, trace = d.respond("q?")
    assert "unavailable" in body and sources == trace == ""


def test_rate_limiter_sliding_window():
    now = [0.0]
    rl = RateLimiter(limit=2, window_s=10, clock=lambda: now[0])
    assert rl.check("a") == (True, 0) and rl.check("a") == (True, 0)
    allowed, wait = rl.check("a")
    assert not allowed and wait == 11
    assert rl.check("b")[0]                       # limits are per client
    now[0] = 10.0
    assert rl.check("a")[0]                       # oldest request has left the window


def test_formatters_handle_empty_cases():
    from bajeti.agent import Answer

    empty = Answer(question="q", text="t")
    assert warnings_markdown(empty) == "" and trace_markdown(empty) == "_No tools were called._"
    full = Answer(question="q", text="t", invalid_citations=["S9"], hit_step_limit=True)
    assert "removed" in warnings_markdown(full) and "limit" in warnings_markdown(full)


def test_gradio_app_builds(demo):
    pytest.importorskip("gradio")
    ui = build_app(demo([]))
    assert ui.title == "Bajeti Copilot"


def test_gradio_receives_the_request_for_per_visitor_limits(demo):
    gr = pytest.importorskip("gradio")
    from gradio.utils import get_type_hints

    ui = build_app(demo([]))
    assert get_type_hints(ui.bajeti_handler).get("request") is gr.Request


def test_demo_examples_are_not_gold_questions():
    from pathlib import Path

    from bajeti.app import EXAMPLES
    from bajeti.evaluation.gold import load_gold

    gold = {g.question for g in load_gold(Path(__file__).parents[1] / "eval" / "gold.jsonl")}
    assert not gold & set(EXAMPLES)
