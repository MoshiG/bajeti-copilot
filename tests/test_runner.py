import json
from pathlib import Path

import pytest
import yaml

from bajeti.agent import REFUSAL_EN, Agent
from bajeti.config import config_from_dict
from bajeti.evaluation.gold import GoldItem
from bajeti.evaluation.runner import ablation_table, run_ablations, run_config
from bajeti.index.dense import HashingEmbedder
from bajeti.llm import ChatResult, ToolCall
from bajeti.retrieval import OverlapReranker, build_retriever
from bajeti.tools import build_tools

GOLD = [
    GoldItem.from_dict(dict(id="g1", type="numeric", language="sw",
                            question="Wizara ya Afya imetengewa kiasi gani?",
                            numbers=[980.5e9], keywords=["Wizara ya Afya"],
                            evidence=[{"doc_id": "speech_fixture", "pages": [2]}])),
    GoldItem.from_dict(dict(id="g2", type="factual", language="sw",
                            question="Serikali inapendekeza marekebisho gani ya Sheria ya Kodi "
                                     "ya Mapato?",
                            keywords=["uwazi"],
                            evidence=[{"doc_id": "speech_fixture", "pages": [1]}])),
    GoldItem.from_dict(dict(id="g3", type="unanswerable", language="en", answerable=False,
                            question="What was the budget of Kenya in 1975?")),
    GoldItem.from_dict(dict(id="g4", type="numeric", language="en", question="Draft",
                            answer="TODO", evidence=[{"doc_id": "speech_fixture",
                                                      "pages": [9]}])),
]

ANSWERS = {   # what the scripted model says after one search; g2 is deliberately wrong
    "g1": "Wizara ya Afya imetengewa shilingi milioni 980,500 kwa 2025/26 [S1].",
    "g2": "Serikali itapunguza kodi [S1][S9].",
    "g3": f"{REFUSAL_EN}. I searched the speeches.",
}


class QuestionScriptedLLM:
    """First call: search for the question. Second call: the scripted answer for that question."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        question = messages[1]["content"]
        qid = next(g.id for g in GOLD if g.question == question)
        if messages[-1]["role"] == "user":
            return ChatResult(None, [ToolCall("c1", "search_docs", {"query": question})],
                              prompt_tokens=50, completion_tokens=10)
        return ChatResult(ANSWERS[qid], prompt_tokens=80, completion_tokens=20)


@pytest.fixture
def factories():
    llm = QuestionScriptedLLM()

    def rf(cfg):
        return build_retriever(cfg, HashingEmbedder(), OverlapReranker())

    def af(cfg, retriever):
        return Agent(llm, build_tools(cfg, retriever), cfg.agent)

    return rf, af, llm


def test_full_run_scores_every_dimension(built_cfg, factories, tmp_path):
    rf, af, _ = factories
    s = run_config(built_cfg, GOLD, rf, af, results_dir=tmp_path)
    a, r = s["answers"], s["retrieval"]
    assert s["meta"]["questions"] == 3 and s["meta"]["skipped_drafts"] == 1
    assert a["correct"]["k"] == 2                       # g1 and the g3 refusal
    assert a["correct_answerable"]["rate"] == 0.5
    assert a["refusal_accuracy"]["rate"] == 1.0 and a["false_refusals"]["rate"] == 0.0
    assert a["invalid_citation_answers"]["k"] == 1      # g2's invented [S9]
    assert a["by"]["language"]["sw"]["k"] == 1
    assert r["hit@1"]["n"] == 2 and r["diagnosis"]["retrieved"] == 2
    assert a["avg_tokens"] > 0

    recs = [json.loads(line) for line in (tmp_path / "default" / "records.jsonl").open()]
    g1 = next(x for x in recs if x["id"] == "g1")
    assert g1["correct"] and g1["citation_hit"] and g1["retrieval"]["rank"] >= 1
    assert (tmp_path / "default" / "summary.json").exists()


def test_resume_skips_finished_questions(built_cfg, factories, tmp_path):
    rf, af, llm = factories
    run_config(built_cfg, GOLD[:1], rf, af, results_dir=tmp_path)
    before = llm.calls
    s = run_config(built_cfg, GOLD, rf, af, results_dir=tmp_path, resume=True)
    assert llm.calls - before == 4                       # only g2 and g3 asked (2 calls each)
    assert s["meta"]["questions"] == 3


def test_retrieval_only_needs_no_llm(built_cfg, factories, tmp_path):
    rf, _, llm = factories
    s = run_config(built_cfg, GOLD, rf, None, results_dir=tmp_path)
    assert "answers" not in s and llm.calls == 0
    assert (tmp_path / "default" / "retrieval_summary.json").exists()


def test_ablations_produce_comparison_table(built_cfg, factories, tmp_path):
    rf, af, _ = factories
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({**built_cfg.to_dict(),
                                    "paths": {**built_cfg.to_dict()["paths"],
                                              "results_dir": str(tmp_path / "res")}}))
    files = sorted((Path(__file__).parents[1] / "configs" / "ablations").glob("*.yaml"))
    summaries = run_ablations(base, files, GOLD, rf, af)
    assert list(summaries) == ["a1_naive_dense", "a2_hybrid", "a3_hybrid_rerank",
                               "a4_structured", "a5_full"]
    table = (tmp_path / "res" / "ablation.md").read_text()
    assert table.count("\n| a") == 5 and "Correct, answerable (95% CI)" in table
    assert "| a1_naive_dense | naive |  | ✓ |  | search only |" in table
    assert "| a5_full | structured | ✓ | ✓ | ✓ | SQL+opt |" in table


def test_retrieval_only_table_has_no_answer_columns():
    s = {"retrieval": {"k": 6, "hit@1": {"rate": .5, "ci_low": 0, "ci_high": 1},
                       "hit@6": {"rate": 1, "ci_low": 0, "ci_high": 1}, "mrr": .75},
         "meta": {"questions": 2, "settings": {"chunking": "naive", "bm25": True, "dense": False,
                                               "reranker": False, "sql": False,
                                               "optimizer": False}}}
    table = ablation_table({"x": s})
    assert "Correct" not in table and "| x | naive | ✓ |  |  | search only | 50% | 100% | 0.75 |" \
        in table


def test_cli_ablate_retrieval_only(built_cfg, tmp_path, capsys):
    from bajeti import cli

    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(json.dumps({**g.__dict__, "evidence": [e.__dict__ for e in
                                                                    g.evidence]}) + "\n"
                            for g in GOLD))
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({**built_cfg.to_dict(),
                                    "paths": {**built_cfg.to_dict()["paths"],
                                              "results_dir": str(tmp_path / "res")}}))
    assert cli.main(["-c", str(base), "ablate", "--retrieval-only", "--gold", str(gold),
                     "--embedder", "hashing", "--reranker", "overlap", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "a5_full: hit@1" in out and (tmp_path / "res" / "ablation_retrieval.md").exists()
    cfg = config_from_dict(built_cfg.to_dict())
    assert cfg.name == "default"


def test_resume_refuses_to_mix_different_settings(built_cfg, factories, tmp_path):
    rf, af, _ = factories
    run_config(built_cfg, GOLD[:1], rf, af, results_dir=tmp_path)
    changed = config_from_dict({**built_cfg.to_dict(), "llm": {"model": "another-model"}})
    with pytest.raises(ValueError, match="different settings"):
        run_config(changed, GOLD, rf, af, results_dir=tmp_path, resume=True)


def test_resume_reasks_only_edited_questions(built_cfg, factories, tmp_path):
    rf, af, llm = factories
    run_config(built_cfg, GOLD, rf, af, results_dir=tmp_path)
    before = llm.calls
    run_config(built_cfg, GOLD, rf, af, results_dir=tmp_path, resume=True)
    assert llm.calls == before                              # nothing new to ask
    edited = [GoldItem.from_dict({**g.__dict__, "keywords": ["Afya"],
                                  "evidence": [e.__dict__ for e in g.evidence]})
              if g.id == "g1" else g for g in GOLD]
    s = run_config(built_cfg, edited, rf, af, results_dir=tmp_path, resume=True)
    assert llm.calls - before == 2 and s["meta"]["questions"] == 3
    lines = (tmp_path / "default" / "records.jsonl").read_text().splitlines()
    assert len(lines) == 3                                  # the stale g1 record was replaced


def test_summary_records_the_models_that_actually_ran(built_cfg, factories, tmp_path):
    rf, _, _ = factories
    meta = run_config(built_cfg, GOLD, rf, None, results_dir=tmp_path)["meta"]
    assert meta["embedding_model"] == "hashing-512" and meta["reranker_model"] == "overlap"


def test_no_answerable_questions_shows_dashes_not_zero(built_cfg, factories, tmp_path):
    rf, af, _ = factories
    s = run_config(built_cfg, [GOLD[2]], rf, af, results_dir=tmp_path)
    table = ablation_table({"x": s})
    assert "Hit@k" in table and "| – | – |" in table
