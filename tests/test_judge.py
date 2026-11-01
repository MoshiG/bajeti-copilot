import json

import pytest
import yaml

from bajeti.config import config_from_dict
from bajeti.evaluation.judge import (
    Claim,
    Judge,
    cohen_kappa,
    parse_verdict,
    sample_claims,
    split_claims,
    validate_judge,
    validation_markdown,
)
from bajeti.llm import ChatResult, FakeLLM

EXCERPTS = {"S1": "| 52 | Wizara ya Afya | 980,500 |", "S2": "Kodi ya mapato itarekebishwa."}


def test_split_claims_keeps_cited_sentences_with_their_passages():
    answer = ("Afya ilipata TZS 980.5 bilioni [S1]. Hii ni habari njema. "
              "Kodi itarekebishwa. [S2] Pia [S1][S2] zinaonyesha mpango.")
    claims = split_claims(answer, EXCERPTS)
    assert [c.sentence for c in claims] == [
        "Afya ilipata TZS 980.5 bilioni.", "Kodi itarekebishwa.", "Pia zinaonyesha mpango."]
    assert claims[1].source_ids == ["S2"]                 # citation after the full stop
    assert claims[2].passages == [EXCERPTS["S1"], EXCERPTS["S2"]]


def test_unknown_citation_ids_are_ignored():
    assert split_claims("Jibu [S9].", EXCERPTS) == []


@pytest.mark.parametrize("raw, label", [
    ('{"label": "supported", "reason": "matches"}', "supported"),
    ('```json\n{"label": "Partial", "reason": "x"}\n```', "partial"),
    ("Unsupported: the figure differs.", "unsupported"),
    ("supported or unsupported?", "unparseable"),
    ("no idea", "unparseable"),
])
def test_parse_verdict(raw, label):
    assert parse_verdict(raw).label == label


def test_judge_sends_claim_and_passages():
    llm = FakeLLM(['{"label": "supported", "reason": "same amount"}'])
    v = Judge(llm).judge(Claim("Afya ilipata TZS 980.5 bilioni.", ["S1"], [EXCERPTS["S1"]]))
    assert v.label == "supported"
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "[S1] | 52 | Wizara ya Afya | 980,500 |" in prompt and "CLAIM:" in prompt
    assert llm.calls[0]["tools"] is None


def test_cohen_kappa_known_values():
    assert cohen_kappa(["a", "b"] * 5, ["a", "b"] * 5) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert cohen_kappa([], []) == 0.0


def test_validate_judge_reports_agreement_and_confusion():
    labelled = [
        {"claim_id": "q1-0", "sentence": "s", "sources": ["S1"], "passages": ["p"],
         "human_label": "supported"},
        {"claim_id": "q2-0", "sentence": "s", "sources": ["S1"], "passages": ["p"],
         "human_label": "unsupported"},
        {"claim_id": "q3-0", "sentence": "s", "sources": ["S1"], "passages": ["p"],
         "human_label": ""},                               # unlabelled: skipped
    ]
    llm = FakeLLM(['{"label": "supported"}', '{"label": "supported"}'])
    v = validate_judge(Judge(llm), labelled)
    assert v["n"] == 2 and v["agreed"] == 1 and v["confusion"]["unsupported"]["supported"] == 1
    assert "agreed with human labels on 1 of 2 claims" in validation_markdown(v)
    with pytest.raises(ValueError):
        validate_judge(Judge(FakeLLM([])), labelled[2:])


def test_sample_claims_is_seeded_and_skips_refusals():
    records = [
        {"id": "q1", "answerable": True, "refused": False, "answer": "A [S1]. B [S1].",
         "citations": [{"id": "S1", "excerpt": "e"}]},
        {"id": "q2", "answerable": True, "refused": True, "answer": "x [S1].",
         "citations": [{"id": "S1", "excerpt": "e"}]},
    ]
    first = sample_claims(records, 5)
    assert len(first) == 2 and {c["question_id"] for c in first} == {"q1"}
    assert sample_claims(records, 5) == first and first[0]["human_label"] == ""


def test_config_accepts_optional_judge_section():
    cfg = config_from_dict({"judge": {"model": "big", "base_url": "http://j/v1"}})
    assert cfg.judge.model == "big"
    assert config_from_dict({}).judge is None
    with pytest.raises(ValueError, match="judge"):
        config_from_dict({"judge": {"modle": "typo"}})


def test_runner_and_cli_judge_flow(built_cfg, tmp_path, monkeypatch, capsys):
    from bajeti import cli, llm
    from bajeti.agent import Agent
    from bajeti.evaluation.gold import GoldItem
    from bajeti.evaluation.runner import ablation_table, run_config
    from bajeti.index.dense import HashingEmbedder
    from bajeti.llm import ToolCall
    from bajeti.retrieval import OverlapReranker, build_retriever
    from bajeti.tools import build_tools

    gold = [GoldItem.from_dict(dict(
        id="g1", type="numeric", language="sw", question="Wizara ya Afya imetengewa kiasi gani?",
        numbers=[980.5e9], keywords=["Afya"],
        evidence=[{"doc_id": "speech_fixture", "pages": [2]}]))]
    answer_llm = FakeLLM([
        ChatResult(None, [ToolCall("c1", "search_docs", {"query": "Wizara ya Afya"})]),
        ChatResult("Wizara ya Afya ilipata shilingi bilioni 980.5 [S1]. Hii ni kubwa [S1]."),
    ])
    judge = Judge(FakeLLM(['{"label": "supported"}', '{"label": "unsupported"}']))
    s = run_config(built_cfg, gold,
                   lambda cfg: build_retriever(cfg, HashingEmbedder(), OverlapReranker()),
                   lambda cfg, r: Agent(answer_llm, build_tools(cfg, r), cfg.agent),
                   results_dir=tmp_path, judge=judge)
    j = s["answers"]["judge"]
    assert j["claims_supported"]["k"] == 1 and j["claims_supported"]["n"] == 2
    assert j["answers_fully_supported"]["rate"] == 0.0
    assert "Fully supported answers (judge)" in ablation_table({"default": s})

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump({**built_cfg.to_dict(),
                                        "paths": {**built_cfg.to_dict()["paths"],
                                                  "results_dir": str(tmp_path)},
                                        "judge": {"model": "judge-model"}}))
    labels = tmp_path / "labels.jsonl"
    assert cli.main(["-c", str(cfg_file), "export-claims", "--run", "default",
                     "--out", str(labels)]) == 0
    rows = [json.loads(line) for line in labels.open()]
    assert len(rows) == 2 and rows[0]["passages"][0].startswith("| Fungu")
    with pytest.raises(SystemExit, match="exists"):          # your labels are never overwritten
        cli.main(["-c", str(cfg_file), "export-claims", "--run", "default", "--out", str(labels)])

    for r, lab in zip(rows, ["supported", "supported"], strict=True):
        r["human_label"] = lab
    labels.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(llm, "build_llm", lambda cfg: FakeLLM(['{"label": "supported"}'] * 2))
    assert cli.main(["-c", str(cfg_file), "judge-validate", "--labels", str(labels)]) == 0
    assert "agreed with human labels on 2 of 2" in capsys.readouterr().out
    assert (tmp_path / "judge_validation.md").exists()


def test_cli_judge_flag_requires_config(built_cfg, tmp_path):
    from bajeti import cli

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(yaml.safe_dump(built_cfg.to_dict()))
    with pytest.raises(SystemExit, match="judge"):
        cli.main(["-c", str(cfg_file), "judge-validate", "--labels", str(tmp_path / "x")])


def test_uncited_claims_bullets_and_abbreviations():
    answer = ("Kwa mujibu wa Dkt. Mwigulu, bajeti imeongezeka [S1].\n"
              "- Afya ilipata TZS 980.5 bilioni [S1]\n"
              "- Elimu ilipata kiasi kikubwa zaidi mwaka huu.\n"
              "Here is the summary:")
    claims = split_claims(answer, EXCERPTS, include_uncited=True)
    assert [c.sentence for c in claims] == [
        "Kwa mujibu wa Dkt. Mwigulu, bajeti imeongezeka.",
        "Afya ilipata TZS 980.5 bilioni",
        "Elimu ilipata kiasi kikubwa zaidi mwaka huu."]
    assert claims[2].source_ids == []
    assert len(split_claims(answer, EXCERPTS)) == 2          # default: cited claims only


def test_judge_answer_counts_uncited_sentences_without_a_model_call():
    llm = FakeLLM(['{"label": "supported"}'])
    out = Judge(llm).judge_answer(
        "Afya ilipata TZS 980.5 bilioni [S1]. Elimu nayo ilipata fedha nyingi sana.", EXCERPTS)
    assert [c["label"] for c in out] == ["supported", "uncited"] and len(llm.calls) == 1


def test_grouped_citations_are_split():
    claims = split_claims("Afya na kodi zimeelezwa [S1, S2].", EXCERPTS)
    assert claims[0].source_ids == ["S1", "S2"]
