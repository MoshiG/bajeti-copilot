import json
from pathlib import Path

import pytest

from bajeti.corpus import load_manifest
from bajeti.evaluation.gold import GoldItem, load_gold, validate_gold

ROOT = Path(__file__).resolve().parents[1]


def _item(**kw):
    base = dict(id="q1", type="numeric", language="sw", question="Fungu 46 lilipata kiasi gani?",
                numbers=[1.25e12], keywords=["Elimu"],
                evidence=[{"doc_id": "speech_fixture", "pages": [2]}])
    return GoldItem.from_dict({**base, **kw})


def test_repo_gold_file_is_valid():
    items = load_gold(ROOT / "eval" / "gold.jsonl")
    doc_ids = {d.id for d in load_manifest(ROOT / "data" / "corpus.yaml").documents}
    report = validate_gold(items, doc_ids)
    assert report.ok, report.errors
    assert report.drafts > 0 and report.counts["type"]["unanswerable"] >= 3


def test_valid_item_passes():
    report = validate_gold([_item()], {"speech_fixture"})
    assert report.ok and report.ready == 1 and not report.warnings[:-1]


@pytest.mark.parametrize("kw, msg", [
    ({"type": "trivia"}, "type must be"),
    ({"language": "fr"}, "language"),
    ({"split": "train"}, "split"),
    ({"evidence": []}, "needs evidence"),
    ({"numbers": [], "keywords": []}, "numbers or keywords"),
    ({"evidence": [{"doc_id": "nope", "pages": [1]}]}, "unknown doc_id"),
    ({"evidence": [{"doc_id": "speech_fixture", "pages": []}]}, "no pages"),
    ({"type": "unanswerable"}, "answerable=false"),
    ({"answerable": False}, "answerable=false"),
])
def test_validation_errors(kw, msg):
    report = validate_gold([_item(**kw)], {"speech_fixture"})
    assert any(msg in e for e in report.errors), report.errors


def test_duplicates_and_drafts():
    report = validate_gold([_item(), _item(), _item(id="q2", answer="TODO: fill")],
                           {"speech_fixture"})
    assert any("duplicate" in e for e in report.errors)
    assert report.drafts == 1


def test_low_unanswerable_share_warns():
    report = validate_gold([_item(id=f"q{i}") for i in range(12)], {"speech_fixture"})
    assert any("unanswerable" in w for w in report.warnings)


def test_load_reports_line_numbers(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "a", "type": "factual", "language": "en",
                             "question": "q", "keywords": ["x"]}) + "\n{broken\n")
    with pytest.raises(ValueError, match=":2:"):
        load_gold(p)
