"""Run the gold set through a configuration, or through all five ablation configurations.

Per configuration, two layers are measured:
  retrieval  the retriever alone (no LLM): hit@1/3/k, MRR, and where gold evidence was lost:
             not_retrieved (no retriever found it) or ranked_out (found, but not in the top k)
  answers    the full agent: correctness, citation hit/precision, invented citations,
             refusal accuracy on unanswerable questions, false refusals, tool calls, latency

Retrieval-only runs need no LLM, so they are free to repeat on a laptop. Answer runs write one
record per question as they go, and --resume skips finished questions, so a GPU session that
drops halfway loses nothing.

eval/results/<config>/records.jsonl   one line per question (answer, citations, diagnostics)
eval/results/<config>/summary.json    aggregated metrics with 95% intervals
eval/results/ablation.md              the comparison table for the README
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bajeti.agent import Agent
from bajeti.config import Config, load_config
from bajeti.evaluation.gold import GoldItem
from bajeti.evaluation.judge import Judge
from bajeti.evaluation.metrics import first_relevant_rank, matches_evidence, rate, score_answer
from bajeti.retrieval import Retriever

AgentFactory = Callable[[Config, Retriever], Agent]
RetrieverFactory = Callable[[Config], Retriever]


def retrieval_record(item: GoldItem, retriever: Retriever) -> dict[str, Any]:
    top_k = retriever.cfg.top_k
    hits = retriever.search(item.question, top_k=retriever.cfg.candidates)
    meta = [(h.chunk.doc_id, h.chunk.page_start, h.chunk.page_end) for h in hits]
    rank = first_relevant_rank(meta, item.evidence)
    lost = "retrieved" if 0 < rank <= top_k else "ranked_out" if rank else "not_retrieved"
    return {"rank": rank, "top_k": top_k, "diagnosis": lost,
            "top_hits": [h.chunk.citation for h in hits[:3]]}


def answer_record(item: GoldItem, agent: Agent, judge: Judge | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    ans = agent.answer(item.question)
    seconds = time.perf_counter() - started
    rec: dict[str, Any] = {
        "answer": ans.text, "refused": ans.refused, "uncited": ans.uncited,
        "invalid_citations": len(ans.invalid_citations), "hit_step_limit": ans.hit_step_limit,
        "unverified_figures": ans.unverified_figures,
        "tool_calls": len(ans.steps), "tools": [s.tool for s in ans.steps],
        "tool_errors": sum(s.error for s in ans.steps), "llm_calls": ans.llm_calls,
        "prompt_tokens": ans.prompt_tokens, "completion_tokens": ans.completion_tokens,
        "seconds": round(seconds, 2),
        "citations": [{"id": c.id, "citation": c.citation, "doc_id": c.doc_id, "page": c.page,
                       "excerpt": c.excerpt} for c in ans.citations],
    }
    if judge and item.answerable and not ans.refused:
        claims = judge.judge_answer(ans.text, {c.id: c.excerpt for c in ans.citations})
        rec["claims"] = claims
        rec["claims_supported"] = (sum(c["label"] == "supported" for c in claims) / len(claims)
                                   if claims else None)
    if item.answerable:
        rec.update(score_answer(ans.text, item.numbers, item.keywords))
        rec["correct"] = rec["correct"] and not ans.refused
        good = [matches_evidence(c.doc_id, c.page, c.page, item.evidence) for c in ans.citations]
        rec["citation_hit"] = any(good)
        rec["citation_precision"] = sum(good) / len(good) if good else 0.0
    else:
        rec["correct"] = ans.refused
    return rec


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def run_config(
    cfg: Config,
    gold: list[GoldItem],
    retriever_factory: RetrieverFactory,
    agent_factory: AgentFactory | None = None,
    results_dir: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    judge: Judge | None = None,
) -> dict[str, Any]:
    """Evaluate one configuration. agent_factory=None means retrieval-only."""
    out = Path(results_dir or cfg.paths.results_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    kind = "records" if agent_factory else "retrieval"
    records_path = out / f"{kind}.jsonl"

    items = [g for g in gold if not g.is_draft]
    retriever = retriever_factory(cfg)
    agent = agent_factory(cfg, retriever) if agent_factory else None
    models = actual_models(retriever)
    fingerprint = run_fingerprint(cfg, models, agent is not None, judge)
    fp_path = out / f"{kind}.run.json"
    qhashes = {i.id: question_hash(i) for i in items}
    if resume and records_path.exists():
        previous = json.loads(fp_path.read_text()) if fp_path.exists() else None
        if previous != fingerprint:
            raise ValueError(f"{records_path} was produced with different settings or models; "
                             "rerun without --resume")
        # Keep only answers to questions that still exist unchanged; edited ones are re-asked.
        done = {qid: r for qid, r in _load_records(records_path).items()
                if r.get("qhash") == qhashes.get(qid)}
    else:
        done = {}
    with open(records_path, "w", encoding="utf-8") as fh:   # drop stale or removed records
        for item in items:
            if item.id in done:
                fh.write(json.dumps(done[item.id], ensure_ascii=False) + "\n")
    fp_path.write_text(json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(records_path, "a", encoding="utf-8") as fh:
        for n, item in enumerate(items, start=1):
            if item.id in done:
                continue
            rec: dict[str, Any] = {"id": item.id, "type": item.type, "language": item.language,
                                   "answerable": item.answerable, "qhash": qhashes[item.id]}
            if item.answerable:
                rec["retrieval"] = retrieval_record(item, retriever)
            if agent:
                rec.update(answer_record(item, agent, judge))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()                    # a dropped GPU session keeps finished questions
            done[item.id] = rec
            if progress:
                progress(f"[{cfg.name}] {n}/{len(items)} {item.id}")

    records = [done[i.id] for i in items if i.id in done]
    summary = summarise(records, with_answers=agent is not None)
    summary["meta"] = {
        "config": cfg.name, "model": cfg.llm.model if agent else None,
        "embedding_model": models["embedding_model"],   # what actually ran, overrides included
        "reranker_model": models["reranker_model"],
        "settings": {"chunking": cfg.ingest.chunking, "bm25": cfg.retrieval.use_bm25,
                     "dense": cfg.retrieval.use_dense, "reranker": cfg.retrieval.use_reranker,
                     "sql": cfg.tools.enable_sql, "optimizer": cfg.tools.enable_optimizer},
        "questions": len(items), "skipped_drafts": len(gold) - len(items),
        "judge_model": judge_model(judge),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
    }
    (out / ("summary.json" if agent else "retrieval_summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(records: list[dict[str, Any]], with_answers: bool) -> dict[str, Any]:
    answerable = [r for r in records if r["answerable"]]
    ranks = [r["retrieval"]["rank"] for r in answerable]
    k = answerable[0]["retrieval"]["top_k"] if answerable else None
    s: dict[str, Any] = {"retrieval": {
        "hit@1": rate([0 < x <= 1 for x in ranks]),
        "hit@3": rate([0 < x <= 3 for x in ranks]),
        f"hit@{k or 'k'}": rate([0 < x <= k for x in ranks]),
        "mrr": _mean([1 / x if 0 < x <= k else 0.0 for x in ranks]),
        "diagnosis": {d: sum(r["retrieval"]["diagnosis"] == d for r in answerable)
                      for d in ("retrieved", "ranked_out", "not_retrieved")},
        "k": k,
    }}
    if not with_answers:
        return s
    unanswerable = [r for r in records if not r["answerable"]]
    s["answers"] = {
        "correct": rate([r["correct"] for r in records]),
        "correct_answerable": rate([r["correct"] for r in answerable]),
        "numbers_ok": rate([r["numbers_ok"] for r in answerable]),
        "citation_hit": rate([r["citation_hit"] for r in answerable]),
        "citation_precision": _mean([r["citation_precision"] for r in answerable]),
        "invalid_citation_answers": rate([r["invalid_citations"] > 0 for r in records]),
        "uncited": rate([r["uncited"] for r in answerable]),
        "unverified_figure_answers": rate([bool(r.get("unverified_figures")) for r in records]),
        "refusal_accuracy": rate([r["refused"] for r in unanswerable]),
        "false_refusals": rate([r["refused"] for r in answerable]),
        "avg_tool_calls": _mean([r["tool_calls"] for r in records]),
        "tool_error_rate": _mean([r["tool_errors"] / r["tool_calls"] if r["tool_calls"] else 0
                                  for r in records]),
        "avg_seconds": _mean([r["seconds"] for r in records]),
        "avg_tokens": _mean([r["prompt_tokens"] + r["completion_tokens"] for r in records]),
    }
    claims = [c for r in records for c in r.get("claims", [])]
    if claims:
        judged = [r for r in records if r.get("claims")]
        s["answers"]["judge"] = {
            "claims_supported": rate([c["label"] == "supported" for c in claims]),
            "claims_unsupported": rate([c["label"] == "unsupported" for c in claims]),
            "claims_unparseable": rate([c["label"] == "unparseable" for c in claims]),
            "claims_uncited": rate([c["label"] == "uncited" for c in claims]),
            "answers_fully_supported": rate([all(c["label"] == "supported" for c in r["claims"])
                                             for r in judged]),
        }
    by: dict[str, dict[str, Any]] = {}
    for key in ("type", "language"):
        groups: dict[str, list[bool]] = {}
        for r in records:
            groups.setdefault(r[key], []).append(r["correct"])
        by[key] = {g: rate(v) for g, v in sorted(groups.items())}
    s["answers"]["by"] = by
    return s


def run_ablations(
    base_config: str | Path | list[str | Path],
    ablation_files: list[Path],
    gold: list[GoldItem],
    retriever_factory: RetrieverFactory,
    agent_factory: AgentFactory | None = None,
    results_dir: str | Path | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    judge: Judge | None = None,
) -> dict[str, dict[str, Any]]:
    bases = list(base_config) if isinstance(base_config, list | tuple) else [base_config]
    summaries = {}
    for path in sorted(ablation_files):
        cfg = load_config(*bases, path)
        summaries[cfg.name] = run_config(cfg, gold, retriever_factory, agent_factory,
                                         results_dir, resume, progress, judge)
    out = Path(results_dir or load_config(*bases).paths.results_dir)
    table = ablation_table(summaries)
    (out / ("ablation.md" if agent_factory else "ablation_retrieval.md")).write_text(
        table + "\n", encoding="utf-8")
    return summaries


def _pct(r: dict[str, Any], ci: bool = False) -> str:
    if not r.get("n", 1):
        return "–"                                  # no questions of this kind
    text = f"{r['rate'] * 100:.0f}%"
    return f"{text} ({r['ci_low'] * 100:.0f}–{r['ci_high'] * 100:.0f})" if ci else text


def actual_models(retriever: Retriever) -> dict[str, str | None]:
    dense = retriever.indexes.dense
    return {"embedding_model": dense.embedder.name if dense else None,
            "reranker_model": retriever.reranker.name if retriever.cfg.use_reranker else None}


def question_hash(item: GoldItem) -> str:
    return hashlib.sha256(json.dumps(asdict(item), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


def run_fingerprint(cfg: Config, models: dict[str, str | None], with_answers: bool,
                    judge: Judge | None) -> dict[str, Any]:
    """Settings and models behind a run; --resume refuses to mix runs that differ.
    Gold edits are handled per question (see question_hash)."""
    settings = cfg.to_dict()
    settings.pop("paths", None)
    if settings.get("llm"):
        settings["llm"].pop("base_url", None)       # same model served elsewhere is fine
        settings["llm"].pop("api_key_env", None)
    return {"settings": settings, "models": models, "answers": with_answers,
            "judge": judge_model(judge)}


def judge_model(judge: Judge | None) -> str | None:
    cfg = getattr(getattr(judge, "llm", None), "cfg", None)
    return getattr(cfg, "model", None) if judge else None


def ablation_table(summaries: dict[str, dict[str, Any]]) -> str:
    with_answers = all("answers" in s for s in summaries.values())
    with_judge = with_answers and any("judge" in s["answers"] for s in summaries.values())
    k = next(iter(summaries.values()))["retrieval"]["k"] if summaries else None
    k = k or "k"
    head = ["Config", "Chunking", "BM25", "Dense", "Rerank", "Tools", "Hit@1", f"Hit@{k}", "MRR"]
    if with_answers:
        head += ["Correct, answerable (95% CI)", "Citation hit", "Invented cites",
                 "Unverified figs", "Refusal acc", "False refusals", "Avg s"]
    if with_judge:
        head += ["Fully supported answers (judge)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, s in summaries.items():
        st, rt = s["meta"]["settings"], s["retrieval"]
        tools = "+".join(t for t, on in (("SQL", st["sql"]), ("opt", st["optimizer"])) if on)
        row = [name, st["chunking"], "✓" if st["bm25"] else "", "✓" if st["dense"] else "",
               "✓" if st["reranker"] else "", tools or "search only",
               _pct(rt["hit@1"]), _pct(rt[f"hit@{k}"]), f"{rt['mrr']:.2f}"]
        if with_answers:
            a = s["answers"]
            row += [_pct(a["correct_answerable"], ci=True), _pct(a["citation_hit"]),
                    _pct(a["invalid_citation_answers"]), _pct(a["unverified_figure_answers"]),
                    _pct(a["refusal_accuracy"]),
                    _pct(a["false_refusals"]), f"{a['avg_seconds']:.1f}"]
            if with_judge:
                row += [_pct(a["judge"]["answers_fully_supported"], ci=True)
                        if "judge" in a else "–"]
        lines.append("| " + " | ".join(row) + " |")
    n = next(iter(summaries.values()))["meta"]["questions"] if summaries else 0
    lines.append("")
    note = (" Correctness covers answerable questions; refusal accuracy covers unanswerable "
            "ones." if with_answers else "")
    lines.append(f"{n} gold questions.{note} Intervals are Wilson 95%; differences inside "
                 "overlapping intervals may be noise.")
    return "\n".join(lines)


def records_for_review(path: str | Path, only_failures: bool = True) -> list[dict[str, Any]]:
    """Load records for error analysis (e.g. wrong answers with their retrieval diagnosis)."""
    recs = list(_load_records(Path(path)).values())
    return [r for r in recs if not r.get("correct", True)] if only_failures else recs

