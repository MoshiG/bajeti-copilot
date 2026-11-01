"""Command-line entry point. Later parts add ingest, index, ask and eval subcommands."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from bajeti.config import load_config
from bajeti.llm import build_llm


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(*args.config)
    print(json.dumps(cfg.to_dict(), indent=2))
    return 0


def _cmd_ping(args: argparse.Namespace) -> int:
    """Check the configured LLM endpoint answers — run this first on a fresh GPU pod."""
    cfg = load_config(*args.config)
    llm = build_llm(cfg.llm)
    result = llm.chat([{"role": "user", "content": "Reply with the single word: tayari"}])
    print(f"{cfg.llm.model} @ {cfg.llm.base_url} -> {result.content!r}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from bajeti.corpus import fetch_all, load_manifest

    cfg = load_config(*args.config)
    results = fetch_all(load_manifest(args.manifest), cfg.paths.raw_dir, set(args.only or []))
    for r in results:
        size = f"{r.bytes / 1e6:.1f} MB" if r.bytes else ""
        print(f"{r.status:15} {r.id:22} {size:>9}  {r.message}")
    failed = [r for r in results if r.status == "failed"]
    manual = [r for r in results if r.status == "manual_missing"]
    if manual:
        print(f"\n{len(manual)} document(s) need a manual download (see messages above).")
    return 1 if failed else 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from bajeti.corpus import load_manifest
    from bajeti.ingest.pipeline import ingest_corpus

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(*args.config)
    reports = ingest_corpus(cfg, load_manifest(args.manifest), set(args.only or []), args.force)
    print(f"{'document':22} {'status':8} {'pages':>5} {'ocr':>4} {'tables':>6} "
          f"{'naive':>6} {'struct':>6}")
    for r in reports:
        print(f"{r.doc_id:22} {r.status:8} {r.pages:>5} {r.ocr_pages:>4} {r.tables:>6} "
              f"{r.chunks.get('naive', 0):>6} {r.chunks.get('structured', 0):>6}  {r.message}")
    return 1 if any(r.status == "failed" for r in reports) else 0


def _cmd_index(args: argparse.Namespace) -> int:
    from bajeti.corpus import load_manifest
    from bajeti.index.build import build_indexes
    from bajeti.index.dense import build_embedder

    cfg = load_config(*args.config)
    embedder = build_embedder(args.embedder or cfg.retrieval.embedding_model)
    report = build_indexes(cfg, load_manifest(args.manifest), embedder)
    print(f"embedding model: {report['embedding_model']}")
    for strategy, r in report["dense"].items():
        detail = (f"{r['chunks']} chunks, dim {r['dim']}, {r['seconds']}s"
                  if r["status"] == "ok" else r["message"])
        print(f"  dense/{strategy:11} {r['status']:8} {detail}")
    db = report["budget_db"]
    print(f"  budget.sqlite        {db['tables']} tables, {db['cells']} cells, "
          f"{db['allocation_rows']} allocation rows" if db else "  budget.sqlite        skipped")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Show what retrieval returns for a query, with each hit's rank in every stage."""
    from bajeti.index.dense import build_embedder
    from bajeti.retrieval import Filters, build_reranker, build_retriever

    cfg = load_config(*args.config)
    embedder = build_embedder(args.embedder) if args.embedder else None
    reranker = build_reranker(args.reranker) if args.reranker else None
    retriever = build_retriever(cfg, embedder, reranker)
    hits = retriever.search(args.query, args.k,
                            Filters(args.year, args.doc_type, args.language))
    if not hits:
        print("no results")
        return 1
    for i, h in enumerate(hits, start=1):
        ranks = " ".join(f"{k}={v}" for k, v in h.ranks.items())
        c = h.chunk
        print(f"{i:>2}. {c.citation} [{c.kind}] score={h.score:.3f} ({ranks})")
        if c.section:
            print(f"    section: {c.section}")
        body = c.text.split("\n", 1)[-1].replace("\n", " ")
        print(f"    {body[:160]}{'…' if len(body) > 160 else ''}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question end to end, printing sources and (optionally) the tool trace."""
    from bajeti.agent import build_agent
    from bajeti.index.dense import build_embedder
    from bajeti.retrieval import build_reranker, build_retriever

    cfg = load_config(*args.config)
    embedder = build_embedder(args.embedder) if args.embedder else None
    reranker = build_reranker(args.reranker) if args.reranker else None
    agent = build_agent(cfg, retriever=build_retriever(cfg, embedder, reranker))
    answer = agent.answer(args.question)
    if args.trace:
        for i, s in enumerate(answer.steps, start=1):
            flag = " (error)" if s.error else " (cached)" if s.cached else ""
            print(f"--- step {i}: {s.tool}({json.dumps(s.arguments, ensure_ascii=False)}){flag}")
            print(s.result[:600] + ("…" if len(s.result) > 600 else ""))
        print("---")
    print(answer.text)
    if answer.citations:
        print("\nSources:\n" + answer.sources_markdown())
    if answer.invalid_citations:
        print(f"\n(removed invented citations: {', '.join(answer.invalid_citations)})")
    return 0


def _factories(args: argparse.Namespace, with_answers: bool):
    """Build retriever/agent factories that load each model once and reuse it across configs."""
    from bajeti.agent import build_agent
    from bajeti.index.dense import build_embedder
    from bajeti.retrieval import build_reranker, build_retriever

    embedders: dict[str, object] = {}
    rerankers: dict[str, object] = {}

    def retriever_factory(cfg):
        # Note: not dict.setdefault(name, build(...)), which would load the model every time.
        embedder = reranker = None
        if cfg.retrieval.use_dense:
            e_name = args.embedder or cfg.retrieval.embedding_model
            if e_name not in embedders:
                embedders[e_name] = build_embedder(e_name)
            embedder = embedders[e_name]
        if cfg.retrieval.use_reranker:
            r_name = args.reranker or cfg.retrieval.reranker_model
            if r_name not in rerankers:
                rerankers[r_name] = build_reranker(r_name)
            reranker = rerankers[r_name]
        return build_retriever(cfg, embedder, reranker)

    def agent_factory(cfg, retriever):
        return build_agent(cfg, retriever=retriever)

    return retriever_factory, (agent_factory if with_answers else None)


def _print_summary(name: str, s: dict) -> None:
    rt = s["retrieval"]
    k = rt["k"] or "k"
    print(f"{name}: hit@1 {rt['hit@1']['rate']:.0%}, hit@{k} {rt[f'hit@{k}']['rate']:.0%}, "
          f"MRR {rt['mrr']:.2f}, lost: {rt['diagnosis']}")
    if "answers" in s:
        a = s["answers"]
        print(f"{' ' * len(name)}  correct {a['correct']['rate']:.0%} "
              f"(95% CI {a['correct']['ci_low']:.0%}–{a['correct']['ci_high']:.0%}), "
              f"citation hit {a['citation_hit']['rate']:.0%}, "
              f"refusal acc {a['refusal_accuracy']['rate']:.0%}, "
              f"false refusals {a['false_refusals']['rate']:.0%}")


def _build_judge(cfg):
    from bajeti.evaluation.judge import Judge
    from bajeti.llm import build_llm

    if cfg.judge is None:
        raise SystemExit("--judge needs a `judge:` section in the config (see default.yaml)")
    return Judge(build_llm(cfg.judge))


def _select(gold, args):
    if args.split != "all":
        gold = [g for g in gold if g.split == args.split]
    return gold[: args.limit]


def _cmd_gold_check(args: argparse.Namespace) -> int:
    from bajeti.corpus import load_manifest
    from bajeti.evaluation.gold import load_gold, validate_gold

    cfg = load_config(*args.config)
    items = load_gold(args.gold or cfg.paths.gold_set)
    doc_ids = {d.id for d in load_manifest(args.manifest).documents}
    report = validate_gold(items, doc_ids)
    print(f"{len(items)} questions: {report.ready} ready, {report.drafts} drafts (TODO)")
    print(f"by type: {dict(report.counts['type'])}; by language: "
          f"{dict(report.counts['language'])}; by split: {dict(report.counts['split'])}")
    for w in report.warnings:
        print(f"warning: {w}")
    for e in report.errors:
        print(f"error:   {e}")
    return 0 if report.ok else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    from bajeti.evaluation.gold import load_gold
    from bajeti.evaluation.runner import run_config

    cfg = load_config(*args.config)
    gold = _select(load_gold(args.gold or cfg.paths.gold_set), args)
    rf, af = _factories(args, with_answers=not args.retrieval_only)
    judge = _build_judge(cfg) if args.judge else None
    summary = run_config(cfg, gold, rf, af, resume=args.resume,
                         progress=None if args.quiet else print, judge=judge)
    _print_summary(cfg.name, summary)
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from bajeti.evaluation.gold import load_gold
    from bajeti.evaluation.runner import run_ablations

    base = args.config                       # every -c layers into the base; overlays go on top
    cfg = load_config(*base)
    gold = _select(load_gold(args.gold or cfg.paths.gold_set), args)
    files = sorted(Path(args.ablations).glob("*.yaml"))
    rf, af = _factories(args, with_answers=not args.retrieval_only)
    judge = _build_judge(cfg) if args.judge else None
    summaries = run_ablations(base, files, gold, rf, af, resume=args.resume,
                              progress=None if args.quiet else print, judge=judge)
    for name, s in summaries.items():
        _print_summary(name, s)
    table = "ablation_retrieval.md" if args.retrieval_only else "ablation.md"
    print(f"\ntable written to {Path(cfg.paths.results_dir) / table}")
    return 0


def _cmd_export_claims(args: argparse.Namespace) -> int:
    from pathlib import Path

    from bajeti.evaluation.judge import sample_claims
    from bajeti.evaluation.runner import records_for_review

    cfg = load_config(*args.config)
    path = Path(cfg.paths.results_dir) / args.run / "records.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `bajeti ... eval` for {args.run} first")
    claims = sample_claims(records_for_review(path, only_failures=False), args.n)
    if not claims:
        raise SystemExit(f"no cited claims in {path}; nothing to label")
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite your labels")
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in claims),
                   encoding="utf-8")
    print(f"wrote {len(claims)} claims to {out}; set each human_label to "
          "supported, partial or unsupported")
    return 0


def _cmd_judge_validate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from bajeti.evaluation.judge import validate_judge, validation_markdown

    cfg = load_config(*args.config)
    judge = _build_judge(cfg)                  # fail fast if no judge is configured
    with open(args.labels, encoding="utf-8") as fh:
        labelled = [json.loads(line) for line in fh if line.strip()]
    result = validate_judge(judge, labelled)
    out = Path(cfg.paths.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge_validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    md = validation_markdown(result)
    (out / "judge_validation.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from bajeti.agent import build_agent
    from bajeti.app import Demo, RateLimiter, build_app
    from bajeti.index.dense import build_embedder
    from bajeti.retrieval import build_reranker, build_retriever

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(*args.config)
    embedder = build_embedder(args.embedder) if args.embedder else None
    reranker = build_reranker(args.reranker) if args.reranker else None
    agent = build_agent(cfg, retriever=build_retriever(cfg, embedder, reranker))
    demo = Demo(agent, RateLimiter(args.rate_limit, args.rate_window))
    build_app(demo).launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bajeti", description="Bajeti Copilot")
    parser.add_argument(
        "-c", "--config", action="append", default=None,
        help="YAML config; repeat to layer overlays, e.g. -c default.yaml -c ablations/a2.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print the resolved configuration").set_defaults(func=_cmd_config)
    sub.add_parser("ping", help="check the LLM endpoint responds").set_defaults(func=_cmd_ping)

    for name, func, helptext in [
        ("fetch", _cmd_fetch, "download the corpus listed in data/corpus.yaml"),
        ("ingest", _cmd_ingest, "parse PDFs and write naive + structured chunks"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--manifest", default="data/corpus.yaml")
        p.add_argument("--only", nargs="*", help="limit to these document ids")
        p.set_defaults(func=func)
        if name == "ingest":
            p.add_argument("--force", action="store_true", help="re-parse cached pages")

    p = sub.add_parser("index", help="build dense vectors and the SQLite budget database")
    p.add_argument("--manifest", default="data/corpus.yaml")
    p.add_argument("--embedder", help="override the embedding model, e.g. 'hashing' for a "
                                      "quick CPU smoke run")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("search", help="inspect retrieval results for a query")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None, help="number of hits (default: top_k)")
    p.add_argument("--year", help="fiscal year filter, e.g. 2025/26")
    p.add_argument("--doc-type", choices=["speech", "finance_act", "estimates"])
    p.add_argument("--language", choices=["sw", "en"])
    p.add_argument("--embedder", help="override embedder, e.g. 'hashing'")
    p.add_argument("--reranker", help="override reranker, e.g. 'overlap'")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("ask", help="answer a question with tools and citations")
    p.add_argument("question")
    p.add_argument("--trace", action="store_true", help="print every tool call and result")
    p.add_argument("--embedder", help="override embedder, e.g. 'hashing'")
    p.add_argument("--reranker", help="override reranker, e.g. 'overlap'")
    p.set_defaults(func=_cmd_ask)

    p = sub.add_parser("gold-check", help="validate the gold question set")
    p.add_argument("--gold", help="path to gold.jsonl (default: paths.gold_set)")
    p.add_argument("--manifest", default="data/corpus.yaml")
    p.set_defaults(func=_cmd_gold_check)

    for name, func, helptext in [
        ("eval", _cmd_eval, "evaluate one configuration on the gold set"),
        ("ablate", _cmd_ablate, "evaluate every config in configs/ablations/"),
    ]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--gold", help="path to gold.jsonl (default: paths.gold_set)")
        p.add_argument("--retrieval-only", action="store_true",
                       help="score retrieval only: no LLM, runs on a laptop")
        p.add_argument("--resume", action="store_true", help="skip questions already scored")
        p.add_argument("--limit", type=int, default=None, help="first N questions only")
        p.add_argument("--split", choices=["dev", "test", "all"], default="all",
                       help="dev while tuning prompts; test for the numbers you report")
        p.add_argument("--embedder", help="override embedder, e.g. 'hashing'")
        p.add_argument("--reranker", help="override reranker, e.g. 'overlap'")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--judge", action="store_true",
                       help="score each cited claim with the judge model (config: judge)")
        if name == "ablate":
            p.add_argument("--ablations", default="configs/ablations")
        p.set_defaults(func=func)

    p = sub.add_parser("export-claims", help="sample answer claims for human labelling")
    p.add_argument("--run", default="a5_full", help="config whose records to sample")
    p.add_argument("-n", type=int, default=30)
    p.add_argument("--out", default="eval/judge_labels.jsonl")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_export_claims)

    p = sub.add_parser("judge-validate", help="compare the judge with your human labels")
    p.add_argument("--labels", default="eval/judge_labels.jsonl")
    p.set_defaults(func=_cmd_judge_validate)

    p = sub.add_parser("demo", help="launch the Gradio demo")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside Docker")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public link")
    p.add_argument("--rate-limit", type=int, default=10, help="questions per visitor per window")
    p.add_argument("--rate-window", type=float, default=600, help="window in seconds")
    p.add_argument("--embedder", help="override embedder, e.g. 'hashing'")
    p.add_argument("--reranker", help="override reranker, e.g. 'overlap'")
    p.set_defaults(func=_cmd_demo)

    args = parser.parse_args(argv)
    args.config = args.config or ["configs/default.yaml"]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
