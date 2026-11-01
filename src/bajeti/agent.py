"""The copilot agent: a tool-calling loop with verified citations and honest refusals.

    question ─> LLM ─┬─ tool calls ─> run tools ─> results back to LLM ─┐
                     │        ^                                         │
                     │        └──────────── up to max_steps ────────────┘
                     └─ final answer ─> check citations ─> Answer

Guarantees, each enforced in code rather than only requested in the prompt:
  - every [S#] in the answer refers to evidence a tool actually returned; invented ids are
    removed and counted, so evaluation can measure citation hallucination
  - an answer with no citations gets one repair turn; if it still has none it is flagged
  - "not in the documents" is a first-class outcome (Answer.refused), in Swahili or English
  - tool errors go back to the model as text; the loop never crashes on a bad tool call
  - repeated identical tool calls are served from cache instead of looping
  - after max_steps the model must answer with the evidence it has
  - every figure in the answer is checked against the tool results; figures found nowhere in
    the evidence are listed in Answer.unverified_figures (they may be calculated, or misquoted)
  - over-long questions are rejected before any model call
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from bajeti.config import AgentConfig
from bajeti.llm import LLMClient, Message, ToolCall
from bajeti.numbers import unverified_figures
from bajeti.tools import Source, SourceRegistry, Tool, ToolResult

MAX_QUESTION_CHARS = 1000

REFUSAL_EN = "I could not find this in the budget documents"
REFUSAL_SW = "Sikuweza kupata taarifa hii katika nyaraka za bajeti"
CITATION = re.compile(r"\[(S\d+)\]")
GROUPED_CITATION = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)+\s*\]")


def normalize_citations(text: str) -> str:
    """'[S1, S9]' and '[S1; S2]' -> '[S1][S9]', so every id in a group is checked."""
    return GROUPED_CITATION.sub(
        lambda m: "".join(f"[{sid}]" for sid in re.findall(r"S\d+", m.group(0))), text)

SYSTEM_PROMPT = f"""\
You are Bajeti Copilot, an assistant that answers questions about Tanzania's public budget \
documents (budget speeches, Finance Acts and expenditure estimates).

Rules:
1. Answer only from tool results. Never use outside knowledge for figures, dates or measures.
2. Cite every factual sentence with the source ids shown in tool results, like [S2] or [S1][S4].
3. For amounts, totals and comparisons, use query_budget instead of doing arithmetic yourself, \
and select doc_id and page_no so the figures can be cited.
4. For policy, tax measures and explanations, use search_docs. Search in the language most \
likely to match the documents; try the other language if results are poor.
5. For prioritisation under a budget ceiling, first find the costs, then call rank_interventions.
6. Always state the fiscal year and the unit (TZS) of any figure.
7. Reply in the language of the question (Swahili or English).
8. If the documents do not contain the answer, reply with exactly: \
"{REFUSAL_EN}." (English) or "{REFUSAL_SW}." (Swahili), then say briefly what you searched.
"""

REPAIR_PROMPT = (
    "Your answer has no [S#] citations. Rewrite it so every factual sentence cites the tool "
    "results, or, if the results do not support an answer, use the not-found reply."
)
FINAL_PROMPT = (
    "You have reached the tool-call limit. Answer now using only the evidence above, with "
    "citations, or use the not-found reply."
)


@dataclass
class Step:
    tool: str
    arguments: dict[str, Any]
    result: str
    error: bool
    cached: bool = False


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Source] = field(default_factory=list)   # in order of first use
    refused: bool = False
    steps: list[Step] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited: bool = False            # final answer still had no citations after repair
    hit_step_limit: bool = False
    unverified_figures: list[str] = field(default_factory=list)
    rejected: bool = False           # input refused before reaching the model
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def sources_markdown(self) -> str:
        return "\n".join(f"[{s.id}] {s.citation}" for s in self.citations)


def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    return t.startswith(REFUSAL_EN.lower()) or t.startswith(REFUSAL_SW.lower())


class Agent:
    def __init__(self, llm: LLMClient, tools: list[Tool], cfg: AgentConfig) -> None:
        self.llm, self.cfg = llm, cfg
        self.tools = {t.name: t for t in tools}

    def _run_tool(self, call: ToolCall, reg: SourceRegistry,
                  cache: dict[str, ToolResult]) -> tuple[ToolResult, bool]:
        key = json.dumps([call.name, call.arguments], sort_keys=True, default=str)
        if key in cache:
            return cache[key], True
        tool = self.tools.get(call.name)
        if tool is None:
            result = ToolResult(f"Error: unknown tool '{call.name}'. Available: "
                                f"{', '.join(self.tools)}.", error=True)
        elif "_error" in call.arguments:
            result = ToolResult("Error: the tool arguments were not valid JSON. Call the tool "
                                "again with a JSON object matching its parameters.", error=True)
        else:
            try:
                result = tool.run(call.arguments, reg)
            except Exception as exc:  # last line of defence; tools are written not to raise
                result = ToolResult(f"Error: tool failed: {exc}", error=True)
        cache[key] = result
        return result, False

    def _chat(self, messages: list[Message], answer: Answer, with_tools: bool):
        specs = [t.spec() for t in self.tools.values()] if with_tools else None
        result = self.llm.chat(messages, specs)
        answer.llm_calls += 1
        answer.prompt_tokens += result.prompt_tokens
        answer.completion_tokens += result.completion_tokens
        return result

    def answer(self, question: str) -> Answer:
        answer = Answer(question=question, text="")
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            answer.rejected = True
            answer.text = (f"Please ask a question of 1 to {MAX_QUESTION_CHARS} characters. / "
                           f"Tafadhali uliza swali lenye herufi 1 hadi {MAX_QUESTION_CHARS}.")
            return answer
        reg, cache = SourceRegistry(), {}
        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": question}]
        final: str | None = None

        for _ in range(self.cfg.max_steps):
            result = self._chat(messages, answer, with_tools=True)
            if not result.wants_tools:
                final = result.content or ""
                break
            messages.append({"role": "assistant", "content": result.content, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in result.tool_calls]})
            for call in result.tool_calls:
                out, cached = self._run_tool(call, reg, cache)
                content = out.content + ("\n(Same call as before; result repeated.)"
                                         if cached else "")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                answer.steps.append(Step(call.name, call.arguments, out.content, out.error,
                                         cached))
        else:
            answer.hit_step_limit = True
            messages.append({"role": "user", "content": FINAL_PROMPT})
            final = self._chat(messages, answer, with_tools=False).content or ""

        answer.refused = is_refusal(final)
        if (self.cfg.require_citations and not answer.refused and reg.sources
                and not CITATION.search(normalize_citations(final))):
            messages += [{"role": "assistant", "content": final},
                         {"role": "user", "content": REPAIR_PROMPT}]
            final = self._chat(messages, answer, with_tools=False).content or ""
            answer.refused = is_refusal(final)

        answer.text, answer.citations, answer.invalid_citations = _resolve_citations(final, reg)
        answer.uncited = (self.cfg.require_citations and not answer.refused
                          and not answer.citations)
        if not answer.refused:
            evidence = [s.result for s in answer.steps if not s.error] + [question]
            answer.unverified_figures = unverified_figures(answer.text, evidence)
        return answer


def _resolve_citations(text: str, reg: SourceRegistry) -> tuple[str, list[Source], list[str]]:
    """Keep citations that point to real evidence; strip and report the rest."""
    text = normalize_citations(text)
    used: list[Source] = []
    invalid: list[str] = []
    for sid in CITATION.findall(text):
        if sid in reg:
            if reg.sources[sid] not in used:
                used.append(reg.sources[sid])
        elif sid not in invalid:
            invalid.append(sid)
    for sid in invalid:
        text = text.replace(f"[{sid}]", "")
    return re.sub(r"[ \t]+([.,;:])", r"\1", text).strip(), used, invalid


def build_agent(cfg, llm: LLMClient | None = None, retriever=None) -> Agent:
    from bajeti.llm import build_llm
    from bajeti.retrieval import build_retriever
    from bajeti.tools import build_tools

    retriever = retriever or build_retriever(cfg)
    return Agent(llm or build_llm(cfg.llm), build_tools(cfg, retriever), cfg.agent)
