"""Gradio demo. The logic is plain functions (rate limiting, answering, formatting warnings) so
it is tested without Gradio; build_app() is a thin layer over respond().

What the demo shows besides the answer:
  - sources, each with document and page
  - warnings when an answer has no citations, contains figures not found in the evidence,
    had invented citations removed, or ran out of tool steps
  - the tool trace, collapsed, for anyone who wants to see how the answer was found
Questions are not stored. Each visitor is rate limited.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

from bajeti.agent import MAX_QUESTION_CHARS, Agent, Answer

try:  # module level on purpose: Gradio resolves the `gr.Request` hint by name to inject it
    import gradio as gr
except ImportError:  # the demo is optional; the logic below works without it
    gr = None

log = logging.getLogger(__name__)

DISCLAIMER = (
    "**Research demo** over public Ministry of Finance documents (budget speeches, Finance Acts, "
    "expenditure estimates). Not official advice: check every figure in the cited document. "
    "Questions are not stored.  \n"
    "*Kielelezo cha utafiti kinachotumia nyaraka za umma za Wizara ya Fedha. Si ushauri rasmi: "
    "hakiki kila takwimu katika nyaraka iliyotajwa. Maswali hayahifadhiwi.*"
)
EXAMPLES = [   # deliberately not gold questions, so the demo never leaks the test set
    "Ni hatua gani za kodi zilitangazwa katika hotuba ya bajeti ya 2024/25?",
    "Which projects under Vote 46 have the largest development allocations in 2026/27?",
    "Mapato ya ndani yalitarajiwa kufikia kiasi gani mwaka 2025/26?",
    "What does the Finance Act, 2021 change about excise duty?",
]


class RateLimiter:
    """At most `limit` requests per `window_s` seconds per client (sliding window)."""

    def __init__(self, limit: int = 10, window_s: float = 600,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, int]:
        """(allowed, seconds until the next request would be allowed)."""
        now = self.clock()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, int(self.window - (now - hits[0])) + 1
            hits.append(now)
            return True, 0


def warnings_markdown(ans: Answer) -> str:
    notes = []
    if ans.uncited:
        notes.append("⚠️ **No citations.** Do not rely on this answer. / "
                     "*Hakuna marejeo; usitegemee jibu hili.*")
    if ans.unverified_figures:
        figs = ", ".join(f"`{f}`" for f in ans.unverified_figures)
        notes.append(f"⚠️ **Figures not found in the sources:** {figs}. They may be calculated "
                     "or misquoted; check the cited pages. / *Takwimu hizi hazikupatikana katika "
                     "vyanzo; hakiki kurasa zilizotajwa.*")
    if ans.invalid_citations:
        notes.append(f"ℹ️ {len(ans.invalid_citations)} citation(s) that pointed to nothing were "
                     "removed.")
    if ans.hit_step_limit:
        notes.append("ℹ️ The search limit was reached; the answer may be incomplete.")
    return "\n\n".join(notes)


def sources_markdown(ans: Answer) -> str:
    if not ans.citations:
        return ""
    return "**Sources / Vyanzo**\n\n" + "\n".join(
        f"- **[{s.id}]** {s.citation}" for s in ans.citations)


def trace_markdown(ans: Answer, max_chars: int = 700) -> str:
    if not ans.steps:
        return "_No tools were called._"
    parts = []
    for i, s in enumerate(ans.steps, start=1):
        flag = " — error" if s.error else " — repeated" if s.cached else ""
        result = s.result if len(s.result) <= max_chars else s.result[:max_chars] + " …"
        parts.append(f"**{i}. `{s.tool}`**{flag}\n\n`{s.arguments}`\n\n```\n{result}\n```")
    return "\n\n".join(parts)


@dataclass
class Demo:
    agent: Agent
    limiter: RateLimiter

    def respond(self, question: str, client: str = "anonymous") -> tuple[str, str, str]:
        """(answer + warnings, sources, trace) for one question."""
        if not question or not question.strip():
            return "Please type a question. / *Tafadhali andika swali.*", "", ""
        if len(question) > MAX_QUESTION_CHARS:
            return (f"Please keep questions under {MAX_QUESTION_CHARS} characters. / "
                    f"*Swali lisizidi herufi {MAX_QUESTION_CHARS}.*", "", "")
        allowed, wait = self.limiter.check(client)
        if not allowed:
            return (f"Too many questions; try again in {wait} seconds. / "
                    f"*Maswali mengi mno; jaribu tena baada ya sekunde {wait}.*", "", "")
        try:
            ans = self.agent.answer(question)
        except Exception:   # endpoint down, timeout: tell the user, keep details in the log
            log.exception("answering failed")
            return ("The model service is unavailable right now. Please try again later. / "
                    "*Huduma haipatikani kwa sasa; jaribu tena baadaye.*", "", "")
        warn = warnings_markdown(ans)
        body = ans.text + (f"\n\n---\n{warn}" if warn else "")
        return body, sources_markdown(ans), trace_markdown(ans)


def build_app(demo: Demo):
    if gr is None:
        raise ImportError("the demo needs Gradio: pip install -e '.[ui]'")

    def handle(question: str, request: gr.Request = None):
        client = getattr(getattr(request, "client", None), "host", None) or "anonymous"
        return demo.respond(question, client)

    ui_handler = handle   # exposed for tests: Gradio must see `request` to pass client info

    with gr.Blocks(title="Bajeti Copilot") as ui:
        gr.Markdown("# Bajeti Copilot\nAsk about Tanzania's public budget documents, in "
                    "Swahili or English. / *Uliza kuhusu nyaraka za bajeti, kwa Kiswahili au "
                    "Kiingereza.*")
        gr.Markdown(DISCLAIMER)
        question = gr.Textbox(label="Question / Swali", lines=2,
                              max_length=MAX_QUESTION_CHARS)
        ask = gr.Button("Ask / Uliza", variant="primary")
        answer = gr.Markdown()
        sources = gr.Markdown()
        with gr.Accordion("How this answer was found / Jinsi jibu lilivyopatikana", open=False):
            trace = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=question)
        ask.click(handle, inputs=question, outputs=[answer, sources, trace])
        question.submit(handle, inputs=question, outputs=[answer, sources, trace])
    ui.queue(default_concurrency_limit=2)   # protects a single GPU from bursts
    ui.bajeti_handler = ui_handler
    return ui
