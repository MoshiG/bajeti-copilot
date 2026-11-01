from types import SimpleNamespace

import pytest

from bajeti.config import LLMConfig
from bajeti.llm import ChatResult, FakeLLM, OpenAICompatClient, _parse_arguments


def _fake_openai(message, usage=None):
    """Mimic the shape of an OpenAI SDK client closely enough for the adapter."""
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, captured


def test_plain_answer_is_returned():
    msg = SimpleNamespace(content="tayari", tool_calls=None)
    client, captured = _fake_openai(msg, SimpleNamespace(prompt_tokens=5, completion_tokens=1))
    llm = OpenAICompatClient(LLMConfig(model="m"), client=client)

    result = llm.chat([{"role": "user", "content": "hi"}])

    assert result.content == "tayari" and not result.wants_tools
    assert result.prompt_tokens == 5
    assert captured["model"] == "m" and "tools" not in captured


def test_tool_calls_are_parsed():
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="query_budget", arguments='{"sql": "SELECT 1"}'),
    )
    client, captured = _fake_openai(SimpleNamespace(content=None, tool_calls=[tc]))
    llm = OpenAICompatClient(LLMConfig(), client=client)

    result = llm.chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])

    assert result.wants_tools
    assert result.tool_calls[0].name == "query_budget"
    assert result.tool_calls[0].arguments == {"sql": "SELECT 1"}
    assert captured["tool_choice"] == "auto"


def test_broken_tool_json_fails_soft():
    assert _parse_arguments('{"sql": ') ["_error"] == "invalid_json"
    assert _parse_arguments(None) == {}


def test_fake_llm_scripts_and_records():
    fake = FakeLLM(["first", ChatResult(content="second")])
    assert fake.chat([{"role": "user", "content": "a"}]).content == "first"
    assert fake.chat([{"role": "user", "content": "b"}]).content == "second"
    assert len(fake.calls) == 2
    with pytest.raises(RuntimeError):
        fake.chat([])
