import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench2  # noqa: E402


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_urlopen(body, seen):
    def _urlopen(req, timeout=None):
        seen.append(json.loads(req.data))
        return _Resp(json.dumps(body).encode())
    return _urlopen


def _body(message):
    return {"choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 5}}


def test_payload_carries_effort_when_set(monkeypatch):
    seen = []
    monkeypatch.setattr(bench2, "REASONING_EFFORT", "xhigh")
    monkeypatch.setattr(bench2, "urlopen", _fake_urlopen(_body({"content": "ok", "reasoning_content": "abc"}), seen))
    r = bench2.api_call([{"role": "user", "content": "x"}], model="m")
    assert seen[0]["reasoning_effort"] == "xhigh"
    assert r["content"] == "ok"
    assert r["reasoning_chars"] == 3


def test_payload_unchanged_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr(bench2, "REASONING_EFFORT", None)
    monkeypatch.setattr(bench2, "SAMPLING", {})
    monkeypatch.setattr(bench2, "urlopen", _fake_urlopen(_body({"content": "ok"}), seen))
    r = bench2.api_call([{"role": "user", "content": "x"}], model="m")
    assert set(seen[0]) == {"model", "messages", "temperature", "max_tokens"}
    assert seen[0]["temperature"] == 0
    assert r["reasoning_chars"] == 0


def test_payload_carries_vendor_sampling(monkeypatch):
    seen = []
    vendor = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
    monkeypatch.setattr(bench2, "TEMPERATURE", 1.0)
    monkeypatch.setattr(bench2, "SAMPLING", vendor)
    monkeypatch.setattr(bench2, "urlopen", _fake_urlopen(_body({"content": "ok"}), seen))
    bench2.api_call([{"role": "user", "content": "x"}], model="m")
    assert seen[0]["temperature"] == 1.0
    assert {k: seen[0][k] for k in vendor} == vendor


def test_sampling_from_args_drops_unset():
    assert bench2.sampling_from_args(top_p=None, top_k=20, min_p=0.0, presence_penalty=None,
                                     repetition_penalty=None) == {"top_k": 20, "min_p": 0.0}


def test_split_reasoning_reads_reasoning_alias():
    assert bench2.split_reasoning({"content": "ok", "reasoning": "zz"}) == ("ok", "zz")


def test_split_reasoning_strips_inline_think():
    content, reasoning = bench2.split_reasoning({"content": "<think>abcd</think>\n```python\nx = 1\n```"})
    assert content == "```python\nx = 1\n```"
    assert reasoning == "abcd"


def test_split_reasoning_handles_null_content():
    assert bench2.split_reasoning({"content": None, "reasoning_content": "r"}) == ("", "r")


def test_error_path_reports_zero_reasoning(monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("down")
    monkeypatch.setattr(bench2, "urlopen", _boom)
    r = bench2.api_call([{"role": "user", "content": "x"}], model="m")
    assert r["finish_reason"] == "error" and r["reasoning_chars"] == 0


import tool_call_bench  # noqa: E402


def test_tool_bench_extra_body(monkeypatch):
    monkeypatch.setattr(tool_call_bench, "REASONING_EFFORT", None)
    monkeypatch.setattr(tool_call_bench, "SAMPLING_EXTRA", {})
    assert tool_call_bench.extra_body() is None
    monkeypatch.setattr(tool_call_bench, "REASONING_EFFORT", "xhigh")
    assert tool_call_bench.extra_body() == {"reasoning_effort": "xhigh"}
    monkeypatch.setattr(tool_call_bench, "SAMPLING_EXTRA", {"top_k": 20, "min_p": 0.0})
    assert tool_call_bench.extra_body() == {"top_k": 20, "min_p": 0.0, "reasoning_effort": "xhigh"}


def test_tool_bench_reasoning_text_reads_sdk_extras():
    from types import SimpleNamespace
    # O SDK da OpenAI guarda campos fora do schema em model_extra; alguns servidores usam o atributo direto.
    assert tool_call_bench.reasoning_text(SimpleNamespace(model_extra={"reasoning_content": "abc"})) == "abc"
    assert tool_call_bench.reasoning_text(SimpleNamespace(reasoning="zz", model_extra={})) == "zz"
    assert tool_call_bench.reasoning_text(SimpleNamespace(model_extra=None)) == ""
