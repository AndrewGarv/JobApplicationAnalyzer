"""Tests that run with no API key and no network: a fake model replays a scripted
conversation so we can check the loop, the tools and memory."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.loop import run_agent
from agent.memory import Memory
from agent.tools import ToolRunner
from agent.trace import Tracer


def text(t):
    return NS(type="text", text=t)


def tool(id_, name, input_):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def response(content, stop):
    return NS(content=content, stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    """Returns scripted responses in order and records what the loop sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot the history: the loop keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_loop_runs_tools_and_saves_to_memory(tmp_path):
    memory = Memory(tmp_path / "m.db")
    runner = ToolRunner(memory, resume_text="Python dev, 3 years")
    client = FakeClient([
        response([text("Checking memory and resume."),
                  tool("t1", "search_past_applications", {"query": "backend python"}),
                  tool("t2", "read_resume", {})], "tool_use"),
        response([tool("t3", "save_application", {
            "company": "Acme", "role": "Backend Engineer",
            "summary": "Python backend role.", "emphasized": ["migrations", "logging"]})], "tool_use"),
        response([text("## Backend Engineer at Acme\nDone.")], "end_turn"),
    ])

    out = run_agent(client, "fake-model", "https://x/job", runner, Tracer(tmp_path / "logs"))

    assert out.startswith("## Backend Engineer at Acme")
    assert len(client.calls) == 3
    # Both tool results went back together, matched to the right tool call ids.
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[1]["content"] == "Python dev, 3 years"
    # Memory persisted and is searchable.
    assert memory.search("backend")[0]["company"] == "Acme"


def test_tool_errors_go_back_to_model_instead_of_crashing(tmp_path):
    runner = ToolRunner(Memory(tmp_path / "m.db"), "", offline=True)
    output, is_error = runner.run("fetch_page", {"url": "https://not-cached.example"})
    assert is_error and "offline mode" in output


def test_step_limit_stops_runaway_loops(tmp_path):
    runner = ToolRunner(Memory(tmp_path / "m.db"), "resume")
    looping = [response([tool(f"t{i}", "read_resume", {})], "tool_use") for i in range(3)]
    out = run_agent(FakeClient(looping), "m", "u", runner, Tracer(tmp_path / "logs"), max_steps=3)
    assert "Stopped after 3 steps" in out


def test_memory_finds_similar_roles(tmp_path):
    m = Memory(tmp_path / "m.db")
    m.save_application("Shopify", "Backend Developer", summary="Ruby and Python services")
    m.save_application("Figma", "Product Designer", summary="Design systems")
    hits = m.search("python backend engineer")
    assert [h["company"] for h in hits] == ["Shopify"]
