"""The agent loop, written by hand (no framework).

How it works:
  1. Send the conversation + tool list to the model.
  2. If the model asks to use tools, run them and send the results back.
  3. Repeat until the model answers without asking for a tool, or we hit max_steps.

That's the whole idea behind "agents". Frameworks like LangGraph add structure
on top, but this loop is the core, and knowing it cold is the interview story.
"""

from .tools import TOOL_SCHEMAS, ToolRunner
from .trace import Tracer

SYSTEM_PROMPT = """You are a job application research assistant working for one candidate.

Given a job posting URL, work through these stages, choosing tools as you need them:
1. Fetch and understand the posting: company, role title, must-have and nice-to-have skills.
2. Research the company: what it builds, who it serves, anything recent or distinctive.
3. Read the candidate's resume, and search past applications for similar roles.
4. Match: which resume items best fit this role? Be honest about gaps.
5. Save the application to memory.
6. Give the final answer.

Rules:
- Never invent experience the resume doesn't support. Reword and reframe only.
- If a tool fails, try another route (a different URL, a search) before giving up.
- If a similar past application exists, mention it and say what you emphasized there.

Final answer format (markdown):
## <Role> at <Company>
**Fit summary:** 2-3 sentences, including gaps.
**Tailored resume bullets:** 3-5 bullets.
**Why this company:** one short paragraph grounded in what you found.
**Memory:** what you found in past applications, or "First application of this kind."
"""


def run_agent(client, model: str, posting_url: str, tools: ToolRunner,
              tracer: Tracer, max_steps: int = 15, max_tokens: int = 2000) -> str:
    messages = [{"role": "user", "content": f"Here's the job posting: {posting_url}"}]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        tracer.model_call(step, response.usage)

        # Keep the model's turn in the history exactly as returned.
        messages.append({"role": "assistant", "content": response.content})

        text_parts = [b.text for b in response.content if b.type == "text"]
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason != "tool_use" or not tool_calls:
            final = "\n".join(text_parts).strip() or "(the model returned no text)"
            tracer.final(final)
            return final

        for t in text_parts:
            tracer.thinking(t)

        # Run every tool the model asked for, then send all results back in one message.
        results = []
        for call in tool_calls:
            tracer.tool_call(call.name, call.input)
            output, is_error = tools.run(call.name, call.input)
            tracer.tool_result(call.name, output, is_error)
            results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": output,
                "is_error": is_error,
            })
        messages.append({"role": "user", "content": results})

    stopped = f"Stopped after {max_steps} steps without a final answer. Check the log."
    tracer.final(stopped)
    return stopped
