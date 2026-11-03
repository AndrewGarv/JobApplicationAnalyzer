"""The tools the agent can call.

Each tool has two parts:
  1. A schema (name, description, input_schema) that tells the model what it can do.
  2. A Python function that actually does it.

Tool descriptions matter a lot: they're the model's only instructions for when
and how to use each tool.
"""

import hashlib
import json
import os
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from .memory import Memory

CACHE_DIR = Path("data/cache")


# ---------------------------------------------------------------- caching
# Fetches and searches are cached on disk. Re-running the same posting costs
# nothing extra, and --offline mode replays the cache for a reliable demo.

def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{kind}-{h}.json"


def _cached(kind: str, key: str, fn, offline: bool) -> str:
    path = _cache_path(kind, key)
    if path.exists():
        return json.loads(path.read_text())["result"]
    if offline:
        raise RuntimeError(f"offline mode: nothing cached for {kind} {key!r}")
    result = fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "key": key, "result": result}))
    return result


# ---------------------------------------------------------------- implementations

def fetch_page(url: str, max_chars: int = 12000) -> str:
    resp = httpx.get(url, follow_redirects=True, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (job-agent personal project)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return text[:max_chars] or "(page had no readable text — it may need JavaScript)"


def web_search(query: str, max_results: int = 5) -> str:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return "web_search is not configured (no TAVILY_API_KEY). Work with fetch_page and what you have."
    from tavily import TavilyClient
    results = TavilyClient(api_key=key).search(query, max_results=max_results)
    lines = [f"- {r['title']} ({r['url']}): {r.get('content', '')[:400]}"
             for r in results.get("results", [])]
    return "\n".join(lines) or "No results."


# ---------------------------------------------------------------- schemas

TOOL_SCHEMAS = [
    {
        "name": "fetch_page",
        "description": "Fetch a web page and return its readable text. Use for the job posting "
                       "itself and for company pages (about, careers, engineering blog).",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full https URL"}},
            "required": ["url"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web. Use to learn what the company does, its product, recent "
                       "news, and its tech stack. Keep queries short and specific.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_resume",
        "description": "Return the candidate's full resume. Call this before drafting anything.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_past_applications",
        "description": "Search memory of applications the candidate already made. Call this early "
                       "with the role title and key skills, so you can stay consistent with (or "
                       "deliberately differ from) what was emphasized before.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "e.g. 'backend python fintech'"}},
            "required": ["query"],
        },
    },
    {
        "name": "save_application",
        "description": "Save this application to long-term memory. Call exactly once, at the end, "
                       "right before giving your final answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "role": {"type": "string"},
                "url": {"type": "string"},
                "summary": {"type": "string", "description": "2-3 sentences: the role and why it fits"},
                "emphasized": {"type": "array", "items": {"type": "string"},
                               "description": "The resume points you chose to emphasize"},
            },
            "required": ["company", "role", "summary", "emphasized"],
        },
    },
]


class ToolRunner:
    """Routes a tool call from the model to the right Python function."""

    def __init__(self, memory: Memory, resume_text: str, offline: bool = False):
        self.memory = memory
        self.resume_text = resume_text
        self.offline = offline

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (result_text, is_error). Errors go back to the model instead of crashing,
        so it can recover (e.g. try a different URL)."""
        try:
            return self._dispatch(name, args), False
        except Exception as e:  # noqa: BLE001 — every failure becomes feedback for the model
            return f"{type(e).__name__}: {e}", True

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "fetch_page":
            return _cached("fetch", args["url"], lambda: fetch_page(args["url"]), self.offline)
        if name == "web_search":
            return _cached("search", args["query"], lambda: web_search(args["query"]), self.offline)
        if name == "read_resume":
            return self.resume_text
        if name == "search_past_applications":
            hits = self.memory.search(args["query"])
            return json.dumps(hits, indent=2) if hits else "No similar past applications."
        if name == "save_application":
            app_id = self.memory.save_application(
                company=args["company"], role=args["role"], url=args.get("url", ""),
                summary=args["summary"], emphasized=args.get("emphasized", []),
            )
            return f"Saved as application #{app_id}."
        raise ValueError(f"Unknown tool: {name}")
