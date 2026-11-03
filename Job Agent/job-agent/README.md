# Job Application Research Agent

Give it a job posting URL. It reads the posting, researches the company, compares the role
against your resume, checks what you've applied to before, and drafts tailored resume bullets
plus a "why this company" paragraph. Then it remembers the application for next time.

Built with a hand-written tool-calling loop on the Claude API (no agent framework), SQLite
long-term memory, and step-level tracing.

```
            ┌────────────── agent loop (agent/loop.py) ──────────────┐
 posting ──▶│ model decides next step ─▶ tool call ─▶ result ─▶ ...  │──▶ tailored bullets
   URL      └───────┬──────────────┬──────────────┬──────────────────┘
                    ▼              ▼              ▼
              fetch_page      web_search     read_resume
             (cached)         (cached)
                    search_past_applications / save_application
                              ▼
                     SQLite memory (data/memory.db)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your ANTHROPIC_API_KEY (and optionally TAVILY_API_KEY)
cp data/resume.example.md data/resume.md   # then paste in your real resume
```

Set a monthly spend limit in the Claude Console before your first run.

## Usage

```bash
python main.py https://company.com/careers/some-job    # research a posting
python main.py https://company.com/careers/some-job --offline   # replay cached pages (demo)
python main.py --history                               # what's in memory
pytest                                                 # offline tests, no API key needed
```

Each run prints its steps live and saves a full log to `logs/run-<time>.jsonl`.

## Project layout

| File | What it does |
|---|---|
| `agent/loop.py` | The agent loop and system prompt. Start reading here. |
| `agent/tools.py` | Tool schemas (what the model sees) and implementations, with disk caching. |
| `agent/memory.py` | SQLite + full-text search memory of past applications. |
| `agent/trace.py` | Live step output and JSONL logs, with token counts. |
| `tests/test_agent.py` | A fake model that replays scripted turns, so the loop is tested without an API key. |

## Design decisions

- **Hand-written loop instead of a framework**, so every step is visible and debuggable.
- **Tool errors are returned to the model** instead of crashing, so it can recover (e.g. try another URL).
- **A step limit** stops runaway loops and runaway costs.
- **Caching** makes re-runs free and gives a reliable offline demo.

## Limitations and next steps

- Pages that need JavaScript to render (some job boards) come back empty. A headless browser would fix this.
- Memory search is keyword-based. Embeddings (e.g. Chroma) would match "backend" with "server-side".
- Next: a small eval set of postings with pass/fail checks, and a Streamlit UI showing the live trace.
