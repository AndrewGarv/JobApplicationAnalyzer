"""Run the job application agent.

    python main.py https://example.com/jobs/123        # research a posting
    python main.py https://example.com/jobs/123 --offline   # replay cached pages (demo)
    python main.py --history                            # show what's in memory
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.loop import run_agent
from agent.memory import Memory
from agent.tools import ToolRunner
from agent.trace import Tracer, console


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Job application research agent")
    parser.add_argument("url", nargs="?", help="Job posting URL")
    parser.add_argument("--resume", default="data/resume.md", help="Path to your resume (markdown/text)")
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "claude-haiku-5-5"))
    parser.add_argument("--offline", action="store_true", help="Only use cached pages and searches")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history", action="store_true", help="List saved applications and exit")
    args = parser.parse_args()

    memory = Memory()

    if args.history:
        apps = memory.all()
        if not apps:
            console.print("No applications saved yet.")
        for a in apps:
            console.print(f"[bold]#{a['id']}[/bold] {a['created_at']}  {a['role']} @ {a['company']}")
            console.print(f"   [dim]{a['summary']}[/dim]")
        return

    if not args.url:
        parser.error("give a job posting URL (or --history)")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Missing ANTHROPIC_API_KEY. Copy .env.example to .env and add your key.")

    import anthropic

    resume_text = Path(args.resume).read_text()
    tools = ToolRunner(memory, resume_text, offline=args.offline)
    tracer = Tracer()

    console.rule(f"[bold]Researching {args.url}")
    run_agent(anthropic.Anthropic(), args.model, args.url, tools, tracer, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
