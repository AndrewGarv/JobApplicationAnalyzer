"""Step-by-step logging: prints each agent step live and saves it to a JSONL file.

The live output is what makes the demo look good. The saved log is what lets you
debug a bad run afterwards (and talk about observability in interviews).
"""

import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


class Tracer:
    def __init__(self, log_dir: str | Path = "logs"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = Path(log_dir) / f"run-{stamp}.jsonl"
        self.input_tokens = 0
        self.output_tokens = 0
        self.start = time.time()

    def _write(self, event: dict) -> None:
        event["t"] = round(time.time() - self.start, 2)
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def model_call(self, step: int, usage) -> None:
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        self.input_tokens += inp
        self.output_tokens += out
        self._write({"type": "model_call", "step": step, "input_tokens": inp, "output_tokens": out})

    def thinking(self, text: str) -> None:
        if text.strip():
            console.print(f"[dim]💭 {text.strip()[:300]}[/dim]")
            self._write({"type": "assistant_text", "text": text})

    def tool_call(self, name: str, args: dict) -> None:
        shown = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in args.items())
        console.print(f"[bold cyan]→ {name}[/bold cyan]({shown})")
        self._write({"type": "tool_call", "name": name, "args": args})

    def tool_result(self, name: str, result: str, is_error: bool) -> None:
        color = "red" if is_error else "green"
        console.print(f"  [{color}]← {len(result)} chars{' (error)' if is_error else ''}[/{color}]")
        self._write({"type": "tool_result", "name": name, "is_error": is_error, "result": result[:2000]})

    def final(self, text: str) -> None:
        console.print(Panel(text, title="Result", border_style="green"))
        console.print(
            f"[dim]tokens in/out: {self.input_tokens}/{self.output_tokens} · "
            f"{time.time() - self.start:.1f}s · log: {self.path}[/dim]"
        )
        self._write({"type": "final", "text": text})
