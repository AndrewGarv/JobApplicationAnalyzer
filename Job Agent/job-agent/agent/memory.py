"""Long-term memory: a SQLite database of past applications.

This is the "memory beyond prompt context" part. Anything saved here survives
between runs, and the agent can search it with full-text search (SQLite FTS5).

Upgrade idea: swap the FTS search for embeddings (e.g. Chroma) so "backend
engineer" also matches "server-side developer".
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class Memory:
    def __init__(self, db_path: str | Path = "data/memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                url TEXT,
                summary TEXT,
                emphasized TEXT  -- JSON list of points you emphasized
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS applications_fts USING fts5(
                company, role, summary, emphasized,
                content='applications', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS applications_ai AFTER INSERT ON applications BEGIN
                INSERT INTO applications_fts(rowid, company, role, summary, emphasized)
                VALUES (new.id, new.company, new.role, new.summary, new.emphasized);
            END;
            """
        )
        self.conn.commit()

    def save_application(self, company: str, role: str, url: str = "",
                         summary: str = "", emphasized: list[str] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO applications (created_at, company, role, url, summary, emphasized) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), company, role,
             url, summary, json.dumps(emphasized or [])),
        )
        self.conn.commit()
        return cur.lastrowid

    def search(self, query: str, limit: int = 5) -> list[dict]:
        # Turn free text into an OR query of words so partial matches still hit.
        words = [w for w in "".join(c if c.isalnum() else " " for c in query).split() if len(w) > 2]
        if not words:
            return []
        fts_query = " OR ".join(f'"{w}"' for w in words)
        rows = self.conn.execute(
            "SELECT a.* FROM applications_fts f JOIN applications a ON a.id = f.rowid "
            "WHERE applications_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def all(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM applications ORDER BY id DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["emphasized"] = json.loads(d.get("emphasized") or "[]")
        return d
