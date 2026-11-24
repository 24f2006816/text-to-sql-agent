"""Database access: download the Chinook sample, describe the schema, run read-only queries."""

from __future__ import annotations

import hashlib
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHINOOK_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/"
    "master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
)
DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "chinook.sqlite"


def ensure_chinook(path: Path = DEFAULT_DB) -> Path:
    """Download the Chinook SQLite database once and cache it locally."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(CHINOOK_URL, tmp)  # noqa: S310 - fixed https URL
        with sqlite3.connect(tmp) as conn:  # fail fast if the download is not a SQLite file
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        tmp.rename(path)
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    ms: float


class Database:
    """A SQLite database opened read-only, with a row cap and a time limit per query."""

    def __init__(self, path: Path | str, max_rows: int = 1000, timeout_s: float = 5.0):
        self.path = Path(path)
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.conn.set_authorizer(_read_only_authorizer)

    def tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def schema(self, sample_rows: int = 0) -> str:
        """CREATE statements for every table, optionally followed by a few example rows."""
        parts = []
        for table in self.tables():
            (sql,) = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            parts.append(" ".join(sql.split()))
            if sample_rows:
                cur = self.conn.execute(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}')
                cols = [d[0] for d in cur.description]
                lines = [" | ".join(cols)] + [" | ".join(_short(v) for v in row) for row in cur.fetchall()]
                parts.append(f"/* {sample_rows} example rows from {table}:\n" + "\n".join(lines) + "\n*/")
        return "\n\n".join(parts)

    def run(self, sql: str) -> QueryResult:
        deadline = time.perf_counter() + self.timeout_s

        def watchdog() -> int:
            return 1 if time.perf_counter() > deadline else 0  # non-zero aborts the query

        self.conn.set_progress_handler(watchdog, 10_000)
        start = time.perf_counter()
        try:
            cur = self.conn.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc):
                raise TimeoutError(f"query took longer than {self.timeout_s:g}s") from exc
            raise
        finally:
            self.conn.set_progress_handler(None, 0)
        columns = [d[0] for d in cur.description] if cur.description else []
        truncated = len(rows) > self.max_rows
        return QueryResult(columns, rows[: self.max_rows], truncated, (time.perf_counter() - start) * 1000)


_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _read_only_authorizer(action, *_args):
    """Second line of defence after the SQL guard: SQLite itself refuses anything but reads."""
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def _short(value, limit: int = 40) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
