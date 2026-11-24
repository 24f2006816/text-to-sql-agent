"""Ask a question from the command line: python -m sqlagent "Which genre sells the most?" """

from __future__ import annotations

import argparse

from .agent import SQLAgent
from .db import Database, ensure_chinook
from .llm import build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Chinook music-store database a question.")
    parser.add_argument("question")
    parser.add_argument("--model", default=None, help="hf:<model id> or openai:<model>")
    parser.add_argument("--db", default=None, help="path to a SQLite file (default: Chinook)")
    args = parser.parse_args()

    db = Database(args.db or ensure_chinook())
    answer = SQLAgent(build_llm(args.model), db).ask(args.question)
    for i, attempt in enumerate(answer.attempts, 1):
        print(f"-- attempt {i}{' (failed: ' + attempt.error + ')' if attempt.error else ''}\n{attempt.sql}\n")
    if answer.result is None:
        print("No valid query found.")
        return
    print(" | ".join(answer.result.columns))
    for row in answer.result.rows[:20]:
        print(" | ".join("NULL" if v is None else str(v) for v in row))
    if len(answer.result.rows) > 20:
        print(f"... {len(answer.result.rows) - 20} more rows")


if __name__ == "__main__":
    main()
