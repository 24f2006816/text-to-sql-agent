"""Benchmark the agent on labelled questions and report execution accuracy per strategy.

Usage: python -m sqlagent.bench [--model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct] [--limit N] [--out results]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

from .agent import STRATEGIES, SQLAgent, Strategy
from .db import Database, ensure_chinook, file_sha256
from .llm import LLM, build_llm
from .scoring import lenient_match, order_matters, strict_match

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "benchmark" / "chinook.jsonl"


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_strategy(llm: LLM, db: Database, questions: list[dict], strategy: Strategy) -> dict:
    agent = SQLAgent(llm, db, strategy)
    records = []
    for q in questions:
        gold = db.run(q["sql"])
        answer = agent.ask(q["question"])
        pred_rows = answer.result.rows if answer.result else None
        ordered = order_matters(q["sql"], gold.rows)
        records.append({
            "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
            "gold_sql": q["sql"], "pred_sql": answer.sql,
            "ran": answer.ok,
            "strict": answer.ok and strict_match(gold.rows, pred_rows, ordered),
            "lenient": answer.ok and lenient_match(gold.rows, pred_rows, ordered),
            "attempts": len(answer.attempts),
            "repaired": answer.ok and len(answer.attempts) > 1 and answer.attempts[0].error is not None,
            "errors": [a.error for a in answer.attempts if a.error],
            "seconds": round(answer.seconds, 2),
        })
    return {"strategy": strategy.name, "summary": summarise(records), "records": records}


def summarise(records: list[dict]) -> dict:
    def rate(rs, key):
        return round(sum(bool(r[key]) for r in rs) / len(rs), 4) if rs else None

    by_diff = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(r)
    return {
        "n": len(records),
        "execution_accuracy": rate(records, "lenient"),
        "strict_accuracy": rate(records, "strict"),
        "valid_sql_rate": rate(records, "ran"),
        "repaired_queries": sum(r["repaired"] for r in records),
        "by_difficulty": {d: {"n": len(rs), "execution_accuracy": rate(rs, "lenient")} for d, rs in by_diff.items()},
        "avg_seconds": round(sum(r["seconds"] for r in records) / len(records), 2) if records else None,
    }


def to_markdown(report: dict) -> str:
    lines = [f"Model: `{report['model']}` | {report['questions']} questions | {report['hardware']}", "",
             "| Strategy | Execution accuracy | Easy | Medium | Hard | Strict | Valid SQL | Repaired "
             "| Avg s/question |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        s = run["summary"]
        d = s["by_difficulty"]
        lines.append(
            f"| {run['strategy']} | **{s['execution_accuracy']:.0%}** | {d['easy']['execution_accuracy']:.0%} | "
            f"{d['medium']['execution_accuracy']:.0%} | {d['hard']['execution_accuracy']:.0%} | "
            f"{s['strict_accuracy']:.0%} | {s['valid_sql_rate']:.0%} | {s['repaired_queries']} | {s['avg_seconds']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="hf:<model id> or openai:<model>")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    db = Database(ensure_chinook())
    questions = load_questions()[: args.limit]
    llm = build_llm(args.model)
    started = time.time()
    runs = [run_strategy(llm, db, questions, s) for s in STRATEGIES]
    report = {
        "model": llm.name, "questions": len(questions), "database_sha256": file_sha256(db.path),
        "hardware": f"{platform.machine()} CPU" if getattr(llm, "device", "cpu") == "cpu" else "GPU",
        "minutes": round((time.time() - started) / 60, 1), "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    slug = llm.name.replace("/", "__")
    (args.out / f"{slug}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / f"{slug}.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
