from collections import Counter

from conftest import ScriptedLLM

from sqlagent.agent import BASELINE
from sqlagent.bench import load_questions, run_strategy


def test_benchmark_is_well_formed(db):
    questions = load_questions()
    assert len(questions) == 50
    assert len({q["id"] for q in questions}) == 50
    assert Counter(q["difficulty"] for q in questions) == {"easy": 15, "medium": 20, "hard": 15}
    for q in questions:  # every gold query runs and has an answer
        assert db.run(q["sql"]).rows, q["id"]


def test_perfect_model_scores_100_percent(db):
    questions = load_questions()[:10]
    llm = ScriptedLLM([f"```sql\n{q['sql']}\n```" for q in questions])
    summary = run_strategy(llm, db, questions, BASELINE)["summary"]
    assert summary["execution_accuracy"] == 1.0 and summary["strict_accuracy"] == 1.0
