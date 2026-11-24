from conftest import ScriptedLLM

from sqlagent.agent import BASELINE, SELF_CORRECTING, SQLAgent


def fenced(sql):
    return f"```sql\n{sql}\n```"


def test_self_correction_repairs_a_failed_query(db):
    llm = ScriptedLLM([fenced("SELECT COUNT(*) FROM Customers"), fenced("SELECT COUNT(*) FROM Customer")])
    answer = SQLAgent(llm, db, SELF_CORRECTING).ask("How many customers are there?")
    assert answer.ok and answer.result.rows == [(59,)]
    assert len(answer.attempts) == 2 and "no such table" in answer.attempts[0].error
    assert "no such table" in llm.calls[1][-1]["content"]  # the error was shown to the model


def test_baseline_does_not_retry(db):
    llm = ScriptedLLM([fenced("SELECT COUNT(*) FROM Customers")])
    answer = SQLAgent(llm, db, BASELINE).ask("How many customers are there?")
    assert not answer.ok and len(answer.attempts) == 1


def test_unsafe_sql_is_never_run_and_gets_repaired(db):
    llm = ScriptedLLM([fenced("DROP TABLE Customer"), fenced("SELECT COUNT(*) FROM Customer")])
    answer = SQLAgent(llm, db, SELF_CORRECTING).ask("How many customers?")
    assert answer.result.rows == [(59,)]
    assert "UnsafeSQL" in answer.attempts[0].error


def test_empty_result_triggers_a_check(db):
    llm = ScriptedLLM([fenced("SELECT COUNT(*) FROM Customer WHERE Country = 'brazil' GROUP BY Country"),
                       fenced("SELECT COUNT(*) FROM Customer WHERE Country = 'Brazil'")])
    answer = SQLAgent(llm, db, SELF_CORRECTING).ask("How many customers live in Brazil?")
    assert answer.result.rows == [(5,)]


def test_model_can_stand_by_an_empty_result(db):
    sql = "SELECT Name FROM Artist WHERE Name = 'Nobody'"
    llm = ScriptedLLM([fenced(sql), fenced(sql)])
    answer = SQLAgent(llm, db, SELF_CORRECTING).ask("Is there an artist called Nobody?")
    assert answer.ok and answer.result.rows == [] and len(answer.attempts) == 2


def test_gives_up_after_max_repairs(db):
    llm = ScriptedLLM([fenced("SELECT nope FROM Track")] * 3)
    answer = SQLAgent(llm, db, SELF_CORRECTING).ask("?")
    assert not answer.ok and len(answer.attempts) == 3
