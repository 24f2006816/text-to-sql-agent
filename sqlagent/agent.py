"""The agent loop: prompt with the schema, generate SQL, check it, run it, repair on failure."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .db import Database, QueryResult
from .guard import UnsafeSQL, check_sql, extract_sql
from .llm import LLM, Message

SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite queries.
Rules:
- Use only the tables and columns in the schema. Use SQLite syntax (e.g. strftime for dates).
- Write exactly one read-only SELECT query (WITH is fine). Never modify data.
- Return only the columns the question asks for, plus anything needed to make the answer clear.
- Reply with the query in a ```sql code block and nothing else."""


@dataclass(frozen=True)
class Strategy:
    """What context the model sees and how many times it may repair a failed query."""

    name: str
    sample_rows: int = 0
    max_repairs: int = 0
    repair_empty: bool = True


BASELINE = Strategy("schema only")
WITH_EXAMPLES = Strategy("schema + example rows", sample_rows=3)
SELF_CORRECTING = Strategy("schema + example rows + self-correction", sample_rows=3, max_repairs=2)
STRATEGIES = (BASELINE, WITH_EXAMPLES, SELF_CORRECTING)


@dataclass
class Attempt:
    sql: str
    error: str | None = None
    rows: int | None = None


@dataclass
class Answer:
    question: str
    sql: str | None
    result: QueryResult | None
    attempts: list[Attempt] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.result is not None


class SQLAgent:
    def __init__(self, llm: LLM, db: Database, strategy: Strategy = SELF_CORRECTING):
        self.llm = llm
        self.db = db
        self.strategy = strategy
        self._schema = db.schema(sample_rows=strategy.sample_rows)

    def messages(self, question: str) -> list[Message]:
        user = f"Schema:\n{self._schema}\n\nQuestion: {question}"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        messages = self.messages(question)
        answer = Answer(question, sql=None, result=None)
        last_good: tuple[str, QueryResult] | None = None
        for attempt_no in range(self.strategy.max_repairs + 1):
            reply = self.llm.generate(messages)
            sql = extract_sql(reply)
            attempt = Attempt(sql)
            answer.attempts.append(attempt)
            problem = None
            try:
                result = self.db.run(check_sql(sql))
            except (UnsafeSQL, TimeoutError, Exception) as exc:  # sqlite errors carry useful text
                attempt.error = f"{type(exc).__name__}: {exc}"
                problem = f"The query failed with this error:\n{attempt.error}\nWrite a corrected query."
            else:
                attempt.rows = len(result.rows)
                last_good = (sql, result)
                if result.rows or not self.strategy.repair_empty:
                    break
                problem = ("The query ran but returned no rows. If that is not the right answer, check the "
                           "filters, joins and literal values against the example rows and write a corrected "
                           "query. If it is right, repeat the same query.")
                if attempt_no and sql == answer.attempts[-2].sql:
                    break  # the model stands by its query
            if attempt_no == self.strategy.max_repairs:
                break
            messages = messages + [
                {"role": "assistant", "content": f"```sql\n{sql}\n```"},
                {"role": "user", "content": problem},
            ]
        if last_good:
            answer.sql, answer.result = last_good
        else:
            answer.sql = answer.attempts[-1].sql
        answer.seconds = time.perf_counter() - start
        return answer
