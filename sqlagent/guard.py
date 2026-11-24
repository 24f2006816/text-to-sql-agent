"""Static checks on generated SQL before it reaches the database."""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

FORBIDDEN = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Pragma,
)


class UnsafeSQL(ValueError):
    """Raised when generated SQL is not a single read-only query."""


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply: a ```sql fenced block if present, else the reply itself."""
    fenced = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = fenced[-1] if fenced else text
    sql = sql.strip()
    if not fenced:
        match = re.search(r"\b(WITH|SELECT)\b", sql, flags=re.I)
        if match:
            sql = sql[match.start():]
    return sql.rstrip().rstrip(";").strip()


def check_sql(sql: str) -> str:
    """Return the SQL unchanged if it is exactly one SELECT/WITH statement, else raise UnsafeSQL."""
    if not sql:
        raise UnsafeSQL("empty query")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise UnsafeSQL(f"could not parse SQL: {str(exc).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise UnsafeSQL("only one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With, exp.Subquery)):
        raise UnsafeSQL(f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN):
            raise UnsafeSQL(f"{node.key.upper()} is not allowed")
    return sql
