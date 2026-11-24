import sqlite3

import pytest


def test_schema_lists_every_table(db):
    schema = db.schema()
    for table in ("Album", "Artist", "Invoice", "Track"):
        assert f"CREATE TABLE [{table}]" in schema


def test_schema_with_example_rows(db):
    assert "3 example rows from Genre" in db.schema(sample_rows=3)


def test_database_itself_refuses_writes(db):
    # Even if a write slipped past the SQL guard, SQLite's authorizer and read-only mode block it.
    with pytest.raises(sqlite3.DatabaseError):
        db.run("DELETE FROM Genre")


def test_row_cap(db):
    result = db.run("SELECT * FROM Track")
    assert len(result.rows) == db.max_rows and result.truncated


def test_runaway_query_times_out(db):
    db.timeout_s = 0.2
    try:
        with pytest.raises(TimeoutError):
            db.run("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n")
    finally:
        db.timeout_s = 5.0
