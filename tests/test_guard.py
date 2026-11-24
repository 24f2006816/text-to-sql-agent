import pytest

from sqlagent.guard import UnsafeSQL, check_sql, extract_sql


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM Customer",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT Name FROM Artist UNION SELECT Name FROM Genre",
])
def test_read_only_queries_pass(sql):
    assert check_sql(sql) == sql


@pytest.mark.parametrize("sql", [
    "DELETE FROM Customer",
    "DROP TABLE Customer",
    "UPDATE Track SET UnitPrice = 0",
    "INSERT INTO Genre VALUES (99, 'x')",
    "SELECT 1; DROP TABLE Customer",
    "PRAGMA table_info(Customer)",
    "ATTACH DATABASE 'x.db' AS x",
    "",
])
def test_writes_and_multiple_statements_are_rejected(sql):
    with pytest.raises(UnsafeSQL):
        check_sql(sql)


def test_extract_sql_from_fenced_block():
    reply = "Here you go:\n```sql\nSELECT Name FROM Genre;\n```\nThis lists genres."
    assert extract_sql(reply) == "SELECT Name FROM Genre"


def test_extract_sql_without_fence_skips_preamble():
    assert extract_sql("Sure. SELECT COUNT(*) FROM Track;") == "SELECT COUNT(*) FROM Track"
