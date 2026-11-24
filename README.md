# Text-to-SQL Agent

[![CI](https://github.com/24f2006816/text-to-sql-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/24f2006816/text-to-sql-agent/actions/workflows/ci.yml)
[![Benchmark](https://github.com/24f2006816/text-to-sql-agent/actions/workflows/benchmark.yml/badge.svg)](https://github.com/24f2006816/text-to-sql-agent/actions/workflows/benchmark.yml)

**Ask a database questions in plain English.** An open LLM (Qwen2.5-Coder) writes SQLite, a guard lets through only read-only queries, and when a query fails the agent shows the model the database's error and lets it repair the query. A 50-question benchmark measures how often the answer is actually right.

<!-- RESULTS -->

## How it works

```
question ──► prompt (schema + example rows) ──► LLM ──► extract SQL ──► guard ──► run (read-only, 5 s cap)
                     ▲                                                              │
                     └──────────── error message or "no rows" (max 2 repairs) ◄─────┘
```

1. **Prompt.** The model sees every table's `CREATE` statement and, in the better strategies, three example rows per table, so it learns real values such as `'USA'` or the date format.
2. **Guard.** [`sqlglot`](https://github.com/tobymao/sqlglot) parses the reply. Anything that is not exactly one `SELECT`/`WITH` statement (`DROP`, `UPDATE`, `PRAGMA`, `ATTACH`, stacked statements) is rejected before it reaches the database.
3. **Defence in depth.** The database is opened read-only, and a SQLite authorizer denies every action except reads, so a write is refused even if the guard missed it. A progress handler kills queries that run longer than 5 seconds, and results are capped at 1,000 rows.
4. **Self-correction.** If the query fails (wrong column, syntax error, blocked statement) or returns nothing, the error goes back to the model as a new turn, up to two times. If the model repeats a query that returned no rows, the agent accepts it as the answer.

## Benchmark

`benchmark/chinook.jsonl` holds 50 questions about the [Chinook](https://github.com/lerocha/chinook-database) music-store database (11 tables: artists, albums, tracks, invoices, customers, employees), each with a hand-written gold query:

| Difficulty | Questions | Example |
| --- | --- | --- |
| Easy | 15 | How many customers live in Brazil? |
| Medium | 20 | Which country has the highest total sales? |
| Hard | 15 | Which sales support agent generated the most revenue through their customers? |

Every gold query is checked in the test suite to run and return rows, and no "top 1" question has a tie.

**Execution accuracy** runs both queries and compares the answers, not the SQL text. Values are compared to two decimals, column names are ignored, and row order counts only when the question asks for a ranking of several rows. The headline number is lenient about extra columns (returning `Name, COUNT(*)` when only `Name` was asked for is correct); the strict number requires exactly the gold columns.

Reproduce: `python -m sqlagent.bench --model hf:Qwen/Qwen2.5-Coder-1.5B-Instruct`, or re-run the **Benchmark** workflow from the Actions tab.

## Run it

```bash
pip install -e ".[hf]"          # transformers + torch; the Chinook database downloads on first use
python -m sqlagent "Which genre brings in the most revenue?"
python -m sqlagent.bench --limit 10
```

Any OpenAI-compatible endpoint also works (OpenAI, Groq, a local vLLM or Ollama server):

```bash
OPENAI_BASE_URL=http://localhost:11434/v1 python -m sqlagent --model openai:qwen2.5-coder:7b "How many albums does AC/DC have?"
```

Point it at your own SQLite file with `--db path/to/file.sqlite`.

## Tests

`pytest` (31 tests, no model download) covers the SQL guard, the read-only database, the timeout, the scorer, the repair loop with a scripted model, and the benchmark file itself. CI runs them with `ruff` on Python 3.10 and 3.12.

## Project layout

```
sqlagent/agent.py     prompt, generate → guard → run → repair loop, strategies
sqlagent/guard.py     SQL extraction and the read-only check (sqlglot)
sqlagent/db.py        read-only SQLite with authorizer, timeout and row cap; schema description
sqlagent/llm.py       transformers and OpenAI-compatible back ends
sqlagent/scoring.py   execution-accuracy comparison
sqlagent/bench.py     benchmark runner and report
benchmark/            50 labelled questions
space/                Hugging Face Space demo (Gradio, ZeroGPU)
```

## Limitations

- 50 questions on one database is a small benchmark. Treat differences of a few points as noise.
- The model sees the whole schema. For databases with hundreds of tables, add a table-selection step first.
- Execution accuracy can count a lucky wrong query as correct when it happens to return the same rows.

## License

MIT. The Chinook database is by Luis Rocha and is downloaded from its repository, not redistributed here.
