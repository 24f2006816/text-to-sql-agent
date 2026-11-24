"""Hugging Face Space for text-to-sql-agent (runs on the free ZeroGPU tier).

The Space holds only this file, requirements.txt and README.md. At start-up it downloads the
agent's source from GitHub, so the demo always runs the code that is in the repository.
"""

import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

try:
    import spaces  # must be imported before torch on ZeroGPU
except ImportError:  # running locally
    spaces = None

import gradio as gr
import pandas as pd

REPO = os.getenv("SOURCE_REPO", "24f2006816/text-to-sql-agent")
SOURCE_DIR = Path("agent-src")


def download_source() -> None:
    if SOURCE_DIR.exists():
        return
    url = f"https://codeload.github.com/{REPO}/tar.gz/refs/heads/main"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    top = archive.getnames()[0].split("/")[0]
    archive.extractall("tmp-src", filter="data")
    Path("tmp-src", top).rename(SOURCE_DIR)


download_source()
sys.path.insert(0, str(SOURCE_DIR))

from sqlagent.agent import SELF_CORRECTING, SQLAgent  # noqa: E402
from sqlagent.db import Database, ensure_chinook  # noqa: E402
from sqlagent.llm import DEFAULT_MODEL, TransformersLLM  # noqa: E402

MODEL = os.getenv("MODEL_ID", DEFAULT_MODEL)
llm = TransformersLLM(MODEL, device="cuda" if spaces else None)
db = Database(ensure_chinook(SOURCE_DIR / "data" / "chinook.sqlite"))
agent = SQLAgent(llm, db, SELF_CORRECTING)

if spaces:
    llm.generate = spaces.GPU(duration=60)(llm.generate)


def ask(question: str):
    question = (question or "").strip()
    if not question:
        return "", "Type a question first.", pd.DataFrame()
    answer = agent.ask(question)
    log = []
    for i, attempt in enumerate(answer.attempts, 1):
        status = f"failed: `{attempt.error}`" if attempt.error else f"ran, {attempt.rows} row(s)"
        log.append(f"**Attempt {i}** ({status})\n```sql\n{attempt.sql}\n```")
    if answer.result is None:
        return answer.sql or "", "\n\n".join(log) + "\n\nNo valid query after repairs.", pd.DataFrame()
    table = pd.DataFrame(answer.result.rows, columns=answer.result.columns)
    note = f"\n\n{len(table)} row(s) in {answer.seconds:.1f}s"
    return answer.sql, "\n\n".join(log) + note, table


EXAMPLES = [
    "Which genre brings in the most revenue?",
    "Who are the top 5 customers by total spending?",
    "How many tracks have never been purchased?",
    "What were the total sales in each year?",
    "Which sales support agent generated the most revenue?",
]

with gr.Blocks(title="Text-to-SQL Agent") as demo:
    gr.Markdown(
        "# Text-to-SQL Agent\nAsk the **Chinook** music-store database a question in plain English. "
        f"`{MODEL}` writes SQLite; the agent blocks anything but read-only queries, runs it, and if the "
        "query fails it shows the model the error and lets it repair the query (up to 2 times). "
        "[Code and benchmark](https://github.com/24f2006816/text-to-sql-agent)"
    )
    question = gr.Textbox(label="Question", placeholder="Which artist sold the most tracks?")
    go = gr.Button("Ask", variant="primary")
    sql = gr.Code(label="Final SQL", language="sql")
    with gr.Row():
        table = gr.Dataframe(label="Result")
    log = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=question)
    go.click(ask, question, [sql, log, table])
    question.submit(ask, question, [sql, log, table])

if __name__ == "__main__":
    demo.launch()
