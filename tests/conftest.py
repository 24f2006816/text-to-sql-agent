import pytest

from sqlagent.db import Database, ensure_chinook


class ScriptedLLM:
    """Returns canned replies in order, and records the conversations it was shown."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture(scope="session")
def db():
    return Database(ensure_chinook())
