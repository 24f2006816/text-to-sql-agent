"""Execution accuracy: does the predicted query return the same answer as the gold query?

Strict: identical rows (as a multiset, or in order when the gold query orders more than one row).
Lenient: the same, after ignoring extra columns the prediction added, e.g. returning
`Name, COUNT(*)` when only `Name` was asked for. Column names never matter, only values.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter


def _norm(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, int) and not isinstance(value, bool):
        return round(float(value), 2)
    return value


def _rows(rows) -> list[tuple]:
    return [tuple(_norm(v) for v in row) for row in rows]


def order_matters(gold_sql: str, gold_rows: list) -> bool:
    return len(gold_rows) > 1 and re.search(r"\border\s+by\b", gold_sql, re.I) is not None


def _same(a: list[tuple], b: list[tuple], ordered: bool) -> bool:
    return a == b if ordered else Counter(a) == Counter(b)


def strict_match(gold_rows, pred_rows, ordered: bool) -> bool:
    return _same(_rows(gold_rows), _rows(pred_rows), ordered)


def lenient_match(gold_rows, pred_rows, ordered: bool) -> bool:
    gold, pred = _rows(gold_rows), _rows(pred_rows)
    if len(gold) != len(pred) or not gold:
        return len(gold) == len(pred) == 0
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
        return False
    gold_cols = [sorted(map(repr, col)) for col in zip(*gold, strict=True)]
    pred_cols = [sorted(map(repr, col)) for col in zip(*pred, strict=True)]
    candidates = [[k for k in range(n_pred) if pred_cols[k] == gold_cols[j]] for j in range(n_gold)]
    if any(not c for c in candidates):
        return False
    for combo in itertools.product(*candidates):
        if len(set(combo)) != n_gold:
            continue
        projected = [tuple(row[k] for k in combo) for row in pred]
        if _same(gold, projected, ordered):
            return True
    return False
