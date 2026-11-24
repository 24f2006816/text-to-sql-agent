from sqlagent.scoring import lenient_match, order_matters, strict_match


def test_extra_columns_are_lenient_but_not_strict():
    gold = [("Rock",)]
    pred = [("Rock", 1297)]
    assert lenient_match(gold, pred, ordered=False)
    assert not strict_match(gold, pred, ordered=False)


def test_column_order_and_names_do_not_matter():
    assert lenient_match([("a", 1), ("b", 2)], [(2, "b"), (1, "a")], ordered=False)


def test_floats_compared_to_two_decimals():
    assert strict_match([(4.731834,)], [(4.73,)], ordered=False)


def test_order_checked_only_when_gold_orders_several_rows():
    assert order_matters("SELECT x FROM t ORDER BY x DESC LIMIT 5", [(1,), (2,)])
    assert not order_matters("SELECT x FROM t ORDER BY x DESC LIMIT 1", [(1,)])
    assert not lenient_match([(1,), (2,)], [(2,), (1,)], ordered=True)


def test_wrong_values_fail():
    assert not lenient_match([("Rock",)], [("Jazz",)], ordered=False)
    assert not lenient_match([(1,), (2,)], [(1,)], ordered=False)
