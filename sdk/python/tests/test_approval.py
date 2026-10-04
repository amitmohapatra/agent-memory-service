"""``approve_when``: the argument shape decisions are pooled by, the expression grammar, and
how an expression evaluates against a call's arguments (``trellis.memory.approval``)."""

from __future__ import annotations

from typing import Any

import pytest

from trellis.memory.approval import MAX_EXPRESSION_CHARS, arg_shape, evaluate, parse, when_shape

# --------------------------------------------------------------------------- arg_shape


def test_no_arguments_have_an_empty_shape() -> None:
    assert arg_shape(None) == ""
    assert arg_shape({}) == ""


def test_a_shape_names_each_argument_with_its_kind_sorted_by_name() -> None:
    args: dict[str, Any] = {
        "vendor": "acme",
        "note": None,
        "urgent": True,
        "draft": False,
        "lines": [1, 2],
        "pair": (1, 2),
        "meta": {"k": "v"},
    }
    assert arg_shape(args) == (
        "draft:bool:false,lines:list,meta:obj,note:null,pair:list,urgent:bool:true,vendor:str"
    )


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        (0, "num:1e0"),
        (7, "num:1e0"),
        (120, "num:1e2"),
        (12_000, "num:1e4"),
        (-450, "num:1e2"),
        (0.05, "num:1e-2"),
        (1.5, "num:1e0"),
    ],
)
def test_a_number_s_kind_is_its_order_of_magnitude(value: float, kind: str) -> None:
    assert arg_shape({"amount": value}) == f"amount:{kind}"


def test_the_shape_carries_no_argument_values() -> None:
    shape = arg_shape({"customer": "Jane Doe", "iban": "DE89370400440532013000"})
    assert "Jane" not in shape and "DE89" not in shape
    assert shape == "customer:str,iban:str"


def test_when_shape_is_true_exactly_for_calls_of_that_shape() -> None:
    small = {"amount": 120, "currency": "EUR"}
    rule = when_shape(arg_shape(small))
    assert rule == 'shape == "amount:num:1e2,currency:str"'
    assert evaluate(rule, {"amount": 450, "currency": "USD"})
    assert not evaluate(rule, {"amount": 12_000, "currency": "EUR"})
    assert not evaluate(rule, {"amount": 120})


def test_when_shape_quotes_a_shape_safely() -> None:
    rule = when_shape('odd"name:str')
    assert evaluate(rule, {'odd"name': "x"})


# --------------------------------------------------------------------------- parsing


def test_an_expression_parses_into_a_tree() -> None:
    assert parse('amount >= 1000 and currency == "EUR"') == (
        "and",
        ("cmp", ">=", ("path", "amount"), ("lit", 1000)),
        ("cmp", "==", ("path", "currency"), ("lit", "EUR")),
    )


def test_and_binds_tighter_than_or_and_not_tighter_than_and() -> None:
    assert parse("a == 1 or b == 2 and not c == 3") == (
        "or",
        ("cmp", "==", ("path", "a"), ("lit", 1)),
        (
            "and",
            ("cmp", "==", ("path", "b"), ("lit", 2)),
            ("not", ("cmp", "==", ("path", "c"), ("lit", 3))),
        ),
    )


def test_numbers_strings_lists_and_dotted_paths_are_operands() -> None:
    assert parse('po.total > -1.5e3 and region in ["eu", 2, 3.0]') == (
        "and",
        ("cmp", ">", ("path", "po.total"), ("lit", -1500.0)),
        ("cmp", "in", ("path", "region"), ("list", [("lit", "eu"), ("lit", 2), ("lit", 3.0)])),
    )


def test_a_string_may_contain_escaped_quotes() -> None:
    assert parse(r'name == "say \"hi\""') == ("cmp", "==", ("path", "name"), ("lit", 'say "hi"'))


def test_whitespace_does_not_matter() -> None:
    assert parse("  amount>1  ") == parse("amount > 1")


def test_an_empty_list_and_a_trailing_comma_are_lists() -> None:
    assert parse("x in []") == ("cmp", "in", ("path", "x"), ("list", []))
    assert parse("x in [1,]") == ("cmp", "in", ("path", "x"), ("list", [("lit", 1)]))


def test_true_and_false_stand_alone_as_conditions() -> None:
    assert parse("true") == ("lit", True)
    assert parse("false or (true)") == ("or", ("lit", False), ("lit", True))
    assert parse("true and x == 1") == (
        "and",
        ("lit", True),
        ("cmp", "==", ("path", "x"), ("lit", 1)),
    )


def test_a_boolean_followed_by_an_operator_is_compared() -> None:
    assert parse("true == urgent") == ("cmp", "==", ("lit", True), ("path", "urgent"))


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("amount", "expected a comparison"),
        ("amount ==", "unexpected end"),
        ("(amount == 1", r"missing \)"),
        ("x in [1 2]", "expected , or ]"),
        ("amount == )", r"unexpected '\)'"),
        ("amount == 1 2", "unexpected 2"),
        ("amount == 1 @ 2", "unexpected input at 11"),
        ("", "unexpected end"),
        ("and", "unexpected 'and'"),
    ],
)
def test_an_expression_that_does_not_parse_is_refused(expression: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse(expression)


def test_an_overlong_expression_is_refused() -> None:
    longest = "x == 1" + " " * (MAX_EXPRESSION_CHARS - len("x == 1"))
    assert parse(longest) == ("cmp", "==", ("path", "x"), ("lit", 1))
    with pytest.raises(ValueError, match="longer than"):
        parse(longest + " ")


def test_an_expression_that_does_not_parse_fails_closed_on_evaluation() -> None:
    with pytest.raises(ValueError):
        evaluate("amount >", {"amount": 1})


# --------------------------------------------------------------------------- evaluation


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("amount == 1200", True),
        ("amount != 1200", False),
        ("amount < 1200", False),
        ("amount <= 1200", True),
        ("amount > 1000", True),
        ("amount >= 1201", False),
        ('currency in ["EUR", "GBP"]', True),
        ('"UR" in currency', True),
        ('currency in ["USD"]', False),
    ],
)
def test_each_comparison_operator_evaluates(expression: str, expected: bool) -> None:
    assert evaluate(expression, {"amount": 1200, "currency": "EUR"}) is expected


def test_the_documented_examples_hold() -> None:
    assert evaluate('amount >= 1000 and currency == "EUR"', {"amount": 1200, "currency": "EUR"})
    assert evaluate('shape == "amount:num:1e3"', {"amount": 1200})


def test_a_dotted_path_reads_into_nested_arguments() -> None:
    args = {"po": {"vendor": {"country": "DE"}, "total": 50}}
    assert evaluate('po.vendor.country == "DE"', args)
    assert not evaluate("po.total > 100", args)


def test_a_missing_argument_makes_its_comparison_false() -> None:
    assert not evaluate("amount > 0", {})
    assert not evaluate("amount != 0", {})
    assert not evaluate("0 == amount", {})
    assert not evaluate("po.total > 0", {"po": 5}), "a path through a non-object is missing"
    assert not evaluate("po.missing == 1", {"po": {"total": 1}})


def test_values_that_do_not_compare_make_the_comparison_false() -> None:
    assert not evaluate("amount > 10", {"amount": "lots"})
    assert not evaluate("amount in 5", {"amount": 5})


def test_boolean_connectives_follow_their_truth_tables() -> None:
    args = {"a": 1, "b": 2}
    assert evaluate("a == 1 or b == 3", args)
    assert evaluate("a == 0 or b == 2", args)
    assert not evaluate("a == 0 or b == 0", args)
    assert evaluate("a == 1 and b == 2", args)
    assert not evaluate("a == 1 and b == 0", args)
    assert evaluate("not a == 0", args)
    assert not evaluate("not not a == 0", args)
    assert evaluate("not (a == 1 and b == 0)", args)


def test_literal_conditions_evaluate_to_themselves() -> None:
    assert evaluate("true", {}) is True
    assert evaluate("false", {}) is False
    assert evaluate("urgent == true", {"urgent": True})
    assert not evaluate("urgent == false", {"urgent": True})


def test_shape_compares_against_the_calls_argument_shape() -> None:
    assert evaluate('shape == ""', {})
    assert evaluate('shape != "amount:num:1e2"', {"amount": 12_000})
