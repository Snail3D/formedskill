"""Tests for formedskill.conditions — show_when evaluator."""

import pytest
from formedskill.conditions import evaluate_condition, should_show_field


class TestEvaluateCondition:
    def test_empty_condition_always_true(self):
        assert evaluate_condition("", {}) is True
        assert evaluate_condition(None, {}) is True

    def test_equality_match(self):
        assert evaluate_condition("mode == generate", {"mode": "generate"}) is True

    def test_equality_no_match(self):
        assert evaluate_condition("mode == generate", {"mode": "file"}) is False

    def test_equality_field_missing(self):
        assert evaluate_condition("mode == generate", {}) is False

    def test_inequality_match(self):
        assert evaluate_condition("mode != file", {"mode": "generate"}) is True

    def test_inequality_no_match(self):
        assert evaluate_condition("mode != file", {"mode": "file"}) is False

    def test_inequality_field_missing(self):
        # Missing field doesn't equal value -> True
        assert evaluate_condition("mode != file", {}) is True

    def test_in_list_match(self):
        assert evaluate_condition(
            "mode in [generate, image, photos]", {"mode": "image"}
        ) is True

    def test_in_list_no_match(self):
        assert evaluate_condition(
            "mode in [generate, image, photos]", {"mode": "file"}
        ) is False

    def test_in_list_field_missing(self):
        assert evaluate_condition("mode in [generate, image]", {}) is False

    def test_bare_field_truthy(self):
        assert evaluate_condition("color", {"color": "black"}) is True

    def test_bare_field_empty_string(self):
        assert evaluate_condition("color", {"color": ""}) is False

    def test_bare_field_none(self):
        assert evaluate_condition("color", {"color": None}) is False

    def test_bare_field_missing(self):
        assert evaluate_condition("color", {}) is False

    def test_bare_field_false(self):
        assert evaluate_condition("auto_print", {"auto_print": False}) is False

    def test_bare_field_zero(self):
        # 0 is falsy
        assert evaluate_condition("count", {"count": 0}) is False

    def test_whitespace_trimmed(self):
        assert evaluate_condition("  mode == generate  ", {"mode": "generate"}) is True

    def test_in_list_with_spaces(self):
        assert evaluate_condition(
            "mode in [ generate , image , photos ]", {"mode": "photos"}
        ) is True


class TestShouldShowField:
    def test_none_condition_always_true(self):
        assert should_show_field(None, {}) is True

    def test_empty_condition_always_true(self):
        assert should_show_field("", {}) is True

    def test_delegates_to_evaluate_condition(self):
        assert should_show_field("mode == generate", {"mode": "generate"}) is True
        assert should_show_field("mode == generate", {"mode": "file"}) is False
