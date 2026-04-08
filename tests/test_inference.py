"""Tests for formedskill.inference — answer extraction and type coercion."""

import pytest
from formedskill.schema import Field
from formedskill.inference import (
    extract_answer,
    _clean_raw,
    _is_sentinel,
    _coerce_number,
    _coerce_boolean,
    _coerce_json_array,
    _coerce_options,
    _resolve_alias,
)


def make_field(**kwargs) -> Field:
    defaults = {"id": "test", "ask": "Test?", "type": "text"}
    defaults.update(kwargs)
    return Field.from_dict(defaults)


# ── _clean_raw ────────────────────────────────────────────────────────────────

class TestCleanRaw:
    def test_strips_whitespace(self):
        assert _clean_raw("  hello  ") == "hello"

    def test_strips_double_quotes(self):
        assert _clean_raw('"generate"') == "generate"

    def test_strips_single_quotes(self):
        assert _clean_raw("'generate'") == "generate"

    def test_strips_markdown_bold(self):
        assert _clean_raw("**generate**") == "generate"

    def test_strips_backticks(self):
        assert _clean_raw("`generate`") == "generate"

    def test_strips_trailing_punctuation(self):
        assert _clean_raw("generate.") == "generate"

    def test_preserves_internal_content(self):
        assert _clean_raw("dragon figurine") == "dragon figurine"

    def test_empty_string(self):
        assert _clean_raw("") == ""


# ── _is_sentinel ──────────────────────────────────────────────────────────────

class TestIsSentinel:
    def test_default_is_sentinel(self):
        assert _is_sentinel("default") is True
        assert _is_sentinel("DEFAULT") is True

    def test_none_is_sentinel(self):
        assert _is_sentinel("none") is True
        assert _is_sentinel("NONE") is True

    def test_na_is_sentinel(self):
        assert _is_sentinel("n/a") is True
        assert _is_sentinel("N/A") is True

    def test_not_specified_is_sentinel(self):
        assert _is_sentinel("not specified") is True

    def test_empty_string_is_sentinel(self):
        assert _is_sentinel("") is True

    def test_real_value_not_sentinel(self):
        assert _is_sentinel("generate") is False
        assert _is_sentinel("PLA") is False
        assert _is_sentinel("50") is False


# ── _coerce_number ────────────────────────────────────────────────────────────

class TestCoerceNumber:
    def test_integer(self):
        assert _coerce_number("50", 10) == 50

    def test_float(self):
        assert _coerce_number("3.14", 1) == 3.14

    def test_returns_int_for_whole_float(self):
        assert _coerce_number("50.0", 10) == 50
        assert isinstance(_coerce_number("50.0", 10), int)

    def test_strips_mm_suffix(self):
        assert _coerce_number("80mm", 50) == 80

    def test_cm_conversion(self):
        assert _coerce_number("5cm", 50) == 50

    def test_invalid_returns_default(self):
        assert _coerce_number("dragon", 50) == 50

    def test_empty_returns_default(self):
        assert _coerce_number("", 50) == 50


# ── _coerce_boolean ───────────────────────────────────────────────────────────

class TestCoerceBoolean:
    def test_yes(self):
        assert _coerce_boolean("yes", None) is True

    def test_true(self):
        assert _coerce_boolean("true", None) is True

    def test_1(self):
        assert _coerce_boolean("1", None) is True

    def test_no(self):
        assert _coerce_boolean("no", None) is False

    def test_false(self):
        assert _coerce_boolean("false", None) is False

    def test_0(self):
        assert _coerce_boolean("0", None) is False

    def test_case_insensitive(self):
        assert _coerce_boolean("YES", None) is True
        assert _coerce_boolean("False", None) is False

    def test_unknown_uses_default(self):
        assert _coerce_boolean("maybe", True) is True
        assert _coerce_boolean("maybe", False) is False


# ── _coerce_json_array ────────────────────────────────────────────────────────

class TestCoerceJsonArray:
    def test_valid_json_array(self):
        result = _coerce_json_array('["a", "b", "c"]', None)
        assert result == ["a", "b", "c"]

    def test_embedded_json_array(self):
        result = _coerce_json_array('Here are the paths: ["x.jpg", "y.jpg"]', None)
        assert result == ["x.jpg", "y.jpg"]

    def test_comma_separated_fallback(self):
        result = _coerce_json_array("a.jpg, b.jpg, c.jpg", None)
        assert result == ["a.jpg", "b.jpg", "c.jpg"]

    def test_newline_separated_fallback(self):
        result = _coerce_json_array("a.jpg\nb.jpg\nc.jpg", None)
        assert result == ["a.jpg", "b.jpg", "c.jpg"]

    def test_single_item_wrapped(self):
        result = _coerce_json_array("a.jpg", None)
        assert result == ["a.jpg"]

    def test_empty_returns_default(self):
        assert _coerce_json_array("", ["default"]) == ["default"]
        assert _coerce_json_array("", None) == []


# ── _coerce_options ───────────────────────────────────────────────────────────

class TestCoerceOptions:
    def make_options_field(self, options):
        return make_field(
            id="mode", type="options",
            options={k: v for k, v in options.items()},
            default=None,
        )

    def test_exact_match(self):
        f = self.make_options_field({"generate": "Create", "file": "File"})
        assert _coerce_options("generate", f) == "generate"

    def test_case_insensitive_match(self):
        f = self.make_options_field({"PLA": "Plastic", "PETG": "Strong"})
        assert _coerce_options("pla", f) == "PLA"

    def test_no_match_returns_default(self):
        f = self.make_options_field({"a": "A", "b": "B"})
        assert _coerce_options("z", f) is None

    def test_prefix_match(self):
        f = self.make_options_field({"generate": "Create", "file": "File"})
        assert _coerce_options("gen", f) == "generate"


# ── _resolve_alias ────────────────────────────────────────────────────────────

class TestResolveAlias:
    def make_alias_field(self, aliases):
        return make_field(
            id="printer", type="options",
            options={"auto": "Auto", "SN123": "Printer"},
            aliases=aliases,
        )

    def test_exact_alias(self):
        f = self.make_alias_field({"P2D2": "SN123"})
        assert _resolve_alias("P2D2", f) == "SN123"

    def test_case_insensitive_alias(self):
        f = self.make_alias_field({"P2D2": "SN123"})
        assert _resolve_alias("p2d2", f) == "SN123"

    def test_no_alias_match(self):
        f = self.make_alias_field({"P2D2": "SN123"})
        assert _resolve_alias("unknown", f) == "unknown"

    def test_no_aliases_returns_unchanged(self):
        f = make_field(id="topic", type="text")
        assert _resolve_alias("anything", f) == "anything"


# ── extract_answer (full pipeline) ────────────────────────────────────────────

class TestExtractAnswer:
    def test_text_field(self):
        f = make_field(id="topic", type="text")
        assert extract_answer("Mariana Trench", f) == "Mariana Trench"

    def test_sentinel_returns_default(self):
        f = make_field(id="topic", type="text", default="fallback")
        assert extract_answer("DEFAULT", f) == "fallback"

    def test_number_field(self):
        f = make_field(id="size", type="number", default=50)
        assert extract_answer("80", f) == 80

    def test_number_field_with_unit(self):
        f = make_field(id="size", type="number", default=50)
        assert extract_answer("80mm", f) == 80

    def test_boolean_true(self):
        f = make_field(id="auto", type="boolean", default=True)
        assert extract_answer("yes", f) is True

    def test_boolean_false(self):
        f = make_field(id="auto", type="boolean", default=True)
        assert extract_answer("no", f) is False

    def test_options_field(self):
        f = make_field(
            id="mode", type="options", default="generate",
            options={"generate": "Gen", "file": "File"},
        )
        assert extract_answer("generate", f) == "generate"

    def test_json_array_field(self):
        f = make_field(id="images", type="json_array")
        result = extract_answer('["a.jpg", "b.jpg"]', f)
        assert result == ["a.jpg", "b.jpg"]

    def test_alias_resolution_in_pipeline(self):
        f = make_field(
            id="printer", type="options", default="auto",
            options={"auto": "Auto", "SN123": "Printer A"},
            aliases={"P2D2": "SN123"},
        )
        assert extract_answer("P2D2", f) == "SN123"

    def test_chatty_response_cleaned(self):
        f = make_field(id="topic", type="text")
        # Model adds explanation — should still extract cleanly
        assert extract_answer("  Mariana Trench  ", f) == "Mariana Trench"
