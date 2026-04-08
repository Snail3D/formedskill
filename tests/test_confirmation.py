"""Tests for formedskill.confirmation — template rendering."""

import pytest
from formedskill.schema import SkillForm, Confirmation
from formedskill.confirmation import (
    render_confirmation,
    _render_template,
    _process_conditionals,
    confirmation_is_enabled,
)


def make_form(confirmation_data=None, fields=None):
    data = {
        "skill": {"name": "test"},
        "action": {"type": "http", "method": "POST", "url": "http://x"},
        "fields": fields or [{"id": "topic", "ask": "Topic?"}],
    }
    if confirmation_data is not None:
        data["confirmation"] = confirmation_data
    return SkillForm.from_dict(data)


class TestRenderTemplate:
    def test_simple_substitution(self):
        result = _render_template("Hello {{name}}!", {"name": "world"})
        assert result == "Hello world!"

    def test_missing_field_empty_string(self):
        result = _render_template("Hello {{name}}!", {})
        assert result == "Hello !"

    def test_none_value_empty_string(self):
        result = _render_template("{{color}}", {"color": None})
        assert result == ""

    def test_list_value_joined(self):
        result = _render_template("{{images}}", {"images": ["a.jpg", "b.jpg"]})
        assert result == "a.jpg, b.jpg"

    def test_multiple_substitutions(self):
        result = _render_template("{{a}} and {{b}}", {"a": "x", "b": "y"})
        assert result == "x and y"


class TestProcessConditionals:
    def test_truthy_field_included(self):
        template = "{% if color %}Color: {{color}}{% endif %}"
        result = _process_conditionals(template, {"color": "black"})
        assert "Color: {{color}}" in result

    def test_falsy_field_excluded(self):
        template = "{% if color %}Color: {{color}}{% endif %}"
        result = _process_conditionals(template, {"color": None})
        assert "Color" not in result

    def test_missing_field_excluded(self):
        template = "{% if color %}Color: {{color}}{% endif %}"
        result = _process_conditionals(template, {})
        assert "Color" not in result

    def test_equality_condition_true(self):
        template = '{% if mode == "generate" %}Generating{% endif %}'
        result = _process_conditionals(template, {"mode": "generate"})
        assert "Generating" in result

    def test_equality_condition_false(self):
        template = '{% if mode == "generate" %}Generating{% endif %}'
        result = _process_conditionals(template, {"mode": "file"})
        assert "Generating" not in result

    def test_multiline_block(self):
        template = "Line 1\n{% if show %}Line 2\nLine 3{% endif %}\nLine 4"
        result = _process_conditionals(template, {"show": True})
        assert "Line 2" in result
        assert "Line 3" in result

    def test_multiline_block_excluded(self):
        template = "Line 1\n{% if show %}Line 2\nLine 3{% endif %}\nLine 4"
        result = _process_conditionals(template, {"show": False})
        assert "Line 2" not in result
        assert "Line 4" in result


class TestRenderConfirmation:
    def test_with_template(self):
        form = make_form(
            confirmation_data={
                "enabled": True,
                "template": "Ready to run {{topic}}!\nProceed?",
            }
        )
        result = render_confirmation(form, {"topic": "dragons"})
        assert "dragons" in result
        assert "Proceed?" in result

    def test_disabled_returns_empty(self):
        form = make_form(confirmation_data={"enabled": False})
        result = render_confirmation(form, {"topic": "test"})
        assert result == ""

    def test_no_confirmation_uses_default(self):
        form = make_form()  # No confirmation key
        result = render_confirmation(form, {"topic": "dragons"})
        assert "test" in result  # skill name
        assert "topic" in result

    def test_conditional_field_shown(self):
        form = make_form(
            confirmation_data={
                "enabled": True,
                "template": "{% if color %}Color: {{color}}{% endif %}",
            }
        )
        result = render_confirmation(form, {"color": "black"})
        assert "Color: black" in result

    def test_conditional_field_hidden(self):
        form = make_form(
            confirmation_data={
                "enabled": True,
                "template": "{% if color %}Color: {{color}}{% endif %}",
            }
        )
        result = render_confirmation(form, {})
        assert "Color" not in result


class TestConfirmationIsEnabled:
    def test_enabled_by_default(self):
        form = make_form()  # no confirmation key
        assert confirmation_is_enabled(form) is True

    def test_explicitly_enabled(self):
        form = make_form(confirmation_data={"enabled": True, "template": ""})
        assert confirmation_is_enabled(form) is True

    def test_explicitly_disabled(self):
        form = make_form(confirmation_data={"enabled": False})
        assert confirmation_is_enabled(form) is False
