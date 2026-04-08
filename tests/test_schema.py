"""Tests for formedskill.schema — SkillForm loading and validation."""

import pytest
from pathlib import Path

from formedskill.schema import (
    Field,
    Action,
    Confirmation,
    SkillForm,
    SkillMeta,
    Validation,
    load_skill,
    validate_skill,
    _load_yaml,
    _minimal_yaml_parse,
)

SKILLS_DIR = Path(__file__).parent.parent / "formedskill" / "skills"


# ── _minimal_yaml_parse ───────────────────────────────────────────────────────

class TestMinimalYamlParse:
    def test_simple_key_value(self):
        result = _minimal_yaml_parse("name: hello\nversion: 1.0.0\n")
        assert result["name"] == "hello"
        assert result["version"] == "1.0.0"

    def test_nested_dict(self):
        text = "skill:\n  name: test\n  version: 2.0.0\n"
        result = _minimal_yaml_parse(text)
        assert result["skill"]["name"] == "test"
        assert result["skill"]["version"] == "2.0.0"

    def test_boolean_values(self):
        result = _minimal_yaml_parse("required: true\nenabled: false\n")
        assert result["required"] is True
        assert result["enabled"] is False

    def test_null_value(self):
        result = _minimal_yaml_parse("default: null\n")
        assert result["default"] is None

    def test_integer_value(self):
        result = _minimal_yaml_parse("count: 42\n")
        assert result["count"] == 42

    def test_list_items(self):
        text = "tags:\n  - foo\n  - bar\n  - baz\n"
        result = _minimal_yaml_parse(text)
        assert result["tags"] == ["foo", "bar", "baz"]

    def test_quoted_string(self):
        result = _minimal_yaml_parse('url: "http://localhost:8080"\n')
        assert result["url"] == "http://localhost:8080"

    def test_comments_ignored(self):
        text = "# This is a comment\nname: test  # inline comment\n"
        result = _minimal_yaml_parse(text)
        assert result["name"] == "test"


# ── Field ─────────────────────────────────────────────────────────────────────

class TestField:
    def test_minimal_field(self):
        f = Field.from_dict({"id": "topic", "ask": "What topic?"})
        assert f.id == "topic"
        assert f.ask == "What topic?"
        assert f.type == "text"
        assert f.required is False
        assert f.default is None
        assert f.options == {}
        assert f.aliases == {}

    def test_options_field(self):
        f = Field.from_dict({
            "id": "mode",
            "ask": "Mode?",
            "type": "options",
            "options": {"a": "Option A", "b": "Option B"},
            "required": True,
        })
        assert f.type == "options"
        assert f.required is True
        assert "a" in f.options
        assert "b" in f.options

    def test_type_inferred_from_options(self):
        f = Field.from_dict({
            "id": "mode",
            "ask": "Mode?",
            "options": {"x": "X", "y": "Y"},
        })
        assert f.type == "options"

    def test_number_field_with_validation(self):
        f = Field.from_dict({
            "id": "size",
            "ask": "Size?",
            "type": "number",
            "default": 50,
            "validation": {"min": 5, "max": 500},
        })
        assert f.type == "number"
        assert f.default == 50
        assert f.validation.min == 5
        assert f.validation.max == 500

    def test_boolean_field(self):
        f = Field.from_dict({
            "id": "auto_print",
            "ask": "Auto print?",
            "type": "boolean",
            "default": True,
        })
        assert f.type == "boolean"
        assert f.default is True

    def test_json_array_field(self):
        f = Field.from_dict({
            "id": "images",
            "ask": "Image paths?",
            "type": "json_array",
        })
        assert f.type == "json_array"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid field type"):
            Field.from_dict({"id": "x", "ask": "X?", "type": "invalid_type"})

    def test_aliases(self):
        f = Field.from_dict({
            "id": "printer",
            "ask": "Which printer?",
            "type": "options",
            "options": {"auto": "Auto", "SN123": "Printer A"},
            "aliases": {"P2D2": "SN123"},
        })
        assert f.aliases["P2D2"] == "SN123"

    def test_show_when(self):
        f = Field.from_dict({
            "id": "prompt",
            "ask": "Prompt?",
            "show_when": "mode == generate",
        })
        assert f.show_when == "mode == generate"


# ── Action ────────────────────────────────────────────────────────────────────

class TestAction:
    def test_http_action(self):
        a = Action.from_dict({
            "type": "http",
            "method": "POST",
            "url": "http://localhost:8080/api/start",
        })
        assert a.type == "http"
        assert a.method == "POST"
        assert a.url == "http://localhost:8080/api/start"

    def test_method_uppercased(self):
        a = Action.from_dict({"type": "http", "method": "post", "url": "http://x"})
        assert a.method == "POST"

    def test_shell_action(self):
        a = Action.from_dict({
            "type": "shell",
            "command": "echo {{name}}",
        })
        assert a.type == "shell"
        assert a.command == "echo {{name}}"

    def test_tool_call_action(self):
        a = Action.from_dict({
            "type": "tool_call",
            "tool_name": "my_tool",
        })
        assert a.type == "tool_call"
        assert a.tool_name == "my_tool"

    def test_invalid_action_type(self):
        with pytest.raises(ValueError, match="Invalid action type"):
            Action.from_dict({"type": "ftp"})

    def test_payload_map(self):
        a = Action.from_dict({
            "type": "http",
            "method": "POST",
            "url": "http://x",
            "payload_map": {"prompt": "description"},
        })
        assert a.payload_map["prompt"] == "description"

    def test_default_timeout(self):
        a = Action.from_dict({"type": "http", "method": "GET", "url": "http://x"})
        assert a.timeout == 60


# ── SkillForm ─────────────────────────────────────────────────────────────────

class TestSkillForm:
    def _minimal_data(self):
        return {
            "skill": {"name": "test", "version": "1.0.0"},
            "action": {"type": "http", "method": "POST", "url": "http://x"},
            "fields": [{"id": "topic", "ask": "Topic?"}],
        }

    def test_from_dict_minimal(self):
        form = SkillForm.from_dict(self._minimal_data())
        assert form.name == "test"
        assert len(form.fields) == 1

    def test_missing_skill_raises(self):
        data = self._minimal_data()
        del data["skill"]
        with pytest.raises(ValueError, match="Missing required.*'skill'"):
            SkillForm.from_dict(data)

    def test_missing_action_raises(self):
        data = self._minimal_data()
        del data["action"]
        with pytest.raises(ValueError, match="Missing required.*'action'"):
            SkillForm.from_dict(data)

    def test_missing_fields_raises(self):
        data = self._minimal_data()
        del data["fields"]
        with pytest.raises(ValueError, match="Missing required.*'fields'"):
            SkillForm.from_dict(data)

    def test_name_property(self):
        form = SkillForm.from_dict(self._minimal_data())
        assert form.name == "test"

    def test_description_property(self):
        data = self._minimal_data()
        data["skill"]["description"] = "A test skill"
        form = SkillForm.from_dict(data)
        assert form.description == "A test skill"


# ── validate_skill ────────────────────────────────────────────────────────────

class TestValidateSkill:
    def _make_form(self, fields=None, action=None):
        data = {
            "skill": {"name": "test"},
            "action": action or {"type": "http", "method": "POST", "url": "http://x"},
            "fields": fields or [{"id": "topic", "ask": "Topic?"}],
        }
        return SkillForm.from_dict(data)

    def test_valid_form_has_no_errors(self):
        form = self._make_form()
        assert validate_skill(form) == []

    def test_duplicate_field_ids(self):
        form = self._make_form(fields=[
            {"id": "x", "ask": "X?"},
            {"id": "x", "ask": "X again?"},
        ])
        errors = validate_skill(form)
        assert any("Duplicate" in e for e in errors)

    def test_show_when_references_nonexistent_field(self):
        form = self._make_form(fields=[
            {"id": "topic", "ask": "Topic?"},
            {"id": "detail", "ask": "Detail?", "show_when": "nonexistent == true"},
        ])
        errors = validate_skill(form)
        assert any("nonexistent" in e for e in errors)

    def test_options_field_with_no_options(self):
        form = self._make_form(fields=[
            {"id": "mode", "ask": "Mode?", "type": "options"},
        ])
        errors = validate_skill(form)
        assert any("options" in e.lower() for e in errors)

    def test_alias_pointing_to_invalid_option(self):
        form = self._make_form(fields=[
            {
                "id": "printer",
                "ask": "Printer?",
                "type": "options",
                "options": {"auto": "Auto"},
                "aliases": {"P2D2": "INVALID_KEY"},
            }
        ])
        errors = validate_skill(form)
        assert any("INVALID_KEY" in e for e in errors)

    def test_http_action_missing_url(self):
        data = {
            "skill": {"name": "test"},
            "action": {"type": "http", "method": "POST", "url": ""},
            "fields": [{"id": "x", "ask": "X?"}],
        }
        # Empty URL — SkillForm.from_dict succeeds, validate_skill catches it
        form = SkillForm.from_dict(data)
        errors = validate_skill(form)
        assert any("url" in e.lower() for e in errors)


# ── load_skill (integration) ──────────────────────────────────────────────────

class TestLoadSkill:
    def test_load_snailprint(self):
        path = SKILLS_DIR / "snailprint.yaml"
        if not path.exists():
            pytest.skip("snailprint.yaml not found")
        form = load_skill(path)
        assert form.name == "snailprint"
        assert len(form.fields) > 5
        assert form.action.type == "http"

    def test_load_snailstudio(self):
        path = SKILLS_DIR / "snailstudio.yaml"
        if not path.exists():
            pytest.skip("snailstudio.yaml not found")
        form = load_skill(path)
        assert form.name == "snailstudio"
        assert len(form.fields) >= 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_skill("/nonexistent/path/skill.yaml")
