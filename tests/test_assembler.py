"""Tests for formedskill.assembler — payload assembly and action execution."""

import json
import pytest
from formedskill.schema import SkillForm, Action
from formedskill.assembler import assemble_payload, preview_action, _template_substitute


def make_form(action_data=None, fields=None):
    data = {
        "skill": {"name": "test"},
        "action": action_data or {"type": "http", "method": "POST", "url": "http://localhost:8080/api"},
        "fields": fields or [{"id": "topic", "ask": "Topic?"}],
    }
    return SkillForm.from_dict(data)


class TestAssemblePayload:
    def test_basic_collected_values(self):
        form = make_form()
        collected = {"topic": "dragons", "segments": 6}
        payload = assemble_payload(form.action, collected)
        assert payload == {"topic": "dragons", "segments": 6}

    def test_none_values_excluded(self):
        form = make_form()
        collected = {"topic": "dragons", "color": None}
        payload = assemble_payload(form.action, collected)
        assert "color" not in payload

    def test_payload_map_remaps_key(self):
        form = make_form(action_data={
            "type": "http",
            "method": "POST",
            "url": "http://x",
            "payload_map": {"prompt": "description"},
        })
        collected = {"prompt": "a dragon", "filament": "PLA"}
        payload = assemble_payload(form.action, collected)
        assert "description" in payload
        assert payload["description"] == "a dragon"
        assert "prompt" not in payload
        assert payload["filament"] == "PLA"

    def test_payload_map_missing_field_ignored(self):
        form = make_form(action_data={
            "type": "http",
            "method": "POST",
            "url": "http://x",
            "payload_map": {"prompt": "description"},
        })
        # "prompt" not in collected — map is just skipped
        collected = {"topic": "test"}
        payload = assemble_payload(form.action, collected)
        assert payload == {"topic": "test"}

    def test_empty_collected(self):
        form = make_form()
        assert assemble_payload(form.action, {}) == {}


class TestTemplateSubstitute:
    def test_simple_substitution(self):
        result = _template_substitute("echo {{name}}", {"name": "hello"})
        assert result == "echo hello"

    def test_missing_key_unchanged(self):
        result = _template_substitute("echo {{name}}", {})
        assert result == "echo {{name}}"

    def test_list_value_json_encoded(self):
        result = _template_substitute("files={{images}}", {"images": ["a.jpg", "b.jpg"]})
        assert result == 'files=["a.jpg", "b.jpg"]'

    def test_multiple_substitutions(self):
        result = _template_substitute("{{a}} and {{b}}", {"a": "x", "b": "y"})
        assert result == "x and y"


class TestPreviewAction:
    def test_http_preview(self):
        form = make_form()
        collected = {"topic": "dragons"}
        preview = preview_action(form, collected)
        assert "POST" in preview
        assert "http://localhost:8080/api" in preview
        assert "dragons" in preview

    def test_shell_preview(self):
        form = make_form(action_data={
            "type": "shell",
            "command": "echo {{topic}}",
        })
        collected = {"topic": "test"}
        preview = preview_action(form, collected)
        assert "echo test" in preview

    def test_tool_call_preview(self):
        form = make_form(action_data={
            "type": "tool_call",
            "tool_name": "my_tool",
        })
        collected = {"topic": "test"}
        preview = preview_action(form, collected)
        assert "my_tool" in preview
