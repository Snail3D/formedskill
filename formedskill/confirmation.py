"""
formedskill.confirmation — Render confirmation templates before action execution.

Supports Jinja2-style {{field_id}} substitution and basic {% if field %} conditionals.
No Jinja2 dependency — pure stdlib implementation sufficient for skill templates.
"""

from __future__ import annotations

import re
from typing import Any

from formedskill.schema import Confirmation, SkillForm


def render_confirmation(
    form: SkillForm, collected: dict[str, Any]
) -> str:
    """
    Render the skill's confirmation template with collected values.

    Returns the rendered string, or a default summary if no template is defined.
    If confirmation is disabled, returns empty string.
    """
    conf = form.confirmation

    if conf is not None and not conf.enabled:
        return ""

    template = (conf.template if conf and conf.template else None) or _default_template(form)
    return _render_template(template, collected)


def _render_template(template: str, values: dict[str, Any]) -> str:
    """
    Render a template string with {{field}} substitution and {% if %} blocks.

    Supported syntax:
      {{field_id}}                   — substitute value or empty string
      {% if field_id %}...{% endif %} — include block only if field is truthy
      {% if field_id == "value" %}...{% endif %} — include block if field equals value
    """
    # Process {% if %} / {% endif %} blocks first
    result = _process_conditionals(template, values)

    # Substitute {{field_id}} placeholders
    def sub(m: re.Match) -> str:
        key = m.group(1).strip()
        val = values.get(key)
        if val is None:
            return ""
        if isinstance(val, list):
            return ", ".join(str(x) for x in val)
        return str(val)

    result = re.sub(r"\{\{(\w+)\}\}", sub, result)

    # Clean up blank lines left by skipped conditionals
    lines = result.splitlines()
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


def _process_conditionals(template: str, values: dict[str, Any]) -> str:
    """
    Process {% if %} / {% endif %} blocks recursively.

    Handles:
      {% if field %}
      {% if field == "value" %}
      {% if field != "value" %}
    """
    # Pattern: {% if CONDITION %}...{% endif %}
    pattern = re.compile(
        r"\{%\s*if\s+(.+?)\s*%\}(.*?)\{%\s*endif\s*%\}",
        re.DOTALL,
    )

    def replacer(m: re.Match) -> str:
        condition = m.group(1).strip()
        block = m.group(2)

        if _eval_template_condition(condition, values):
            # Recurse to handle nested ifs
            return _process_conditionals(block, values)
        return ""

    # Keep replacing until no more matches (handles nested blocks)
    prev = None
    result = template
    while prev != result:
        prev = result
        result = pattern.sub(replacer, result)

    return result


def _eval_template_condition(condition: str, values: dict[str, Any]) -> bool:
    """Evaluate a simple template condition."""
    condition = condition.strip()

    # field == "value" or field == value
    eq_match = re.match(r'^(\w+)\s*==\s*["\']?([^"\']+)["\']?$', condition)
    if eq_match:
        field, expected = eq_match.group(1), eq_match.group(2).strip()
        actual = values.get(field)
        return actual is not None and str(actual) == expected

    # field != "value"
    neq_match = re.match(r'^(\w+)\s*!=\s*["\']?([^"\']+)["\']?$', condition)
    if neq_match:
        field, expected = neq_match.group(1), neq_match.group(2).strip()
        actual = values.get(field)
        return actual is None or str(actual) != expected

    # Bare field name — truthy check
    if re.match(r'^[a-zA-Z_]\w*$', condition):
        val = values.get(condition)
        return bool(val) if val is not None else False

    return False


def _default_template(form: SkillForm) -> str:
    """Generate a sensible default confirmation template from field definitions."""
    lines = [f"Ready to run: {form.meta.name}"]
    lines.append("")
    for f in form.fields:
        lines.append(f"  {f.id}: {{{{{f.id}}}}}")
    lines.append("")
    lines.append("Proceed? (yes/no)")
    return "\n".join(lines)


def confirmation_is_enabled(form: SkillForm) -> bool:
    """Return True if the form has a confirmation step enabled."""
    if form.confirmation is None:
        return True  # Default: enabled
    return form.confirmation.enabled
