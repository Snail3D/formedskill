"""
formedskill.conditions — show_when evaluator.

Supports:
  field == value       (equality)
  field != value       (inequality)
  field in [a, b, c]  (membership)
  field               (truthy check)
"""

from __future__ import annotations

import re
from typing import Any


def evaluate_condition(condition: str, collected: dict[str, Any]) -> bool:
    """
    Evaluate a show_when condition string against already-collected values.

    Returns True if the field should be shown, False if it should be skipped.
    If condition is empty/None, always returns True.
    """
    if not condition:
        return True

    condition = condition.strip()

    # field == value
    if "==" in condition:
        field, _, value = condition.partition("==")
        field = field.strip()
        value = value.strip()
        actual = collected.get(field)
        if actual is None:
            return False
        return str(actual).strip() == value

    # field != value
    if "!=" in condition:
        field, _, value = condition.partition("!=")
        field = field.strip()
        value = value.strip()
        actual = collected.get(field)
        if actual is None:
            return True  # field not set means it doesn't equal value
        return str(actual).strip() != value

    # field in [a, b, c]
    if " in " in condition:
        parts = condition.split(" in ", 1)
        field = parts[0].strip()
        values_str = parts[1].strip()
        # Strip surrounding brackets
        values_str = values_str.strip("[]")
        values = [v.strip() for v in values_str.split(",") if v.strip()]
        actual = collected.get(field)
        if actual is None:
            return False
        return str(actual).strip() in values

    # Bare field name — truthy check
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", condition):
        val = collected.get(condition)
        return bool(val) if val is not None else False

    # Unknown condition syntax — default to showing the field
    return True


def should_show_field(field_show_when: str | None, collected: dict[str, Any]) -> bool:
    """Convenience wrapper. Returns True if field should be presented to LLM."""
    if not field_show_when:
        return True
    return evaluate_condition(field_show_when, collected)
