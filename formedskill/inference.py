"""
formedskill.inference — Answer extraction, default application, alias resolution,
type coercion for all 5 field types.

Handles chatty models that add explanations, "DEFAULT"/"NONE" sentinels,
and edge cases like empty strings.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from formedskill.schema import Field

# Sentinel strings LLMs use to signal "I don't know" or "use default"
_DEFAULT_SENTINELS = frozenset({
    "default", "none", "n/a", "not specified", "not mentioned",
    "unknown", "unspecified", "not provided", "null", "nil",
    "not applicable", "not available", "not given", "na",
})


def extract_answer(raw: str, field: Field) -> Any:
    """
    Full extraction pipeline for a single field:
      1. Strip whitespace and surrounding quotes
      2. Detect DEFAULT/NONE sentinels -> apply field default
      3. Resolve aliases
      4. Type coerce
      5. For options fields, fuzzy-match against valid keys

    Returns the extracted value, or field.default if extraction fails.
    """
    cleaned = _clean_raw(raw)

    # Sentinel check
    if _is_sentinel(cleaned):
        return field.default

    # Alias resolution (before type coercion)
    resolved = _resolve_alias(cleaned, field)

    # Type coercion
    coerced = _coerce(resolved, field)

    return coerced


def _clean_raw(raw: str) -> str:
    """
    Clean a raw LLM response.

    Handles:
    - Leading/trailing whitespace
    - Surrounding quotes (single or double)
    - "The answer is X" / "Value: X" style prefixes
    - Markdown bold/italic
    """
    text = raw.strip()

    # Strip markdown formatting
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"`+", "", text)

    # Strip surrounding quotes
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1].strip()

    # Strip common chatty prefixes: "The value is X", "Answer: X", etc.
    prefixes = [
        r"^(?:the\s+)?(?:answer|value|result)\s+is\s*:?\s*",
        r"^(?:field\s+\w+\s*:?\s*)",
        r"^(?:\w+\s*:\s*)",  # "mode: generate" -> "generate"
    ]
    for pattern in prefixes:
        m = re.match(pattern, text, re.IGNORECASE)
        if m and m.end() < len(text):
            candidate = text[m.end():].strip().strip('"\'')
            # Only apply if the remaining part looks like a single value
            if "\n" not in candidate and len(candidate) < 200:
                text = candidate
                break

    # Remove trailing punctuation that isn't part of the value
    text = text.rstrip(".,;!")

    return text.strip()


def _is_sentinel(text: str) -> bool:
    """Return True if text is a sentinel meaning 'use the default'."""
    lower = text.lower().strip()
    if lower in _DEFAULT_SENTINELS or lower == "":
        return True
    # Catch verbose non-answers from chatty models:
    # "(Note: no color was mentioned...)" / "(No reviewers specified)" / "(empty)"
    verbose_patterns = [
        r"^\s*\(note[\s:]",          # (Note: ...)
        r"^\s*\(no\s+.+(provided|mentioned|specified|given)\)",  # (No X specified)
        r"^\s*\(empty\)",            # (empty)
        r"^\s*\(not\s+",            # (Not specified...)
        r"^\s*\(none\)",            # (None)
        r"^\s*\(blank\)",           # (blank)
        r"^\s*\(n/a\)",             # (N/A)
        r"not\s+(specified|mentioned|provided|given)$",  # ends with "not specified"
        r"no\s+\w+\s+(was\s+)?(specified|mentioned|provided|given)",  # "no X was specified"
        r"^blank$",                  # just "blank"
        r"^empty$",                  # just "empty"
        r"</think>",                 # leaked reasoning tokens
        r"<think>",
    ]
    for pat in verbose_patterns:
        if re.search(pat, lower):
            return True
    return False


def _resolve_alias(text: str, field: Field) -> str:
    """Resolve alias mappings. Case-sensitive first, then case-insensitive fallback."""
    if not field.aliases:
        return text

    # Exact match
    if text in field.aliases:
        return field.aliases[text]

    # Case-insensitive match
    text_lower = text.lower()
    for alias, target in field.aliases.items():
        if alias.lower() == text_lower:
            return target

    return text


def _coerce(text: str, field: Field) -> Any:
    """Type-coerce the cleaned text to the field's declared type."""
    if field.type == "text":
        return text if text else field.default

    elif field.type == "number":
        return _coerce_number(text, field.default)

    elif field.type == "options":
        return _coerce_options(text, field)

    elif field.type == "boolean":
        return _coerce_boolean(text, field.default)

    elif field.type == "json_array":
        return _coerce_json_array(text, field.default)

    # Unknown type — return as text
    return text


def _coerce_number(text: str, default: Any) -> Any:
    """Parse a number from text, supporting unit conversions (cm -> mm, inches -> mm)."""
    # Strip non-numeric suffixes: "50mm" -> "50", "6cm" -> need conversion
    text = text.strip()

    # Unit conversion
    cm_match = re.match(r"^([\d.]+)\s*cm$", text, re.IGNORECASE)
    if cm_match:
        try:
            return int(float(cm_match.group(1)) * 10)
        except ValueError:
            pass

    inch_match = re.match(r"^([\d.]+)\s*(?:in|inch|inches|\")?$", text, re.IGNORECASE)
    if inch_match and ("in" in text.lower() or '"' in text):
        try:
            return int(float(inch_match.group(1)) * 25.4)
        except ValueError:
            pass

    # Strip units from the end
    numeric_str = re.sub(r"[a-zA-Z\s\"']+$", "", text).strip()

    try:
        val = float(numeric_str)
        # Return int if whole number
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return default


def _coerce_options(text: str, field: Field) -> Any:
    """
    Match text against valid option keys. Case-insensitive fuzzy matching.
    Returns matched key or field.default if no match.
    """
    if not field.options:
        return text or field.default

    # Exact match
    if text in field.options:
        return text

    # Case-insensitive exact match
    text_lower = text.lower()
    for key in field.options:
        if key.lower() == text_lower:
            return key

    # Prefix match (model returned partial key)
    for key in field.options:
        if key.lower().startswith(text_lower) or text_lower.startswith(key.lower()):
            return key

    # Check if text appears in an option description
    for key, desc in field.options.items():
        if text_lower in desc.lower():
            return key

    # No match — return default
    return field.default


def _coerce_boolean(text: str, default: Any) -> Any:
    """Convert text to bool. Handles yes/no/true/false/1/0."""
    text_lower = text.lower().strip()
    if text_lower in {"yes", "true", "1", "on", "enabled", "y", "affirmative"}:
        return True
    if text_lower in {"no", "false", "0", "off", "disabled", "n", "negative"}:
        return False
    return default if default is not None else False


def _coerce_json_array(text: str, default: Any) -> Any:
    """
    Parse text as a JSON array. Falls back to wrapping in a list or
    splitting by commas/newlines.
    """
    text = text.strip()
    if not text:
        return default if default is not None else []

    # Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return [result]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract a JSON array from within the text
    array_match = re.search(r"\[.*?\]", text, re.DOTALL)
    if array_match:
        try:
            result = json.loads(array_match.group())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Split by newlines or commas
    if "\n" in text:
        items = [line.strip().strip("-").strip().strip('"\'') for line in text.splitlines()]
        items = [item for item in items if item]
        return items if items else (default or [])

    if "," in text:
        items = [item.strip().strip('"\'') for item in text.split(",")]
        items = [item for item in items if item]
        return items if items else (default or [])

    # Single item — wrap in list
    return [text]


def apply_default(value: Any, field: Field) -> Any:
    """Return field.default if value is None or empty sentinel."""
    if value is None:
        return field.default
    if isinstance(value, str) and _is_sentinel(value):
        return field.default
    return value
