"""
formedskill.schema — SkillForm dataclass, YAML loader, schema validation.

Zero required dependencies. Falls back to a minimal YAML parser if PyYAML
is not installed (handles the simple cases we actually use in skill files).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── YAML loading ─────────────────────────────────────────────────────────────

def _load_yaml(text: str) -> dict:
    """Load YAML, preferring PyYAML but falling back to a minimal parser."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        return _minimal_yaml_parse(text)


def _minimal_yaml_parse(text: str) -> dict:
    """
    Minimal YAML parser sufficient for formedskill skill files.

    Handles:
    - Nested dicts via indentation
    - Lists (- item)
    - Quoted and unquoted scalars
    - Multiline scalars (|, >-)
    - Inline dicts/lists are NOT supported — install PyYAML for those
    """
    lines = text.splitlines()
    result, _ = _parse_block(lines, 0, 0)
    return result


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s or s.lower() == "null" or s == "~":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    # Quoted string — no comment stripping inside quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    # Strip inline comments from unquoted scalars: "value  # comment" -> "value"
    if " #" in s:
        s = s[:s.index(" #")].rstrip()
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _get_indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _parse_block(lines: list[str], start: int, base_indent: int) -> tuple[Any, int]:
    """Parse a YAML block starting at line `start` with `base_indent` indentation."""
    result: dict | list | None = None
    i = start

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = _get_indent(raw)

        # We've returned to a higher (or equal) level — stop
        if indent < base_indent:
            break

        # List item
        if stripped.startswith("- "):
            if result is None:
                result = []
            if not isinstance(result, list):
                break
            value_str = stripped[2:].strip()
            if ":" in value_str and not value_str.startswith('"'):
                # Inline dict item — treat as mapping
                sub: dict = {}
                key, _, val = value_str.partition(":")
                val = val.strip()
                if val:
                    sub[key.strip()] = _parse_scalar(val)
                i += 1
                # Collect more keys at this item's indent
                item_indent = indent + 2
                while i < len(lines):
                    r2 = lines[i]
                    s2 = r2.strip()
                    if not s2 or s2.startswith("#"):
                        i += 1
                        continue
                    if _get_indent(r2) < item_indent:
                        break
                    if ":" in s2:
                        # Handle quoted keys with colons
                        if s2.startswith('"') and '":' in s2:
                            eq = s2.index('":', 1)
                            k2 = s2[1:eq]
                            v2 = s2[eq + 2:].strip()
                        elif s2.startswith("'") and "':" in s2:
                            eq = s2.index("':", 1)
                            k2 = s2[1:eq]
                            v2 = s2[eq + 2:].strip()
                        else:
                            k2, _, v2 = s2.partition(":")
                            v2 = v2.strip()
                        if v2:
                            sub[k2.strip()] = _parse_scalar(v2)
                        else:
                            # Nested block
                            nested, i = _parse_block(lines, i + 1, _get_indent(r2) + 2)
                            sub[k2.strip()] = nested
                            continue
                    i += 1
                result.append(sub)
                continue
            else:
                result.append(_parse_scalar(value_str))
                i += 1
                continue

        # Bare "- " (next line is block)
        if stripped == "-":
            if result is None:
                result = []
            nested, i = _parse_block(lines, i + 1, indent + 2)
            result.append(nested)  # type: ignore[union-attr]
            continue

        # Key: value
        if ":" in stripped:
            if result is None:
                result = {}
            if not isinstance(result, dict):
                break

            # Handle quoted keys that contain colons, e.g. "clone:eric": "value"
            if stripped.startswith('"') and '":' in stripped:
                end_quote = stripped.index('":', 1)
                key = stripped[1:end_quote]  # strip surrounding quotes
                val_part = stripped[end_quote + 2:].strip()
            elif stripped.startswith("'") and "':" in stripped:
                end_quote = stripped.index("':", 1)
                key = stripped[1:end_quote]
                val_part = stripped[end_quote + 2:].strip()
            else:
                key, _, val_part = stripped.partition(":")
                key = key.strip()
                val_part = val_part.strip()

            # Multiline block scalar (| or >-)
            if val_part in ("|", ">", ">-", "|-"):
                block_lines = []
                i += 1
                child_indent: int | None = None
                while i < len(lines):
                    bl = lines[i]
                    bs = bl.rstrip()
                    if not bs:
                        block_lines.append("")
                        i += 1
                        continue
                    bi = _get_indent(bl)
                    if child_indent is None:
                        child_indent = bi
                    if bi < child_indent:
                        break
                    block_lines.append(bl[child_indent:].rstrip())
                    i += 1
                sep = "\n" if val_part in ("|", "|-") else " "
                result[key] = sep.join(block_lines).strip()
                continue

            if not val_part:
                # Nested block
                next_indent = indent + 2
                # Peek ahead to find actual indent
                j = i + 1
                while j < len(lines):
                    peeked = lines[j]
                    ps = peeked.strip()
                    if ps and not ps.startswith("#"):
                        next_indent = _get_indent(peeked)
                        break
                    j += 1
                nested, i = _parse_block(lines, i + 1, next_indent)
                result[key] = nested
                continue

            result[key] = _parse_scalar(val_part)
            i += 1
            continue

        i += 1

    return result if result is not None else {}, i


# ── Dataclasses ───────────────────────────────────────────────────────────────

VALID_FIELD_TYPES = {"text", "number", "options", "boolean", "json_array"}
VALID_ACTION_TYPES = {"http", "shell", "tool_call"}
VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


@dataclass
class Validation:
    min: Optional[float] = None
    max: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_pattern: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Validation":
        return cls(
            min=data.get("min"),
            max=data.get("max"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            pattern=data.get("pattern"),
            min_items=data.get("min_items"),
            max_items=data.get("max_items"),
            item_pattern=data.get("item_pattern"),
        )


@dataclass
class Field:
    id: str
    ask: str
    type: str = "text"
    options: dict[str, str] = field(default_factory=dict)
    required: bool = False
    default: Any = None
    infer_from: str = ""
    show_when: Optional[str] = None
    aliases: dict[str, str] = field(default_factory=dict)
    validation: Optional[Validation] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Field":
        field_type = data.get("type", "text")
        # If options present with no explicit type, infer "options"
        if "options" in data and field_type == "text":
            field_type = "options"
        if field_type not in VALID_FIELD_TYPES:
            raise ValueError(f"Invalid field type '{field_type}' for field '{data.get('id')}'. "
                             f"Valid types: {sorted(VALID_FIELD_TYPES)}")

        raw_options = data.get("options", {})
        if isinstance(raw_options, dict):
            options = {str(k): str(v) for k, v in raw_options.items()}
        else:
            options = {}

        validation = None
        if "validation" in data:
            validation = Validation.from_dict(data["validation"])

        return cls(
            id=str(data["id"]),
            ask=str(data.get("ask", "")),
            type=field_type,
            options=options,
            required=bool(data.get("required", False)),
            default=data.get("default"),
            infer_from=str(data.get("infer_from", "")),
            show_when=data.get("show_when"),
            aliases=dict(data.get("aliases", {})),
            validation=validation,
        )


@dataclass
class Action:
    type: str = "http"
    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 60
    command: Optional[str] = None
    tool_name: Optional[str] = None
    payload_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        action_type = data.get("type", "http")
        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action type '{action_type}'. Valid: {sorted(VALID_ACTION_TYPES)}")

        method = str(data.get("method", "POST")).upper()
        if action_type == "http" and method not in VALID_HTTP_METHODS:
            raise ValueError(f"Invalid HTTP method '{method}'")

        return cls(
            type=action_type,
            method=method,
            url=str(data.get("url", "")),
            headers=dict(data.get("headers", {})),
            timeout=int(data.get("timeout", 60)),
            command=data.get("command"),
            tool_name=data.get("tool_name"),
            payload_map=dict(data.get("payload_map", {})),
        )


@dataclass
class Confirmation:
    enabled: bool = True
    template: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Confirmation":
        return cls(
            enabled=bool(data.get("enabled", True)),
            template=str(data.get("template", "")),
        )


@dataclass
class SkillMeta:
    name: str
    version: str = "1.0.0"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SkillMeta":
        return cls(
            name=str(data["name"]),
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            tags=list(data.get("tags", [])),
            platforms=list(data.get("platforms", [])),
        )


VALID_STRATEGIES = {"auto", "step-by-step", "batched"}


@dataclass
class SkillForm:
    meta: SkillMeta
    action: Action
    fields: list[Field]
    preamble: str = ""  # Short context paragraph injected before each extraction step
    confirmation: Optional[Confirmation] = None
    source_path: Optional[Path] = None
    strategy: str = "auto"  # "auto", "step-by-step", or "batched"

    @classmethod
    def from_dict(cls, data: dict, source_path: Optional[Path] = None) -> "SkillForm":
        if "skill" not in data:
            raise ValueError("Missing required top-level key 'skill'")
        if "action" not in data:
            raise ValueError("Missing required top-level key 'action'")
        if "fields" not in data:
            raise ValueError("Missing required top-level key 'fields'")

        meta = SkillMeta.from_dict(data["skill"])
        action = Action.from_dict(data["action"])

        raw_fields = data["fields"]
        if not isinstance(raw_fields, list):
            raise ValueError("'fields' must be a list")
        fields = [Field.from_dict(f) for f in raw_fields]

        confirmation = None
        if "confirmation" in data:
            confirmation = Confirmation.from_dict(data["confirmation"])

        preamble = str(data.get("preamble", "")).strip()

        strategy = str(data.get("strategy", "auto"))
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. Valid values: {sorted(VALID_STRATEGIES)}"
            )

        return cls(
            meta=meta,
            action=action,
            fields=fields,
            preamble=preamble,
            confirmation=confirmation,
            source_path=source_path,
            strategy=strategy,
        )

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def description(self) -> str:
        return self.meta.description


# ── Loader ────────────────────────────────────────────────────────────────────

def load_skill(path: str | Path) -> SkillForm:
    """Load and validate a skill YAML file. Returns a SkillForm."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    text = path.read_text(encoding="utf-8")
    data = _load_yaml(text)
    form = SkillForm.from_dict(data, source_path=path)
    errors = validate_skill(form)
    if errors:
        raise ValueError(f"Skill validation failed for '{path}':\n" + "\n".join(f"  - {e}" for e in errors))
    return form


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_skill(form: SkillForm) -> list[str]:
    """
    Validate a SkillForm. Returns a list of error strings (empty = valid).

    Checks:
    - Required fields have 'ask'
    - All show_when references point to valid field IDs
    - Options fields have at least one option
    - Alias values are valid option keys (for options fields)
    - No duplicate field IDs
    - Action type matches required keys
    """
    errors: list[str] = []
    field_ids: set[str] = set()

    for f in form.fields:
        # Duplicate IDs
        if f.id in field_ids:
            errors.append(f"Duplicate field id: '{f.id}'")
        field_ids.add(f.id)

        # ask is required
        if not f.ask:
            errors.append(f"Field '{f.id}' missing 'ask'")

        # options fields need options
        if f.type == "options" and not f.options:
            errors.append(f"Field '{f.id}' has type 'options' but no options defined")

        # aliases must map to valid option keys (for options fields)
        if f.aliases and f.type == "options":
            for alias, target in f.aliases.items():
                if target not in f.options:
                    errors.append(
                        f"Field '{f.id}' alias '{alias}' -> '{target}' is not a valid option key. "
                        f"Valid keys: {sorted(f.options.keys())}"
                    )

    # Validate show_when references
    for f in form.fields:
        if f.show_when:
            ref_field = _extract_show_when_field(f.show_when)
            if ref_field and ref_field not in field_ids:
                errors.append(
                    f"Field '{f.id}' show_when references unknown field '{ref_field}'"
                )

    # Action-type-specific checks
    if form.action.type == "http" and not form.action.url:
        errors.append("HTTP action requires 'url'")
    if form.action.type == "shell" and not form.action.command:
        errors.append("Shell action requires 'command'")
    if form.action.type == "tool_call" and not form.action.tool_name:
        errors.append("tool_call action requires 'tool_name'")

    return errors


def _extract_show_when_field(condition: str) -> Optional[str]:
    """Extract the field name referenced in a show_when condition."""
    condition = condition.strip()
    if "==" in condition:
        return condition.split("==")[0].strip()
    if "!=" in condition:
        return condition.split("!=")[0].strip()
    if " in " in condition:
        return condition.split(" in ")[0].strip()
    # Bare field name
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", condition):
        return condition
    return None
