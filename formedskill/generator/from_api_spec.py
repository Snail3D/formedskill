"""
formedskill.generator.from_api_spec — Generate a SKILL.yaml from an OpenAPI/Swagger JSON spec.

Parses the spec to extract endpoints, parameters, and request body schema,
then uses an LLM to generate infer_from hints and assemble the final YAML.

Usage:
    from formedskill.generator.from_api_spec import generate_from_api_spec
    yaml_text = generate_from_api_spec(
        spec_path="my-api.json",
        endpoint_path="/api/print/start",  # optional: target specific endpoint
        model="llama3",
        endpoint="http://localhost:11434",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from formedskill.client import LLMClient
from formedskill.schema import _load_yaml, SkillForm, validate_skill


_SYSTEM_PROMPT = """\
You are an expert at designing guided-form skill definitions for the FormedSkill framework.
FormedSkill breaks complex API calls into step-by-step form filling so small LLMs (3B parameters) achieve 100% accuracy.

You will receive a structured summary of an API endpoint. Your task is to generate a complete SKILL.yaml.

Output ONLY valid YAML. No markdown fences, no explanations.

YAML SCHEMA:
skill:
  name: <snake_case_name>
  version: "1.0.0"
  description: <one-line description>
  tags: [tag1, tag2]

action:
  type: http
  method: <HTTP_METHOD>
  url: "<full_url>"
  timeout: 60

fields:
  - id: <snake_case_id>
    ask: "<clear question for the user>"
    type: <text|number|options|boolean|json_array>
    required: <true|false>
    infer_from: >-
      <hint for the LLM: how to extract this value from natural language>
    options:  # only for type: options
      key1: "Plain English description"
      key2: "Plain English description"
    default: <default_value_if_any>
    show_when: "<field_id> == <value>"  # only when field is conditional

confirmation:
  enabled: true

RULES:
1. Every field MUST have a concise, actionable infer_from hint
2. Enum/const parameters become type: options — list ALL options with descriptions
3. String parameters with pattern constraints become type: text with infer_from guidance
4. Integer/number parameters become type: number
5. Boolean parameters become type: boolean
6. Array parameters become type: json_array
7. Required parameters have required: true
8. Optional parameters must have a sensible default
9. Group related optional fields under a show_when if there is a mode/type discriminator
10. Keep field IDs short and snake_case (strip prefixes like "request_" or "body_")
"""

_USER_TEMPLATE = """\
Generate a SKILL.yaml for this API endpoint:

{endpoint_summary}

Output ONLY the complete YAML, nothing else.
"""


def generate_from_api_spec(
    spec_path: str | Path,
    endpoint_path: Optional[str] = None,
    method: Optional[str] = None,
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    timeout: int = 120,
    temperature: float = 0.2,
) -> str:
    """
    Generate a SKILL.yaml from an OpenAPI/Swagger JSON spec file.

    Args:
        spec_path: Path to OpenAPI JSON or YAML spec file
        endpoint_path: Target endpoint path (e.g. "/api/print/start").
                       If None, picks the first POST endpoint found.
        method: HTTP method filter (e.g. "POST"). Defaults to POST.
        model: LLM model name
        endpoint: OpenAI-compatible endpoint URL
        api_key: Optional Bearer token
        timeout: Request timeout in seconds
        temperature: Sampling temperature

    Returns:
        YAML text of the generated skill definition

    Raises:
        FileNotFoundError: If spec file not found
        ValueError: If spec is invalid or generation fails
        RuntimeError: If LLM call fails
    """
    spec_path = Path(spec_path)
    if not spec_path.exists():
        raise FileNotFoundError(f"OpenAPI spec not found: {spec_path}")

    spec = _load_spec(spec_path)
    summary = _extract_endpoint_summary(spec, endpoint_path=endpoint_path, method=method or "POST")

    client = LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        default_temperature=temperature,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(endpoint_summary=summary)},
    ]

    raw, _stats = client.chat_completion(messages, temperature=temperature)
    yaml_text = _extract_yaml(raw)
    _validate_generated_yaml(yaml_text)

    return yaml_text


def _load_spec(path: Path) -> dict[str, Any]:
    """Load an OpenAPI spec from JSON or YAML file."""
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        data = _load_yaml(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse spec as JSON: {e}") from e

    return data


def _extract_endpoint_summary(
    spec: dict[str, Any],
    endpoint_path: Optional[str],
    method: str = "POST",
) -> str:
    """
    Extract a structured text summary of one endpoint from an OpenAPI spec.
    Handles both OpenAPI 3.x and Swagger 2.x.
    """
    paths = spec.get("paths", {})
    if not paths:
        raise ValueError("OpenAPI spec has no 'paths' defined")

    # Detect spec version
    openapi_version = spec.get("openapi", "") or spec.get("swagger", "")
    is_v3 = openapi_version.startswith("3")

    # Pick target endpoint
    target_path, target_method, operation = _find_operation(
        paths, endpoint_path, method.lower()
    )

    # Build base URL
    base_url = _extract_base_url(spec, is_v3)
    full_url = base_url.rstrip("/") + target_path

    # Extract info
    info = spec.get("info", {})
    api_title = info.get("title", "API")
    api_description = info.get("description", "")

    lines = [
        f"API: {api_title}",
        f"Endpoint: {target_method.upper()} {full_url}",
    ]
    if api_description:
        lines.append(f"Description: {api_description[:200]}")

    op_summary = operation.get("summary", "") or operation.get("description", "")
    if op_summary:
        lines.append(f"Operation: {op_summary[:200]}")

    lines.append("")

    # Path parameters
    path_params = _extract_parameters(operation, spec, location="path")
    if path_params:
        lines.append("PATH PARAMETERS:")
        for p in path_params:
            lines.append(_format_param(p))

    # Query parameters
    query_params = _extract_parameters(operation, spec, location="query")
    if query_params:
        lines.append("QUERY PARAMETERS:")
        for p in query_params:
            lines.append(_format_param(p))

    # Request body
    body_fields = _extract_request_body(operation, spec, is_v3)
    if body_fields:
        lines.append("REQUEST BODY FIELDS:")
        for p in body_fields:
            lines.append(_format_param(p))

    return "\n".join(lines)


def _find_operation(
    paths: dict,
    target_path: Optional[str],
    method: str,
) -> tuple[str, str, dict]:
    """Find a matching operation in the paths dict."""
    method = method.lower()

    if target_path:
        path_ops = paths.get(target_path, {})
        if not path_ops:
            raise ValueError(f"Endpoint '{target_path}' not found in spec. Available: {list(paths.keys())[:10]}")
        if method in path_ops:
            return target_path, method, path_ops[method]
        # Try any method
        for m, op in path_ops.items():
            if isinstance(op, dict) and "responses" in op:
                return target_path, m, op
        raise ValueError(f"No operations found for '{target_path}'")

    # Auto-pick: prefer POST, then any method
    for pth, ops in paths.items():
        if isinstance(ops, dict) and method in ops:
            return pth, method, ops[method]

    # Fallback: first operation found
    for pth, ops in paths.items():
        if isinstance(ops, dict):
            for m, op in ops.items():
                if isinstance(op, dict) and "responses" in op:
                    return pth, m, op

    raise ValueError("No operations found in OpenAPI spec")


def _extract_base_url(spec: dict, is_v3: bool) -> str:
    """Extract base URL from spec."""
    if is_v3:
        servers = spec.get("servers", [])
        if servers:
            return servers[0].get("url", "http://localhost")
        return "http://localhost"
    else:
        host = spec.get("host", "localhost")
        base_path = spec.get("basePath", "/")
        schemes = spec.get("schemes", ["http"])
        scheme = schemes[0] if schemes else "http"
        return f"{scheme}://{host}{base_path}"


def _resolve_ref(ref: str, spec: dict) -> dict:
    """Resolve a $ref string like '#/components/schemas/Foo' to its definition."""
    if not ref.startswith("#/"):
        return {}
    parts = ref.lstrip("#/").split("/")
    node = spec
    for part in parts:
        if isinstance(node, dict):
            node = node.get(part, {})
        else:
            return {}
    return node if isinstance(node, dict) else {}


def _resolve_schema(schema: dict, spec: dict) -> dict:
    """Resolve $ref in a schema object."""
    if "$ref" in schema:
        return _resolve_ref(schema["$ref"], spec)
    return schema


def _extract_parameters(operation: dict, spec: dict, location: str) -> list[dict]:
    """Extract parameters of a given location (path, query, header, cookie)."""
    params = operation.get("parameters", [])
    result = []
    for p in params:
        if "$ref" in p:
            p = _resolve_ref(p["$ref"], spec)
        if p.get("in") == location:
            schema = _resolve_schema(p.get("schema", {}), spec)
            result.append({
                "name": p.get("name", ""),
                "required": p.get("required", False),
                "description": p.get("description", ""),
                "type": _schema_to_type(schema),
                "enum": schema.get("enum", []),
                "default": schema.get("default"),
                "format": schema.get("format", ""),
            })
    return result


def _extract_request_body(operation: dict, spec: dict, is_v3: bool) -> list[dict]:
    """Extract request body fields from an operation."""
    if is_v3:
        body = operation.get("requestBody", {})
        if not body:
            return []
        content = body.get("content", {})
        # Prefer application/json
        schema_container = (
            content.get("application/json", {}) or
            next(iter(content.values()), {})
        )
        schema = _resolve_schema(schema_container.get("schema", {}), spec)
    else:
        # Swagger 2.x — body parameter
        params = operation.get("parameters", [])
        for p in params:
            if p.get("in") == "body":
                schema = _resolve_schema(p.get("schema", {}), spec)
                break
        else:
            return []

    return _flatten_schema_properties(schema, spec, required_set=set(schema.get("required", [])))


def _flatten_schema_properties(
    schema: dict,
    spec: dict,
    required_set: set[str],
    prefix: str = "",
    depth: int = 0,
) -> list[dict]:
    """Recursively flatten schema properties into a list of field dicts."""
    if depth > 3:
        return []

    schema = _resolve_schema(schema, spec)
    props = schema.get("properties", {})
    results = []

    for name, prop_schema in props.items():
        prop_schema = _resolve_schema(prop_schema, spec)
        field_name = f"{prefix}{name}" if prefix else name

        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            # Recurse into nested objects
            nested = _flatten_schema_properties(
                prop_schema, spec, set(prop_schema.get("required", [])),
                prefix=f"{field_name}_", depth=depth + 1
            )
            results.extend(nested)
        else:
            results.append({
                "name": field_name,
                "required": name in required_set,
                "description": prop_schema.get("description", ""),
                "type": _schema_to_type(prop_schema),
                "enum": prop_schema.get("enum", []),
                "default": prop_schema.get("default"),
                "format": prop_schema.get("format", ""),
            })

    return results


def _schema_to_type(schema: dict) -> str:
    """Convert a JSON Schema type to a FormedSkill field type."""
    t = schema.get("type", "string")
    fmt = schema.get("format", "")
    has_enum = bool(schema.get("enum"))

    if has_enum:
        return "options"
    if t in ("integer", "number"):
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        return "json_array"
    return "text"


def _format_param(p: dict) -> str:
    """Format a parameter dict as a readable text line."""
    parts = [f"  - {p['name']}"]
    if p.get("required"):
        parts.append("(required)")
    else:
        parts.append("(optional)")

    parts.append(f"type={p['type']}")

    if p.get("enum"):
        opts = ", ".join(str(v) for v in p["enum"][:10])
        parts.append(f"options=[{opts}]")

    if p.get("default") is not None:
        parts.append(f"default={p['default']!r}")

    if p.get("description"):
        parts.append(f"— {p['description'][:100]}")

    return " ".join(parts)


def _extract_yaml(raw: str) -> str:
    """Strip markdown code fences from LLM output if present."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _validate_generated_yaml(yaml_text: str) -> None:
    """Validate the generated YAML against the FormedSkill schema."""
    try:
        data = _load_yaml(yaml_text)
    except Exception as e:
        raise ValueError(f"Generated YAML failed to parse: {e}") from e

    missing = [k for k in ("skill", "action", "fields") if k not in data]
    if missing:
        raise ValueError(f"Generated YAML missing required keys: {', '.join(missing)}")

    try:
        form = SkillForm.from_dict(data)
    except ValueError as e:
        raise ValueError(f"Generated YAML has schema errors: {e}") from e

    errors = validate_skill(form)
    if errors:
        raise ValueError(
            f"Generated YAML has {len(errors)} validation error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
