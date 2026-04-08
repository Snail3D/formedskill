"""
formedskill.generator.from_description — Generate a SKILL.yaml from a natural language description.

Uses an LLM to:
- Identify the API/tool being described
- Break it into fields with types, options, defaults
- Write infer_from hints
- Output valid YAML

Usage:
    from formedskill.generator.from_description import generate_from_description
    yaml_text = generate_from_description(
        description="A skill that sends Slack messages to channels",
        model="llama3",
        endpoint="http://localhost:11434",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from formedskill.client import LLMClient
from formedskill.schema import _load_yaml, SkillForm


_SYSTEM_PROMPT = """\
You are an expert at designing guided-form skill definitions for the FormedSkill framework.
FormedSkill breaks complex API calls into step-by-step form filling so small LLMs can achieve 100% accuracy.

You must output ONLY valid YAML in the exact schema format below. No explanation, no markdown code fences.

YAML SCHEMA:
skill:
  name: <snake_case_name>
  version: "1.0.0"
  description: <one-line description>
  tags: [tag1, tag2]

action:
  type: http  # or shell or tool_call
  method: POST
  url: "http://localhost:PORT/api/endpoint"
  timeout: 60

fields:
  - id: <snake_case_id>
    ask: "<question to ask the user>"
    type: <text|number|options|boolean|json_array>
    required: true
    infer_from: >-
      <hint for the LLM about how to extract this field from the user message>
    options:  # only for type: options
      key1: "Description of key1"
      key2: "Description of key2"
    default: <default_value>
    show_when: "<field_id> == <value>"  # optional conditional

confirmation:
  enabled: true
  template: |
    Ready to perform: {{field1}}, {{field2}}

FIELD TYPE GUIDE:
- text: free-form string (names, descriptions, prompts)
- number: numeric value with optional min/max validation
- options: closed set of choices (always list them all with descriptions)
- boolean: yes/no toggle
- json_array: list of values

RULES:
1. Every field must have a clear, concise infer_from hint
2. Options fields must list ALL valid options with plain-English descriptions
3. Put the most important/discriminating field first (often a "mode" or "type" field)
4. Use show_when to hide fields that only apply to certain modes
5. Provide sensible defaults where possible
6. Keep field IDs short and snake_case
7. The ask question should be clear to a non-technical user
"""

_USER_TEMPLATE = """\
Generate a complete SKILL.yaml for the following tool/API:

{description}

Requirements:
- Identify all parameters this tool accepts
- Create clear field definitions with types, options where applicable, and defaults
- Write helpful infer_from hints that explain how to extract each field from natural language
- Use show_when conditions if some fields only apply in certain modes
- Output ONLY the YAML, nothing else
"""


def generate_from_description(
    description: str,
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    timeout: int = 120,
    temperature: float = 0.2,
) -> str:
    """
    Generate a SKILL.yaml from a natural language description using an LLM.

    Args:
        description: Natural language description of the tool/API
        model: LLM model name
        endpoint: OpenAI-compatible endpoint URL
        api_key: Optional Bearer token
        timeout: Request timeout in seconds
        temperature: Sampling temperature (lower = more deterministic)

    Returns:
        YAML text of the generated skill definition

    Raises:
        RuntimeError: If the LLM call fails
        ValueError: If the generated YAML is invalid
    """
    client = LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        default_temperature=temperature,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(description=description)},
    ]

    raw, _stats = client.chat_completion(messages, temperature=temperature)

    yaml_text = _extract_yaml(raw)
    _validate_generated_yaml(yaml_text)

    return yaml_text


def _extract_yaml(raw: str) -> str:
    """Strip markdown code fences from LLM output if present."""
    text = raw.strip()

    # Strip ```yaml ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```yaml or ```)
        lines = lines[1:]
        # Remove last ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def _validate_generated_yaml(yaml_text: str) -> None:
    """
    Validate the generated YAML against the FormedSkill schema.
    Raises ValueError with a helpful message if invalid.
    """
    try:
        data = _load_yaml(yaml_text)
    except Exception as e:
        raise ValueError(f"Generated YAML failed to parse: {e}") from e

    # Basic structure check
    missing = [k for k in ("skill", "action", "fields") if k not in data]
    if missing:
        raise ValueError(
            f"Generated YAML missing required keys: {', '.join(missing)}. "
            f"The LLM may have included extra text — try again with a clearer description."
        )

    try:
        form = SkillForm.from_dict(data)
    except ValueError as e:
        raise ValueError(f"Generated YAML has schema errors: {e}") from e

    from formedskill.schema import validate_skill
    errors = validate_skill(form)
    if errors:
        raise ValueError(
            f"Generated YAML has {len(errors)} validation error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
