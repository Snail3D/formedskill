"""
formedskill.gen_tests — Auto-generate test cases from a SKILL.yaml.

Uses an LLM to generate diverse user messages that exercise different
field combinations, plus adversarial cases (ambiguous, missing info,
conflicting params). Output format matches benchmarks/test_cases.json.

Usage:
    from formedskill.gen_tests import generate_tests
    test_cases = generate_tests(
        skill_path="snailprint.yaml",
        model="llama3",
        endpoint="http://localhost:11434",
        count=30,
    )
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from formedskill.client import LLMClient
from formedskill.schema import load_skill, SkillForm


_SYSTEM_PROMPT = """\
You are an expert at generating test cases for guided-form skill definitions.

Your job: given a SKILL.yaml definition, generate realistic and diverse test cases
that exercise different field combinations, edge cases, and natural language variations.

Each test case must be a JSON object with exactly these fields:
{
  "id": "<skill_name>_N",
  "user_message": "<realistic natural language request>",
  "description": "<one-line summary of what this tests>",
  "expected_fields": {
    "<field_id>": <expected_value>,
    ...
  }
}

IMPORTANT RULES:
1. Include only fields that the user explicitly stated or that have clear defaults
2. expected_fields should only contain fields the test is checking — not ALL fields
3. Use the EXACT option key strings (not descriptions) for options fields
4. Numbers should be numeric (80, not "80mm")
5. Booleans should be true/false (not "yes"/"no")
6. Be realistic — write messages a real user would actually send
7. Cover different modes/types if the skill has a mode field
8. Include some adversarial cases (see below)

ADVERSARIAL CASES TO INCLUDE:
- Ambiguous: message could map to multiple option values
- Missing info: user omits a required field (test that defaults are used)
- Conflicting: user gives contradictory info
- Verbose: user gives a lot of irrelevant context
- Terse: one or two words only
- Units: measurements in different units (cm, inches, feet)
- Synonyms: using synonyms for option values (e.g. "flexible" instead of "TPU")

Output a JSON array of test case objects. No explanation, no markdown fences.
"""

_USER_TEMPLATE = """\
Generate {count} diverse test cases for this skill:

SKILL YAML:
{yaml_text}

FIELD SUMMARY:
{field_summary}

Generate exactly {count} test cases as a JSON array. Mix regular, edge case, and adversarial scenarios.
Cover all modes/types. Output ONLY the JSON array, nothing else.
"""


def generate_tests(
    skill_path: str | Path,
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    count: int = 30,
    timeout: int = 180,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Auto-generate test cases from a SKILL.yaml file.

    Args:
        skill_path: Path to the skill YAML file
        model: LLM model name
        endpoint: OpenAI-compatible endpoint URL
        api_key: Optional Bearer token
        count: Number of test cases to generate
        timeout: Request timeout in seconds
        temperature: Sampling temperature (higher = more diverse)

    Returns:
        Dict in test_cases.json format:
        {
          "skills": [
            {
              "name": str,
              "skill_yaml_path": str,
              "tests": [...]
            }
          ]
        }

    Raises:
        FileNotFoundError: If skill YAML not found
        ValueError: If skill YAML is invalid or generation fails
        RuntimeError: If LLM call fails
    """
    skill_path = Path(skill_path)
    form = load_skill(skill_path)

    yaml_text = skill_path.read_text(encoding="utf-8")
    field_summary = _build_field_summary(form)

    client = LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        default_temperature=temperature,
        timeout=timeout,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                count=count,
                yaml_text=yaml_text,
                field_summary=field_summary,
            ),
        },
    ]

    raw, _stats = client.chat_completion(messages, temperature=temperature)
    test_cases = _parse_test_cases(raw, form)

    # Ensure IDs are unique and skill-prefixed
    skill_name = form.name.lower().replace(" ", "_").replace("-", "_")
    seen_ids: set[str] = set()
    for i, tc in enumerate(test_cases, start=1):
        base_id = f"{skill_name}_{i}"
        if tc.get("id") in seen_ids or not tc.get("id", "").startswith(skill_name):
            tc["id"] = base_id
        seen_ids.add(tc["id"])

    return {
        "skills": [
            {
                "name": form.name,
                "skill_yaml_path": str(skill_path.resolve()),
                "tests": test_cases,
            }
        ]
    }


def _build_field_summary(form: SkillForm) -> str:
    """Build a concise summary of all fields for the LLM prompt."""
    lines = []
    for f in form.fields:
        line = f"- {f.id} ({f.type})"
        if f.required:
            line += " REQUIRED"
        if f.default is not None:
            line += f" default={f.default!r}"
        if f.show_when:
            line += f" [shown when: {f.show_when}]"
        if f.options:
            opts = ", ".join(f.options.keys())
            line += f" options=[{opts}]"
        if f.infer_from:
            hint = f.infer_from[:80].replace("\n", " ")
            line += f"\n  hint: {hint}"
        lines.append(line)
    return "\n".join(lines)


def _parse_test_cases(raw: str, form: SkillForm) -> list[dict[str, Any]]:
    """
    Parse the LLM's JSON array response into a list of test case dicts.
    Handles markdown fences and extracts the first JSON array found.
    """
    text = raw.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return _normalize_cases(result, form)
        if isinstance(result, dict) and "tests" in result:
            return _normalize_cases(result["tests"], form)
    except json.JSONDecodeError:
        pass

    # Try to find first [...] block
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return _normalize_cases(result, form)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse test cases from LLM response. "
        f"Response preview: {text[:200]}"
    )


def _normalize_cases(cases: list[Any], form: SkillForm) -> list[dict[str, Any]]:
    """Normalize and validate each test case dict."""
    valid_field_ids = {f.id for f in form.fields}
    normalized = []

    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            continue

        user_msg = str(case.get("user_message", "")).strip()
        if not user_msg:
            continue

        expected = case.get("expected_fields", {})
        if not isinstance(expected, dict):
            expected = {}

        # Filter to only valid field IDs
        expected = {
            k: v for k, v in expected.items()
            if k in valid_field_ids
        }

        normalized.append({
            "id": str(case.get("id", f"test_{i + 1}")),
            "user_message": user_msg,
            "description": str(case.get("description", f"Test case {i + 1}")),
            "expected_fields": expected,
        })

    return normalized


def save_test_cases(test_data: dict[str, Any], output_path: str | Path) -> Path:
    """
    Save test cases to a JSON file.

    Args:
        test_data: Dict from generate_tests()
        output_path: Output file path

    Returns:
        Path to the saved file
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(test_data, indent=2, default=str), encoding="utf-8")
    return out
