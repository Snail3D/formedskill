"""
formedskill.optimizer — Karpathy-loop hill-climbing optimizer for SKILL.yaml.

Iteratively improves a skill definition by:
1. Running benchmark against test cases to find failures
2. Asking LLM to mutate the YAML to fix failures
3. Keeping the mutation if it scores better
4. Repeating until target accuracy or iteration limit reached

Usage:
    from formedskill.optimizer import optimize
    best_yaml, best_score = optimize(
        skill_path="snailprint.yaml",
        test_cases=test_data,  # dict from gen_tests or loaded JSON
        model="llama3",
        endpoint="http://localhost:11434",
        iterations=20,
        target=0.98,
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from formedskill.client import LLMClient
from formedskill.schema import _load_yaml, load_skill, SkillForm, validate_skill


_MUTATION_SYSTEM = """\
You are an expert at improving guided-form skill definitions for the FormedSkill framework.

You will receive:
1. A SKILL.yaml definition
2. Test cases that FAILED — showing what the user said, what was expected, and what the model actually extracted
3. A summary of what went wrong

Your task: modify the SKILL.yaml to fix these failures.

WHAT YOU CAN CHANGE:
- infer_from hints: make them clearer, more specific, add examples
- Field order: put discriminating fields first
- Option descriptions: make them clearer, add synonyms/examples
- Default values: correct wrong defaults
- aliases: add natural language aliases for options
- ask questions: make them clearer
- show_when conditions: fix incorrect conditions

WHAT YOU MUST NOT CHANGE:
- The action (type, method, url, command)
- Field IDs (these map to API parameters)
- The overall structure and required fields

Output ONLY the complete modified YAML. No explanation, no markdown fences.
"""

_MUTATION_USER_TEMPLATE = """\
CURRENT SKILL.yaml:
{yaml_text}

FAILED TEST CASES ({failure_count} failures, current accuracy: {accuracy_pct}%):
{failure_summary}

DIAGNOSIS:
{diagnosis}

Modify the SKILL.yaml to fix these failures. Output ONLY the complete modified YAML.
"""


@dataclass
class OptimizationResult:
    """Result from one optimization run."""
    best_yaml: str
    best_score: float
    iterations_run: int
    initial_score: float
    score_history: list[float] = field(default_factory=list)
    change_history: list[str] = field(default_factory=list)
    converged: bool = False


def optimize(
    skill_path: str | Path,
    test_cases: dict[str, Any],
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    iterations: int = 20,
    target: float = 0.98,
    timeout: int = 180,
    temperature: float = 0.3,
    on_iteration: Optional[Callable[[int, float, str], None]] = None,
) -> OptimizationResult:
    """
    Hill-climbing optimizer for SKILL.yaml definitions.

    Args:
        skill_path: Path to the skill YAML file (used as starting point)
        test_cases: Test data dict (format from gen_tests.generate_tests())
        model: LLM model name
        endpoint: OpenAI-compatible endpoint URL
        api_key: Optional Bearer token
        iterations: Maximum number of optimization iterations
        target: Target accuracy (0.0–1.0). Stop early if reached.
        timeout: Per-LLM-call timeout in seconds
        temperature: Sampling temperature for mutations
        on_iteration: Optional callback(iter_num, score, description) called after each iteration

    Returns:
        OptimizationResult with best YAML, score, and history
    """
    skill_path = Path(skill_path)
    best_yaml = skill_path.read_text(encoding="utf-8")

    client = LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        default_temperature=temperature,
        timeout=timeout,
    )

    # Extract skill tests from the test_cases structure
    skill_tests = _extract_tests(test_cases, skill_path)

    # Baseline score
    best_score = _score_yaml(best_yaml, skill_tests, endpoint, model, api_key, timeout)
    initial_score = best_score
    score_history = [best_score]
    change_history = ["baseline"]

    if on_iteration:
        on_iteration(0, best_score, "baseline")

    converged = False

    for i in range(1, iterations + 1):
        if best_score >= target:
            converged = True
            break

        # Identify failing test cases
        failures = _get_failures(best_yaml, skill_tests, endpoint, model, api_key, timeout)

        if not failures:
            converged = True
            break

        # Ask LLM to mutate the YAML
        diagnosis = _diagnose_failures(failures)
        failure_summary = _format_failures(failures)

        messages = [
            {"role": "system", "content": _MUTATION_SYSTEM},
            {
                "role": "user",
                "content": _MUTATION_USER_TEMPLATE.format(
                    yaml_text=best_yaml,
                    failure_count=len(failures),
                    accuracy_pct=int(best_score * 100),
                    failure_summary=failure_summary,
                    diagnosis=diagnosis,
                ),
            },
        ]

        try:
            raw, _stats = client.chat_completion(messages, temperature=temperature)
            mutated_yaml = _extract_yaml(raw)
        except RuntimeError:
            # LLM call failed — skip this iteration
            score_history.append(best_score)
            change_history.append("llm_error")
            continue

        # Validate the mutation
        if not _is_valid_yaml(mutated_yaml):
            score_history.append(best_score)
            change_history.append("invalid_yaml")
            continue

        # Score the mutation
        new_score = _score_yaml(mutated_yaml, skill_tests, endpoint, model, api_key, timeout)
        change_desc = _describe_change(best_yaml, mutated_yaml, new_score, best_score)

        score_history.append(new_score)
        change_history.append(change_desc)

        # Keep if better
        if new_score > best_score:
            best_yaml = mutated_yaml
            best_score = new_score

        if on_iteration:
            on_iteration(i, new_score, change_desc)

    return OptimizationResult(
        best_yaml=best_yaml,
        best_score=best_score,
        iterations_run=len(score_history) - 1,
        initial_score=initial_score,
        score_history=score_history,
        change_history=change_history,
        converged=converged,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_tests(test_cases: dict[str, Any], skill_path: Path) -> list[dict[str, Any]]:
    """
    Extract the test list from a test_cases dict.
    Matches by skill_yaml_path or falls back to the first skill.
    """
    skills = test_cases.get("skills", [])
    if not skills:
        return []

    # Try to match by path
    skill_path_str = str(skill_path.resolve())
    for skill in skills:
        yaml_path = skill.get("skill_yaml_path", "")
        if yaml_path and Path(yaml_path).resolve() == skill_path.resolve():
            return skill.get("tests", [])

    # Fallback: return first skill's tests
    return skills[0].get("tests", [])


def _score_yaml(
    yaml_text: str,
    tests: list[dict[str, Any]],
    endpoint: str,
    model: str,
    api_key: Optional[str],
    timeout: int,
) -> float:
    """Run all test cases against the YAML and return overall accuracy (0.0–1.0)."""
    if not tests:
        return 0.0

    from formedskill.schema import _load_yaml, SkillForm
    from formedskill.runtime import FormRunner

    try:
        data = _load_yaml(yaml_text)
        form = SkillForm.from_dict(data)
    except (ValueError, Exception):
        return 0.0

    runner = FormRunner(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        temperature=0.1,
        timeout=timeout,
        verbose=False,
    )

    total_correct = 0
    total_fields = 0

    for tc in tests:
        user_message = tc.get("user_message", "")
        expected = tc.get("expected_fields", {})
        if not user_message or not expected:
            continue

        try:
            result = runner.run_step_by_step(form, user_message)
            collected = result.collected
        except RuntimeError:
            total_fields += len(expected)
            continue

        for key, exp_val in expected.items():
            got_val = collected.get(key)
            total_correct += int(_score_field(exp_val, got_val))
            total_fields += 1

    return total_correct / total_fields if total_fields > 0 else 0.0


def _get_failures(
    yaml_text: str,
    tests: list[dict[str, Any]],
    endpoint: str,
    model: str,
    api_key: Optional[str],
    timeout: int,
) -> list[dict[str, Any]]:
    """Return list of test cases where at least one field was wrong."""
    if not tests:
        return []

    from formedskill.schema import _load_yaml, SkillForm
    from formedskill.runtime import FormRunner

    try:
        data = _load_yaml(yaml_text)
        form = SkillForm.from_dict(data)
    except Exception:
        return list(tests)

    runner = FormRunner(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        temperature=0.1,
        timeout=timeout,
        verbose=False,
    )

    failures = []

    for tc in tests:
        user_message = tc.get("user_message", "")
        expected = tc.get("expected_fields", {})
        if not user_message or not expected:
            continue

        try:
            result = runner.run_step_by_step(form, user_message)
            collected = result.collected
        except RuntimeError as e:
            failures.append({
                **tc,
                "collected": {},
                "field_errors": {k: {"expected": v, "got": None} for k, v in expected.items()},
                "error": str(e),
            })
            continue

        field_errors = {}
        for key, exp_val in expected.items():
            got_val = collected.get(key)
            if not _score_field(exp_val, got_val):
                field_errors[key] = {"expected": exp_val, "got": got_val}

        if field_errors:
            failures.append({
                **tc,
                "collected": collected,
                "field_errors": field_errors,
            })

    return failures


def _score_field(expected: Any, got: Any) -> bool:
    """Fuzzy field scoring matching the benchmark harness."""
    if expected is None and got is None:
        return True
    if expected is None or got is None:
        return False
    if isinstance(expected, str) and isinstance(got, str):
        return expected.lower() in got.lower() or got.lower() in expected.lower()
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(expected - got) < 1
    return str(expected).lower() == str(got).lower()


def _diagnose_failures(failures: list[dict[str, Any]]) -> str:
    """Generate a concise diagnosis of the failure patterns."""
    field_failure_counts: dict[str, int] = {}
    for f in failures:
        for field_id in f.get("field_errors", {}):
            field_failure_counts[field_id] = field_failure_counts.get(field_id, 0) + 1

    if not field_failure_counts:
        return "Unknown failures — possibly LLM errors."

    sorted_fields = sorted(field_failure_counts.items(), key=lambda x: -x[1])
    lines = ["Most problematic fields:"]
    for fid, count in sorted_fields[:5]:
        lines.append(f"  - {fid}: failed {count}/{len(failures)} times")

    # Look for patterns
    all_errors = []
    for f in failures:
        for fid, err in f.get("field_errors", {}).items():
            all_errors.append((fid, err.get("expected"), err.get("got")))

    if all_errors:
        lines.append("\nExample errors:")
        for fid, exp, got in all_errors[:4]:
            lines.append(f"  - {fid}: expected={exp!r}, got={got!r}")

    return "\n".join(lines)


def _format_failures(failures: list[dict[str, Any]]) -> str:
    """Format failures for the LLM mutation prompt."""
    lines = []
    for i, f in enumerate(failures[:10], start=1):
        lines.append(f"\n[Failure {i}]")
        lines.append(f"  User: {f.get('user_message', '')[:100]}")
        for fid, err in f.get("field_errors", {}).items():
            lines.append(f"  {fid}: expected={err['expected']!r}, got={err['got']!r}")
    return "\n".join(lines)


def _describe_change(old_yaml: str, new_yaml: str, new_score: float, old_score: float) -> str:
    """Produce a short description of what changed between two YAML versions."""
    delta = new_score - old_score
    direction = "▲" if delta > 0 else ("▼" if delta < 0 else "=")

    # Count changed lines
    old_lines = set(old_yaml.splitlines())
    new_lines = set(new_yaml.splitlines())
    added = len(new_lines - old_lines)
    removed = len(old_lines - new_lines)

    parts = [f"{direction} {int(new_score * 100)}%"]
    if added or removed:
        parts.append(f"+{added}/-{removed} lines")

    # Detect what kind of change
    old_data = _safe_load_yaml(old_yaml)
    new_data = _safe_load_yaml(new_yaml)

    changes = _diff_yaml_changes(old_data, new_data)
    if changes:
        parts.append(changes)

    return " ".join(parts)


def _safe_load_yaml(text: str) -> dict:
    try:
        return _load_yaml(text)
    except Exception:
        return {}


def _diff_yaml_changes(old: dict, new: dict) -> str:
    """Detect high-level changes between two YAML dicts."""
    changes = []

    old_fields = {f.get("id"): f for f in old.get("fields", []) if isinstance(f, dict)}
    new_fields = {f.get("id"): f for f in new.get("fields", []) if isinstance(f, dict)}

    for fid in old_fields:
        if fid not in new_fields:
            continue
        of = old_fields[fid]
        nf = new_fields[fid]

        if of.get("infer_from") != nf.get("infer_from"):
            changes.append(f"updated infer_from:{fid}")
        if of.get("options") != nf.get("options"):
            changes.append(f"updated options:{fid}")
        if of.get("default") != nf.get("default"):
            changes.append(f"changed default:{fid}")
        if of.get("aliases") != nf.get("aliases"):
            changes.append(f"+aliases:{fid}")

    # Check field order
    old_order = list(old_fields.keys())
    new_order = list(new_fields.keys())
    if old_order != new_order:
        changes.append("reordered fields")

    return ", ".join(changes[:3]) if changes else ""


def _extract_yaml(raw: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _is_valid_yaml(yaml_text: str) -> bool:
    """Return True if the YAML parses and passes basic FormedSkill validation."""
    try:
        data = _load_yaml(yaml_text)
        if not all(k in data for k in ("skill", "action", "fields")):
            return False
        form = SkillForm.from_dict(data)
        errors = validate_skill(form)
        return len(errors) == 0
    except Exception:
        return False
