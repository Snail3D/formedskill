"""
formedskill.runtime — FormRunner: the core execution engine.

Two modes:
  run_step_by_step  — one LLM call per field (proven approach, 3B = 100%)
  run_single_shot   — one LLM call for all fields (faster, less reliable)

Ported directly from /tmp/guided_form_skill.py with full type support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from formedskill.client import LLMClient
from formedskill.conditions import should_show_field
from formedskill.inference import extract_answer
from formedskill.schema import Field, SkillForm

# System prompt for extraction calls — keep short and directive
_EXTRACTION_SYSTEM = (
    "You extract single values from user messages. "
    "Respond with ONLY the value. One word or short phrase. Nothing else."
)

_SINGLE_SHOT_SYSTEM = (
    "You fill out forms by extracting information from user messages. "
    "Respond ONLY with the field values in the exact format requested. "
    "No explanations, no extra text."
)


@dataclass
class StepResult:
    """Result for a single form field extraction."""
    id: str
    answer: Any
    skipped: bool = False  # True when show_when condition was False
    elapsed: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FormResult:
    """Complete result from a form run."""
    collected: dict[str, Any]
    step_results: list[StepResult]
    total_elapsed: float
    total_prompt_tokens: int
    total_completion_tokens: int
    mode: str  # "step-by-step" or "single-shot"

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "collected": self.collected,
            "mode": self.mode,
            "total_elapsed": round(self.total_elapsed, 3),
            "total_tokens": self.total_tokens,
            "steps": [
                {
                    "id": s.id,
                    "answer": s.answer,
                    "skipped": s.skipped,
                    "elapsed": round(s.elapsed, 3),
                    "tokens": s.prompt_tokens + s.completion_tokens,
                }
                for s in self.step_results
            ],
        }


class FormRunner:
    """
    Runs a SkillForm against an LLM endpoint to extract structured parameters
    from a natural language user message.

    Args:
        endpoint: OpenAI-compatible base URL (e.g. "http://localhost:11434")
        model: Model name
        temperature: Sampling temperature for extraction (default 0.1 — deterministic)
        api_key: Optional Bearer token
        timeout: Per-request timeout in seconds
        verbose: Print step-by-step progress to stdout
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        timeout: int = 120,
        verbose: bool = False,
    ):
        self.client = LLMClient(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
            default_temperature=temperature,
            timeout=timeout,
        )
        self.temperature = temperature
        self.verbose = verbose

    def run_step_by_step(self, form: SkillForm, user_message: str) -> FormResult:
        """
        Fill the form one field at a time. One LLM call per visible field.

        This is the proven approach. Even 3B models hit 100% accuracy because
        each call asks exactly one simple question with full context.
        """
        collected: dict[str, Any] = {}
        step_results: list[StepResult] = []
        total_elapsed = 0.0
        total_prompt = 0
        total_completion = 0

        for f in form.fields:
            # Evaluate show_when — skip fields whose condition isn't met
            if not should_show_field(f.show_when, collected):
                step_results.append(StepResult(id=f.id, answer=None, skipped=True))
                continue

            prompt = _build_step_prompt(f, collected, user_message)
            messages = [
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": prompt},
            ]

            if self.verbose:
                print(f"  [{f.id}] asking: {f.ask[:60]}...")

            raw, stats = self.client.chat_completion(
                messages, temperature=self.temperature
            )

            answer = extract_answer(raw, f)

            if self.verbose:
                raw_preview = repr(raw[:40])
                print(f"  [{f.id}] raw={raw_preview} -> {answer!r}")

            if answer is not None:
                collected[f.id] = answer

            step_results.append(
                StepResult(
                    id=f.id,
                    answer=answer,
                    skipped=False,
                    elapsed=stats["elapsed"],
                    prompt_tokens=stats["prompt_tokens"],
                    completion_tokens=stats["completion_tokens"],
                )
            )
            total_elapsed += stats["elapsed"]
            total_prompt += stats["prompt_tokens"]
            total_completion += stats["completion_tokens"]

        return FormResult(
            collected=collected,
            step_results=step_results,
            total_elapsed=total_elapsed,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            mode="step-by-step",
        )

    def run_single_shot(self, form: SkillForm, user_message: str) -> FormResult:
        """
        Fill the entire form in one LLM call.

        Faster but less reliable on smaller models. Use step-by-step for
        production accuracy.
        """
        prompt = _build_single_shot_prompt(form, user_message)
        messages = [
            {"role": "system", "content": _SINGLE_SHOT_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        raw, stats = self.client.chat_completion(
            messages, temperature=self.temperature
        )

        collected, step_results = _parse_single_shot_response(raw, form)

        return FormResult(
            collected=collected,
            step_results=step_results,
            total_elapsed=stats["elapsed"],
            total_prompt_tokens=stats["prompt_tokens"],
            total_completion_tokens=stats["completion_tokens"],
            mode="single-shot",
        )


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_step_prompt(f: Field, collected: dict[str, Any], user_message: str) -> str:
    """Build a minimal prompt for extracting a single field value."""
    parts = [f'User said: "{user_message}"']

    if collected:
        parts.append(f"Already collected: {json.dumps(collected)}")

    parts.append(f"\nExtract the value for: {f.id}")

    if f.infer_from:
        parts.append(f"Hint: {f.infer_from}")

    if f.options:
        opts = list(f.options.keys())
        parts.append(f"Valid options: {', '.join(opts)}")

    if f.default is not None:
        parts.append(f"Default (if not specified by user): {f.default}")

    parts.append("\nRespond with ONLY the value, nothing else. No quotes, no explanation.")

    return "\n".join(parts)


def _build_single_shot_prompt(form: SkillForm, user_message: str) -> str:
    """Build a prompt that asks the model to fill out all fields at once."""
    parts = [
        f'A user said: "{user_message}"',
        "",
        f"Fill out this form for the {form.name} skill by extracting information from the user's message.",
        "For each field, provide ONLY the value — no explanation. If the user didn't specify, write DEFAULT.",
        "Respond in this exact format:",
        "",
    ]

    for f in form.fields:
        field_line = f"{f.id}: "
        if f.options:
            opts = " | ".join(f"{k} ({v})" for k, v in f.options.items())
            field_line += f"[{opts}]"
        elif f.type == "number":
            field_line += f"(number, default: {f.default if f.default is not None else 'none'})"
        elif f.type == "boolean":
            field_line += "(yes/no)"
        elif f.type == "json_array":
            field_line += "(JSON array of strings)"
        else:
            field_line += f"(text, default: {f.default if f.default is not None else 'none'})"

        if f.infer_from:
            field_line += f"\n  Hint: {f.infer_from}"
        if f.show_when:
            field_line += f"\n  Only fill if: {f.show_when}"

        parts.append(field_line)
        parts.append("")

    return "\n".join(parts)


def _parse_single_shot_response(
    raw: str, form: SkillForm
) -> tuple[dict[str, Any], list[StepResult]]:
    """
    Parse the model's single-shot response (key: value lines) into
    a collected dict and step results.
    """
    from formedskill.inference import extract_answer, _is_sentinel, _clean_raw

    # Build field lookup
    field_map = {f.id: f for f in form.fields}

    collected: dict[str, Any] = {}
    step_results: list[StepResult] = []

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, _, val_str = line.partition(":")
        key = key.strip().lower()
        val_str = val_str.strip()

        if key not in field_map:
            continue

        f = field_map[key]
        cleaned = _clean_raw(val_str)

        if _is_sentinel(cleaned):
            answer = f.default
        else:
            answer = extract_answer(val_str, f)

        if answer is not None:
            collected[key] = answer

        step_results.append(StepResult(id=key, answer=answer, skipped=False))

    # Ensure all fields have a result entry
    seen = {s.id for s in step_results}
    for f in form.fields:
        if f.id not in seen:
            step_results.append(StepResult(id=f.id, answer=f.default, skipped=True))

    return collected, step_results
