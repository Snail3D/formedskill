"""
formedskill.forge — Full pipeline: source -> generate -> gen_tests -> optimize -> save.

Chains all components together:
  forge(source) -> generate() -> gen_tests() -> optimize() -> save()

Usage:
    from formedskill.forge import forge
    result = forge(
        source="my-api.json",        # OpenAPI spec, natural language, or existing YAML
        output_dir="./skills",
        model="llama3",
        endpoint="http://localhost:11434",
        test_count=30,
        iterations=20,
        target=0.98,
        on_progress=my_callback,     # optional progress hook
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional


@dataclass
class ForgeProgress:
    """Progress event emitted during the forge pipeline."""
    stage: Literal["generate", "gen_tests", "baseline", "optimize", "save"]
    step: int           # current step within stage (0-based)
    total: int          # total steps in stage
    message: str        # human-readable status
    detail: str = ""    # optional extra detail (e.g. "12 fields extracted")
    score: float = 0.0  # current accuracy (0.0-1.0), only during optimize


@dataclass
class ForgeResult:
    """Result from a complete forge run."""
    skill_yaml_path: Path
    test_cases_path: Path
    initial_score: float
    final_score: float
    iterations_run: int
    field_count: int
    test_count: int
    elapsed: float
    converged: bool
    skill_name: str


def forge(
    source: str | Path,
    output_dir: str | Path = ".",
    model: str = "llama3",
    endpoint: str = "http://localhost:11434",
    api_key: Optional[str] = None,
    test_count: int = 30,
    iterations: int = 20,
    target: float = 0.98,
    timeout: int = 180,
    output_name: Optional[str] = None,
    on_progress: Optional[Callable[[ForgeProgress], None]] = None,
) -> ForgeResult:
    """
    Run the full forge pipeline on a source.

    Source types (auto-detected):
    - Existing SKILL.yaml: skip generation, go straight to gen_tests + optimize
    - OpenAPI JSON/YAML file: generate from spec
    - Natural language string (not a file path): generate from description
    - File ending in .md: generate from description (reads file content)

    Args:
        source: Input source (file path or natural language description)
        output_dir: Directory to write output files
        model: LLM model name
        endpoint: OpenAI-compatible endpoint URL
        api_key: Optional Bearer token
        test_count: Number of test cases to generate
        iterations: Max optimization iterations
        target: Target accuracy to stop early (0.0-1.0)
        timeout: Per-LLM-call timeout in seconds
        output_name: Override output filename (without extension)
        on_progress: Optional callback receiving ForgeProgress events

    Returns:
        ForgeResult with paths and metrics
    """
    t_start = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _emit(stage, step, total, message, detail="", score=0.0):
        if on_progress:
            on_progress(ForgeProgress(
                stage=stage, step=step, total=total,
                message=message, detail=detail, score=score,
            ))

    # ── Step 1: Generate or load SKILL.yaml ──────────────────────────────────
    source_path = Path(source) if isinstance(source, (str, Path)) and Path(str(source)).exists() else None

    if source_path and source_path.suffix in (".yaml", ".yml"):
        # Existing SKILL.yaml — skip generation
        _emit("generate", 1, 1, "Loading existing skill definition")
        from formedskill.schema import load_skill
        form = load_skill(source_path)
        yaml_text = source_path.read_text(encoding="utf-8")
        skill_name = output_name or form.name.lower().replace(" ", "_")
        _emit("generate", 1, 1, "Skill loaded", detail=f"{len(form.fields)} fields")
        skill_yaml_path = source_path
        save_yaml = False  # Don't overwrite the original
    else:
        # Need to generate a new YAML
        _emit("generate", 0, 3, "Generating skill definition...")

        if source_path and source_path.suffix in (".json",):
            # OpenAPI spec (JSON)
            _emit("generate", 1, 3, "Parsing OpenAPI spec...")
            from formedskill.generator.from_api_spec import generate_from_api_spec
            yaml_text = generate_from_api_spec(
                spec_path=source_path,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
            )
        else:
            # Natural language description or .md file
            if source_path and source_path.suffix == ".md":
                description = source_path.read_text(encoding="utf-8")
            else:
                description = str(source)
            _emit("generate", 1, 3, "Sending description to LLM...")
            from formedskill.generator.from_description import generate_from_description
            yaml_text = generate_from_description(
                description=description,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                timeout=timeout,
            )

        from formedskill.schema import _load_yaml, SkillForm
        data = _load_yaml(yaml_text)
        form = SkillForm.from_dict(data)
        skill_name = output_name or form.name.lower().replace(" ", "_")

        _emit("generate", 2, 3, "Validating generated YAML...",
              detail=f"{len(form.fields)} fields extracted")

        # Save generated YAML
        skill_yaml_path = output_dir / f"{skill_name}.yaml"
        skill_yaml_path.write_text(yaml_text, encoding="utf-8")
        save_yaml = True

        _emit("generate", 3, 3, "Skill definition ready",
              detail=f"{len(form.fields)} fields")

    # ── Step 2: Generate test cases ───────────────────────────────────────────
    _emit("gen_tests", 0, 2, f"Generating {test_count} test cases...")

    from formedskill.gen_tests import generate_tests, save_test_cases
    test_data = generate_tests(
        skill_path=skill_yaml_path,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        count=test_count,
        timeout=timeout,
        temperature=0.7,
    )

    actual_count = len(test_data.get("skills", [{}])[0].get("tests", []))
    _emit("gen_tests", 1, 2, "Test cases generated",
          detail=f"{actual_count} test cases")

    tests_path = output_dir / f"{skill_name}_tests.json"
    save_test_cases(test_data, tests_path)

    _emit("gen_tests", 2, 2, "Test cases saved", detail=str(tests_path))

    # ── Step 3: Baseline benchmark ────────────────────────────────────────────
    _emit("baseline", 0, 1, "Running baseline benchmark...")

    from formedskill.optimizer import _score_yaml, _extract_tests
    skill_tests = _extract_tests(test_data, skill_yaml_path)
    current_yaml = skill_yaml_path.read_text(encoding="utf-8")
    baseline_score = _score_yaml(current_yaml, skill_tests, endpoint, model, api_key, timeout)

    _emit("baseline", 1, 1, "Baseline complete",
          detail=f"{int(baseline_score * 100)}% accuracy",
          score=baseline_score)

    # ── Step 4: Optimize ──────────────────────────────────────────────────────
    _emit("optimize", 0, iterations, "Starting optimization loop...", score=baseline_score)

    iteration_log: list[tuple[int, float, str]] = []

    def _on_iter(iter_num: int, score: float, desc: str) -> None:
        iteration_log.append((iter_num, score, desc))
        _emit("optimize", iter_num, iterations,
              f"Iteration {iter_num}", detail=desc, score=score)

    from formedskill.optimizer import optimize
    opt_result = optimize(
        skill_path=skill_yaml_path,
        test_cases=test_data,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        iterations=iterations,
        target=target,
        timeout=timeout,
        on_iteration=_on_iter,
    )

    # ── Step 5: Save results ──────────────────────────────────────────────────
    _emit("save", 0, 2, "Saving optimized skill...")

    # Write the best YAML back to disk
    skill_yaml_path.write_text(opt_result.best_yaml, encoding="utf-8")

    # Update test cases file with final path
    test_data["skills"][0]["skill_yaml_path"] = str(skill_yaml_path.resolve())
    save_test_cases(test_data, tests_path)

    _emit("save", 2, 2, "All done",
          detail=f"{int(opt_result.best_score * 100)}% accuracy, {opt_result.iterations_run} iterations")

    elapsed = time.monotonic() - t_start

    return ForgeResult(
        skill_yaml_path=skill_yaml_path,
        test_cases_path=tests_path,
        initial_score=opt_result.initial_score,
        final_score=opt_result.best_score,
        iterations_run=opt_result.iterations_run,
        field_count=len(form.fields),
        test_count=actual_count,
        elapsed=elapsed,
        converged=opt_result.converged,
        skill_name=skill_name,
    )
