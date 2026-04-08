"""
formedskill.cli — Command-line interface entry point.

Commands:
  formedskill run <skill.yaml> "<message>" [--model MODEL] [--endpoint URL] [--mode step-by-step|single-shot] [--dry-run] [--no-confirm]
  formedskill validate <skill.yaml>
  formedskill list <directory>
  formedskill forge <source> [--output-dir DIR] [--model MODEL] [--endpoint URL]
  formedskill optimize <skill.yaml> <tests.json> [--model MODEL] [--endpoint URL]
  formedskill gen-tests <skill.yaml> [--model MODEL] [--endpoint URL] [--count N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


# ── run ───────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Run a skill form against a user message."""
    from formedskill.schema import load_skill
    from formedskill.runtime import FormRunner
    from formedskill.confirmation import render_confirmation, confirmation_is_enabled
    from formedskill.assembler import execute_action, preview_action
    from formedskill.tui import (
        bold, bold_cyan, bold_green, bold_red, cyan, dim, green, red, yellow,
        ok, err, warn, info, table, print_box, kv, accuracy_str, COLOR,
    )

    try:
        form = load_skill(args.skill)
    except FileNotFoundError as e:
        err(str(e))
        return 1
    except ValueError as e:
        err(f"Validation error: {e}")
        return 1

    runner = FormRunner(
        endpoint=args.endpoint,
        model=args.model,
        verbose=args.verbose,
    )

    # Resolve effective strategy: CLI flag overrides form-level strategy
    effective_strategy = args.strategy if args.strategy != "auto" else form.strategy

    # Header
    print(file=sys.stderr)
    print(f"  {bold_cyan('⚡ FormedSkill Run')}", file=sys.stderr)
    kv("Skill:", f"{bold(form.name)} v{form.meta.version}")
    kv("Model:", args.model)
    kv("Strategy:", effective_strategy)
    print(file=sys.stderr)

    t_start = time.monotonic()

    try:
        if args.strategy == "single-shot":
            result = runner.run_single_shot(form, args.message)
        elif args.strategy == "batched" or effective_strategy == "batched":
            result = runner.run_batched(form, args.message)
        elif args.strategy == "step-by-step" or effective_strategy == "step-by-step":
            result = runner.run_step_by_step(form, args.message)
        else:
            # auto — delegate to run_auto which picks based on model + form.strategy
            result = runner.run_auto(form, args.message)
    except RuntimeError as e:
        err(f"LLM error: {e}")
        return 1

    # Show collected fields in a box
    box_lines = []
    for step in result.step_results:
        if step.skipped:
            continue
        val_str = str(step.answer) if step.answer is not None else dim("(none)")
        elapsed_str = dim(f"({step.elapsed:.1f}s)") if step.elapsed > 0 else ""
        field_label = f"{cyan(step.id):<20}"
        arrow = dim("->")
        box_lines.append(f"{field_label} {arrow} {bold(val_str):<30} {elapsed_str}")

    print_box(box_lines, title="Filling form", width=60)
    print(file=sys.stderr)

    # Stats
    elapsed = time.monotonic() - t_start
    info(f"Completed in {elapsed:.1f}s  |  {result.total_tokens} tokens")
    print(file=sys.stderr)

    # Confirmation step
    if not args.no_confirm and confirmation_is_enabled(form):
        rendered = render_confirmation(form, result.collected)
        if rendered:
            conf_lines = [line for line in rendered.splitlines() if line.strip()]
            print_box(conf_lines, title="Confirmation", width=50)
            print(file=sys.stderr)
            if not args.yes:
                try:
                    answer = input("  Proceed? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print(file=sys.stderr)
                    warn("Aborted.")
                    return 0
                if answer not in ("y", "yes"):
                    warn("Aborted.")
                    return 0

    # Dry run
    if args.dry_run:
        print(file=sys.stderr)
        warn("DRY RUN — action not executed")
        preview = preview_action(form, result.collected)
        print_box(preview.splitlines(), title="Would send", width=60)
        return 0

    # Execute action
    action_result = execute_action(form, result.collected)

    if args.json:
        print(json.dumps(action_result, indent=2, default=str))
    else:
        if action_result["success"]:
            method = form.action.method
            url = form.action.url or form.action.command or form.action.tool_name or ""
            print(file=sys.stderr)
            ok(f"{method} {url}")
            response = action_result.get("response") or action_result.get("stdout", "")
            if response:
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(str(response)[:2000])
        else:
            print(file=sys.stderr)
            err(f"Action failed: {action_result.get('error', 'Unknown error')}")
            return 1

    return 0


# ── validate ──────────────────────────────────────────────────────────────────

def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a skill YAML file against the schema."""
    from formedskill.schema import load_skill, validate_skill, _load_yaml, SkillForm
    from formedskill.tui import ok, err, bold, cyan, dim, green, yellow, kv

    path = Path(args.skill)
    if not path.exists():
        err(f"File not found: {path}")
        return 1

    try:
        text = path.read_text(encoding="utf-8")
        data = _load_yaml(text)
    except Exception as e:
        err(f"YAML parse error: {e}")
        return 1

    try:
        form = SkillForm.from_dict(data, source_path=path)
    except ValueError as e:
        err(f"Schema error: {e}")
        return 1

    errors = validate_skill(form)
    if errors:
        print(file=sys.stderr)
        err(f"{form.name} — {len(errors)} error(s):")
        for e in errors:
            print(f"    {yellow('!')} {e}", file=sys.stderr)
        return 1

    print(file=sys.stderr)
    mode_count = sum(1 for f in form.fields if f.options) if form.fields else 0
    cond_count = sum(1 for f in form.fields if f.show_when)
    action_str = f"{form.action.type.upper()} {form.action.url or form.action.command or form.action.tool_name or ''}"

    ok(f"{bold(form.name)} v{form.meta.version} — {len(form.fields)} fields, {cond_count} conditional")
    kv("Action:", action_str)
    if form.confirmation and form.confirmation.enabled:
        kv("Confirm:", green("enabled"))
    else:
        kv("Confirm:", dim("disabled"))
    print(file=sys.stderr)

    return 0


# ── list ──────────────────────────────────────────────────────────────────────

def cmd_list(args: argparse.Namespace) -> int:
    """List available skill YAML files in a directory."""
    from formedskill.schema import _load_yaml, SkillForm
    from formedskill.tui import bold, dim, red, yellow, cyan, table, err, warn

    directory = Path(args.directory)
    if not directory.exists():
        err(f"Directory not found: {directory}")
        return 1

    yaml_files = sorted(
        list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
    )

    skill_rows: list[list[str]] = []
    invalid_rows: list[tuple[Path, str]] = []

    for path in yaml_files:
        try:
            text = path.read_text(encoding="utf-8")
            data = _load_yaml(text)
            if "skill" in data and "fields" in data:
                form = SkillForm.from_dict(data, source_path=path)
                rel = path.relative_to(directory) if path.is_relative_to(directory) else path
                cond = sum(1 for f in form.fields if f.show_when)
                skill_rows.append([
                    bold(form.name),
                    form.meta.version,
                    str(len(form.fields)),
                    f"{cond} conditional" if cond else "",
                    form.description[:40] + ("..." if len(form.description) > 40 else ""),
                ])
        except Exception as e:
            rel = path.relative_to(directory) if path.is_relative_to(directory) else path
            invalid_rows.append((rel, str(e)[:60]))

    if not skill_rows and not invalid_rows:
        warn(f"No skill files found in {directory}")
        return 0

    print(file=sys.stderr)
    if skill_rows:
        table(
            headers=["Skill", "Version", "Fields", "Conditions", "Description"],
            rows=skill_rows,
        )

    if invalid_rows:
        print(file=sys.stderr)
        for path, errmsg in invalid_rows:
            print(f"  {red('✗')} {path}: {dim(errmsg)}", file=sys.stderr)

    print(file=sys.stderr)
    return 0


# ── forge ─────────────────────────────────────────────────────────────────────

def cmd_forge(args: argparse.Namespace) -> int:
    """Full pipeline: source -> generate -> gen_tests -> optimize -> save."""
    from formedskill.tui import (
        bold, bold_cyan, bold_green, bold_magenta, bold_red, bold_yellow,
        cyan, dim, green, red, yellow, magenta,
        ok, err, warn, info, table, kv, progress_bar, accuracy_str, COLOR,
    )
    from formedskill.forge import forge, ForgeProgress

    source = args.source
    source_path = Path(source) if Path(source).exists() else None

    # Detect source type label
    if source_path:
        if source_path.suffix in (".yaml", ".yml"):
            source_label = f"{source_path.name} (existing SKILL.yaml)"
        elif source_path.suffix == ".json":
            source_label = f"{source_path.name} (OpenAPI JSON)"
        elif source_path.suffix in (".yaml", ".yml"):
            source_label = f"{source_path.name} (OpenAPI YAML)"
        elif source_path.suffix == ".md":
            source_label = f"{source_path.name} (Markdown)"
        else:
            source_label = str(source_path.name)
    else:
        desc_preview = source[:50] + "..." if len(source) > 50 else source
        source_label = f'"{desc_preview}" (natural language)'

    # Print banner
    print(file=sys.stderr)
    banner_lines = [
        f"  {bold_magenta('⚡ FormedSkill Forge')}",
        f"  {dim('API doc -> optimized skill in one shot')}",
    ]
    for line in banner_lines:
        print(line, file=sys.stderr)

    _box_top = "  ┌─────────────────────────────────────────┐"
    _box_bot = "  └─────────────────────────────────────────┘"

    print(file=sys.stderr)
    kv("Source:", source_label)
    kv("Model:", args.model)
    kv("Target:", f"{int(args.target * 100)}% accuracy")
    kv("Tests:", str(args.count))
    kv("Iters:", str(args.iterations))
    print(file=sys.stderr)

    # Progress state
    iteration_rows: list[list[str]] = []
    stage_steps = {"generate": 0, "gen_tests": 0, "baseline": 0, "optimize": 0, "save": 0}
    last_stage = [None]

    def _on_progress(p: ForgeProgress) -> None:
        # Print stage header on transition
        if p.stage != last_stage[0]:
            last_stage[0] = p.stage
            stage_names = {
                "generate": "Step 1/4: Generating skill definition",
                "gen_tests": "Step 2/4: Generating test cases",
                "baseline":  "Step 3/4: Baseline benchmark",
                "optimize":  "Step 4/4: Optimizing (Karpathy loop)",
                "save":      "Saving results",
            }
            print(f"  {bold(stage_names.get(p.stage, p.stage))}...", file=sys.stderr)

        if p.stage == "optimize" and p.step > 0:
            # Accumulate iteration table rows
            score_str = accuracy_str(p.score)
            detail = p.detail or ""
            # Check if this iter index already recorded
            if len(iteration_rows) < p.step:
                iteration_rows.append([str(p.step), score_str, detail])
            else:
                iteration_rows[p.step - 1] = [str(p.step), score_str, detail]

        elif p.stage in ("generate", "gen_tests", "baseline") and p.total > 0:
            bar = progress_bar(p.step, p.total)
            detail = f"  {dim(p.detail)}" if p.detail else ""
            print(f"  {bar}{detail}", file=sys.stderr)

    try:
        result = forge(
            source=args.source,
            output_dir=args.output_dir,
            model=args.model,
            endpoint=args.endpoint,
            api_key=getattr(args, "api_key", None),
            test_count=args.count,
            iterations=args.iterations,
            target=args.target,
            timeout=args.timeout,
            output_name=getattr(args, "output_name", None),
            on_progress=_on_progress,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(file=sys.stderr)
        err(str(e))
        return 1

    # Print optimization table
    if iteration_rows:
        print(file=sys.stderr)
        table(
            headers=["Iter", "Accuracy", "Changes"],
            rows=iteration_rows,
        )

    # Summary
    print(file=sys.stderr)
    ok(f"Saved: {bold(str(result.skill_yaml_path))} ({int(result.final_score * 100)}% accuracy, {result.iterations_run} iterations)")
    ok(f"Tests: {bold(str(result.test_cases_path))} ({result.test_count} cases)")
    print(file=sys.stderr)

    delta = result.final_score - result.initial_score
    if delta > 0:
        improvement = f"+{int(delta * 100)}pp vs baseline"
        print(f"  {bold_green('Your model now handles this API with ' + str(int(result.final_score * 100)) + '% accuracy.')}", file=sys.stderr)
        print(f"  {dim('(' + improvement + ')')}", file=sys.stderr)
    elif result.final_score >= args.target:
        print(f"  {bold_green('Target accuracy reached!')}", file=sys.stderr)

    print(file=sys.stderr)
    return 0


# ── optimize ──────────────────────────────────────────────────────────────────

def cmd_optimize(args: argparse.Namespace) -> int:
    """Hill-climbing optimizer for an existing skill + test suite."""
    from formedskill.tui import (
        bold, bold_cyan, bold_green, cyan, dim, green, red, yellow,
        ok, err, warn, info, table, kv, progress_bar, accuracy_str,
    )
    from formedskill.optimizer import optimize, OptimizationResult

    skill_path = Path(args.skill)
    if not skill_path.exists():
        err(f"Skill file not found: {skill_path}")
        return 1

    tests_path = Path(args.tests)
    if not tests_path.exists():
        err(f"Test cases file not found: {tests_path}")
        return 1

    try:
        with open(tests_path, encoding="utf-8") as f:
            test_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        err(f"Failed to load test cases: {e}")
        return 1

    print(file=sys.stderr)
    print(f"  {bold_cyan('⚡ FormedSkill Optimize')}", file=sys.stderr)
    kv("Skill:", str(skill_path))
    kv("Tests:", str(tests_path))
    kv("Model:", args.model)
    kv("Target:", f"{int(args.target * 100)}%")
    kv("Max iters:", str(args.iterations))
    print(file=sys.stderr)

    iteration_rows: list[list[str]] = []

    def _on_iter(iter_num: int, score: float, desc: str) -> None:
        score_str = accuracy_str(score)
        iteration_rows.append([str(iter_num), score_str, desc])
        if iter_num == 0:
            print(f"  Baseline: {score_str}", file=sys.stderr)
        else:
            bar = progress_bar(iter_num, args.iterations)
            print(f"  {bar}  iter {iter_num}: {score_str}  {dim(desc)}", file=sys.stderr)

    try:
        result = optimize(
            skill_path=skill_path,
            test_cases=test_data,
            model=args.model,
            endpoint=args.endpoint,
            iterations=args.iterations,
            target=args.target,
            timeout=args.timeout,
            on_iteration=_on_iter,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        err(str(e))
        return 1

    # Print table
    if iteration_rows:
        print(file=sys.stderr)
        table(
            headers=["Iter", "Accuracy", "Changes"],
            rows=iteration_rows,
        )

    # Save the best YAML
    skill_path.write_text(result.best_yaml, encoding="utf-8")

    print(file=sys.stderr)
    ok(f"Saved optimized skill: {bold(str(skill_path))}")
    ok(f"Final accuracy: {bold_green(str(int(result.best_score * 100)) + '%')} "
       f"{dim('(was ' + str(int(result.initial_score * 100)) + '%)')}")
    ok(f"Iterations: {result.iterations_run}")
    if result.converged:
        ok("Converged (target reached or no more failures)")
    print(file=sys.stderr)

    return 0


# ── gen-tests ─────────────────────────────────────────────────────────────────

def cmd_gen_tests(args: argparse.Namespace) -> int:
    """Auto-generate test cases from a SKILL.yaml."""
    from formedskill.tui import (
        bold, bold_cyan, bold_green, cyan, dim, green, red,
        ok, err, warn, info, kv, progress_bar,
    )
    from formedskill.gen_tests import generate_tests, save_test_cases

    skill_path = Path(args.skill)
    if not skill_path.exists():
        err(f"Skill file not found: {skill_path}")
        return 1

    print(file=sys.stderr)
    print(f"  {bold_cyan('⚡ FormedSkill Gen-Tests')}", file=sys.stderr)
    kv("Skill:", str(skill_path))
    kv("Model:", args.model)
    kv("Count:", str(args.count))
    print(file=sys.stderr)

    info(f"Generating {args.count} test cases via LLM...")
    print(file=sys.stderr)

    try:
        test_data = generate_tests(
            skill_path=skill_path,
            model=args.model,
            endpoint=args.endpoint,
            count=args.count,
            timeout=args.timeout,
            temperature=0.7,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        err(str(e))
        return 1

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        stem = skill_path.stem
        out_path = skill_path.parent / f"{stem}_tests.json"

    try:
        save_test_cases(test_data, out_path)
    except OSError as e:
        err(f"Failed to save test cases: {e}")
        return 1

    actual_count = len(test_data.get("skills", [{}])[0].get("tests", []))

    print(file=sys.stderr)
    ok(f"Generated {bold(str(actual_count))} test cases")
    ok(f"Saved to: {bold(str(out_path))}")
    print(file=sys.stderr)

    # Show a preview of first 3 cases
    tests = test_data.get("skills", [{}])[0].get("tests", [])
    if tests:
        info("Preview (first 3 cases):")
        print(file=sys.stderr)
        for tc in tests[:3]:
            msg = tc.get("user_message", "")[:70]
            desc = tc.get("description", "")
            expected = tc.get("expected_fields", {})
            expected_str = ", ".join(f"{k}={v!r}" for k, v in list(expected.items())[:3])
            print(f"  {dim(tc['id'])}: {msg}", file=sys.stderr)
            print(f"    {dim('-> ' + expected_str)}", file=sys.stderr)
        print(file=sys.stderr)

    return 0


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="formedskill",
        description="Guided form filling for LLM tool calls. 3B models score 100%.",
    )
    parser.add_argument("--version", action="version", version="formedskill 0.1.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run a skill against a user message")
    run_p.add_argument("skill", metavar="SKILL.yaml", help="Path to skill definition file")
    run_p.add_argument("message", metavar="MESSAGE", help="User's natural language request")
    run_p.add_argument("--model", "-m", default="llama3",
                       help="Model name (default: llama3)")
    run_p.add_argument("--endpoint", "-e", default="http://localhost:11434",
                       help="OpenAI-compatible endpoint (default: http://localhost:11434)")
    run_p.add_argument("--strategy", choices=["auto", "step-by-step", "batched", "single-shot"],
                       default="auto", help="Execution strategy (default: auto)")
    run_p.add_argument("--dry-run", action="store_true",
                       help="Show what would be sent without executing the action")
    run_p.add_argument("--no-confirm", action="store_true",
                       help="Skip the confirmation step")
    run_p.add_argument("--yes", "-y", action="store_true",
                       help="Auto-confirm without prompting")
    run_p.add_argument("--json", action="store_true",
                       help="Output action result as JSON")
    run_p.add_argument("--verbose", "-v", action="store_true",
                       help="Print each extraction step")
    run_p.set_defaults(func=cmd_run)

    # ── validate ─────────────────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate a skill YAML file")
    val_p.add_argument("skill", metavar="SKILL.yaml", help="Path to skill definition file")
    val_p.set_defaults(func=cmd_validate)

    # ── list ─────────────────────────────────────────────────────────────────
    list_p = sub.add_parser("list", help="List available skills in a directory")
    list_p.add_argument("directory", metavar="DIRECTORY", nargs="?", default=".",
                        help="Directory to search (default: current directory)")
    list_p.set_defaults(func=cmd_list)

    # ── forge ─────────────────────────────────────────────────────────────────
    forge_p = sub.add_parser(
        "forge",
        help="Full pipeline: source -> generate -> gen-tests -> optimize -> save",
    )
    forge_p.add_argument(
        "source",
        metavar="SOURCE",
        help=(
            "Input source: path to OpenAPI JSON/YAML, existing SKILL.yaml, "
            ".md file, or a natural language description in quotes"
        ),
    )
    forge_p.add_argument("--output-dir", "-o", default=".",
                         help="Output directory for generated files (default: .)")
    forge_p.add_argument("--output-name", default=None,
                         help="Override output filename stem (without extension)")
    forge_p.add_argument("--model", "-m", default="llama3",
                         help="Model name (default: llama3)")
    forge_p.add_argument("--endpoint", "-e", default="http://localhost:11434",
                         help="OpenAI-compatible endpoint (default: http://localhost:11434)")
    forge_p.add_argument("--count", "-n", type=int, default=30,
                         help="Number of test cases to generate (default: 30)")
    forge_p.add_argument("--iterations", "-i", type=int, default=20,
                         help="Max optimization iterations (default: 20)")
    forge_p.add_argument("--target", "-t", type=float, default=0.98,
                         help="Target accuracy to stop early (default: 0.98)")
    forge_p.add_argument("--timeout", type=int, default=180,
                         help="Per-LLM-call timeout in seconds (default: 180)")
    forge_p.set_defaults(func=cmd_forge)

    # ── optimize ──────────────────────────────────────────────────────────────
    opt_p = sub.add_parser(
        "optimize",
        help="Karpathy-loop optimizer for an existing skill + test suite",
    )
    opt_p.add_argument("skill", metavar="SKILL.yaml", help="Path to skill YAML file")
    opt_p.add_argument("tests", metavar="TESTS.json", help="Path to test cases JSON file")
    opt_p.add_argument("--model", "-m", default="llama3",
                       help="Model name (default: llama3)")
    opt_p.add_argument("--endpoint", "-e", default="http://localhost:11434",
                       help="OpenAI-compatible endpoint (default: http://localhost:11434)")
    opt_p.add_argument("--iterations", "-i", type=int, default=20,
                       help="Max optimization iterations (default: 20)")
    opt_p.add_argument("--target", "-t", type=float, default=0.98,
                       help="Target accuracy to stop early (default: 0.98)")
    opt_p.add_argument("--timeout", type=int, default=180,
                       help="Per-LLM-call timeout in seconds (default: 180)")
    opt_p.set_defaults(func=cmd_optimize)

    # ── gen-tests ─────────────────────────────────────────────────────────────
    gen_p = sub.add_parser(
        "gen-tests",
        help="Auto-generate test cases from a SKILL.yaml",
    )
    gen_p.add_argument("skill", metavar="SKILL.yaml", help="Path to skill YAML file")
    gen_p.add_argument("--output", "-o", default=None,
                       help="Output JSON file (default: <skill_stem>_tests.json)")
    gen_p.add_argument("--model", "-m", default="llama3",
                       help="Model name (default: llama3)")
    gen_p.add_argument("--endpoint", "-e", default="http://localhost:11434",
                       help="OpenAI-compatible endpoint (default: http://localhost:11434)")
    gen_p.add_argument("--count", "-n", type=int, default=30,
                       help="Number of test cases to generate (default: 30)")
    gen_p.add_argument("--timeout", type=int, default=180,
                       help="Per-LLM-call timeout in seconds (default: 180)")
    gen_p.set_defaults(func=cmd_gen_tests)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
