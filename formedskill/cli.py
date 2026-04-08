"""
formedskill.cli — Command-line interface entry point.

Commands:
  formedskill run <skill.yaml> "<message>" [--model MODEL] [--endpoint URL] [--mode step-by-step|single-shot] [--dry-run] [--no-confirm]
  formedskill validate <skill.yaml>
  formedskill list <directory>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def cmd_run(args: argparse.Namespace) -> int:
    """Run a skill form against a user message."""
    from formedskill.schema import load_skill
    from formedskill.runtime import FormRunner
    from formedskill.confirmation import render_confirmation, confirmation_is_enabled
    from formedskill.assembler import execute_action, preview_action

    try:
        form = load_skill(args.skill)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return 1

    runner = FormRunner(
        endpoint=args.endpoint,
        model=args.model,
        verbose=args.verbose,
    )

    print(f"Running: {form.name} ({args.mode})", file=sys.stderr)
    print(f"Message: {args.message[:80]}", file=sys.stderr)
    print(file=sys.stderr)

    try:
        if args.mode == "single-shot":
            result = runner.run_single_shot(form, args.message)
        else:
            result = runner.run_step_by_step(form, args.message)
    except RuntimeError as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 1

    # Print step summary
    print(f"Collected ({result.total_elapsed:.1f}s, {result.total_tokens} tokens):", file=sys.stderr)
    for step in result.step_results:
        if step.skipped:
            print(f"  {step.id}: [skipped]", file=sys.stderr)
        else:
            print(f"  {step.id}: {step.answer}", file=sys.stderr)
    print(file=sys.stderr)

    # Confirmation step
    if not args.no_confirm and confirmation_is_enabled(form):
        rendered = render_confirmation(form, result.collected)
        if rendered:
            print(rendered)
            print()
            if not args.yes:
                try:
                    answer = input("Proceed? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nAborted.", file=sys.stderr)
                    return 0
                if answer not in ("y", "yes"):
                    print("Aborted.", file=sys.stderr)
                    return 0

    # Dry run — show what would happen
    if args.dry_run:
        print("--- DRY RUN (action not executed) ---")
        print(preview_action(form, result.collected))
        return 0

    # Execute action
    action_result = execute_action(form, result.collected)

    if args.json:
        print(json.dumps(action_result, indent=2, default=str))
    else:
        if action_result["success"]:
            print(f"Success ({action_result['action_type']})")
            response = action_result.get("response") or action_result.get("stdout", "")
            if response:
                if isinstance(response, dict):
                    print(json.dumps(response, indent=2))
                else:
                    print(str(response)[:2000])
        else:
            print(f"Error: {action_result.get('error', 'Unknown error')}", file=sys.stderr)
            return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a skill YAML file against the schema."""
    from formedskill.schema import load_skill, validate_skill, _load_yaml
    from pathlib import Path

    path = Path(args.skill)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    try:
        text = path.read_text(encoding="utf-8")
        data = _load_yaml(text)
    except Exception as e:
        print(f"YAML parse error: {e}", file=sys.stderr)
        return 1

    try:
        from formedskill.schema import SkillForm
        form = SkillForm.from_dict(data, source_path=path)
    except ValueError as e:
        print(f"Schema error: {e}", file=sys.stderr)
        return 1

    errors = validate_skill(form)
    if errors:
        print(f"Validation FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"Valid: {form.name} v{form.meta.version}")
    print(f"  Fields: {len(form.fields)}")
    print(f"  Action: {form.action.type} {form.action.url or form.action.command or form.action.tool_name or ''}")
    conditional_count = sum(1 for f in form.fields if f.show_when)
    print(f"  Conditional fields: {conditional_count}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available skill YAML files in a directory."""
    from formedskill.schema import _load_yaml, SkillForm
    from pathlib import Path

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        return 1

    # Find all YAML files recursively
    yaml_files = sorted(
        list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
    )

    skill_files: list[tuple[Path, SkillForm | None, str]] = []
    for path in yaml_files:
        try:
            text = path.read_text(encoding="utf-8")
            data = _load_yaml(text)
            if "skill" in data and "fields" in data:
                form = SkillForm.from_dict(data, source_path=path)
                skill_files.append((path, form, ""))
        except Exception as e:
            skill_files.append((path, None, str(e)))

    if not skill_files:
        print(f"No guided-form skill files found in {directory}")
        return 0

    print(f"Found {len(skill_files)} skill(s) in {directory}:")
    print()
    for path, form, error in skill_files:
        rel = path.relative_to(directory) if path.is_relative_to(directory) else path
        if form:
            print(f"  {form.name} ({form.meta.version})")
            print(f"    File:   {rel}")
            print(f"    Desc:   {form.description[:80]}")
            print(f"    Fields: {len(form.fields)}, Action: {form.action.type}")
        else:
            print(f"  [invalid] {rel}: {error[:80]}")
        print()

    return 0


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
    run_p.add_argument(
        "--model", "-m",
        default="llama3",
        help="Model name (default: llama3)",
    )
    run_p.add_argument(
        "--endpoint", "-e",
        default="http://localhost:11434",
        help="OpenAI-compatible endpoint (default: http://localhost:11434)",
    )
    run_p.add_argument(
        "--mode",
        choices=["step-by-step", "single-shot"],
        default="step-by-step",
        help="Execution mode (default: step-by-step)",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without executing the action",
    )
    run_p.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation step",
    )
    run_p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm without prompting",
    )
    run_p.add_argument(
        "--json",
        action="store_true",
        help="Output action result as JSON",
    )
    run_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each extraction step",
    )
    run_p.set_defaults(func=cmd_run)

    # ── validate ─────────────────────────────────────────────────────────────
    val_p = sub.add_parser("validate", help="Validate a skill YAML file")
    val_p.add_argument("skill", metavar="SKILL.yaml", help="Path to skill definition file")
    val_p.set_defaults(func=cmd_validate)

    # ── list ─────────────────────────────────────────────────────────────────
    list_p = sub.add_parser("list", help="List available skills in a directory")
    list_p.add_argument(
        "directory",
        metavar="DIRECTORY",
        nargs="?",
        default=".",
        help="Directory to search (default: current directory)",
    )
    list_p.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
