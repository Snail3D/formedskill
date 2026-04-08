#!/usr/bin/env python3
"""
Basic formedskill usage example.

Runs a skill form step-by-step against a local Ollama endpoint.
No LLM calls are made — this just shows the API surface.

To actually run with a model:
  python examples/basic_usage.py
"""

from pathlib import Path
from formedskill import load_skill, FormRunner
from formedskill.assembler import preview_action
from formedskill.confirmation import render_confirmation


def main():
    # Load the built-in snailprint skill
    skill_path = Path(__file__).parent.parent / "formedskill" / "skills" / "snailprint.yaml"
    form = load_skill(skill_path)

    print(f"Skill: {form.name} v{form.meta.version}")
    print(f"Description: {form.description}")
    print(f"Fields: {len(form.fields)}")
    print(f"Action: {form.action.type} {form.action.url}")
    print()

    # Show what fields would be asked
    print("Fields:")
    for f in form.fields:
        cond = f" [show_when: {f.show_when}]" if f.show_when else ""
        opts = f" options={list(f.options.keys())}" if f.options else ""
        print(f"  {f.id} ({f.type}){opts}{cond}")
        print(f"    ask: {f.ask}")
    print()

    # Simulate collected values (without LLM)
    collected = {
        "mode": "generate",
        "prompt": "dragon figurine",
        "filament": "PLA",
        "color": "black",
        "scale_mm": 80,
        "engine": "hunyuan",
        "printer": "auto",
        "auto_print": True,
    }

    print("Simulated collected values:")
    for k, v in collected.items():
        print(f"  {k}: {v}")
    print()

    # Render confirmation
    print("Confirmation:")
    print(render_confirmation(form, collected))
    print()

    # Preview action
    print("Action preview (dry run):")
    print(preview_action(form, collected))
    print()

    # To actually run with a model, uncomment:
    # runner = FormRunner(
    #     endpoint="http://localhost:11434",
    #     model="llama3",
    #     verbose=True,
    # )
    # result = runner.run_step_by_step(form, "Print me a dragon in black PLA, 80mm")
    # print(result.to_dict())


if __name__ == "__main__":
    main()
