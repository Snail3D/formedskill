"""
FormedSkill Benchmark Harness

Compares monolithic (SKILL.md stuffed into system prompt) vs guided-form
(FormedSkill step-by-step YAML) approaches across multiple skills and
test cases.

Usage:
    python benchmarks/run_benchmark.py --test-cases benchmarks/test_cases.json \
        --model nemotron-cascade-2 --port 11434
    python benchmarks/run_benchmark.py --test-cases benchmarks/test_cases.json \
        --model gemma4:moe-chat --port 11435
    python benchmarks/run_benchmark.py --test-cases benchmarks/test_cases.json \
        --model nemotron-cascade-2 --port 11434 --mode formedskill-only
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Allow importing formedskill from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_field(expected: Any, got: Any) -> bool:
    """Fuzzy field scoring — handles strings, numbers, and fallback."""
    if expected is None and got is None:
        return True
    if expected is None or got is None:
        return False
    if isinstance(expected, str) and isinstance(got, str):
        return expected.lower() in got.lower() or got.lower() in expected.lower()
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        return abs(expected - got) < 1
    return str(expected).lower() == str(got).lower()


def score_result(expected_fields: dict[str, Any], collected: dict[str, Any]) -> dict:
    """
    Score a collected result against expected fields.

    Returns:
        {
          "per_field": {"field_id": True/False, ...},
          "correct": int,
          "total": int,
          "accuracy": float (0-1),
          "hallucinated_keys": [keys in collected but not in expected]
        }
    """
    per_field = {}
    for key, exp_val in expected_fields.items():
        got_val = collected.get(key)
        per_field[key] = score_field(exp_val, got_val)

    correct = sum(1 for v in per_field.values() if v)
    total = len(per_field)
    accuracy = correct / total if total > 0 else 0.0

    expected_keys = set(expected_fields.keys())
    collected_keys = set(collected.keys())
    hallucinated = sorted(collected_keys - expected_keys)

    return {
        "per_field": per_field,
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4),
        "hallucinated_keys": hallucinated,
    }


# ── LLM helpers (stdlib only) ─────────────────────────────────────────────────

def _llm_call(
    endpoint: str,
    model: str,
    messages: list[dict],
    temperature: float = 0.1,
    timeout: int = 120,
) -> tuple[str, dict]:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint using urllib.request.

    Returns (response_text, stats_dict).
    """
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e
    elapsed = time.monotonic() - start

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    )
    usage = data.get("usage", {})
    stats = {
        "elapsed": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return content, stats


def _parse_json_from_response(text: str) -> dict:
    """
    Extract a JSON object from a model response.

    Tries in order:
    1. Parse the full text as JSON.
    2. Extract the first {...} block.
    3. Return empty dict on failure.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return {}


# ── Monolithic mode ───────────────────────────────────────────────────────────

def run_monolithic(
    skill_md_path: str,
    user_message: str,
    endpoint: str,
    model: str,
    timeout: int = 120,
) -> dict:
    """
    Run the monolithic approach: stuff SKILL.md into the system prompt,
    ask the model to output JSON parameters in a single shot.

    Returns:
        {
          "collected": {...},
          "raw_response": str,
          "elapsed": float,
          "prompt_tokens": int,
          "completion_tokens": int,
          "error": str | None
        }
    """
    result = {
        "collected": {},
        "raw_response": "",
        "elapsed": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
    }

    # Load SKILL.md
    if not skill_md_path:
        result["error"] = "No SKILL.md path provided"
        return result
    md_path = Path(skill_md_path).expanduser()
    if not md_path.exists() or md_path.is_dir():
        result["error"] = f"SKILL.md not found: {skill_md_path}"
        return result

    skill_md_content = md_path.read_text(encoding="utf-8")

    system_prompt = (
        "You are Hermes Agent. Here is a skill definition:\n\n"
        f"{skill_md_content}\n\n"
        "The user wants to perform an action. Generate ONLY the exact command or API call. "
        "Output the parameters as JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        raw, stats = _llm_call(endpoint, model, messages, timeout=timeout)
    except RuntimeError as e:
        result["error"] = str(e)
        return result

    result["raw_response"] = raw
    result["elapsed"] = stats["elapsed"]
    result["prompt_tokens"] = stats["prompt_tokens"]
    result["completion_tokens"] = stats["completion_tokens"]
    result["collected"] = _parse_json_from_response(raw)

    return result


# ── FormedSkill mode ──────────────────────────────────────────────────────────

def run_formedskill(
    skill_yaml_path: str,
    user_message: str,
    endpoint: str,
    model: str,
    timeout: int = 120,
) -> dict:
    """
    Run the FormedSkill guided-form approach using the runtime step-by-step engine.

    Returns:
        {
          "collected": {...},
          "elapsed": float,
          "prompt_tokens": int,
          "completion_tokens": int,
          "steps": [...],
          "error": str | None
        }
    """
    result: dict[str, Any] = {
        "collected": {},
        "elapsed": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "steps": [],
        "error": None,
    }

    yaml_path = Path(skill_yaml_path).expanduser()
    if not yaml_path.exists():
        result["error"] = f"SKILL.yaml not found: {skill_yaml_path}"
        return result

    try:
        from formedskill.schema import load_skill
        from formedskill.runtime import FormRunner
    except ImportError as e:
        result["error"] = f"formedskill import failed: {e}"
        return result

    try:
        form = load_skill(yaml_path)
    except (FileNotFoundError, ValueError) as e:
        result["error"] = f"Failed to load skill YAML: {e}"
        return result

    try:
        runner = FormRunner(
            endpoint=endpoint,
            model=model,
            temperature=0.1,
            timeout=timeout,
            verbose=False,
        )
        form_result = runner.run_step_by_step(form, user_message)
    except RuntimeError as e:
        result["error"] = f"FormedSkill runtime error: {e}"
        return result

    result["collected"] = form_result.collected
    result["elapsed"] = form_result.total_elapsed
    result["prompt_tokens"] = form_result.total_prompt_tokens
    result["completion_tokens"] = form_result.total_completion_tokens
    result["steps"] = [
        {
            "id": s.id,
            "answer": s.answer,
            "skipped": s.skipped,
            "elapsed": s.elapsed,
            "tokens": s.prompt_tokens + s.completion_tokens,
        }
        for s in form_result.step_results
    ]

    return result


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    test_cases_path: str,
    model: str,
    port: int,
    host: str = "localhost",
    mode: str = "both",
    timeout: int = 120,
) -> dict:
    """
    Run all test cases and return the full benchmark result structure.

    mode: "both" | "monolithic-only" | "formedskill-only"
    """
    endpoint = f"http://{host}:{port}"

    with open(test_cases_path, encoding="utf-8") as f:
        test_data = json.load(f)

    skills = test_data.get("skills", [])
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    benchmark = {
        "meta": {
            "model": model,
            "endpoint": endpoint,
            "timestamp": timestamp,
            "mode": mode,
            "test_cases_path": test_cases_path,
        },
        "skills": [],
        "summary": {},
    }

    run_mono = mode in ("both", "monolithic-only")
    run_fs = mode in ("both", "formedskill-only")

    total_mono_correct = 0
    total_mono_fields = 0
    total_fs_correct = 0
    total_fs_fields = 0

    for skill_def in skills:
        skill_name = skill_def.get("name", "unknown")
        skill_md_path = skill_def.get("skill_md_path", "") or ""
        skill_yaml_path = skill_def.get("skill_yaml_path") or skill_def.get("builtin_yaml_path", "") or ""
        # Expand ~ in paths
        if skill_md_path:
            skill_md_path = str(Path(skill_md_path).expanduser())
        if skill_yaml_path:
            skill_yaml_path = str(Path(skill_yaml_path).expanduser())
        tests = skill_def.get("tests", [])

        print(f"\n{'='*60}")
        print(f"Skill: {skill_name}  ({len(tests)} tests)")
        print(f"{'='*60}")

        skill_record: dict[str, Any] = {
            "name": skill_name,
            "skill_md_path": skill_md_path,
            "skill_yaml_path": skill_yaml_path,
            "tests": [],
            "monolithic": {"correct": 0, "total": 0, "accuracy": 0.0, "elapsed": 0.0},
            "formedskill": {"correct": 0, "total": 0, "accuracy": 0.0, "elapsed": 0.0},
        }

        for tc in tests:
            tc_id = tc.get("id", "?")
            user_message = tc.get("user_message", "")
            expected_fields = tc.get("expected_fields", {})
            description = tc.get("description", "")

            print(f"\n  [{tc_id}] {description}")
            print(f"  User: {user_message[:80]}")

            tc_record: dict[str, Any] = {
                "id": tc_id,
                "user_message": user_message,
                "description": description,
                "expected_fields": expected_fields,
                "monolithic": None,
                "formedskill": None,
            }

            # --- Monolithic ---
            if run_mono:
                print(f"  [monolithic] running...", end=" ", flush=True)
                mono_out = run_monolithic(
                    skill_md_path=skill_md_path,
                    user_message=user_message,
                    endpoint=endpoint,
                    model=model,
                    timeout=timeout,
                )

                if mono_out["error"]:
                    print(f"ERROR: {mono_out['error']}")
                    mono_score = {"correct": 0, "total": len(expected_fields),
                                  "accuracy": 0.0, "per_field": {}, "hallucinated_keys": []}
                else:
                    mono_score = score_result(expected_fields, mono_out["collected"])
                    print(
                        f"{mono_score['correct']}/{mono_score['total']} correct "
                        f"({mono_score['accuracy']*100:.0f}%) "
                        f"in {mono_out['elapsed']:.1f}s"
                    )

                tc_record["monolithic"] = {
                    "collected": mono_out["collected"],
                    "raw_response": mono_out.get("raw_response", ""),
                    "elapsed": mono_out["elapsed"],
                    "prompt_tokens": mono_out["prompt_tokens"],
                    "completion_tokens": mono_out["completion_tokens"],
                    "error": mono_out["error"],
                    "score": mono_score,
                }

                skill_record["monolithic"]["correct"] += mono_score["correct"]
                skill_record["monolithic"]["total"] += mono_score["total"]
                skill_record["monolithic"]["elapsed"] += mono_out["elapsed"]

            # --- FormedSkill ---
            if run_fs:
                print(f"  [formedskill] running...", end=" ", flush=True)
                fs_out = run_formedskill(
                    skill_yaml_path=skill_yaml_path,
                    user_message=user_message,
                    endpoint=endpoint,
                    model=model,
                    timeout=timeout,
                )

                if fs_out["error"]:
                    print(f"ERROR: {fs_out['error']}")
                    fs_score = {"correct": 0, "total": len(expected_fields),
                                "accuracy": 0.0, "per_field": {}, "hallucinated_keys": []}
                else:
                    fs_score = score_result(expected_fields, fs_out["collected"])
                    print(
                        f"{fs_score['correct']}/{fs_score['total']} correct "
                        f"({fs_score['accuracy']*100:.0f}%) "
                        f"in {fs_out['elapsed']:.1f}s"
                    )

                tc_record["formedskill"] = {
                    "collected": fs_out["collected"],
                    "elapsed": fs_out["elapsed"],
                    "prompt_tokens": fs_out["prompt_tokens"],
                    "completion_tokens": fs_out["completion_tokens"],
                    "steps": fs_out["steps"],
                    "error": fs_out["error"],
                    "score": fs_score,
                }

                skill_record["formedskill"]["correct"] += fs_score["correct"]
                skill_record["formedskill"]["total"] += fs_score["total"]
                skill_record["formedskill"]["elapsed"] += fs_out["elapsed"]

            skill_record["tests"].append(tc_record)

        # Compute per-skill accuracy
        if run_mono and skill_record["monolithic"]["total"] > 0:
            mono_acc = skill_record["monolithic"]["correct"] / skill_record["monolithic"]["total"]
            skill_record["monolithic"]["accuracy"] = round(mono_acc, 4)
            total_mono_correct += skill_record["monolithic"]["correct"]
            total_mono_fields += skill_record["monolithic"]["total"]

        if run_fs and skill_record["formedskill"]["total"] > 0:
            fs_acc = skill_record["formedskill"]["correct"] / skill_record["formedskill"]["total"]
            skill_record["formedskill"]["accuracy"] = round(fs_acc, 4)
            total_fs_correct += skill_record["formedskill"]["correct"]
            total_fs_fields += skill_record["formedskill"]["total"]

        print(
            f"\n  Skill summary — "
            f"Monolithic: {skill_record['monolithic']['accuracy']*100:.0f}%  |  "
            f"FormedSkill: {skill_record['formedskill']['accuracy']*100:.0f}%"
        )

        benchmark["skills"].append(skill_record)

    # Overall summary
    overall_mono = (
        round(total_mono_correct / total_mono_fields, 4)
        if total_mono_fields > 0 else None
    )
    overall_fs = (
        round(total_fs_correct / total_fs_fields, 4)
        if total_fs_fields > 0 else None
    )

    benchmark["summary"] = {
        "monolithic_accuracy": overall_mono,
        "formedskill_accuracy": overall_fs,
        "monolithic_correct": total_mono_correct,
        "monolithic_total": total_mono_fields,
        "formedskill_correct": total_fs_correct,
        "formedskill_total": total_fs_fields,
        "improvement_pct": (
            round((overall_fs - overall_mono) * 100, 1)
            if overall_mono is not None and overall_fs is not None
            else None
        ),
    }

    return benchmark


def save_results(benchmark: dict, model: str, results_dir: str) -> str:
    """Save benchmark results to a timestamped JSON file. Returns the file path."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filename
    safe_model = model.replace(":", "_").replace("/", "_").replace(" ", "_")
    timestamp = benchmark["meta"]["timestamp"]
    filename = f"{safe_model}_{timestamp}.json"
    out_path = results_path / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, default=str)

    return str(out_path)


def print_summary(benchmark: dict) -> None:
    """Print a human-readable summary table to stdout."""
    meta = benchmark["meta"]
    summary = benchmark["summary"]

    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Model:     {meta['model']}")
    print(f"Endpoint:  {meta['endpoint']}")
    print(f"Timestamp: {meta['timestamp']}")
    print()

    # Per-skill table
    headers = ["Skill", "Mono %", "FS %", "Delta", "Mono t", "FS t"]
    rows = []
    for sk in benchmark["skills"]:
        mono_acc = sk["monolithic"]["accuracy"]
        fs_acc = sk["formedskill"]["accuracy"]
        delta = (
            f"+{(fs_acc - mono_acc)*100:.0f}%"
            if mono_acc is not None and fs_acc is not None
            else "N/A"
        )
        mono_t = f"{sk['monolithic']['elapsed']:.1f}s"
        fs_t = f"{sk['formedskill']['elapsed']:.1f}s"
        rows.append([
            sk["name"],
            f"{mono_acc*100:.0f}%" if mono_acc is not None else "N/A",
            f"{fs_acc*100:.0f}%" if fs_acc is not None else "N/A",
            delta,
            mono_t,
            fs_t,
        ])

    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))

    print()
    print(f"Overall Monolithic:  {(summary['monolithic_accuracy'] or 0)*100:.1f}%  "
          f"({summary['monolithic_correct']}/{summary['monolithic_total']} fields)")
    print(f"Overall FormedSkill: {(summary['formedskill_accuracy'] or 0)*100:.1f}%  "
          f"({summary['formedskill_correct']}/{summary['formedskill_total']} fields)")
    if summary["improvement_pct"] is not None:
        print(f"Improvement:         +{summary['improvement_pct']}pp")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FormedSkill benchmark: monolithic vs guided-form accuracy"
    )
    parser.add_argument(
        "--test-cases",
        required=True,
        help="Path to test_cases.json",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g. nemotron-cascade-2, gemma4:moe-chat)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Ollama/LLM server port (default: 11434)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="LLM server host (default: localhost)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "monolithic-only", "formedskill-only"],
        default="both",
        help="Which modes to run (default: both)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Directory to save result JSON files",
    )

    args = parser.parse_args()

    if not os.path.exists(args.test_cases):
        print(f"Error: test cases file not found: {args.test_cases}", file=sys.stderr)
        sys.exit(1)

    print(f"FormedSkill Benchmark")
    print(f"Model: {args.model}  |  Port: {args.port}  |  Mode: {args.mode}")

    try:
        benchmark = run_benchmark(
            test_cases_path=args.test_cases,
            model=args.model,
            port=args.port,
            host=args.host,
            mode=args.mode,
            timeout=args.timeout,
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading test cases: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)

    print_summary(benchmark)

    out_path = save_results(benchmark, args.model, args.results_dir)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
