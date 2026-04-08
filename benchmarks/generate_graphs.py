"""
FormedSkill Benchmark Graph Generator

Reads JSON result files from benchmarks/results/ and generates charts
for the GitHub README, saved to docs/images/.

Usage:
    python benchmarks/generate_graphs.py
    python benchmarks/generate_graphs.py --results-dir benchmarks/results
    python benchmarks/generate_graphs.py --results-file benchmarks/results/model_20240101.json
    python benchmarks/generate_graphs.py --all-results  # combine all result files
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR_MONO = "#ff6b6b"       # red — monolithic
COLOR_FS = "#51cf66"         # green — formedskill
COLOR_BG = "#1a1a2e"         # deep dark background
COLOR_PANEL = "#16213e"      # slightly lighter panel
COLOR_TEXT = "#e0e0e0"       # light text
COLOR_GRID = "#2a2a4a"       # subtle grid lines

DOCS_IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"


# ── Matplotlib setup ──────────────────────────────────────────────────────────

def _get_plt():
    """
    Import matplotlib, falling back to ASCII art mode if unavailable.
    Returns (plt, has_mpl) where has_mpl is True when matplotlib is present.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # No display needed — write to file
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_PANEL,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_TEXT,
            "axes.titlecolor": COLOR_TEXT,
            "xtick.color": COLOR_TEXT,
            "ytick.color": COLOR_TEXT,
            "grid.color": COLOR_GRID,
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
            "text.color": COLOR_TEXT,
            "legend.facecolor": COLOR_PANEL,
            "legend.edgecolor": COLOR_GRID,
            "legend.labelcolor": COLOR_TEXT,
            "font.family": "DejaVu Sans",
            "font.size": 11,
        })
        return plt, True
    except ImportError:
        return None, False


# ── Data loading ──────────────────────────────────────────────────────────────

def load_result_file(path: str) -> dict:
    """Load a single benchmark result JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_results(results_dir: str) -> list[dict]:
    """Load all .json result files from results_dir, newest first."""
    results_path = Path(results_dir)
    files = sorted(results_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    results = []
    for fp in files:
        try:
            results.append(load_result_file(str(fp)))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {fp.name}: {e}", file=sys.stderr)
    return results


def _skill_rows(benchmark: dict) -> list[dict]:
    """
    Extract per-skill accuracy rows from a benchmark result.

    Returns list of:
        {name, mono_acc, fs_acc, mono_elapsed, fs_elapsed}
    """
    rows = []
    for sk in benchmark.get("skills", []):
        rows.append({
            "name": sk["name"],
            "mono_acc": sk["monolithic"]["accuracy"],
            "fs_acc": sk["formedskill"]["accuracy"],
            "mono_elapsed": sk["monolithic"]["elapsed"],
            "fs_elapsed": sk["formedskill"]["elapsed"],
        })
    return rows


# ── Chart 1: Accuracy by Approach (grouped bar per skill) ────────────────────

def chart_accuracy_by_skill(benchmark: dict, out_path: Path, plt) -> None:
    """
    Grouped bar chart: monolithic vs FormedSkill accuracy per skill.
    """
    rows = _skill_rows(benchmark)
    if not rows:
        print("  No skill rows — skipping chart_accuracy_by_skill")
        return

    model = benchmark["meta"]["model"]
    skill_names = [r["name"] for r in rows]
    mono_accs = [r["mono_acc"] * 100 for r in rows]
    fs_accs = [r["fs_acc"] * 100 for r in rows]

    n = len(skill_names)
    x = list(range(n))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    fig.patch.set_facecolor(COLOR_BG)

    bars_mono = ax.bar(
        [xi - width / 2 for xi in x], mono_accs,
        width=width, label="Monolithic", color=COLOR_MONO, alpha=0.9, zorder=3
    )
    bars_fs = ax.bar(
        [xi + width / 2 for xi in x], fs_accs,
        width=width, label="FormedSkill", color=COLOR_FS, alpha=0.9, zorder=3
    )

    # Value labels on top of bars
    for bar in bars_mono:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=9, color=COLOR_MONO
        )
    for bar in bars_fs:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=9, color=COLOR_FS
        )

    ax.set_title(f"FormedSkill vs Monolithic Accuracy\n({model})", pad=14, fontsize=14, fontweight="bold")
    ax.set_xlabel("Skill", labelpad=8)
    ax.set_ylabel("Accuracy (%)", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(skill_names, rotation=15 if n > 4 else 0, ha="right" if n > 4 else "center")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 2: Model Size vs Accuracy (multi-model scatter/line) ────────────────

# Known approximate active parameter counts for common local models
_MODEL_PARAMS: dict[str, float] = {
    "gemma3:1b": 1.0,
    "gemma3:4b": 3.0,      # 4B total ~ 3B active
    "gemma4:scout": 3.0,
    "gemma4:moe-chat": 4.0,
    "phi4-mini": 3.8,
    "qwen2.5:3b": 3.0,
    "qwen2.5:7b": 7.0,
    "llama3.2:3b": 3.0,
    "llama3.1:8b": 8.0,
    "mistral:7b": 7.0,
    "nemotron-mini": 4.0,
    "nemotron-cascade-2": 5.1,
    "phi3.5": 3.8,
    "phi4": 14.0,
    "gemma3:12b": 12.0,
    "gemma3:27b": 27.0,
}

def _infer_params(model_name: str) -> float:
    """Best-effort inference of active param count from model name."""
    name_lower = model_name.lower()
    if name_lower in _MODEL_PARAMS:
        return _MODEL_PARAMS[name_lower]
    # Scan prefix matches
    for k, v in _MODEL_PARAMS.items():
        if name_lower.startswith(k.lower()):
            return v
    # Try to parse digits: "model:7b" -> 7.0
    import re
    m = re.search(r"(\d+(?:\.\d+)?)b", name_lower)
    if m:
        return float(m.group(1))
    return 0.0


def chart_model_size_vs_accuracy(results: list[dict], out_path: Path, plt) -> None:
    """
    Scatter/line chart: active params on X, accuracy on Y.
    One series per approach across all provided result files.
    """
    mono_points: list[tuple[float, float, str]] = []  # (params, acc, model)
    fs_points: list[tuple[float, float, str]] = []

    for bm in results:
        model = bm["meta"]["model"]
        params = _infer_params(model)
        if params <= 0:
            continue
        summary = bm.get("summary", {})
        if summary.get("monolithic_accuracy") is not None:
            mono_points.append((params, summary["monolithic_accuracy"] * 100, model))
        if summary.get("formedskill_accuracy") is not None:
            fs_points.append((params, summary["formedskill_accuracy"] * 100, model))

    if not mono_points and not fs_points:
        print("  No model-size data — skipping chart_model_size_vs_accuracy")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(COLOR_BG)

    def _plot_series(points, color, label):
        if not points:
            return
        points_sorted = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points_sorted]
        ys = [p[1] for p in points_sorted]
        labels = [p[2] for p in points_sorted]
        ax.plot(xs, ys, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter(xs, ys, color=color, s=80, zorder=4)
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(
                lbl, (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=color, alpha=0.9
            )

    _plot_series(mono_points, COLOR_MONO, "Monolithic")
    _plot_series(fs_points, COLOR_FS, "FormedSkill")

    ax.set_title("Model Size vs Accuracy\n(FormedSkill closes the gap)", pad=14, fontsize=14, fontweight="bold")
    ax.set_xlabel("Active Parameters (B)", labelpad=8)
    ax.set_ylabel("Accuracy (%)", labelpad=8)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 3: Response Time Comparison ────────────────────────────────────────

def chart_response_time(benchmark: dict, out_path: Path, plt) -> None:
    """
    Grouped bar chart: monolithic vs FormedSkill elapsed time per skill.
    """
    rows = _skill_rows(benchmark)
    if not rows:
        print("  No skill rows — skipping chart_response_time")
        return

    model = benchmark["meta"]["model"]
    skill_names = [r["name"] for r in rows]
    mono_times = [r["mono_elapsed"] for r in rows]
    fs_times = [r["fs_elapsed"] for r in rows]

    n = len(skill_names)
    x = list(range(n))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 2), 6))
    fig.patch.set_facecolor(COLOR_BG)

    bars_mono = ax.bar(
        [xi - width / 2 for xi in x], mono_times,
        width=width, label="Monolithic", color=COLOR_MONO, alpha=0.9, zorder=3
    )
    bars_fs = ax.bar(
        [xi + width / 2 for xi in x], fs_times,
        width=width, label="FormedSkill", color=COLOR_FS, alpha=0.9, zorder=3
    )

    for bar in bars_mono:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.2,
            f"{h:.1f}s", ha="center", va="bottom", fontsize=9, color=COLOR_MONO
        )
    for bar in bars_fs:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.2,
            f"{h:.1f}s", ha="center", va="bottom", fontsize=9, color=COLOR_FS
        )

    ax.set_title(f"Response Time Comparison\n({model})", pad=14, fontsize=14, fontweight="bold")
    ax.set_xlabel("Skill", labelpad=8)
    ax.set_ylabel("Total Elapsed Time (s)", labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(skill_names, rotation=15 if n > 4 else 0, ha="right" if n > 4 else "center")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Chart 4: Overall Summary (horizontal bar hero image) ─────────────────────

def chart_overall_summary(results: list[dict], out_path: Path, plt) -> None:
    """
    Horizontal bar chart showing per-model monolithic vs FormedSkill accuracy.
    This is the README hero image — big, clear, high contrast.
    """
    rows = []
    for bm in results:
        model = bm["meta"]["model"]
        summary = bm.get("summary", {})
        mono = summary.get("monolithic_accuracy")
        fs = summary.get("formedskill_accuracy")
        if mono is not None or fs is not None:
            rows.append({
                "model": model,
                "mono": (mono or 0.0) * 100,
                "fs": (fs or 0.0) * 100,
            })

    if not rows:
        print("  No summary data — skipping chart_overall_summary")
        return

    # Sort by FormedSkill accuracy descending
    rows.sort(key=lambda r: r["fs"], reverse=True)

    n = len(rows)
    fig_h = max(5, n * 1.2 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor(COLOR_BG)

    model_labels = [r["model"] for r in rows]
    mono_vals = [r["mono"] for r in rows]
    fs_vals = [r["fs"] for r in rows]

    y = list(range(n))
    bar_h = 0.35

    bars_fs = ax.barh(
        [yi + bar_h / 2 for yi in y], fs_vals,
        height=bar_h, label="FormedSkill", color=COLOR_FS, alpha=0.9, zorder=3
    )
    bars_mono = ax.barh(
        [yi - bar_h / 2 for yi in y], mono_vals,
        height=bar_h, label="Monolithic", color=COLOR_MONO, alpha=0.9, zorder=3
    )

    for bar, val in zip(bars_fs, fs_vals):
        ax.text(
            val + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", fontsize=10, color=COLOR_FS, fontweight="bold"
        )
    for bar, val in zip(bars_mono, mono_vals):
        ax.text(
            val + 1, bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%", va="center", fontsize=10, color=COLOR_MONO
        )

    ax.set_title(
        "FormedSkill vs Monolithic — Overall Accuracy by Model",
        pad=16, fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Accuracy (%)", labelpad=8, fontsize=12)
    ax.set_yticks(y)
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.set_xlim(0, 115)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── ASCII art fallback ────────────────────────────────────────────────────────

def _bar(value: float, width: int = 40, fill: str = "#", empty: str = ".") -> str:
    filled = round(value / 100 * width)
    return fill * filled + empty * (width - filled)


def ascii_accuracy_by_skill(benchmark: dict) -> str:
    rows = _skill_rows(benchmark)
    model = benchmark["meta"]["model"]
    lines = [
        f"FormedSkill vs Monolithic Accuracy — {model}",
        "=" * 60,
        "",
    ]
    for r in rows:
        lines.append(f"  {r['name']}")
        lines.append(f"    Monolithic  [{_bar(r['mono_acc']*100, 30)}] {r['mono_acc']*100:5.1f}%")
        lines.append(f"    FormedSkill [{_bar(r['fs_acc']*100, 30, fill='=')}] {r['fs_acc']*100:5.1f}%")
        lines.append("")
    return "\n".join(lines)


def ascii_overall_summary(results: list[dict]) -> str:
    lines = [
        "Overall Accuracy by Model",
        "=" * 60,
        "",
        f"  {'Model':<30} {'Monolithic':>12} {'FormedSkill':>12}",
        "  " + "-" * 56,
    ]
    for bm in results:
        model = bm["meta"]["model"]
        summary = bm.get("summary", {})
        mono = (summary.get("monolithic_accuracy") or 0) * 100
        fs = (summary.get("formedskill_accuracy") or 0) * 100
        lines.append(f"  {model:<30} {mono:>11.1f}% {fs:>11.1f}%")
    return "\n".join(lines)


# ── Markdown summary table ────────────────────────────────────────────────────

def generate_markdown_table(results: list[dict]) -> str:
    """
    Generate a markdown table summarising all result files.
    Suitable for pasting directly into README.md.
    """
    lines = [
        "## Benchmark Results",
        "",
        "| Model | Monolithic | FormedSkill | Improvement |",
        "|-------|-----------|-------------|-------------|",
    ]
    for bm in results:
        model = bm["meta"]["model"]
        summary = bm.get("summary", {})
        mono = summary.get("monolithic_accuracy")
        fs = summary.get("formedskill_accuracy")
        imp = summary.get("improvement_pct")
        mono_str = f"{mono*100:.1f}%" if mono is not None else "—"
        fs_str = f"{fs*100:.1f}%" if fs is not None else "—"
        imp_str = f"+{imp:.1f}pp" if imp is not None else "—"
        lines.append(f"| `{model}` | {mono_str} | {fs_str} | {imp_str} |")

    lines.append("")
    lines.append(
        "> *Accuracy = fraction of expected parameter fields correctly extracted. "
        "Fuzzy matching: string contains, numeric within 1, case-insensitive.*"
    )
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts for FormedSkill README"
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--results-file",
        help="Path to a single benchmark result JSON file",
    )
    source_group.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing benchmark result JSON files (default: benchmarks/results/)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DOCS_IMAGES_DIR),
        help="Output directory for PNG charts (default: docs/images/)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown summary table to stdout",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    if args.results_file:
        if not os.path.exists(args.results_file):
            print(f"Error: result file not found: {args.results_file}", file=sys.stderr)
            sys.exit(1)
        results = [load_result_file(args.results_file)]
        print(f"Loaded 1 result file: {args.results_file}")
    else:
        results = load_all_results(args.results_dir)
        if not results:
            print(f"No result files found in: {args.results_dir}", file=sys.stderr)
            print("Run run_benchmark.py first to generate results.", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(results)} result file(s) from {args.results_dir}")

    plt, has_mpl = _get_plt()

    if has_mpl:
        print("\nGenerating charts (matplotlib)...")

        # Chart 1: Accuracy by skill — use the most recent result
        chart_accuracy_by_skill(
            results[0],
            out_dir / "accuracy_by_skill.png",
            plt,
        )

        # Chart 2: Model size vs accuracy — across all results
        chart_model_size_vs_accuracy(
            results,
            out_dir / "model_size_vs_accuracy.png",
            plt,
        )

        # Chart 3: Response time — most recent result
        chart_response_time(
            results[0],
            out_dir / "response_time.png",
            plt,
        )

        # Chart 4: Overall summary hero image — all results
        chart_overall_summary(
            results,
            out_dir / "overall_summary.png",
            plt,
        )

        print(f"\nAll charts saved to: {out_dir}")

    else:
        print("\nmatplotlib not available — printing ASCII art instead.")
        print("Install with: pip install matplotlib\n")
        print(ascii_accuracy_by_skill(results[0]))
        print()
        print(ascii_overall_summary(results))

    if args.markdown or not has_mpl:
        print("\n" + generate_markdown_table(results))

    # Always write the markdown table to a file
    md_path = out_dir.parent / "benchmark_table.md"
    md_path.write_text(generate_markdown_table(results), encoding="utf-8")
    print(f"\nMarkdown table saved to: {md_path}")


if __name__ == "__main__":
    main()
