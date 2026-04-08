"""
formedskill.tui — Terminal UI utilities.

Pure ANSI + Unicode. No third-party dependencies.
Auto-detects color support and gracefully degrades to plain text.
"""

from __future__ import annotations

import os
import sys
from typing import Any


# ── Color support detection ───────────────────────────────────────────────────

def _supports_color() -> bool:
    """Return True if the terminal supports ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


COLOR = _supports_color()


# ── ANSI codes ────────────────────────────────────────────────────────────────

class _Codes:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground colors
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    # Bright variants
    BRED    = "\033[91m"
    BGREEN  = "\033[92m"
    BYELLOW = "\033[93m"
    BCYAN   = "\033[96m"
    BWHITE  = "\033[97m"

C = _Codes()


def _c(code: str, text: str) -> str:
    """Apply an ANSI code if color is supported, otherwise return text as-is."""
    if not COLOR:
        return text
    return f"{code}{text}{C.RESET}"


def bold(text: str) -> str:
    return _c(C.BOLD, text)

def dim(text: str) -> str:
    return _c(C.DIM, text)

def green(text: str) -> str:
    return _c(C.BGREEN, text)

def red(text: str) -> str:
    return _c(C.BRED, text)

def yellow(text: str) -> str:
    return _c(C.BYELLOW, text)

def cyan(text: str) -> str:
    return _c(C.BCYAN, text)

def magenta(text: str) -> str:
    return _c(C.MAGENTA, text)

def dim_white(text: str) -> str:
    return _c(C.DIM, text)

def bold_cyan(text: str) -> str:
    if not COLOR:
        return text
    return f"{C.BOLD}{C.BCYAN}{text}{C.RESET}"

def bold_green(text: str) -> str:
    if not COLOR:
        return text
    return f"{C.BOLD}{C.BGREEN}{text}{C.RESET}"

def bold_red(text: str) -> str:
    if not COLOR:
        return text
    return f"{C.BOLD}{C.BRED}{text}{C.RESET}"

def bold_yellow(text: str) -> str:
    if not COLOR:
        return text
    return f"{C.BOLD}{C.BYELLOW}{text}{C.RESET}"

def bold_magenta(text: str) -> str:
    if not COLOR:
        return text
    return f"{C.BOLD}{C.MAGENTA}{text}{C.RESET}"


# ── Box drawing ───────────────────────────────────────────────────────────────

def box(lines: list[str], title: str = "", width: int = 44) -> str:
    """
    Render a box with Unicode box-drawing characters.

    Example:
      box(["line one", "line two"], title="Header", width=32)
      -> ┌─ Header ────────────────────┐
         │ line one                    │
         │ line two                    │
         └─────────────────────────────┘
    """
    inner = width - 2  # subtract left/right borders

    if title:
        title_str = f" {title} "
        dash_right = "─" * max(0, inner - len(title_str) - 1)
        top = f"┌─{title_str}{dash_right}┐"
    else:
        top = f"┌{'─' * inner}┐"

    bottom = f"└{'─' * inner}┘"

    body_lines = []
    for line in lines:
        # Pad to inner width (strip ANSI for length calc)
        visible_len = _visible_length(line)
        padding = max(0, inner - 2 - visible_len)
        body_lines.append(f"│ {line}{' ' * padding} │")

    return "\n".join([top] + body_lines + [bottom])


def _visible_length(text: str) -> int:
    """Return the visible length of a string, ignoring ANSI escape codes."""
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    return len(ansi_escape.sub('', text))


def print_box(lines: list[str], title: str = "", width: int = 44, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(box(lines, title=title, width=width), file=file)


# ── Banner ────────────────────────────────────────────────────────────────────

FORGE_BANNER = """\
  ┌─────────────────────────────────────────┐
  │  {icon} FormedSkill Forge{pad}│
  │  API doc -> optimized skill in one shot  │
  └─────────────────────────────────────────┘"""

def print_banner(icon: str = "⚡", file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    pad_len = 41 - len(f"  {icon} FormedSkill Forge")
    pad = " " * max(0, pad_len)
    banner = FORGE_BANNER.format(icon=icon, pad=pad)
    if COLOR:
        print(bold_magenta(banner), file=file)
    else:
        print(banner, file=file)
    print(file=file)


# ── Progress bar ──────────────────────────────────────────────────────────────

def progress_bar(current: int, total: int, width: int = 24) -> str:
    """
    Return a progress bar string like: ████████████░░░░ 75%

    Args:
        current: Current value (0 <= current <= total)
        total: Max value
        width: Number of bar characters
    """
    if total <= 0:
        pct = 1.0
    else:
        pct = min(1.0, max(0.0, current / total))

    filled = int(pct * width)
    empty = width - filled
    bar = "█" * filled + "░" * empty
    pct_str = f"{int(pct * 100)}%"

    if COLOR:
        if pct >= 1.0:
            bar_colored = bold_green(bar)
        elif pct >= 0.5:
            bar_colored = cyan(bar)
        else:
            bar_colored = yellow(bar)
        return f"{bar_colored} {dim_white(pct_str)}"
    return f"{bar} {pct_str}"


def print_step(
    step_num: int,
    total_steps: int,
    label: str,
    detail: str = "",
    current: int = 0,
    total: int = 0,
    file: Any = None,
) -> None:
    """
    Print a step line with progress bar.

    Example:
      Step 2/4: Generating test cases...
      ████████████████████████ 100%  50 test cases
    """
    if file is None:
        file = sys.stderr
    step_label = bold(f"  Step {step_num}/{total_steps}:") + f" {label}"
    print(step_label, file=file)
    if total > 0:
        bar = progress_bar(current, total)
        suffix = f"  {dim_white(detail)}" if detail else ""
        print(f"  {bar}{suffix}", file=file)
    print(file=file)


# ── Table ─────────────────────────────────────────────────────────────────────

def table(
    headers: list[str],
    rows: list[list[str]],
    file: Any = None,
) -> None:
    """
    Print a Unicode box-drawing table.

    Example:
      ┌────────────────┬─────────┬────────┐
      │ Skill          │ Version │ Fields │
      ├────────────────┼─────────┼────────┤
      │ snailprint     │ 1.0.0   │ 12     │
      └────────────────┴─────────┴────────┘
    """
    if file is None:
        file = sys.stderr

    if not headers:
        return

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], _visible_length(str(cell)))

    def _sep(left: str, mid: str, right: str, fill: str = "─") -> str:
        parts = [fill * (w + 2) for w in col_widths]
        return left + mid.join(parts) + right

    def _row_line(cells: list[str], color_fn=None) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] if i < len(col_widths) else 10
            visible = _visible_length(str(cell))
            padding = " " * max(0, w - visible)
            if color_fn:
                parts.append(f" {color_fn(str(cell))}{padding} ")
            else:
                parts.append(f" {str(cell)}{padding} ")
        return "│" + "│".join(parts) + "│"

    print(_sep("┌", "┬", "┐"), file=file)
    print(_row_line(headers, color_fn=bold), file=file)
    print(_sep("├", "┼", "┤"), file=file)
    for row in rows:
        print(_row_line(row), file=file)
    print(_sep("└", "┴", "┘"), file=file)


# ── Status indicators ─────────────────────────────────────────────────────────

def ok(msg: str, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(f"  {green('✓')} {msg}", file=file)

def err(msg: str, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(f"  {red('✗')} {msg}", file=file)

def warn(msg: str, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(f"  {yellow('!')} {msg}", file=file)

def info(msg: str, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(f"  {cyan('·')} {msg}", file=file)

def step(msg: str, file: Any = None) -> None:
    if file is None:
        file = sys.stderr
    print(f"  {bold_cyan('⚡')} {msg}", file=file)


# ── Key-value display ─────────────────────────────────────────────────────────

def kv(key: str, value: str, key_width: int = 10, file: Any = None) -> None:
    """Print a key: value pair with aligned values."""
    if file is None:
        file = sys.stderr
    k = dim_white(f"{key:<{key_width}}")
    print(f"  {k}  {value}", file=file)


# ── Accuracy display ──────────────────────────────────────────────────────────

def accuracy_str(score: float) -> str:
    """Return a colored accuracy string like '94% ▲' or '100% ✓'."""
    pct = int(score * 100)
    s = f"{pct}%"
    if score >= 1.0:
        return bold_green(f"{s} ✓")
    elif score > 0.8:
        return green(f"{s} ▲")
    elif score > 0.6:
        return yellow(f"{s} ▲")
    else:
        return red(f"{s}")
