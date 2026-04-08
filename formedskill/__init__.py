"""
formedskill — Guided form filling for LLM tool calls.

Break complex parameter extraction into atomic steps. Even 3B models score 100%.
"""

__version__ = "0.1.0"
__author__ = "Eric Woodard"
__license__ = "MIT"

from formedskill.schema import SkillForm, Field, Action, Confirmation, load_skill
from formedskill.runtime import FormRunner

__all__ = [
    "FormedSkill",
    "FormRunner",
    "SkillForm",
    "Field",
    "Action",
    "Confirmation",
    "load_skill",
    "__version__",
]

# Alias for ergonomics
FormedSkill = SkillForm
