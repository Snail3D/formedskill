"""
formedskill — Guided form filling for LLM tool calls.

Break complex parameter extraction into atomic steps. Even 3B models score 100%.
"""

__version__ = "0.1.0"
__author__ = "Snail3D"
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
    "forge",
    "optimize",
    "generate_tests",
    "generate_from_description",
    "generate_from_api_spec",
    "__version__",
]

# Alias for ergonomics
FormedSkill = SkillForm


def forge(*args, **kwargs):
    """Full pipeline: source -> generate -> gen_tests -> optimize -> save."""
    from formedskill.forge import forge as _forge
    return _forge(*args, **kwargs)


def optimize(*args, **kwargs):
    """Hill-climbing optimizer for SKILL.yaml definitions."""
    from formedskill.optimizer import optimize as _optimize
    return _optimize(*args, **kwargs)


def generate_tests(*args, **kwargs):
    """Auto-generate test cases from a SKILL.yaml."""
    from formedskill.gen_tests import generate_tests as _gen
    return _gen(*args, **kwargs)


def generate_from_description(*args, **kwargs):
    """Generate a SKILL.yaml from a natural language description."""
    from formedskill.generator.from_description import generate_from_description as _gen
    return _gen(*args, **kwargs)


def generate_from_api_spec(*args, **kwargs):
    """Generate a SKILL.yaml from an OpenAPI/Swagger spec."""
    from formedskill.generator.from_api_spec import generate_from_api_spec as _gen
    return _gen(*args, **kwargs)
