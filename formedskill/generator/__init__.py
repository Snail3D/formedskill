"""
formedskill.generator — Auto-generate skill YAML from various sources.

  from_api_spec    — OpenAPI/Swagger JSON -> SKILL.yaml
  from_skill_md    — Hermes SKILL.md -> SKILL.yaml
  from_description — Natural language -> SKILL.yaml
"""

from formedskill.generator.from_description import generate_from_description
from formedskill.generator.from_api_spec import generate_from_api_spec

__all__ = [
    "generate_from_description",
    "generate_from_api_spec",
]
