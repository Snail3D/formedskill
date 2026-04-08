---
name: formedskill
description: Optimize skills with FormedSkill — converts monolithic SKILL.md into structured guided-form YAML for 95-100% accuracy on small models. Use when creating or fixing skills with complex API parameters.
version: 1.0.0
author: snail3d
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [skills, optimization, forms, accuracy, formedskill, forge, skill-creator]
    related_skills: []
---

# FormedSkill — Skill Optimizer

Use FormedSkill when creating or updating skills that have complex API calls, multiple modes, or many parameters. It converts monolithic skill definitions into structured guided-form YAML that achieves 95-100% accuracy on small models (vs ~12% for monolithic prompts).

## When to Use

**Check if FormedSkill applies when:**
- Creating a new skill that calls an API with 3+ parameters
- A skill is producing wrong API calls or hallucinating parameters
- User says "fix this skill", "this skill isn't working", "optimize this"
- A skill has multiple modes (create/update/delete) with different parameters per mode
- The skill involves structured inputs: API calls, CLI commands, form submissions

**Don't use FormedSkill for:**
- Workflow/process skills (TDD, code review, debugging) — these are instructions, not forms
- Skills with no structured API calls
- Simple skills with 1-2 parameters

## Quick Decision

Ask yourself: "Does this skill need the model to construct a specific API call or command with multiple parameters?" If yes → use FormedSkill.

## How to Use

FormedSkill CLI is at: `~/formedskill/`
Run with: `cd ~/formedskill && python3 -m formedskill <command>`

### Creating a New Skill with FormedSkill

1. **Generate the YAML from a description:**
```bash
cd ~/formedskill && python3 -m formedskill forge \
  "A skill that controls Philips Hue lights — on/off, brightness 0-100, color, room selection" \
  --model nemotron-cascade-2 --endpoint http://localhost:11434 \
  --output ~/.hermes/skills/smart-home/hue-lights/SKILL.yaml
```

2. **Or convert an existing SKILL.md:**
```bash
cd ~/formedskill && python3 -m formedskill forge \
  ~/.hermes/skills/category/skill-name/SKILL.md \
  --model nemotron-cascade-2 --endpoint http://localhost:11434 \
  --output ~/.hermes/skills/category/skill-name/SKILL.yaml
```

3. **Validate it:**
```bash
cd ~/formedskill && python3 -m formedskill validate SKILL.yaml
```

4. **Test it:**
```bash
cd ~/formedskill && python3 -m formedskill run SKILL.yaml \
  "test user message" \
  --model nemotron-cascade-2 --endpoint http://localhost:11434
```

### Fixing a Failing Skill

1. **Convert the broken SKILL.md to YAML:**
```bash
cd ~/formedskill && python3 -m formedskill forge \
  ~/.hermes/skills/<category>/<skill>/SKILL.md \
  --output ~/.hermes/skills/<category>/<skill>/SKILL.yaml
```

2. **Generate test cases:**
```bash
cd ~/formedskill && python3 -m formedskill gen-tests SKILL.yaml \
  --count 20 --output tests.json
```

3. **Optimize with Karpathy loop:**
```bash
cd ~/formedskill && python3 -m formedskill optimize SKILL.yaml tests.json \
  --iterations 20 --target 0.98
```

## YAML Schema Quick Reference

```yaml
skill:
  name: my-skill
  version: "1.0.0"
  description: "What it does"
  tags: [tag1, tag2]

strategy: auto  # auto | step-by-step | batched

preamble: >
  Short 2-3 sentence context. Helps model understand domain terms.
  Always include this — adds ~9% accuracy.

action:
  type: http  # http | shell | tool_call
  method: POST
  url: "http://localhost:PORT/api/endpoint"

fields:
  - id: field_name
    ask: "Question"
    type: options  # text | number | options | boolean | json_array
    options:
      key: "Description"
    default: value
    show_when: "other_field == value"
    infer_from: "Hint for extraction"
    aliases:
      "natural name": "actual_value"

confirmation:
  enabled: true
  template: |
    Ready to execute:
    - Field: {{field_name}}
```

## Key Rules
- Always include a `preamble` — 2-3 sentences about what the tool does (+9% accuracy)
- Use `strategy: auto` — lets FormedSkill pick batched for Mamba models, step-by-step for others
- Put the most important routing field first (e.g., `mode`, `action`, `operation`)
- Use `show_when` to hide irrelevant fields per mode
- Use `aliases` for natural language mapping (e.g., "P2D2" → serial number)
- Use `infer_from` with arrow notation: "quick/fast → spar3d. Default hunyuan."
- Do NOT set max_tokens — thinking models need room for reasoning tokens
- SKILL.yaml lives alongside SKILL.md — both coexist, YAML takes priority when FormedSkill integration is active

## Benchmarks
- Monolithic SKILL.md: 12% accuracy on 3B model
- FormedSkill step-by-step: 95% accuracy, 15s
- FormedSkill batched: 100% accuracy, 6s
- Tested on 15 real Hermes skills, 75 test cases
