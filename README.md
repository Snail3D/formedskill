# formedskill

**Break complex LLM tool calls into step-by-step form filling.**

Instead of dumping an entire API schema into a system prompt (which small models fail at), `formedskill` presents **one field at a time**. The LLM fills out a form. The framework assembles the final API call.

**Result: 3B parameter models score 100% on tasks they previously failed.**

```
gemma3:4b  monolithic  →  45% accuracy
gemma3:4b  guided-form → 100% accuracy
```

Zero required dependencies. Works with any OpenAI-compatible endpoint: Ollama, MLX, vLLM, OpenAI.

---

## Install

```bash
pip install formedskill

# With PyYAML (recommended — more robust YAML parsing):
pip install "formedskill[yaml]"
```

Or from source:

```bash
git clone https://github.com/Snail3D/formedskill
cd formedskill
pip install -e .
```

---

## Quickstart

```bash
# Run a built-in skill
formedskill run formedskill/skills/snailprint.yaml \
  "Print me a dragon figurine in black PLA, 80mm" \
  --model gemma4:moe-chat \
  --endpoint http://localhost:11435 \
  --dry-run

# Validate a skill file
formedskill validate myskill.yaml

# List skills in a directory
formedskill list ~/.hermes/skills/
```

```python
from formedskill import FormRunner, load_skill

form = load_skill("snailprint.yaml")
runner = FormRunner(endpoint="http://localhost:11434", model="llama3")
result = runner.run_step_by_step(form, "Print me a dragon in black PLA")

print(result.collected)
# {'mode': 'generate', 'prompt': 'dragon', 'filament': 'PLA',
#  'color': 'black', 'scale_mm': 50, 'engine': 'hunyuan', 'printer': 'auto'}
```

---

## Why It Works

Traditional approach: stuff the entire API schema into the system prompt, ask the model to generate a curl command. Small models hallucinate parameters, get formats wrong, pick wrong options.

`formedskill` approach:

1. **One field per LLM call** — "What filament? (PLA | PETG | TPU | ABS)"
2. **Infer hints** — tell the model exactly how to interpret user input
3. **Conditional fields** — skip fields that don't apply (`show_when`)
4. **Type coercion** — numbers, booleans, JSON arrays, option matching
5. **Alias resolution** — "P2D2" → actual printer serial number

Each extraction call is minimal: user message + one question + hint + valid options. Even a 3B model handles this reliably.

---

## YAML Schema Reference

```yaml
skill:
  name: myskill                     # Required. Unique identifier.
  version: "1.0.0"
  description: "What this skill does"
  tags: [tag1, tag2]
  platforms: [macos, linux]         # Omit = all platforms

action:
  type: http                        # "http" | "shell" | "tool_call"
  method: POST
  url: "http://localhost:8080/api/action"
  headers:
    Content-Type: "application/json"
  timeout: 60
  payload_map:                      # Optional: remap field IDs to payload keys
    prompt: description             # field "prompt" -> payload key "description"

fields:
  - id: mode                        # Required. Unique identifier used in show_when.
    ask: "What type of job?"        # Required. Shown to LLM as the question.
    type: options                   # text | number | options | boolean | json_array
    options:
      generate: "Create from text"
      file: "Use existing file"
    required: true                  # Block form if not answered (default false)
    default: generate               # Used when LLM says DEFAULT/NONE/N/A
    infer_from: >-                  # Hint for the LLM — how to extract this value
      User describes something to create -> generate.
      User mentions a file path -> file.
    show_when: "mode == generate"   # Conditional: only show if condition is true
    aliases:                        # Natural language -> option key mapping
      P2D2: "22E8AJ5C2800915"
    validation:
      min: 5          # number: minimum value
      max: 500        # number: maximum value
      min_items: 2    # json_array: minimum items
      max_items: 8    # json_array: maximum items

confirmation:
  enabled: true
  template: |
    Ready to run {{mode}}:
    - Input: {{prompt}}
    {% if color %}- Color: {{color}}{% endif %}
    Proceed? (yes/no)
```

### Field Types

| Type | Python Type | Notes |
|------|------------|-------|
| `text` | `str` | Free-form |
| `number` | `int` / `float` | Supports unit conversion (cm→mm, inches→mm) |
| `options` | `str` | Must match option key; fuzzy-matched |
| `boolean` | `bool` | yes/true/1 → True, no/false/0 → False |
| `json_array` | `list` | JSON array or comma-separated fallback |

### Condition Syntax (`show_when`)

| Pattern | Meaning |
|---------|---------|
| `field == value` | Show when field equals value |
| `field != value` | Show when field does not equal value |
| `field in [a, b, c]` | Show when field is one of the values |
| `field` | Show when field has any truthy value |

---

## Python API

```python
from formedskill import load_skill, FormRunner
from formedskill.assembler import execute_action
from formedskill.confirmation import render_confirmation

# Load and validate a skill
form = load_skill("snailprint.yaml")

# Run step-by-step (recommended)
runner = FormRunner(
    endpoint="http://localhost:11434",
    model="llama3",
    temperature=0.1,   # Low = deterministic extraction
    verbose=True,      # Print each step
)
result = runner.run_step_by_step(form, "Print a dragon in black PLA")

# result.collected  — dict of extracted values
# result.total_elapsed  — wall time
# result.total_tokens   — total tokens used

# Show confirmation
print(render_confirmation(form, result.collected))

# Execute the action
action_result = execute_action(form, result.collected)
print(action_result["response"])
```

---

## CLI Reference

```bash
# Run a skill
formedskill run SKILL.yaml "user message" [OPTIONS]
  --model MODEL         Model name (default: llama3)
  --endpoint URL        API endpoint (default: http://localhost:11434)
  --mode step-by-step|single-shot   (default: step-by-step)
  --dry-run             Show payload without executing
  --no-confirm          Skip confirmation prompt
  --yes, -y             Auto-confirm
  --json                Output as JSON
  --verbose, -v         Print each extraction step

# Validate
formedskill validate SKILL.yaml

# List skills
formedskill list [DIRECTORY]
```

---

## Hermes Integration

`formedskill` integrates with [Hermes](https://github.com/Snail3D/hermes-agent) as a native skill type. Place `SKILL.yaml` alongside existing `SKILL.md` files:

```
~/.hermes/skills/media/snailprint/
├── SKILL.md      # Existing (kept for reference)
└── SKILL.yaml    # New: guided-form definition
```

See [Section 5 of the design doc](.omc/plans/guided-form-skill-framework.md) for the full integration guide.

---

## License

MIT — see [LICENSE](LICENSE)

Built by [Snail3D](https://github.com/Snail3D)
