"""
formedskill.assembler — Assemble and execute the final action from collected values.

Supports:
  http      — POST/GET/etc to a URL with JSON body
  shell     — Template substitution in command string, then subprocess
  tool_call — Build OpenAI function call format dict

payload_map remaps field IDs to payload keys before execution.
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from typing import Any, Optional

from formedskill.schema import Action, SkillForm


def assemble_payload(
    action: Action, collected: dict[str, Any]
) -> dict[str, Any]:
    """
    Build the final payload dict from collected values, applying payload_map.

    payload_map example:
      payload_map:
        prompt: description   # field "prompt" -> payload key "description"

    Fields with None values are excluded.
    """
    # Start with non-None collected values
    body = {k: v for k, v in collected.items() if v is not None}

    # Apply payload_map remapping
    if action.payload_map:
        remapped: dict[str, Any] = {}
        for field_id, payload_key in action.payload_map.items():
            if field_id in body:
                remapped[payload_key] = body.pop(field_id)
        body.update(remapped)

    return body


def execute_action(
    form: SkillForm, collected: dict[str, Any]
) -> dict[str, Any]:
    """
    Execute the skill's action with the collected values.

    Returns a result dict with:
      - success: bool
      - action_type: str
      - payload: dict (what was sent)
      - response: str or dict (what came back)
      - status_code: int (for HTTP)
      - error: str (on failure)
    """
    action = form.action
    payload = assemble_payload(action, collected)

    if action.type == "http":
        return _execute_http(action, payload)
    elif action.type == "shell":
        return _execute_shell(action, collected)
    elif action.type == "tool_call":
        return _build_tool_call(action, payload)
    else:
        return {
            "success": False,
            "action_type": action.type,
            "payload": payload,
            "error": f"Unknown action type: {action.type}",
        }


def _execute_http(action: Action, payload: dict[str, Any]) -> dict[str, Any]:
    """Execute an HTTP action."""
    headers = {"Content-Type": "application/json"}
    headers.update(action.headers or {})

    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        action.url,
        data=body_bytes if action.method not in ("GET", "HEAD") else None,
        headers=headers,
        method=action.method,
    )

    try:
        with urllib.request.urlopen(req, timeout=action.timeout) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                response_data = json.loads(raw)
            except json.JSONDecodeError:
                response_data = raw

        return {
            "success": True,
            "action_type": "http",
            "method": action.method,
            "url": action.url,
            "payload": payload,
            "status_code": status,
            "response": response_data,
        }

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return {
            "success": False,
            "action_type": "http",
            "method": action.method,
            "url": action.url,
            "payload": payload,
            "status_code": e.code,
            "error": f"HTTP {e.code}: {error_body[:500]}",
        }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "action_type": "http",
            "url": action.url,
            "payload": payload,
            "error": f"Connection error: {e.reason}",
        }
    except Exception as e:
        return {
            "success": False,
            "action_type": "http",
            "url": action.url,
            "payload": payload,
            "error": str(e),
        }


def _execute_shell(action: Action, collected: dict[str, Any]) -> dict[str, Any]:
    """Execute a shell command with {{field}} template substitution."""
    if not action.command:
        return {
            "success": False,
            "action_type": "shell",
            "error": "No command defined in action",
        }

    command = _template_substitute(action.command, collected)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=action.timeout,
        )
        return {
            "success": result.returncode == 0,
            "action_type": "shell",
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "action_type": "shell",
            "command": command,
            "error": f"Command timed out after {action.timeout}s",
        }
    except Exception as e:
        return {
            "success": False,
            "action_type": "shell",
            "command": command,
            "error": str(e),
        }


def _build_tool_call(action: Action, payload: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI function call format dict (does not execute)."""
    tool_call = {
        "type": "function",
        "function": {
            "name": action.tool_name,
            "arguments": json.dumps(payload),
        },
    }
    return {
        "success": True,
        "action_type": "tool_call",
        "tool_name": action.tool_name,
        "payload": payload,
        "tool_call": tool_call,
    }


def _template_substitute(template: str, values: dict[str, Any]) -> str:
    """Replace {{field}} placeholders in a template string."""
    def replacer(m: re.Match) -> str:
        key = m.group(1).strip()
        val = values.get(key)
        if val is None:
            return m.group(0)  # Leave unreplaced
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return str(val)

    return re.sub(r"\{\{(\w+)\}\}", replacer, template)


def preview_action(form: SkillForm, collected: dict[str, Any]) -> str:
    """
    Return a human-readable preview of what the action will do,
    without executing it. Useful for dry-run / confirmation display.
    """
    action = form.action
    payload = assemble_payload(action, collected)

    if action.type == "http":
        lines = [
            f"HTTP {action.method} {action.url}",
            f"Headers: {json.dumps(action.headers or {})}",
            f"Body: {json.dumps(payload, indent=2)}",
        ]
        return "\n".join(lines)

    elif action.type == "shell":
        command = _template_substitute(action.command or "", collected)
        return f"Shell: {command}"

    elif action.type == "tool_call":
        return (
            f"Tool call: {action.tool_name}\n"
            f"Arguments: {json.dumps(payload, indent=2)}"
        )

    return f"Action: {action.type}\nPayload: {json.dumps(payload, indent=2)}"
