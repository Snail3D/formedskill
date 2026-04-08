"""
formedskill.client — Zero-dependency OpenAI-compatible HTTP client.

Uses only urllib.request from stdlib. Works with Ollama, MLX, vLLM, OpenAI,
and any other OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


class LLMClient:
    """
    Minimal OpenAI-compatible chat client.

    Args:
        endpoint: Base URL, e.g. "http://localhost:11435" or "https://api.openai.com"
        model: Model name, e.g. "gemma4:moe-chat" or "gpt-4o-mini"
        api_key: Optional Bearer token (required for OpenAI, ignored for local)
        default_temperature: Default sampling temperature (0.0–2.0)
        timeout: Request timeout in seconds
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3",
        api_key: Optional[str] = None,
        default_temperature: float = 0.1,
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.default_temperature = default_temperature
        self.timeout = timeout

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Call the chat completions endpoint.

        Returns:
            (response_text, stats) where stats contains:
              - elapsed: wall time in seconds
              - prompt_tokens: tokens in the prompt
              - completion_tokens: tokens generated
              - total_tokens: sum
              - model: model name used
        """
        url = f"{self.endpoint}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM endpoint returned HTTP {e.code}: {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach LLM endpoint {url}: {e.reason}"
            ) from e
        elapsed = time.monotonic() - start

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        usage = data.get("usage", {})
        stats: dict[str, Any] = {
            "elapsed": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "model": data.get("model", model or self.model),
        }
        return content, stats

    def unload_model(self, model: Optional[str] = None) -> None:
        """
        Signal Ollama to unload the model from VRAM (keep_alive: 0).
        Silently ignores errors — not all endpoints support this.
        """
        try:
            url = f"{self.endpoint}/api/generate"
            payload = json.dumps({"model": model or self.model, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass


def chat_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    timeout: int = 120,
    api_key: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Module-level convenience function. No class instantiation needed.

    Returns (response_text, stats_dict).
    """
    client = LLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        default_temperature=temperature,
        timeout=timeout,
    )
    return client.chat_completion(messages, temperature=temperature)
