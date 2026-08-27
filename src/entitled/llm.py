"""LLM provider chain: Anthropic API -> claude CLI -> None.

The agent must work with NO provider at all (the Day-6 deterministic
spine is the floor), so every consumer of complete() handles text=None.
Lessons inherited from Incremental's brief-writer:
- claude CLI can return rc=0 with an error message on stdout -> sniff it
- the CLI waits on stdin -> always DEVNULL
- always return a diagnostics dict; silent fallbacks are undebuggable
"""

from __future__ import annotations

import os
import shutil
import subprocess

MODEL = "claude-haiku-4-5"            # cheap + fast; the math is not its job
MAX_TOKENS = 1000

# CLI --model aliases -> API model ids (for the api path); keep in sync with
# bench.PRICING keys. The CLI accepts the alias directly.
MODEL_ALIASES = {"haiku": "claude-haiku-4-5",
                 "sonnet": "claude-sonnet-4-6",
                 "opus": "claude-opus-4-8"}

_ERROR_SNIFF = ("not logged in", "error:", "invalid api key", "rate limit",
                "usage:", "traceback")


def _looks_like_error(text: str) -> bool:
    head = text.strip()[:200].lower()
    return not head or any(s in head for s in _ERROR_SNIFF)


def _via_api(prompt: str, system: str | None, model: str) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL_ALIASES.get(model, model), max_tokens=MAX_TOKENS,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception:
        return None


def _via_cli(prompt: str, system: str | None, model: str) -> str | None:
    exe = shutil.which("claude")
    if not exe:
        return None
    full = f"{system}\n\n{prompt}" if system else prompt
    # the CLI takes the short alias; map an api id back to its alias if given
    alias = next((a for a, m in MODEL_ALIASES.items() if m == model), model)
    try:
        proc = subprocess.run(
            [exe, "-p", full, "--model", alias],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
        if proc.returncode != 0 or _looks_like_error(proc.stdout):
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def complete(prompt: str, system: str | None = None,
             model: str = "haiku") -> tuple[str | None, dict]:
    """Returns (text, diagnostics). text=None means no provider produced
    output — callers MUST have a deterministic fallback. `model` is a tier
    alias (haiku/sonnet/opus) or an explicit api id."""
    diag = {"tried": [], "model": model}
    for name, fn in [("anthropic-api", _via_api), ("claude-cli", _via_cli)]:
        diag["tried"].append(name)
        text = fn(prompt, system, model)
        if text:
            diag["provider"] = name
            return text, diag
    diag["provider"] = None
    return None, diag
