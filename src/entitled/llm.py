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

MODEL = "claude-haiku-4-5-20251001"   # cheap + fast; the math is not its job
MAX_TOKENS = 1000

_ERROR_SNIFF = ("not logged in", "error:", "invalid api key", "rate limit",
                "usage:", "traceback")


def _looks_like_error(text: str) -> bool:
    head = text.strip()[:200].lower()
    return not head or any(s in head for s in _ERROR_SNIFF)


def _via_api(prompt: str, system: str | None) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
    except Exception:
        return None


def _via_cli(prompt: str, system: str | None) -> str | None:
    exe = shutil.which("claude")
    if not exe:
        return None
    full = f"{system}\n\n{prompt}" if system else prompt
    try:
        proc = subprocess.run(
            [exe, "-p", full, "--model", "haiku"],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
        if proc.returncode != 0 or _looks_like_error(proc.stdout):
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def complete(prompt: str, system: str | None = None) -> tuple[str | None, dict]:
    """Returns (text, diagnostics). text=None means no provider produced
    output — callers MUST have a deterministic fallback."""
    diag = {"tried": []}
    for name, fn in [("anthropic-api", _via_api), ("claude-cli", _via_cli)]:
        diag["tried"].append(name)
        text = fn(prompt, system)
        if text:
            diag["provider"] = name
            return text, diag
    diag["provider"] = None
    return None, diag
