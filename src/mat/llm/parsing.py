"""Pull a JSON object out of an LLM response.

Models wrap JSON in prose, fence it in markdown, or emit a trailing comma
even when told not to. Rather than retrying on every such response, repair
the cheap cases here and reserve retries for genuinely broken output.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


class JSONParseError(ValueError):
    """Raised when no valid JSON object could be recovered."""


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from `text`."""
    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # Some models answer a list when asked for {"items": [...]}.
            return {"items": parsed}

    preview = text.strip()[:200]
    raise JSONParseError(f"no JSON object found in response: {preview!r}")


def _candidates(text: str) -> list[str]:
    """Progressively more aggressive repairs, cheapest first."""
    stripped = text.strip()
    candidates = [stripped]

    fenced = FENCE_RE.search(stripped)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for source in list(candidates):
        block = _outermost_block(source)
        if block:
            candidates.append(block)

    candidates.extend(TRAILING_COMMA_RE.sub(r"\1", c) for c in list(candidates))

    seen: dict[str, None] = {}
    for candidate in candidates:
        if candidate:
            seen.setdefault(candidate, None)
    return list(seen)


def _outermost_block(text: str) -> str | None:
    """Slice from the first opening brace/bracket to its matching close."""
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return None

    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
