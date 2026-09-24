"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import sys


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which cannot encode characters that
    turn up routinely in model output - a narrow no-break space in a
    summary was enough to crash a finished extraction at the print step,
    losing a run that had already cost a dozen API calls.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
