"""Phase 0 smoke test: is the environment actually usable?

Run:  python scripts/check_env.py
Reports each dependency as OK / MISSING so setup problems surface now,
not on demo day.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHECKS = [
    ("streamlit", "UI"),
    ("pandas", "tables"),
    ("plotly", "charts"),
    ("pydantic", "schema validation"),
    ("dateutil", "date resolution"),
    ("groq", "LLM provider (primary)"),
    ("ollama", "LLM provider (offline fallback)"),
    ("faster_whisper", "transcription"),
]


def main() -> int:
    failures = 0
    print(f"python {sys.version.split()[0]}\n")

    for module, purpose in CHECKS:
        try:
            importlib.import_module(module)
            print(f"  OK       {module:<18} {purpose}")
        except ImportError:
            failures += 1
            print(f"  MISSING  {module:<18} {purpose}")

    print()
    try:
        from mat.config import settings

        print(f"  provider  : {settings.llm_provider}")
        print(f"  groq key  : {'set' if settings.groq_api_key else 'NOT SET'}")
        print(f"  whisper   : {settings.whisper_model} ({settings.whisper_compute_type})")
        print(f"  db path   : {settings.db_path}")
    except Exception as exc:  # noqa: BLE001 - surface any config error verbatim
        failures += 1
        print(f"  config load FAILED: {exc}")

    print("\nPhase 0 environment:", "READY" if failures == 0 else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
