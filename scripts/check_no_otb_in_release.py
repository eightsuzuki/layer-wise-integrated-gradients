#!/usr/bin/env python3
"""Post-sync guard: unsupported baselines must not appear under utils/ or lig/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("self_output_token",)
SCAN_ROOTS = ("utils", "lig")
SKIP_FILES = {"release_scope.py", "check_no_otb_in_release.py", "strip_otb_from_release.py"}
SKIP_PATH_PARTS = {"scripts/reproduce"}


def main() -> int:
    hits: list[str] = []
    for scan_root in SCAN_ROOTS:
        base = ROOT / scan_root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix != ".py":
                continue
            if path.name in SKIP_FILES:
                continue
            if SKIP_PATH_PARTS & set(path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in FORBIDDEN:
                if token in text:
                    hits.append(f"{path.relative_to(ROOT)}: {token}")
    if hits:
        print("Unsupported baseline references remain in implementation tree:", file=sys.stderr)
        for h in hits:
            print(f"  - {h}", file=sys.stderr)
        return 1
    print("OK: no unsupported baseline references under utils/ or lig/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
