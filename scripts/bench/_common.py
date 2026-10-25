"""Shared bits for the bench scripts: a checklist that prints PASS/FAIL, logs to
var/bench/, and exits nonzero if anything failed. Run from the repo root as
`python -m scripts.bench.<name>`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

OUT = Path("var/bench")


class Report:
    def __init__(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.rows: list[dict] = []
        self.data: dict = {}
        print(f"\n=== {name} ===")

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.rows.append({"check": label, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        return bool(ok)

    def note(self, text: str) -> None:
        self.rows.append({"note": text})
        print(f"  ...    {text}")

    def ask(self, question: str) -> bool:
        """Human-in-the-loop check: the operator looks and answers y/n."""
        ans = input(f"  ?? {question} [y/n] ").strip().lower()
        return self.check(question, ans.startswith("y"))

    def finish(self) -> int:
        bad = [r for r in self.rows if r.get("ok") is False]
        path = OUT / f"{self.name}-{self.stamp}.json"
        path.write_text(json.dumps({"rows": self.rows, "data": self.data}, indent=2, default=str))
        print(f"\n  {len(bad)} failed, log: {path}")
        return 1 if bad else 0


def main(fn) -> None:
    try:
        sys.exit(fn())
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
