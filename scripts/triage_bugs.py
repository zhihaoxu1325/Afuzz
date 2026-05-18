#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/latest/summary.json")
    if path.exists():
        summary = json.loads(path.read_text())
        failures = summary.get("failures", [])
    else:
        result_files = sorted(path.parent.glob("cases/case_*/result.json"))
        if not result_files:
            print(f"[asfuzz] summary not found: {path}", file=sys.stderr)
            summaries = sorted(Path("runs").glob("*/summary.json"))
            if summaries:
                print("[asfuzz] available summaries:", file=sys.stderr)
                for summary_path in summaries:
                    print(f"  {summary_path}", file=sys.stderr)
            else:
                print("[asfuzz] no summaries under runs/. Run a campaign first.", file=sys.stderr)
            return 2
        failures = []
        for result_file in result_files:
            try:
                failures.extend(json.loads(result_file.read_text()).get("failures", []))
            except Exception as exc:
                print(f"[asfuzz] skip unreadable {result_file}: {exc}", file=sys.stderr)
        print(f"[asfuzz] summary not found; triaged {len(result_files)} case result files under {path.parent}", file=sys.stderr)

    print(f"[asfuzz] failures: {len(failures)}")
    by_key = Counter((f.get("backend"), f.get("mr"), f.get("status"), f.get("detail", {}).get("reason")) for f in failures)
    for key, count in by_key.most_common():
        print(count, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
