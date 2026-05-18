from __future__ import annotations

import argparse
import json
from pathlib import Path

from asfuzz.config import load_config
from asfuzz.reporter.html_report import write_html_report
from asfuzz.runner.pipeline import replay_case, run_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asfuzz")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--config", default="configs/default.yaml")
    run_p.add_argument("--resume", action="store_true", help="resume from existing case directories instead of deleting out_dir")

    replay_p = sub.add_parser("replay")
    replay_p.add_argument("--config", default="configs/default.yaml")
    replay_p.add_argument("--case", required=True)

    report_p = sub.add_parser("report")
    report_p.add_argument("--summary", default="runs/latest/summary.json")
    report_p.add_argument("--out", default="runs/latest/report.html")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        cfg = load_config(args.config)
        summary = run_campaign(cfg, resume=args.resume)
        print(json.dumps({"summary": str(Path(cfg.out_dir) / "summary.json"), "failures": len(summary["failures"])}, indent=2))
        return 0
    if args.cmd == "replay":
        cfg = load_config(args.config)
        result = replay_case(cfg, args.case)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ok" else 1
    if args.cmd == "report":
        summary = json.loads(Path(args.summary).read_text())
        write_html_report(summary, args.out)
        print(args.out)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
