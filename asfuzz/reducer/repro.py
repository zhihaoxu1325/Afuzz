from __future__ import annotations

import re
from pathlib import Path


def write_repro(case_dir: str | Path, backend: str, mr: str, variant: str) -> Path:
    case_dir = Path(case_dir)
    path = case_dir / "repro.py"
    content = f"""#!/usr/bin/env python3
from pathlib import Path

from asfuzz.config import load_config
from asfuzz.runner.pipeline import replay_case

ROOT = Path(__file__).resolve().parents[4]
cfg = load_config(ROOT / "configs" / "smoke.yaml")
cfg.backends = [{backend!r}]
cfg.mrs = [{mr!r}]
result = replay_case(cfg, Path(__file__).with_name("spec.json"))
print(result)
raise SystemExit(0 if result["status"] == "ok" else 1)

# Original failing variant: {variant}
"""
    return _write_repro_file(path, content, backend, mr, variant)


def _write_repro_file(path: Path, content: str, backend: str, mr: str, variant: str) -> Path:
    candidates = [path, path.with_name(f"repro_{_safe(backend)}_{_safe(mr)}_{_safe(variant)}.py")]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.write_text(content)
            candidate.chmod(0o755)
            return candidate
        except PermissionError as exc:
            last_error = exc
            try:
                if candidate.exists():
                    candidate.chmod(0o666)
                    candidate.unlink()
                    candidate.write_text(content)
                    candidate.chmod(0o755)
                    return candidate
            except Exception as retry_exc:
                last_error = retry_exc
                continue
    print(f"[asfuzz] warning: failed to write repro {path}: {last_error}", flush=True)
    return path


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:64]
