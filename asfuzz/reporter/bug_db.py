from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from asfuzz.spec.opspec import OpSpec


class BugDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.lock = threading.Lock()
        self._init()

    def _init(self) -> None:
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bugs(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                  backend TEXT,
                  mr TEXT,
                  op_kind TEXT,
                  dtype TEXT,
                  signature TEXT,
                  status TEXT,
                  stderr_hash TEXT,
                  repro_path TEXT,
                  upstream_issue TEXT,
                  detail_json TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS iterations(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                  spec_hash TEXT,
                  backend TEXT,
                  mr TEXT,
                  result TEXT,
                  elapsed_ms REAL
                )
                """
            )
            self.conn.commit()

    def record_iteration(self, spec: OpSpec, backend: str, mr: str, result: str, elapsed_ms: float) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO iterations(spec_hash, backend, mr, result, elapsed_ms) VALUES (?, ?, ?, ?, ?)",
                (spec.signature(), backend, mr, result, elapsed_ms),
            )
            self.conn.commit()

    def record_bug(self, spec: OpSpec, backend: str, mr: str, status: str, repro_path: str, detail: dict) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO bugs(backend, mr, op_kind, dtype, signature, status, stderr_hash, repro_path, upstream_issue, detail_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    backend,
                    mr,
                    spec.op_kind,
                    spec.dtype(),
                    spec.signature(),
                    status,
                    detail.get("stderr_hash", ""),
                    repro_path,
                    "",
                    json.dumps(detail, sort_keys=True),
                ),
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()
