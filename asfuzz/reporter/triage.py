from __future__ import annotations

import hashlib


def stderr_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

