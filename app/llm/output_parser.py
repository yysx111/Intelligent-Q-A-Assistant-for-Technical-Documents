from __future__ import annotations

import json
import re
from typing import Any


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text or ""
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError("未找到JSON对象")
    val = json.loads(m.group(0))
    if not isinstance(val, dict):
        raise ValueError("输出不是JSON对象")
    return val
