"""Save traces as JSON, or as a self-contained replay page."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer.html")
PLACEHOLDER = "/*__TRACE_JSON__*/null"


def save_json(trace: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def trace_to_html(trace: dict) -> str:
    """Return the replay viewer with this trace embedded (opens offline, no server needed)."""
    page = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise RuntimeError("viewer.html is missing the trace placeholder")
    data = json.dumps(trace, ensure_ascii=False).replace("</", "<\\/").replace("<!--", "<\\!--")
    return page.replace(PLACEHOLDER, data, 1)


def export_html(trace: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(trace_to_html(trace), encoding="utf-8")
    return path
