"""Turn one observation into something readable: terminal text or notebook HTML."""
from __future__ import annotations

import html
import os
import sys

BLOCKS = " ▏▎▍▌▋▊▉█"


def show_token(text: str) -> str:
    """Make whitespace visible: spaces as ␣, newlines as ↵."""
    return text.replace("\n", "↵").replace(" ", "␣") or "∅"


def bar(fraction: float, width: int = 20) -> str:
    fraction = max(0.0, min(1.0, fraction))
    units = fraction * width
    full = int(units)
    part = int(round((units - full) * 8))
    if part == 8:
        full, part = full + 1, 0
    out = "█" * full + (BLOCKS[part] if part and full < width else "")
    return out.ljust(width)


def describe(info: dict | None) -> str:
    """One-line description of a feature: Neuronpedia label if known, else the words it pushes toward."""
    if not info:
        return ""
    if info.get("label"):
        return info["label"]
    words = info.get("promotes") or []
    return ("pushes toward: " + ", ".join(words)) if words else ""


def _layers(record: dict) -> list[str]:
    feats = record.get("features") or {}
    if isinstance(feats, list):  # version 1 trace
        return []
    return sorted(feats, key=int)


class Style:
    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
        self.on = enabled

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t): return self._c("1", t)
    def dim(self, t): return self._c("2", t)
    def accent(self, t): return self._c("36", t)
    def good(self, t): return self._c("32", t)
    def warm(self, t): return self._c("33", t)


def format_step(record: dict, text_so_far: str, feature_info: dict, layer_every: int = 1,
                style: Style | None = None, max_features: int = 4, show_layers: bool = True) -> str:
    s = style or Style()
    lines = [s.dim("─" * 72), s.dim("So far: ") + text_so_far[-400:] + s.accent("▌"), ""]

    alts = "  ".join(
        f"{show_token(a['token'])} {a['prob']:.2f}" for a in record["alternatives"] if a["token_id"] != record["token_id"]
    )
    lines.append(f"{s.bold('Chose')} {s.accent(show_token(record['token']))}  p={record['prob']:.2f}"
                 + (s.dim("  (end of reply)") if record.get("stop") else ""))
    if alts:
        lines.append(s.dim(f"  runners-up: {alts}"))

    settled = record.get("settled_layer")
    n = len(record["lens"])
    if show_layers:
        lines.append("")
        lines.append(s.bold("How the guess formed") + s.dim(f"  (top guess per layer · bar = chance of {show_token(record['token'])})"))
        for row in record["lens"]:
            layer = row["layer"]
            if layer % layer_every and layer != n - 1 and layer != settled:
                continue
            guess = show_token(row["top_token"])[:14].ljust(14)
            guess = s.good(guess) if row["top_token"] == record["token"] else guess
            mark = s.accent(" ← settled") if layer == settled else ""
            lines.append(f"  L{layer:02d} {guess} {bar(row['chosen_prob'])} {row['chosen_prob']:.2f}{mark}")
    if settled is not None:
        lines.append(s.dim(f"  Settled at layer {settled} of {n - 1}"))

    for layer in _layers(record):
        rows = record["features"][layer][:max_features]
        if not rows:
            continue
        count = (record.get("active_count") or {}).get(layer, len(rows))
        lines.append("")
        lines.append(s.bold(f"Concepts at layer {layer}") + s.dim(f"  ({count} firing)"))
        top = rows[0]["activation"] or 1.0
        for f in rows:
            info = feature_info.get(f"{layer}:{f['index']}")
            text = describe(info)
            text = text if (info and info.get("label")) else s.dim(text)
            lines.append(f"  #{f['index']:<7}{s.warm(bar(f['activation'] / top, 8))} {f['activation']:7.1f}  {text}")
    return "\n".join(lines)


def step_html(record: dict, text_so_far: str, feature_info: dict, n_steps: int | None = None,
              max_features: int = 5) -> str:
    """Compact, self-styled HTML for live display in Jupyter/Colab/Kaggle (works in light or dark notebooks)."""
    e = html.escape
    chosen = record["token"]
    rows = []
    for row in record["lens"]:
        pct = row["chosen_prob"] * 100
        hit = row["top_token"] == chosen
        settled = row["layer"] == record.get("settled_layer")
        rows.append(
            f'<div style="display:grid;grid-template-columns:3em 8em 1fr 3em;gap:6px;align-items:center;'
            f'font:11.5px ui-monospace,Menlo,monospace;{"background:#e6f4f1;" if settled else ""}">'
            f'<span style="color:#5b6b75">L{row["layer"]:02d}</span>'
            f'<span style="color:{"#0b6b5e" if hit else "#2b3a44"};overflow:hidden;white-space:nowrap">{e(show_token(row["top_token"]))}</span>'
            f'<span style="background:#e3e9ec;height:8px;border-radius:2px"><span style="display:block;height:8px;'
            f'width:{pct:.1f}%;background:#0f8a78;border-radius:2px"></span></span>'
            f'<span style="text-align:right;color:#2b3a44">{row["chosen_prob"]:.2f}</span></div>'
        )
    alts = "".join(
        f'<span style="margin-right:12px">{e(show_token(a["token"]))} <b>{a["prob"]:.2f}</b></span>'
        for a in record["alternatives"]
    )
    blocks = []
    for layer in _layers(record):
        feats = record["features"][layer][:max_features]
        if not feats:
            continue
        top = feats[0]["activation"] or 1.0
        items = []
        for f in feats:
            info = feature_info.get(f"{layer}:{f['index']}") or {}
            text = describe(info)
            text_html = e(text) if info.get("label") else f'<span style="color:#6b7b85">{e(text)}</span>'
            ref = f'<a href="{e(info["url"])}" target="_blank" style="color:#9a4f1d">#{f["index"]}</a>' if info.get("url") else f"#{f['index']}"
            items.append(
                f'<div style="display:grid;grid-template-columns:5.5em 4em 1fr;gap:8px;align-items:center;font-size:12.5px">'
                f'<span style="font-family:ui-monospace,Menlo,monospace">{ref}</span>'
                f'<span style="background:#e3e9ec;height:8px;border-radius:2px"><span style="display:block;height:8px;'
                f'width:{100 * f["activation"] / top:.0f}%;background:#c76b2a;border-radius:2px"></span></span>'
                f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{text_html}</span></div>'
            )
        count = (record.get("active_count") or {}).get(layer, len(feats))
        blocks.append(
            f'<div style="min-width:0"><div style="font-weight:600;margin:0 0 4px;font-size:13px">Layer {layer}'
            f' <span style="font-weight:400;color:#5b6b75">({count} firing)</span></div>{"".join(items)}</div>'
        )
    settled = record.get("settled_layer")
    progress = f"word {record.get('index', 0) + 1}" + (f" of up to {n_steps}" if n_steps else "")
    return (
        '<div style="background:#ffffff;color:#1d2a33;border:1px solid #d6dee3;border-radius:8px;padding:14px 16px;'
        'font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px">'
        f'<div style="color:#5b6b75;font-size:12px">Watching · {progress}</div>'
        f'<div style="font-size:15px;margin:6px 0 10px;white-space:pre-wrap;max-height:9em;overflow:auto">{e(text_so_far[-1500:])}'
        f'<span style="background:#cfeee8;color:#0b6b5e;padding:0 2px;border-radius:2px">{e(chosen)}</span></div>'
        f'<div style="font-size:13px;margin-bottom:10px"><b>Chose</b> {e(show_token(chosen))} '
        f'(p={record["prob"]:.2f}) &nbsp; <span style="color:#5b6b75">{alts}</span></div>'
        '<div style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.3fr);gap:18px">'
        f'<div><div style="font-weight:600;margin-bottom:4px;font-size:13px">How the guess formed'
        f'<span style="font-weight:400;color:#5b6b75">{f" · settled at layer {settled}" if settled is not None else ""}</span></div>'
        f'<div style="display:flex;flex-direction:column;gap:1px">{"".join(rows)}</div></div>'
        f'<div style="display:flex;flex-direction:column;gap:12px;min-width:0">'
        f'<div style="font-weight:600;font-size:13px">Concepts active now</div>{"".join(blocks)}</div>'
        '</div></div>'
    )
