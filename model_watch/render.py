"""Turn one observation into something readable: terminal text or notebook HTML."""
from __future__ import annotations

import html
import os
import sys

BLOCKS = " ▏▎▍▌▋▊▉█"


def show_token(text: str) -> str:
    """Make whitespace visible: leading/trailing spaces as ␣, newlines as ↵."""
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


def format_step(record: dict, text_so_far: str, sae_layer: int | None, layer_every: int = 1, style: Style | None = None) -> str:
    s = style or Style()
    lines = []
    lines.append(s.dim("─" * 64))
    lines.append(s.dim("Text so far: ") + text_so_far + s.accent("▌"))
    lines.append("")

    alts = "  ".join(
        f"{show_token(a['token'])} {a['prob']:.2f}" for a in record["alternatives"] if a["token_id"] != record["token_id"]
    )
    lines.append(f"{s.bold('Chose')} {s.accent(show_token(record['token']))}  p={record['prob']:.2f}")
    if alts:
        lines.append(s.dim(f"  runners-up: {alts}"))
    lines.append("")

    chosen = show_token(record["token"])
    lines.append(s.bold("How the guess formed") + s.dim(f"  (top guess at each layer · bar = chance of {chosen})"))
    settled = record.get("settled_layer")
    n = len(record["lens"])
    for row in record["lens"]:
        layer = row["layer"]
        if layer % layer_every and layer != n - 1 and layer != settled:
            continue
        guess = show_token(row["top_token"])[:14].ljust(14)
        is_chosen = row["top_token"] == record["token"]
        guess = s.good(guess) if is_chosen else guess
        mark = s.accent(" ← settled") if layer == settled else ""
        lines.append(f"  L{layer:02d} {guess} {bar(row['chosen_prob'])} {row['chosen_prob']:.2f}{mark}")
    if settled is not None:
        lines.append(s.dim(f"  Settled at layer {settled} of {n - 1}"))
    lines.append("")

    if record.get("features"):
        lines.append(s.bold(f"Active concepts at layer {sae_layer}") + s.dim(f"  ({record['active_count']} features firing)"))
        top = record["features"][0]["activation"] or 1.0
        for f in record["features"]:
            label = f["label"] or s.dim("(no label)")
            lines.append(f"  #{f['index']:<6} {bar(f['activation'] / top, 10)} {f['activation']:7.2f}  {label}")
    return "\n".join(lines)


def step_html(record: dict, text_so_far: str, sae_layer: int | None, n_steps: int | None = None) -> str:
    """Compact, self-styled HTML for live display in Jupyter/Colab (works in light or dark notebooks)."""
    e = html.escape
    chosen = record["token"]
    rows = []
    for row in record["lens"]:
        pct = row["chosen_prob"] * 100
        hit = row["top_token"] == chosen
        settled = row["layer"] == record.get("settled_layer")
        rows.append(
            f'<div style="display:grid;grid-template-columns:3em 9em 1fr 3.5em;gap:6px;align-items:center;'
            f'font:12px ui-monospace,Menlo,monospace;{"background:#e6f4f1;" if settled else ""}">'
            f'<span style="color:#5b6b75">L{row["layer"]:02d}</span>'
            f'<span style="color:{"#0b6b5e" if hit else "#2b3a44"};overflow:hidden;white-space:nowrap">{e(show_token(row["top_token"]))}</span>'
            f'<span style="background:#e3e9ec;height:9px;border-radius:2px"><span style="display:block;height:9px;'
            f'width:{pct:.1f}%;background:#0f8a78;border-radius:2px"></span></span>'
            f'<span style="text-align:right;color:#2b3a44">{row["chosen_prob"]:.2f}</span></div>'
        )
    alts = "".join(
        f'<span style="margin-right:12px">{e(show_token(a["token"]))} <b>{a["prob"]:.2f}</b></span>'
        for a in record["alternatives"]
    )
    feats = ""
    if record.get("features"):
        top = record["features"][0]["activation"] or 1.0
        items = []
        for f in record["features"]:
            label = e(f["label"]) if f["label"] else '<i style="color:#7a8a94">no label</i>'
            link = f'<a href="{e(f["url"])}" target="_blank" style="color:#0f6f8a">#{f["index"]}</a>' if f.get("url") else f"#{f['index']}"
            items.append(
                f'<div style="display:grid;grid-template-columns:5.5em 6em 1fr;gap:8px;align-items:center;font-size:13px">'
                f'<span style="font-family:ui-monospace,Menlo,monospace">{link}</span>'
                f'<span style="background:#e3e9ec;height:9px;border-radius:2px"><span style="display:block;height:9px;'
                f'width:{100 * f["activation"] / top:.0f}%;background:#c76b2a;border-radius:2px"></span></span>'
                f'<span>{label}</span></div>'
            )
        feats = (
            f'<div style="margin-top:14px"><div style="font-weight:600;margin-bottom:6px">Active concepts at layer {sae_layer}'
            f' <span style="font-weight:400;color:#5b6b75">({record["active_count"]} firing)</span></div>{"".join(items)}</div>'
        )
    settled = record.get("settled_layer")
    progress = f" · token {record.get('index', 0) + 1}" + (f" of {n_steps}" if n_steps else "")
    return (
        '<div style="background:#ffffff;color:#1d2a33;border:1px solid #d6dee3;border-radius:8px;padding:14px 16px;'
        'font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:760px">'
        f'<div style="color:#5b6b75;font-size:12px">Watching{progress}</div>'
        f'<div style="font-size:15px;margin:6px 0 10px;white-space:pre-wrap">{e(text_so_far)}'
        f'<span style="background:#cfeee8;color:#0b6b5e;padding:0 2px;border-radius:2px">{e(chosen)}</span></div>'
        f'<div style="font-size:13px;margin-bottom:10px"><b>Chose</b> {e(show_token(chosen))} '
        f'(p={record["prob"]:.2f}) &nbsp; <span style="color:#5b6b75">{alts}</span></div>'
        f'<div style="font-weight:600;margin-bottom:4px">How the guess formed'
        f'<span style="font-weight:400;color:#5b6b75"> · bar = chance of {e(show_token(chosen))}'
        f'{f" · settled at layer {settled}" if settled is not None else ""}</span></div>'
        f'<div style="display:flex;flex-direction:column;gap:2px">{"".join(rows)}</div>{feats}</div>'
    )
