#!/usr/bin/env python3
"""Watch Gemma-2 2B write, one word at a time, in your terminal.

Examples:
  python watch.py --prompt "The capital of France is"
  python watch.py --prompt "def fibonacci(n):" --tokens 30 --step
  python watch.py --prompt "The Marine Corps was founded in" --html run.html --json run.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from model_watch import ModelWatcher, WatchConfig, export_html, format_step, save_json
from model_watch.render import Style


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Watch a language model choose each word.")
    p.add_argument("--prompt", "-p", required=True, help="Text for the model to continue")
    p.add_argument("--tokens", "-n", type=int, default=20, help="How many words (tokens) to generate (default 20)")
    p.add_argument("--step", action="store_true", help="Pause after each token until you press Enter")
    p.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between tokens")
    p.add_argument("--layer-every", type=int, default=1, help="Show every Nth layer in the guess view (default: all)")
    p.add_argument("--json", help="Save the full trace as JSON")
    p.add_argument("--html", help="Save a replay page you can open in a browser")
    p.add_argument("--no-labels", action="store_true", help="Skip Neuronpedia label lookups (offline mode)")
    p.add_argument("--model", default=WatchConfig.model_id, help="Hugging Face model id")
    p.add_argument("--sae-layer", type=int, default=WatchConfig.sae_layer)
    p.add_argument("--sae-width", default=WatchConfig.sae_width)
    p.add_argument("--sae-l0", type=int, default=WatchConfig.sae_l0)
    p.add_argument("--top-features", type=int, default=WatchConfig.top_k_features)
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--no-color", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = WatchConfig(
        model_id=args.model,
        sae_layer=args.sae_layer,
        sae_width=args.sae_width,
        sae_l0=args.sae_l0,
        top_k_features=args.top_features,
        labels=not args.no_labels,
        device=args.device,
        dtype=args.dtype,
    )
    watcher = ModelWatcher.load(cfg, hf_token=os.environ.get("HF_TOKEN"))
    style = Style(False if args.no_color else None)

    text = args.prompt

    def on_step(record, _steps):
        nonlocal text
        print(format_step(record, text, watcher.sae_layer, args.layer_every, style), flush=True)
        text += record["token"]
        if args.step and sys.stdin.isatty():
            input(style.dim("  Enter for next word, Ctrl+C to stop "))
        elif args.delay:
            time.sleep(args.delay)

    try:
        trace = watcher.trace(args.prompt, args.tokens, on_step=on_step)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130

    print(style.dim("─" * 64))
    print(style.bold("Final text: ") + text)
    if args.json:
        print(f"Saved trace: {save_json(trace, args.json)}")
    if args.html:
        print(f"Saved replay page: {export_html(trace, args.html)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
