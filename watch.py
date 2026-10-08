#!/usr/bin/env python3
"""Watch a Gemma model write, one word at a time, in your terminal.

Examples:
  python watch.py --prompt "Is AI a black box?"                       # Gemma 3 4B chat (default)
  python watch.py --preset gemma-3-1b-it --prompt "Explain superposition" --device cpu
  python watch.py --prompt "Why is the sky blue?" --step                # Enter for each word
  python watch.py --prompt "..." --compact --html run.html --json run.json
  python watch.py --interactive                                         # keep chatting, every reply traced
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from model_watch import PRESETS, ModelWatcher, WatchConfig, export_html, format_step, save_json
from model_watch.core import DEFAULT_PRESET
from model_watch.render import Style


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Watch a language model choose each word.")
    p.add_argument("--prompt", "-p", help="Your message (chat models) or text to continue (base models)")
    p.add_argument("--interactive", "-i", action="store_true", help="Keep chatting; every reply is traced")
    p.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS), help=f"Model and dictionaries (default {DEFAULT_PRESET})")
    p.add_argument("--tokens", "-n", type=int, default=200, help="Maximum words (tokens) per reply (default 200)")
    p.add_argument("--step", action="store_true", help="Pause after each word until you press Enter")
    p.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between words")
    p.add_argument("--compact", action="store_true", help="Hide the per-layer table; show the choice and concepts only")
    p.add_argument("--quiet", action="store_true", help="Show only the text as it is written; inspect the replay page after")
    p.add_argument("--layer-every", type=int, default=2, help="In the per-layer table, show every Nth layer (default 2)")
    p.add_argument("--features", type=int, default=4, help="Concepts shown per layer in the terminal (default 4)")
    p.add_argument("--json", help="Save the trace as JSON (in interactive mode, one file per turn)")
    p.add_argument("--html", help="Save a replay page (in interactive mode, one file per turn)")
    p.add_argument("--no-labels", action="store_true", help="Skip Neuronpedia lookups; keep the offline 'pushes toward' labels")
    p.add_argument("--label-lookups", type=int, default=400, help="Max Neuronpedia lookups after each reply, done only when saving with --html or --json (default 400)")
    p.add_argument("--sae-layers", help="Comma-separated layers to read concepts from (default: the preset's four)")
    p.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    p.add_argument("--device-map", choices=["auto"], help="Split a large model across all GPUs (for example Kaggle's 2x T4)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args(argv)
    if not args.prompt and not args.interactive:
        p.error("give --prompt, or use --interactive")
    return args


def numbered(path: str | None, turn: int, interactive: bool) -> str | None:
    if not path or not interactive:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.stem}-turn{turn}{p.suffix}"))


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = WatchConfig(
        preset=args.preset,
        sae_layers=tuple(int(x) for x in args.sae_layers.split(",")) if args.sae_layers else None,
        labels=not args.no_labels,
        device=args.device,
        device_map=args.device_map,
        dtype=args.dtype,
    )
    watcher = ModelWatcher.load(cfg, hf_token=os.environ.get("HF_TOKEN"))
    style = Style(False if args.no_color else None)
    messages: list[dict] = []
    turn = 0

    while True:
        if args.interactive:
            try:
                user = input(style.bold("\nYou: ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                return 0
        else:
            user = args.prompt
        turn += 1
        prompt = (messages + [{"role": "user", "content": user}]) if watcher.chat else user
        reply = ""

        def on_step(record, _steps):
            nonlocal reply
            if args.quiet:
                if not record.get("stop"):
                    sys.stdout.write(record["token"])
                    sys.stdout.flush()
            else:
                print(format_step(record, reply, watcher.feature_info, args.layer_every, style,
                                  args.features, show_layers=not args.compact), flush=True)
            if not record.get("stop"):
                reply += record["token"]
            if args.step and sys.stdin.isatty():
                input(style.dim("  Enter for next word, Ctrl+C to stop "))
            elif args.delay:
                time.sleep(args.delay)

        if args.quiet:
            print(style.bold("Model: "), end="")
        try:
            trace = watcher.trace(prompt, args.tokens, on_step=on_step)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 130

        print("\n" + style.dim("─" * 72))
        if not args.quiet:
            print(style.bold("Reply: ") + trace["reply"])
        if watcher.chat:
            messages = prompt + [{"role": "assistant", "content": trace["reply"]}]
        if not args.no_labels and (args.html or args.json):
            watcher.add_labels(trace, max_lookups=args.label_lookups)
        if args.json:
            print(f"Saved trace: {save_json(trace, numbered(args.json, turn, args.interactive))}")
        if args.html:
            print(f"Saved replay page: {export_html(trace, numbered(args.html, turn, args.interactive))}")
        if not args.interactive:
            return 0


if __name__ == "__main__":
    sys.exit(main())
