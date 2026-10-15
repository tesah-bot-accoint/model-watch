"""The trace workflow as a small explicit graph.

A run moves through named nodes (encode -> generate -> assemble). Each node
reads and writes one typed RunState; the edges are plain functions of that
state. After every node, and after every generated word, the state is
checkpointed to a JSONL file, so an interrupted run can be saved as a partial
trace or resumed. Nothing here touches the network or the model.

Checkpoints hold the conversation, so they are written owner-only (0600),
deleted when a run finishes, and pruned after `keep_days` if a run never does.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

# Status values, in order: new, encoded, generating, generated, assembled; or failed.


@dataclass
class RunState:
    run_id: str
    created: str
    prompt: Union[str, list]  # what the caller passed; private
    max_new_tokens: int
    stop_at_eos: bool = True
    use_cache: bool = True
    status: str = "new"
    prompt_ids: list = field(default_factory=list)  # the exact ids the model read
    exact_history: Optional[bool] = None  # chat continued from the previous turn's exact ids?
    model: Optional[str] = None  # which model produced the steps; checked on resume
    steps: list = field(default_factory=list)
    error: Optional[str] = None
    trace: Optional[dict] = None  # set by the assemble node; never checkpointed

    @classmethod
    def new(cls, prompt, max_new_tokens: int, stop_at_eos: bool = True, use_cache: bool = True,
            model: Optional[str] = None) -> "RunState":
        return cls(run_id=uuid.uuid4().hex[:12], created=_dt.datetime.now().isoformat(timespec="seconds"),
                   prompt=prompt, max_new_tokens=max_new_tokens, stop_at_eos=stop_at_eos, use_cache=use_cache,
                   model=model)

    def header(self) -> dict:
        data = asdict(self)
        data.pop("steps")
        data.pop("trace")
        return data


@dataclass(frozen=True)
class Node:
    run: Callable[[RunState], None]
    next: Callable[[RunState], Optional[str]]  # name of the next node, or None to stop


def describe_error(err: BaseException) -> str:
    text = str(err)
    return f"{type(err).__name__}: {text}" if text else type(err).__name__


def run_graph(nodes: dict, start: str, state: RunState,
              on_checkpoint: Optional[Callable[[RunState], None]] = None) -> RunState:
    """Run nodes from `start` until an edge returns None. On any error, mark the state failed and re-raise."""
    name: Optional[str] = start
    while name is not None:
        node = nodes[name]
        try:
            node.run(state)
        except BaseException as err:  # includes KeyboardInterrupt: the partial state must still be recorded
            state.status = "failed"
            state.error = describe_error(err)
            if on_checkpoint:
                try:
                    on_checkpoint(state)
                except Exception:
                    pass  # never hide the original error behind a checkpoint problem
            raise
        if on_checkpoint:
            on_checkpoint(state)
        name = node.next(state)
    return state


class Checkpointer:
    """Appends run events to <directory>/<run_id>.jsonl, readable by the owner only."""

    def __init__(self, directory: Union[str, Path], keep_days: float = 7.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.dir, 0o700)
        except OSError:
            pass
        self.keep_days = keep_days
        self.failed = False  # a write failed; this file can't be trusted for resume
        self.prune()

    def path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.jsonl"

    def _append(self, run_id: str, event: dict) -> None:
        """Best effort: a checkpoint problem (full disk, removed folder) warns once and never stops the run."""
        if self.failed:
            return
        line = json.dumps(event, ensure_ascii=False) + "\n"
        try:
            fd = os.open(self.path(run_id), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as err:
            self.failed = True
            warnings.warn(f"Checkpointing to {self.dir} stopped ({err}); the reply continues without it.",
                          RuntimeWarning, stacklevel=3)

    def start(self, state: RunState) -> None:
        self.failed = False
        self._append(state.run_id, {"event": "start", **state.header()})

    def step(self, state: RunState, record: dict) -> None:
        self._append(state.run_id, {"event": "step", "step": record})

    def status(self, state: RunState) -> None:
        self._append(state.run_id, {"event": "status", "status": state.status, "error": state.error,
                                    "max_new_tokens": state.max_new_tokens})

    def finish(self, state: RunState) -> None:
        try:
            self.path(state.run_id).unlink()
        except OSError:
            pass

    def prune(self) -> None:
        cutoff = time.time() - self.keep_days * 86400
        for path in self.dir.glob("*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass


def load_checkpoint(path: Union[str, Path]) -> RunState:
    """Rebuild a RunState from a checkpoint file written by Checkpointer."""
    state: Optional[RunState] = None
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                break  # a torn last line from a crash; keep what came before
            kind = event.get("event")
            if kind == "start":
                fields = {k: v for k, v in event.items() if k in RunState.__dataclass_fields__}
                state = RunState(**fields)
            elif state is None:
                raise ValueError(f"{path} does not start with a run header (line {n})")
            elif kind == "step":
                state.steps.append(event["step"])
            elif kind == "status":
                state.status, state.error = event.get("status", state.status), event.get("error")
                state.max_new_tokens = event.get("max_new_tokens", state.max_new_tokens)
    if state is None:
        raise ValueError(f"{path} is not a Model Watch checkpoint")
    return state


__all__ = ["RunState", "Node", "run_graph", "Checkpointer", "load_checkpoint", "describe_error"]
