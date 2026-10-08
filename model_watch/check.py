"""Check that a loaded model, its concept dictionaries, the labels and the live viewer all work.

Run it once after loading a real model:

    python watch.py --preset qwen3-1.7b --check
    # or in a notebook:  from model_watch.check import run_checks; run_checks(watcher)

Each line says PASS, WARN or FAIL in plain words. Paste the report back to
whoever is helping you if anything fails. It only observes the model.
"""
from __future__ import annotations

import json
import time
import urllib.request

import torch

SAMPLE = ("The Eiffel Tower is in Paris, the capital of France. It was finished in 1889 and is about "
          "330 metres tall. Every year, millions of people climb its stairs or take the lifts to the top.")


class Report:
    def __init__(self, log=print):
        self.rows: list[tuple[str, str, str]] = []
        self.log = log

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        self.log(f"{status:<4}  {name}" + (f": {detail}" if detail else ""))

    def guard(self, name: str, fn) -> None:
        """Run one check; an exception becomes a FAIL line instead of stopping the rest."""
        try:
            fn()
        except Exception as err:
            self.add("FAIL", name, f"{type(err).__name__}: {err}")

    @property
    def ok(self) -> bool:
        return all(status != "FAIL" for status, _, _ in self.rows)


def _hidden_states(watcher, ids):
    with torch.no_grad():
        out = watcher.model(input_ids=ids.to(watcher.input_device), output_hidden_states=True)
    return out.logits[0, -1], out.hidden_states


def run_checks(watcher, labels: bool = True, live: bool = True, log=print) -> Report:
    """Run every check and print a report. Returns the Report (report.ok is False if anything failed)."""
    r = Report(log)
    meta = watcher.meta
    log(f"Model Watch self-check · {meta.get('model', '?')} · {meta.get('device', '?')} · {meta.get('dtype', '?')}")
    ids = watcher.tok(SAMPLE, return_tensors="pt").input_ids
    state = {}

    def layers():
        logits, hs = _hidden_states(watcher, ids)
        _, resid, _ = watcher._forward(ids)
        state["hs"] = hs
        worst = 0.0
        for layer in range(watcher.n_layers - 1):  # HF stores the last one after the final norm
            want = hs[layer + 1][0, -1].to(resid.device).float()
            worst = max(worst, float((resid[layer].float() - want).norm() / want.norm().clamp_min(1e-6)))
        status = "PASS" if worst < 0.02 else "FAIL"
        r.add(status, "Reading every layer", f"{watcher.n_layers} layers captured; largest difference from the model's own record {worst:.1%}")

    def lens():
        record, chosen = watcher.step(ids)
        logits, _ = _hidden_states(watcher, ids)
        probs = torch.softmax(logits.float(), -1)
        same = chosen == int(probs.argmax())
        gap = abs(record["lens"][-1]["chosen_prob"] - float(probs[chosen]))
        status = "PASS" if same and gap < 0.02 else "FAIL"
        r.add(status, "Last layer's guess matches the model's real choice",
              f"next word {record['token']!r} at {record['prob']:.0%}; difference {gap:.3f}")

    r.guard("Reading every layer", layers)
    r.guard("Last layer's guess matches the model's real choice", lens)

    for layer in watcher.sae_layers:
        def dictionary(layer=layer):
            sae = watcher.saes[layer]
            x = state["hs"][layer + 1][0, 1:].to(sae.W_enc.device).float()  # skip the first token: it is unusually large
            with torch.no_grad():
                acts = sae.encode(x)
                recon = sae.decode(acts)
            explained = 1 - float(((x - recon) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
            firing = float((acts > 0).sum(-1).float().mean())
            status = "PASS" if explained >= 0.6 and 1 <= firing <= 2000 else ("WARN" if explained >= 0.3 else "FAIL")
            share = f"explains {explained:.0%} of the layer's activity" if explained > 0 else \
                "explains none of the layer's activity (wrong file or wrong layer?)"
            r.add(status, f"Concept dictionary at layer {layer}",
                  f"{share}; about {firing:.0f} concepts firing per word "
                  f"({sae.n_features} in the dictionary)")
        r.guard(f"Concept dictionary at layer {layer}", dictionary)
    if not watcher.sae_layers:
        r.add("PASS", "Concept dictionaries", "this preset has none (choice and layer guesses only)")

    def stops():
        found = sorted(watcher.stop_ids)
        if not watcher.chat:
            r.add("PASS" if found else "FAIL", "End-of-reply markers", f"{len(found)} known: {found}")
            return
        enc = watcher.tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
                                              tokenize=True, return_dict=True)
        template_ids = torch.as_tensor(enc["input_ids"]).reshape(-1).tolist()
        hit = [i for i in found if i in template_ids]
        r.add("PASS" if hit else "FAIL", "End-of-reply markers",
              f"the chat format ends a reply with {[watcher._decode(i) for i in hit] or 'nothing Model Watch recognises'}")
    r.guard("End-of-reply markers", stops)

    if watcher.thinking is not None:
        def thinking():
            msgs = [{"role": "user", "content": "hi"}]
            on = watcher.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
            off = watcher.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ok = on != off and "</think>" in off
            r.add("PASS" if ok else "FAIL", "Thinking switch",
                  "the chat format turns thinking on and off" if ok else "the chat format ignores enable_thinking")
        r.guard("Thinking switch", thinking)

    if labels and watcher.labelers:
        def neuronpedia():
            layer, labeler = next(iter(sorted(watcher.labelers.items())))
            record, _ = watcher.step(ids)
            rows = record["features"].get(str(layer)) or []
            index = rows[0]["index"] if rows else 0
            ok, label = labeler._fetch(index)
            if not ok:
                r.add("WARN", "Neuronpedia labels", "could not reach neuronpedia.org; gray 'pushes toward' labels still work")
            else:
                r.add("PASS", "Neuronpedia labels", f"layer {layer} concept #{index}: {label!r}" if label
                      else f"reachable; concept #{index} at layer {layer} has no written label")
        r.guard("Neuronpedia labels", neuronpedia)
    elif watcher.labelers:
        r.add("PASS", "Neuronpedia labels", "skipped (labels turned off)")

    if live:
        def live_viewer():
            from .live import LiveServer

            server = LiveServer(watcher, max_new_tokens=5, labels=False, log=lambda *_: None).start(port=0)
            try:
                page = urllib.request.urlopen(server.url, timeout=10).read().decode()
                req = urllib.request.Request(server.url + "api/ask", data=json.dumps({"message": "Say hi."}).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=10).read()
                deadline, d = time.time() + 300, {"busy": True, "final": None, "v": -1}
                while (d["busy"] or d["final"] is None) and time.time() < deadline:
                    d = json.loads(urllib.request.urlopen(f"{server.url}api/state?v={d['v']}", timeout=30).read())
                words = len(d["final"]["steps"]) if d["final"] else 0
                ok = '"api": "api/"' in page and words > 0
                r.add("PASS" if ok else "FAIL", "Live viewer server", f"page served and a {words}-word reply streamed")
            finally:
                server.stop()
        r.guard("Live viewer server", live_viewer)

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**30
        r.add("PASS", "GPU memory", f"peak {peak:.1f} GB on {torch.cuda.device_count()} GPU(s)")
    log("All checks passed." if r.ok else "Some checks failed. Paste this report back to whoever is helping you.")
    return r


__all__ = ["run_checks", "Report"]
