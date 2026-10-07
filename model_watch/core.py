"""Watch a language model choose each word.

For every generated token, ModelWatcher records:

1. The choice: the token picked and its runners-up.
2. How the guess formed (logit lens): each layer's output is read through the
   model's own final norm and output layer, giving "what the model would say
   if it stopped here".
3. Active concepts at several layers: Gemma Scope sparse autoencoders report
   which features fire. Each feature gets a free offline label (the words its
   direction pushes the model toward) and, where Neuronpedia has one, a
   written label.

Across a whole answer these records form a concept timeline: which ideas
stay active, where they switch on and off, and how that lines up with what
the model writes.

This only observes. Nothing in the model is changed.
"""
from __future__ import annotations

import datetime as _dt
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from typing import Iterator, Optional, Union

import torch

from .labels import NeuronpediaLabels
from .sae import JumpReLUSAE

Messages = list  # [{"role": "user" | "assistant", "content": str}, ...]

# Verified against the Hugging Face repos and Neuronpedia, October 2026.
PRESETS: dict[str, dict] = {
    "gemma-3-1b-it": dict(
        model_id="google/gemma-3-1b-it", sae_repo="google/gemma-scope-2-1b-it", sae_format="gemma-scope-2",
        sae_layers=(7, 13, 17, 22), sae_width="16k", sae_l0="medium", chat=True,
        neuronpedia_model="gemma-3-1b-it", neuronpedia_sources={13: "13-gemmascope-2-res-16k"},
    ),
    "gemma-3-4b-it": dict(
        model_id="google/gemma-3-4b-it", sae_repo="google/gemma-scope-2-4b-it", sae_format="gemma-scope-2",
        sae_layers=(9, 17, 22, 29), sae_width="16k", sae_l0="medium", chat=True,
        neuronpedia_model="gemma-3-4b-it", neuronpedia_sources={17: "17-gemmascope-2-res-16k"},
    ),
    "gemma-3-12b-it": dict(
        model_id="google/gemma-3-12b-it", sae_repo="google/gemma-scope-2-12b-it", sae_format="gemma-scope-2",
        sae_layers=(12, 24, 31, 41), sae_width="16k", sae_l0="medium", chat=True,
        neuronpedia_model="gemma-3-12b-it", neuronpedia_sources={12: "12-gemmascope-2-res-16k"},
    ),
    "gemma-2-2b": dict(
        model_id="google/gemma-2-2b", sae_repo="google/gemma-scope-2b-pt-res", sae_format="gemma-scope-1",
        sae_layers=(20,), sae_width="16k", sae_l0=71, chat=False,
        neuronpedia_model="gemma-2-2b", neuronpedia_sources={20: "20-gemmascope-res-16k"},
    ),
}
DEFAULT_PRESET = "gemma-3-4b-it"


@dataclass
class WatchConfig:
    preset: str = DEFAULT_PRESET
    # Anything left as None comes from the preset.
    model_id: Optional[str] = None
    sae_repo: Optional[str] = None
    sae_format: Optional[str] = None  # "gemma-scope-2" or "gemma-scope-1"
    sae_layers: Optional[tuple] = None
    sae_width: Optional[str] = None
    sae_l0: Optional[Union[str, int]] = None
    chat: Optional[bool] = None
    neuronpedia_model: Optional[str] = None
    neuronpedia_sources: Optional[dict] = None
    labels: bool = True
    top_k_tokens: int = 5
    top_k_features: int = 10
    promote_tokens: int = 5
    device: Optional[str] = None  # "cuda", "mps", "cpu"; None picks the best available
    device_map: Optional[str] = None  # "auto" splits a big model across several GPUs (Kaggle 2x T4)
    dtype: str = "auto"  # "auto", "float32", "bfloat16", "float16"

    def resolved(self) -> "WatchConfig":
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}. Choose from: {', '.join(PRESETS)}")
        base = PRESETS[self.preset]
        updates = {f.name: base[f.name] for f in fields(self) if getattr(self, f.name) is None and f.name in base}
        cfg = replace(self, **updates)
        cfg.sae_layers = tuple(int(x) for x in (cfg.sae_layers or ()))
        cfg.neuronpedia_sources = {int(k): v for k, v in (cfg.neuronpedia_sources or {}).items()}
        return cfg


def pick_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        # Gemma can overflow in float16, so auto uses bfloat16 on any GPU
        # (older cards like the T4 emulate it: slower, same answers) and float32 on CPU.
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    table = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in table:
        raise ValueError(f"Unknown dtype {name!r}; use one of auto, {', '.join(table)}")
    return table[name]


def _dtype_kwarg() -> str:
    """transformers renamed torch_dtype to dtype in 4.56."""
    import transformers

    major, minor = (int(re.match(r"\d+", p).group()) for p in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def _decoder(model):
    """Find the stack of transformer layers, for text-only and multimodal Gemma alike."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers") and hasattr(inner, "norm"):
        return inner
    if inner is not None and hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model
    if hasattr(model, "get_decoder"):
        dec = model.get_decoder()
        if hasattr(dec, "layers"):
            return dec
    raise AttributeError("Could not find the decoder layers on this model.")


def _text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def settled_layer(top_ids: list[int], chosen: int) -> Optional[int]:
    """First layer from which the top guess equals the final choice and never changes again."""
    settled = None
    for layer in range(len(top_ids) - 1, -1, -1):
        if top_ids[layer] != chosen:
            break
        settled = layer
    return settled


def feature_key(layer: int, index: int) -> str:
    return f"{layer}:{index}"


class ModelWatcher:
    def __init__(
        self,
        model,
        tokenizer,
        saes: Optional[dict] = None,
        labelers: Optional[dict] = None,
        chat: bool = False,
        top_k_tokens: int = 5,
        top_k_features: int = 10,
        promote_tokens: int = 5,
        meta: Optional[dict] = None,
    ):
        self.model = model.eval()
        self.tok = tokenizer
        decoder = _decoder(model)
        self.layers = list(decoder.layers)
        self.final_norm = decoder.norm
        self.unembed = model.get_output_embeddings()
        self.softcap = getattr(_text_config(model.config), "final_logit_softcapping", None)
        self.home = self.unembed.weight.device  # everything we record is gathered here
        self.input_device = model.get_input_embeddings().weight.device
        self.saes = {int(k): v for k, v in (saes or {}).items()}
        for layer in self.saes:
            if not 0 <= layer < len(self.layers):
                raise ValueError(f"SAE layer {layer} is outside 0..{len(self.layers) - 1}")
        self.labelers = {int(k): v for k, v in (labelers or {}).items()}
        self.chat = chat
        self.top_k_tokens = top_k_tokens
        self.top_k_features = top_k_features
        self.promote_tokens = promote_tokens
        self.meta = meta or {}
        self.feature_info: dict[str, dict] = {}
        self.stop_ids = self._stop_ids()

    # ---- setup ------------------------------------------------------------

    def _stop_ids(self) -> set:
        ids = set()
        eos = getattr(self.tok, "eos_token_id", None)
        ids.update(eos if isinstance(eos, (list, tuple, set)) else [eos])
        convert = getattr(self.tok, "convert_tokens_to_ids", None)
        unk = getattr(self.tok, "unk_token_id", None)
        if convert:
            for name in ("<end_of_turn>", "<eos>"):
                try:
                    tid = convert(name)
                except Exception:
                    tid = None
                if isinstance(tid, int) and tid != unk:
                    ids.add(tid)
        return {i for i in ids if isinstance(i, int)}

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def sae_layers(self) -> list[int]:
        return sorted(self.saes)

    @classmethod
    def load(cls, cfg: Optional[WatchConfig] = None, hf_token: Optional[str] = None, log=print) -> "ModelWatcher":
        import transformers
        from transformers import AutoTokenizer

        cfg = (cfg or WatchConfig()).resolved()
        device = pick_device(cfg.device)
        dtype = pick_dtype(cfg.dtype, device)
        dtype_name = str(dtype).replace("torch.", "")
        log(f"Loading {cfg.model_id} ({dtype_name}, {'split across GPUs' if cfg.device_map else device})...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
        kwargs = {"token": hf_token, "attn_implementation": "eager", _dtype_kwarg(): dtype}
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        model = None
        for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            auto = getattr(transformers, loader, None)
            if auto is None:
                continue
            try:
                model = auto.from_pretrained(cfg.model_id, **kwargs)
                break
            except ValueError:
                continue  # this auto class does not cover the model; try the next
        if model is None:
            raise RuntimeError(f"Could not load {cfg.model_id} with transformers {transformers.__version__}")
        if not cfg.device_map:
            model = model.to(device)
        home = model.get_output_embeddings().weight.device

        saes = {}
        for layer in cfg.sae_layers:
            log(f"Loading concept dictionary for layer {layer}...")
            if cfg.sae_format == "gemma-scope-2":
                saes[layer] = JumpReLUSAE.from_gemma_scope_2(cfg.sae_repo, layer, cfg.sae_width, cfg.sae_l0,
                                                             device=home, token=hf_token)
            else:
                saes[layer] = JumpReLUSAE.from_gemma_scope_1(cfg.sae_repo, layer, cfg.sae_width, cfg.sae_l0,
                                                             device=home, token=hf_token)
        labelers = {
            layer: NeuronpediaLabels(cfg.neuronpedia_model, source, enabled=cfg.labels, log=log)
            for layer, source in cfg.neuronpedia_sources.items()
            if layer in saes
        }
        meta = {
            "preset": cfg.preset,
            "model": cfg.model_id,
            "sae": f"{cfg.sae_repo} ({cfg.sae_format}, width {cfg.sae_width}, L0 {cfg.sae_l0})",
            "sae_layers": list(cfg.sae_layers),
            "neuronpedia_model": cfg.neuronpedia_model,
            "neuronpedia_sources": {str(k): v for k, v in cfg.neuronpedia_sources.items()},
            "device": "multi-GPU" if cfg.device_map else device,
            "dtype": dtype_name,
            "chat": bool(cfg.chat),
        }
        log("Ready.")
        return cls(model, tokenizer, saes, labelers, bool(cfg.chat), cfg.top_k_tokens, cfg.top_k_features,
                   cfg.promote_tokens, meta)

    # ---- one forward pass -------------------------------------------------

    def _forward(self, ids: torch.Tensor, past=None, use_cache: bool = False):
        """Run the model on `ids` (all new tokens since `past`).

        Returns next-token logits, every layer's output at the last position
        (stacked [n_layers, d_model] on self.home), and the updated cache.
        """
        outputs: list[Optional[torch.Tensor]] = [None] * self.n_layers

        def make_hook(i):
            def hook(_module, _inputs, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                outputs[i] = hidden[0, -1].detach().to(self.home)

            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)]
        try:
            with torch.no_grad():
                out = self.model(input_ids=ids.to(self.input_device), past_key_values=past, use_cache=use_cache)
        finally:
            for handle in handles:
                handle.remove()
        cache = getattr(out, "past_key_values", None) if use_cache else None
        return out.logits[0, -1].to(self.home), torch.stack(outputs), cache

    def _lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Logit lens: read every layer through the model's own final norm and output layer."""
        with torch.no_grad():
            norm_device = next(self.final_norm.parameters()).device
            normed = self.final_norm(resid.unsqueeze(0).to(norm_device))[0].to(self.home)
            logits = self.unembed(normed).float()
            if self.softcap:
                logits = self.softcap * torch.tanh(logits / self.softcap)
        return torch.softmax(logits, dim=-1)  # [n_layers, vocab]

    def promotes(self, layer: int, index: int) -> list[str]:
        """Free, offline label: the tokens this feature's direction pushes the output toward."""
        with torch.no_grad():
            direction = self.saes[layer].W_dec[index].to(device=self.home, dtype=self.unembed.weight.dtype)
            scores = (self.unembed.weight @ direction).float()
            top = scores.topk(self.promote_tokens * 3).indices.tolist()
        words, seen = [], set()
        for tid in top:
            text = self._decode(tid).strip()
            if not text or text in seen or (text.startswith("<") and text.endswith(">")):
                continue
            seen.add(text)
            words.append(text)
            if len(words) == self.promote_tokens:
                break
        return words

    def _register(self, layer: int, index: int) -> None:
        key = feature_key(layer, index)
        if key in self.feature_info:
            return
        labeler = self.labelers.get(layer)
        self.feature_info[key] = {
            "layer": layer,
            "index": index,
            "promotes": self.promotes(layer, index),
            "label": labeler.cached(index) if labeler else None,
            "url": labeler.url(index) if labeler else None,
        }

    def _features(self, resid: torch.Tensor) -> tuple[dict, dict]:
        feats, active = {}, {}
        for layer, sae in self.saes.items():
            with torch.no_grad():
                acts = sae.encode(resid[layer].float().unsqueeze(0))[0]
            count = int((acts > 0).sum())
            active[str(layer)] = count
            k = min(self.top_k_features, count)
            if k == 0:
                feats[str(layer)] = []
                continue
            values, indices = acts.topk(k)
            rows = []
            for value, index in zip(values.tolist(), indices.tolist()):
                self._register(layer, index)
                rows.append({"index": index, "activation": round(value, 3)})
            feats[str(layer)] = rows
        return feats, active

    def _decode(self, token_id: int) -> str:
        return self.tok.decode([token_id])

    def _observe(self, final_logits: torch.Tensor, resid: torch.Tensor) -> tuple[dict, int]:
        probs = torch.softmax(final_logits.float(), dim=-1)
        chosen = int(probs.argmax())
        top_p, top_i = probs.topk(min(self.top_k_tokens, probs.numel()))
        lens = self._lens(resid)
        layer_top_p, layer_top_i = lens.max(dim=-1)
        chosen_by_layer = lens[:, chosen]
        top_ids = layer_top_i.tolist()
        features, active = self._features(resid) if self.saes else ({}, {})
        record = {
            "token": self._decode(chosen),
            "token_id": chosen,
            "prob": round(float(probs[chosen]), 5),
            "alternatives": [
                {"token": self._decode(i), "token_id": i, "prob": round(p, 5)}
                for p, i in zip(top_p.tolist(), top_i.tolist())
            ],
            "lens": [
                {
                    "layer": layer,
                    "top_token": self._decode(top_ids[layer]),
                    "top_prob": round(float(layer_top_p[layer]), 5),
                    "chosen_prob": round(float(chosen_by_layer[layer]), 5),
                }
                for layer in range(self.n_layers)
            ],
            "settled_layer": settled_layer(top_ids, chosen),
            "features": features,
            "active_count": active,
            "stop": chosen in self.stop_ids,
        }
        return record, chosen

    # ---- public API -------------------------------------------------------

    def step(self, ids: torch.Tensor) -> tuple[dict, int]:
        """Observe one next-token decision for the full sequence `ids` (shape [1, seq])."""
        logits, resid, _ = self._forward(ids, None)
        return self._observe(logits, resid)

    def encode(self, prompt: Union[str, Messages]) -> torch.Tensor:
        """Token ids for a plain prompt, or for a chat (string = one user message, or a message list)."""
        if self.chat:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
            enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt", return_dict=True)
            ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
        else:
            if not isinstance(prompt, str):
                raise ValueError("This model is not in chat mode; pass a string prompt.")
            ids = self.tok(prompt, return_tensors="pt").input_ids
        return torch.as_tensor(ids).reshape(1, -1)

    # Version 1 name.
    encode_prompt = encode

    def watch(self, prompt: Union[str, Messages], max_new_tokens: int = 200, stop_at_eos: bool = True,
              use_cache: bool = True) -> Iterator[dict]:
        """Generate greedily, yielding one observation per new token as it happens."""
        ids = self.encode(prompt)
        past, feed = None, ids
        for n in range(max_new_tokens):
            logits, resid, past = self._forward(feed, past, use_cache)
            record, chosen = self._observe(logits, resid)
            record["index"] = n
            yield record
            if stop_at_eos and record["stop"]:
                break
            new = torch.tensor([[chosen]])
            ids = torch.cat([ids, new], dim=1)
            feed = new if (use_cache and past is not None) else ids

    def trace(self, prompt: Union[str, Messages], max_new_tokens: int = 200, stop_at_eos: bool = True,
              on_step=None, use_cache: bool = True) -> dict:
        """Run watch() to completion and return a JSON-ready trace for the viewer."""
        prompt_ids = self.encode(prompt)[0].tolist()
        steps = []
        for record in self.watch(prompt, max_new_tokens, stop_at_eos, use_cache):
            steps.append(record)
            if on_step:
                on_step(record, steps)
        used = {feature_key(int(layer), f["index"]) for s in steps for layer, rows in s["features"].items() for f in rows}
        if isinstance(prompt, str):
            display, messages = prompt, ([{"role": "user", "content": prompt}] if self.chat else None)
        else:
            messages = list(prompt)
            display = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return {
            "version": 2,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "meta": {**self.meta, "n_layers": self.n_layers, "sae_layers": self.sae_layers,
                     "labeled_layers": sorted(self.labelers), "chat": self.chat},
            "prompt": display,
            "messages": messages,
            "prompt_tokens": [self._decode(i) for i in prompt_ids],
            "reply": self.reply_text(steps),
            "steps": steps,
            "feature_info": {k: self.feature_info[k] for k in sorted(used)},
        }

    def reply_text(self, steps: list[dict]) -> str:
        return "".join(s["token"] for s in steps if not s.get("stop"))

    def add_labels(self, trace: dict, max_lookups: int = 400, workers: int = 6, log=print) -> int:
        """Fetch Neuronpedia labels for features in this trace (most active first). Returns how many were looked up."""
        totals: dict[str, float] = {}
        for s in trace["steps"]:
            for layer, rows in s["features"].items():
                for f in rows:
                    key = feature_key(int(layer), f["index"])
                    totals[key] = totals.get(key, 0.0) + f["activation"]
        todo = []
        for key in sorted(totals, key=totals.get, reverse=True):
            layer, index = (int(x) for x in key.split(":"))
            labeler = self.labelers.get(layer)
            if labeler and labeler.enabled and not labeler.is_cached(index):
                todo.append((labeler, index))
        todo = todo[:max_lookups]
        if todo:
            log(f"Looking up {len(todo)} labels on Neuronpedia...")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda job: job[0].get(job[1], save=False), todo))
            for labeler in self.labelers.values():
                labeler.save()
        for key, info in trace["feature_info"].items():
            labeler = self.labelers.get(info["layer"])
            if labeler:
                info["label"] = labeler.cached(info["index"])
                self.feature_info.get(key, {}).update(label=info["label"])
        return len(todo)


__all__ = ["WatchConfig", "ModelWatcher", "PRESETS", "DEFAULT_PRESET", "settled_layer", "feature_key",
           "pick_device", "pick_dtype"]
