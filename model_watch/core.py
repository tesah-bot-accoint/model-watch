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

trace() runs as a small graph (see graph.py): encode, generate, assemble,
with the run state checkpointed after each node and each word.

This only observes. Nothing in the model is changed.
"""
from __future__ import annotations

import inspect
import operator
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterator, Optional, Union

import torch

from .graph import Checkpointer, Node, RunState, load_checkpoint, run_graph
from .labels import NeuronpediaLabels, _cache_dir
from .sae import JumpReLUSAE

TRACE_VERSION = 3
ROLES = ("system", "user", "assistant")

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
    model_revision: Optional[str] = None  # Hub branch, tag or commit for the model and tokenizer; None = latest
    sae_revision: Optional[str] = None  # same, for the SAE repo
    checkpoints: Union[bool, str] = True  # checkpoint runs to the cache (True), to a folder (path), or not (False)

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


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


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
        *,
        checkpoints: Union[bool, str, Path] = True,
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
        # Memo of per-feature descriptions, shared across runs (bounded by the SAE sizes). Each trace gets a copy.
        self.feature_info: dict[str, dict] = {}
        self.stop_ids = self._stop_ids()
        self.max_positions = getattr(_text_config(model.config), "max_position_embeddings", None)
        # Hooks are registered on the shared model, so only one forward pass may run at a time.
        self._forward_lock = threading.RLock()
        self._info_lock = threading.RLock()
        self._sanitizes = _accepts(getattr(tokenizer, "apply_chat_template", None), "sanitize_control_tokens")
        self.checkpoints: Optional[Checkpointer] = None
        if checkpoints:
            try:
                self.checkpoints = Checkpointer(_cache_dir() / "runs" if checkpoints is True else checkpoints)
            except OSError as err:  # read-only home, full disk: run without the safety net rather than not at all
                warnings.warn(f"Checkpoints are off: {err}", RuntimeWarning, stacklevel=2)

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
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token, revision=cfg.model_revision)
        kwargs = {"token": hf_token, "revision": cfg.model_revision, "attn_implementation": "eager",
                  _dtype_kwarg(): dtype}
        if cfg.device_map:
            kwargs["device_map"] = cfg.device_map
        model = _load_model(transformers, cfg.model_id, kwargs)
        if not cfg.device_map:
            model = model.to(device)
        home = model.get_output_embeddings().weight.device

        saes = {}
        for layer in cfg.sae_layers:
            log(f"Loading concept dictionary for layer {layer}...")
            if cfg.sae_format == "gemma-scope-2":
                saes[layer] = JumpReLUSAE.from_gemma_scope_2(cfg.sae_repo, layer, cfg.sae_width, cfg.sae_l0,
                                                             device=home, token=hf_token, revision=cfg.sae_revision)
            else:
                saes[layer] = JumpReLUSAE.from_gemma_scope_1(cfg.sae_repo, layer, cfg.sae_width, cfg.sae_l0,
                                                             device=home, token=hf_token, revision=cfg.sae_revision)
        labelers = {
            layer: NeuronpediaLabels(cfg.neuronpedia_model, source, enabled=cfg.labels, log=log)
            for layer, source in cfg.neuronpedia_sources.items()
            if layer in saes
        }
        meta = {
            "preset": cfg.preset,
            "model": cfg.model_id,
            "model_revision": getattr(model.config, "_commit_hash", None) or cfg.model_revision,
            "sae": f"{cfg.sae_repo} ({cfg.sae_format}, width {cfg.sae_width}, L0 {cfg.sae_l0})",
            "sae_revisions": {str(layer): getattr(sae, "revision", None) for layer, sae in saes.items()},
            "sae_layers": list(cfg.sae_layers),
            "neuronpedia_model": cfg.neuronpedia_model,
            "neuronpedia_sources": {str(k): v for k, v in cfg.neuronpedia_sources.items()},
            "device": "multi-GPU" if cfg.device_map else device,
            "dtype": dtype_name,
            "chat": bool(cfg.chat),
        }
        log("Ready.")
        return cls(model, tokenizer, saes, labelers, bool(cfg.chat), cfg.top_k_tokens, cfg.top_k_features,
                   cfg.promote_tokens, meta, checkpoints=cfg.checkpoints)

    # ---- one forward pass -------------------------------------------------

    def _forward(self, ids: torch.Tensor, past=None, use_cache: bool = False):
        """Run the model on `ids` (all new tokens since `past`).

        Returns next-token logits, every layer's output at the last position
        (stacked [n_layers, d_model] on self.home), and the updated cache.
        Thread-safe: concurrent calls take turns, so their hooks never mix.
        """
        ids = _as_batch_of_one(ids)
        outputs: list[Optional[torch.Tensor]] = [None] * self.n_layers

        def make_hook(i):
            def hook(_module, _inputs, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                outputs[i] = hidden[0, -1].detach().to(self.home)

            return hook

        with self._forward_lock:
            handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)]
            try:
                with torch.no_grad():
                    out = self.model(input_ids=ids.to(self.input_device), past_key_values=past, use_cache=use_cache)
            finally:
                for handle in handles:
                    handle.remove()
        missing = [i for i, o in enumerate(outputs) if o is None]
        if missing:
            raise RuntimeError(f"Layers {missing} produced no output during the forward pass.")
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
        return self._promotes_many([(layer, index)])[0]

    def _promotes_many(self, pairs: list) -> list[list[str]]:
        """promotes() for many features with one matrix product, so the output matrix is read once per word."""
        with torch.no_grad():
            dtype = self.unembed.weight.dtype
            directions = torch.stack([self.saes[layer].W_dec[index].to(device=self.home, dtype=dtype)
                                      for layer, index in pairs])
            scores = (directions @ self.unembed.weight.T).float()  # [features, vocab]
            top = scores.topk(min(self.promote_tokens * 3, scores.shape[-1]), dim=-1).indices.tolist()
        return [self._words(ids) for ids in top]

    def _words(self, token_ids: list[int]) -> list[str]:
        words, seen = [], set()
        for tid in token_ids:
            text = self._decode(tid).strip()
            if not text or text in seen or (text.startswith("<") and text.endswith(">")):
                continue
            seen.add(text)
            words.append(text)
            if len(words) == self.promote_tokens:
                break
        return words

    def _register_many(self, pairs: list) -> None:
        """Describe features seen for the first time (shared memo; each trace copies what it uses)."""
        with self._info_lock:
            pairs = [p for p in dict.fromkeys(pairs) if feature_key(*p) not in self.feature_info]
        if not pairs:
            return
        words = self._promotes_many(pairs)
        with self._info_lock:
            for (layer, index), promoted in zip(pairs, words):
                labeler = self.labelers.get(layer)
                cached = labeler is not None and labeler.is_cached(index)
                self.feature_info.setdefault(feature_key(layer, index), {
                    "layer": layer,
                    "index": index,
                    "promotes": promoted,
                    "label": labeler.cached(index) if labeler else None,
                    "label_source": ("cache" if cached else "not_looked_up") if labeler else None,
                    "url": labeler.url(index) if labeler else None,
                })

    def _features(self, resid: torch.Tensor) -> tuple[dict, dict]:
        feats, active, seen = {}, {}, []
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
                seen.append((layer, index))
                rows.append({"index": index, "activation": round(value, 3)})
            feats[str(layer)] = rows
        self._register_many(seen)
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

    # ---- encoding ---------------------------------------------------------

    def _messages(self, prompt) -> list[dict]:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        if not messages:
            raise ValueError("The conversation is empty; send at least one message.")
        for i, m in enumerate(messages):
            if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), (str, list)):
                raise ValueError(f"Message {i} needs a role ({', '.join(ROLES)}) and text content; got {m!r:.120}")
        return messages

    def _control_tokens(self) -> set:
        names = set(getattr(self.tok, "all_special_tokens", None) or [])
        for added in (getattr(self.tok, "added_tokens_decoder", None) or {}).values():
            if getattr(added, "special", False):
                names.add(str(added))
        return {n for n in names if n}

    def _chat_ids(self, messages: list[dict]) -> list[int]:
        """Token ids for a chat. Turn markers typed inside a message are read as plain text, never as turns."""
        kwargs = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
        if self._sanitizes:
            kwargs["sanitize_control_tokens"] = True
        else:  # older transformers: refuse rather than let typed text forge a turn
            control = self._control_tokens()
            for i, m in enumerate(messages):
                text = m["content"] if isinstance(m["content"], str) else str(m["content"])
                found = sorted(t for t in control if t in text)
                if found and m["role"] != "assistant":
                    raise ValueError(f"Message {i} contains {found[0]!r}, which this transformers version would read "
                                     "as a chat control token. Remove it, or upgrade transformers to 5.x, which "
                                     "encodes it as plain text.")
        enc = self.tok.apply_chat_template(messages, **kwargs)
        ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
        return torch.as_tensor(ids).reshape(-1).tolist()

    def _splice_history(self, messages: list[dict], full: list[int], previous: dict) -> Optional[list[int]]:
        """Rebuild a continued chat from the previous turn's exact token ids instead of re-tokenized text.

        The model wrote specific tokens; turning them into text and back can give different ones. This keeps
        the earlier ids and the reply's ids, and takes only the template's closing marker and the new message
        from the fresh encoding. Returns None, meaning "use the fresh encoding", when anything doesn't line up.
        """
        prompt_ids, steps, earlier = previous.get("prompt_token_ids"), previous.get("steps"), previous.get("messages")
        if not prompt_ids or not steps or not isinstance(earlier, list):
            return None
        expected = [(m.get("role"), m.get("content")) for m in earlier] + [("assistant", previous.get("reply"))]
        if len(messages) != len(expected) + 1 or [(m["role"], m["content"]) for m in messages[:-1]] != expected:
            return None
        generated = [s.get("token_id") for s in steps]
        if generated and generated[-1] in self.stop_ids:
            generated = generated[:-1]
        if not all(isinstance(t, int) for t in generated) or any(t in self.stop_ids for t in generated):
            return None
        before = self._chat_ids(self._messages(earlier))  # fresh encoding of the earlier prompt
        if full[:len(before)] != before:
            return None
        end = next((i for i in range(len(before), len(full)) if full[i] in self.stop_ids), None)
        if end is None:
            return None
        return list(prompt_ids) + generated + full[end:]

    def _encode(self, prompt, continue_from: Optional[dict] = None) -> tuple[list[int], Optional[bool]]:
        if not self.chat:
            if not isinstance(prompt, str):
                raise ValueError("This model is not in chat mode; pass a string prompt.")
            if continue_from is not None:
                raise ValueError("continue_from only applies to chat models.")
            return torch.as_tensor(self.tok(prompt, return_tensors="pt").input_ids).reshape(-1).tolist(), None
        messages = self._messages(prompt)
        ids = self._chat_ids(messages)
        if continue_from is None:
            return ids, None
        spliced = self._splice_history(messages, ids, continue_from)
        return (spliced, True) if spliced is not None else (ids, False)

    def encode(self, prompt: Union[str, Messages], continue_from: Optional[dict] = None) -> torch.Tensor:
        """Token ids for a plain prompt, or for a chat (string = one user message, or a message list).

        With `continue_from` (the previous turn's trace), a chat reuses that turn's exact token ids.
        """
        ids, _ = self._encode(prompt, continue_from)
        if not ids:
            raise ValueError("The prompt produced no tokens.")
        return torch.tensor([ids])

    # Version 1 name.
    encode_prompt = encode

    def _budget(self, n_prompt: int, max_new_tokens) -> int:
        """Validate max_new_tokens and keep the whole sequence inside the model's context window."""
        try:
            n = operator.index(max_new_tokens)
        except TypeError:
            raise ValueError(f"max_new_tokens must be a whole number, not {max_new_tokens!r}") from None
        if n < 1:
            raise ValueError(f"max_new_tokens must be at least 1, not {n}")
        if self.max_positions:
            if n_prompt >= self.max_positions:
                raise ValueError(f"The prompt is {n_prompt} tokens; this model reads at most {self.max_positions}.")
            n = min(n, self.max_positions - n_prompt)
        return n

    # ---- generation -------------------------------------------------------

    def _generate(self, ids: torch.Tensor, start: int, max_new_tokens: int, stop_at_eos: bool,
                  use_cache: bool) -> Iterator[dict]:
        past, feed = None, ids
        for n in range(start, max_new_tokens):
            logits, resid, past = self._forward(feed, past, use_cache)
            record, chosen = self._observe(logits, resid)
            record["index"] = n
            yield record
            if stop_at_eos and record["stop"]:
                break
            new = torch.tensor([[chosen]])
            ids = torch.cat([ids, new], dim=1)
            feed = new if (use_cache and past is not None) else ids

    def step(self, ids: torch.Tensor) -> tuple[dict, int]:
        """Observe one next-token decision for the full sequence `ids` (shape [1, seq])."""
        logits, resid, _ = self._forward(ids, None)
        return self._observe(logits, resid)

    def watch(self, prompt: Union[str, Messages], max_new_tokens: int = 200, stop_at_eos: bool = True,
              use_cache: bool = True, *, continue_from: Optional[dict] = None) -> Iterator[dict]:
        """Generate greedily, yielding one observation per new token as it happens."""
        ids = self.encode(prompt, continue_from)
        yield from self._generate(ids, 0, self._budget(ids.shape[1], max_new_tokens), stop_at_eos, use_cache)

    # ---- the trace graph --------------------------------------------------

    def _nodes(self, on_step, continue_from) -> dict:
        def encode(state: RunState) -> None:
            ids, state.exact_history = self._encode(state.prompt, continue_from)
            if not ids:
                raise ValueError("The prompt produced no tokens.")
            state.prompt_ids = ids
            state.max_new_tokens = self._budget(len(ids), state.max_new_tokens)
            state.status = "encoded"
            if self.checkpoints:
                self.checkpoints.start(state)

        def generate(state: RunState) -> None:
            state.status = "generating"
            ids = torch.tensor([state.prompt_ids + [s["token_id"] for s in state.steps]])
            for record in self._generate(ids, len(state.steps), state.max_new_tokens, state.stop_at_eos,
                                         state.use_cache):
                state.steps.append(record)
                if self.checkpoints:
                    self.checkpoints.step(state, record)
                if on_step:
                    on_step(record, state.steps)
            state.status = "generated"

        def assemble(state: RunState) -> None:
            state.trace = self._assemble(state, complete=True)
            state.status = "assembled"

        return {
            "encode": Node(encode, lambda s: "generate"),
            "generate": Node(generate, lambda s: "assemble"),
            "assemble": Node(assemble, lambda s: None),
        }

    def _finished(self, state: RunState) -> bool:
        last = state.steps[-1] if state.steps else None
        return len(state.steps) >= state.max_new_tokens or bool(last and state.stop_at_eos and last.get("stop"))

    def _run(self, state: RunState, start: str, on_step=None, continue_from=None) -> dict:
        checkpoint = self.checkpoints.status if self.checkpoints else None
        try:
            run_graph(self._nodes(on_step, continue_from), start, state, checkpoint)
        except BaseException as err:
            try:
                err.partial_trace = self._assemble(state, complete=False) if state.prompt_ids else None
            except Exception:
                err.partial_trace = None  # never replace the real error with one from building the partial trace
            path = self.checkpoints.path(state.run_id) if self.checkpoints else None
            err.checkpoint = str(path) if path is not None and path.exists() and not self.checkpoints.failed else None
            raise
        if self.checkpoints:
            self.checkpoints.finish(state)
        return state.trace

    def trace(self, prompt: Union[str, Messages], max_new_tokens: int = 200, stop_at_eos: bool = True,
              on_step=None, use_cache: bool = True, *, continue_from: Optional[dict] = None) -> dict:
        """Run the encode -> generate -> assemble graph and return a JSON-ready trace for the viewer.

        If anything stops the run (an error, Ctrl+C, an exception from on_step), the exception is re-raised
        with `partial_trace` (the trace so far, marked complete=False) and `checkpoint` (a file resume() takes).
        """
        state = RunState.new(prompt, max_new_tokens, stop_at_eos, use_cache, model=self.meta.get("model"))
        return self._run(state, "encode", on_step, continue_from)

    def resume(self, checkpoint: Union[str, Path], on_step=None) -> dict:
        """Continue an interrupted trace() from its checkpoint file. Gives the same result as an uninterrupted run."""
        state = load_checkpoint(checkpoint)
        if state.model != self.meta.get("model"):
            raise ValueError(f"This checkpoint was made with {state.model!r}, not {self.meta.get('model')!r}.")
        if not state.prompt_ids:
            raise ValueError("This checkpoint stopped before the prompt was encoded; start a new trace instead.")
        state.status, state.error = "encoded", None
        self._register_many([(int(layer), f["index"]) for s in state.steps
                             for layer, rows in s["features"].items() for f in rows if int(layer) in self.saes])
        return self._run(state, "assemble" if self._finished(state) else "generate", on_step)

    def _assemble(self, state: RunState, complete: bool) -> dict:
        prompt, steps = state.prompt, list(state.steps)
        used = {feature_key(int(layer), f["index"]) for s in steps for layer, rows in s["features"].items() for f in rows}
        if isinstance(prompt, str):
            display, messages = prompt, ([{"role": "user", "content": prompt}] if self.chat else None)
        else:
            messages = [dict(m) for m in prompt]
            display = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        with self._info_lock:
            info = {k: {**self.feature_info[k], "promotes": list(self.feature_info[k]["promotes"])}
                    for k in sorted(used) if k in self.feature_info}
        return {
            "version": TRACE_VERSION,
            "created": state.created,
            "complete": complete,
            "error": state.error,
            "meta": {**self.meta, "n_layers": self.n_layers, "sae_layers": self.sae_layers,
                     "labeled_layers": sorted(self.labelers), "chat": self.chat},
            "prompt": display,
            "messages": messages,
            "prompt_tokens": [self._decode(i) for i in state.prompt_ids],
            "prompt_token_ids": list(state.prompt_ids),
            "exact_history": state.exact_history,
            "reply": self.reply_text(steps),
            "steps": steps,
            "feature_info": info,
        }

    def reply_text(self, steps: list[dict]) -> str:
        """The reply decoded as one sequence, so characters split across tokens (emoji, some scripts) survive."""
        kept = [s for s in steps if not s.get("stop")]
        if all(isinstance(s.get("token_id"), int) for s in kept):
            return self.tok.decode([s["token_id"] for s in kept]) if kept else ""
        return "".join(s["token"] for s in kept)  # very old traces without token ids

    # ---- labels -----------------------------------------------------------

    def add_labels(self, trace: dict, max_lookups: int = 400, workers: int = 6, log=print) -> int:
        """Fetch Neuronpedia labels for features in this trace (most active first). Returns how many were looked up.

        Each feature's `label_source` records where its label came from: cache, network, failed (Neuronpedia
        unreachable), or not_looked_up (labels off, or past max_lookups).
        """
        for labeler in self.labelers.values():
            labeler.reset()  # a previous turn's outage shouldn't silence this one
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
                todo.append((key, labeler, index))
        todo = todo[:max_lookups]
        sources: dict[str, str] = {}

        def fetch(job) -> tuple[str, str]:
            key, labeler, index = job
            if not labeler.enabled:
                return key, "not_looked_up"
            labeler.get(index, save=False)
            return key, "network" if labeler.is_cached(index) else "failed"

        if todo:
            log(f"Looking up {len(todo)} labels on Neuronpedia...")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                sources.update(pool.map(fetch, todo))
            for labeler in self.labelers.values():
                labeler.save()
        for key, info in trace["feature_info"].items():
            labeler = self.labelers.get(info["layer"])
            if not labeler:
                continue
            source = sources.get(key) or ("cache" if labeler.is_cached(info["index"]) else "not_looked_up")
            update = {"label": labeler.cached(info["index"]), "label_source": source}
            info.update(update)
            with self._info_lock:
                self.feature_info.get(key, {}).update(update)
        return len(todo)


def _as_batch_of_one(ids) -> torch.Tensor:
    ids = torch.as_tensor(ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError(f"Expected token ids of shape [1, seq] with seq >= 1, got {tuple(ids.shape)}")
    return ids


def _load_model(transformers, model_id: str, kwargs: dict):
    """Try the text-only auto class, then the multimodal one, keeping each one's error if both fail."""
    errors = []
    for loader in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        auto = getattr(transformers, loader, None)
        if auto is None:
            continue
        try:
            return auto.from_pretrained(model_id, **kwargs)
        except ValueError as err:  # this auto class does not cover the model (or the config is bad); say which
            errors.append((loader, err))
    detail = " | ".join(f"{name}: {err}" for name, err in errors) or "no suitable auto class in this version"
    raise RuntimeError(f"Could not load {model_id} with transformers {transformers.__version__}. {detail}") from (
        errors[-1][1] if errors else None)


__all__ = ["WatchConfig", "ModelWatcher", "PRESETS", "DEFAULT_PRESET", "settled_layer", "feature_key",
           "pick_device", "pick_dtype", "TRACE_VERSION"]
