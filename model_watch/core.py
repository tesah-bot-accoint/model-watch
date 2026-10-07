"""Watch a language model choose each word.

For every generated token, ModelWatcher records three views:

1. The choice: the token picked and its runners-up.
2. How the guess formed (logit lens): each layer's output is pushed through
   the model's own final norm and output layer, giving "what the model would
   say if it stopped here". You see the answer firm up layer by layer.
3. Active concepts: a Gemma Scope sparse autoencoder reads one layer's
   activations and reports which features fire, labeled via Neuronpedia.

This only observes. Nothing in the model is changed.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from .labels import NeuronpediaLabels
from .sae import JumpReLUSAE


@dataclass
class WatchConfig:
    model_id: str = "google/gemma-2-2b"  # base model: Gemma Scope SAEs were trained on it
    sae_repo: str = "google/gemma-scope-2b-pt-res"
    sae_layer: int = 20
    sae_width: str = "16k"
    sae_l0: int = 71
    neuronpedia_model: str = "gemma-2-2b"
    neuronpedia_source: Optional[str] = None  # default: "{layer}-gemmascope-res-{width}"
    labels: bool = True
    top_k_tokens: int = 5
    top_k_features: int = 8
    device: Optional[str] = None  # "cuda", "mps", "cpu"; None picks the best available
    dtype: str = "auto"  # "auto", "float32", "bfloat16", "float16"

    def source_id(self) -> str:
        return self.neuronpedia_source or f"{self.sae_layer}-gemmascope-res-{self.sae_width}"


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
        # Gemma 2 can overflow in float16, so auto only uses bfloat16 or float32.
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
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
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    raise AttributeError("Could not find the decoder layers on this model.")


def settled_layer(top_ids: list[int], chosen: int) -> Optional[int]:
    """First layer from which the top guess equals the final choice and never changes again."""
    settled = None
    for layer in range(len(top_ids) - 1, -1, -1):
        if top_ids[layer] != chosen:
            break
        settled = layer
    return settled


class ModelWatcher:
    def __init__(
        self,
        model,
        tokenizer,
        sae: Optional[JumpReLUSAE] = None,
        sae_layer: Optional[int] = None,
        labeler: Optional[NeuronpediaLabels] = None,
        top_k_tokens: int = 5,
        top_k_features: int = 8,
        meta: Optional[dict] = None,
    ):
        self.model = model.eval()
        self.tok = tokenizer
        decoder = _decoder(model)
        self.layers = list(decoder.layers)
        self.final_norm = decoder.norm
        self.unembed = model.get_output_embeddings()
        self.softcap = getattr(model.config, "final_logit_softcapping", None)
        self.device = next(model.parameters()).device
        self.sae = sae
        self.sae_layer = sae_layer
        if sae is not None and (sae_layer is None or not 0 <= sae_layer < len(self.layers)):
            raise ValueError(f"sae_layer must be between 0 and {len(self.layers) - 1}")
        self.labeler = labeler
        self.top_k_tokens = top_k_tokens
        self.top_k_features = top_k_features
        self.meta = meta or {}
        eos = getattr(tokenizer, "eos_token_id", None)
        self.eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos]) - {None}

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @classmethod
    def load(cls, cfg: Optional[WatchConfig] = None, hf_token: Optional[str] = None, log=print) -> "ModelWatcher":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = cfg or WatchConfig()
        device = pick_device(cfg.device)
        dtype = pick_dtype(cfg.dtype, device)
        log(f"Loading {cfg.model_id} on {device} ({str(dtype).replace('torch.', '')})...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            token=hf_token,
            attn_implementation="eager",  # recommended for Gemma 2's attention soft-capping
            **{_dtype_kwarg(): dtype},
        ).to(device)
        log(f"Loading SAE {cfg.sae_repo} layer {cfg.sae_layer}, width {cfg.sae_width}, L0 {cfg.sae_l0}...")
        sae = JumpReLUSAE.from_hub(cfg.sae_repo, cfg.sae_layer, cfg.sae_width, cfg.sae_l0, device=device, token=hf_token)
        labeler = NeuronpediaLabels(cfg.neuronpedia_model, cfg.source_id(), enabled=cfg.labels, log=log)
        meta = {
            "model": cfg.model_id,
            "sae": f"{cfg.sae_repo} layer_{cfg.sae_layer}/width_{cfg.sae_width}/average_l0_{cfg.sae_l0}",
            "sae_layer": cfg.sae_layer,
            "neuronpedia_source": cfg.source_id(),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
        }
        log("Ready.")
        return cls(model, tokenizer, sae, cfg.sae_layer, labeler, cfg.top_k_tokens, cfg.top_k_features, meta)

    # ---- one forward pass -------------------------------------------------

    def _forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model once; return final next-token logits and every layer's output at the last position."""
        outputs: list[Optional[torch.Tensor]] = [None] * self.n_layers

        def make_hook(i):
            def hook(_module, _inputs, out):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                outputs[i] = hidden[0, -1].detach()

            return hook

        handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(self.layers)]
        try:
            with torch.no_grad():
                logits = self.model(ids, use_cache=False).logits[0, -1]
        finally:
            for handle in handles:
                handle.remove()
        return logits, torch.stack(outputs)  # [n_layers, d_model]

    def _lens(self, resid: torch.Tensor) -> torch.Tensor:
        """Logit lens: read every layer through the model's own final norm and output layer."""
        with torch.no_grad():
            normed = self.final_norm(resid.unsqueeze(0))[0]
            logits = self.unembed(normed).float()
            if self.softcap:
                logits = self.softcap * torch.tanh(logits / self.softcap)
        return torch.softmax(logits, dim=-1)  # [n_layers, vocab]

    def _features(self, row: torch.Tensor) -> tuple[list[dict], int]:
        with torch.no_grad():
            acts = self.sae.encode(row.float().to(self.sae.W_enc.device).unsqueeze(0))[0]
        active = int((acts > 0).sum())
        k = min(self.top_k_features, active)
        if k == 0:
            return [], 0
        values, indices = acts.topk(k)
        feats = []
        for value, index in zip(values.tolist(), indices.tolist()):
            feats.append(
                {
                    "index": index,
                    "activation": round(value, 3),
                    "label": self.labeler.get(index) if self.labeler else None,
                    "url": self.labeler.url(index) if self.labeler else None,
                }
            )
        return feats, active

    def _decode(self, token_id: int) -> str:
        return self.tok.decode([token_id])

    # ---- public API -------------------------------------------------------

    def step(self, ids: torch.Tensor) -> tuple[dict, int]:
        """Observe one next-token decision for the sequence `ids` (shape [1, seq])."""
        final_logits, resid = self._forward(ids)
        probs = torch.softmax(final_logits.float(), dim=-1)
        chosen = int(probs.argmax())
        top_p, top_i = probs.topk(min(self.top_k_tokens, probs.numel()))

        lens = self._lens(resid)
        layer_top_p, layer_top_i = lens.max(dim=-1)
        chosen_by_layer = lens[:, chosen]
        top_ids = layer_top_i.tolist()

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
            "features": [],
            "active_count": 0,
        }
        if self.sae is not None:
            record["features"], record["active_count"] = self._features(resid[self.sae_layer])
        return record, chosen

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        return self.tok(prompt, return_tensors="pt").input_ids.to(self.device)

    def watch(self, prompt: str, max_new_tokens: int = 20, stop_at_eos: bool = True) -> Iterator[dict]:
        """Generate greedily, yielding one observation per new token as it happens."""
        ids = self.encode_prompt(prompt)
        for n in range(max_new_tokens):
            record, chosen = self.step(ids)
            record["index"] = n
            yield record
            if stop_at_eos and chosen in self.eos_ids:
                break
            ids = torch.cat([ids, torch.tensor([[chosen]], device=ids.device)], dim=1)

    def trace(self, prompt: str, max_new_tokens: int = 20, stop_at_eos: bool = True, on_step=None) -> dict:
        """Run watch() to completion and return a JSON-ready trace for the replay viewer."""
        prompt_ids = self.encode_prompt(prompt)[0].tolist()
        steps = []
        for record in self.watch(prompt, max_new_tokens, stop_at_eos):
            steps.append(record)
            if on_step:
                on_step(record, steps)
        return {
            "version": 1,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "meta": {**self.meta, "n_layers": self.n_layers},
            "prompt": prompt,
            "prompt_tokens": [self._decode(i) for i in prompt_ids],
            "steps": steps,
        }


__all__ = ["WatchConfig", "ModelWatcher", "settled_layer", "pick_device", "pick_dtype"]
