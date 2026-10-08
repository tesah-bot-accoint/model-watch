"""A minimal JumpReLU sparse autoencoder, the architecture Gemma Scope 1 and 2 use.

Written out in full so you can read exactly what turns a layer's activations
into a short list of "concepts" (features). No SAELens dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Accepted spellings for each parameter, lower-cased. Gemma Scope 1 uses W_enc etc.
_ALIASES = {
    "W_enc": ("w_enc", "encoder.weight", "w_e"),
    "W_dec": ("w_dec", "decoder.weight", "w_d"),
    "b_enc": ("b_enc", "encoder.bias", "b_e"),
    "b_dec": ("b_dec", "decoder.bias", "b_d"),
    "threshold": ("threshold", "thresholds", "jump_threshold"),
}


def normalize_params(raw: dict) -> dict:
    """Map a checkpoint's tensors onto W_enc [d, n], W_dec [n, d], b_enc [n], b_dec [d], threshold [n].

    Handles different key spellings and transposed weight matrices, orienting
    them using the length of b_dec (the model width d).
    """
    lowered = {k.lower(): v for k, v in raw.items()}
    out = {}
    for name, options in _ALIASES.items():
        for opt in options:
            if opt in lowered:
                out[name] = torch.as_tensor(np.asarray(lowered[opt]) if not torch.is_tensor(lowered[opt]) else lowered[opt])
                break
    missing = set(_ALIASES) - set(out)
    if missing:
        raise ValueError(f"SAE checkpoint is missing {sorted(missing)}; found keys {sorted(raw)}")
    d = out["b_dec"].shape[-1]
    if out["W_enc"].shape[0] != d and out["W_enc"].shape[-1] == d:
        out["W_enc"] = out["W_enc"].T.contiguous()
    if out["W_dec"].shape[-1] != d and out["W_dec"].shape[0] == d:
        out["W_dec"] = out["W_dec"].T.contiguous()
    n = out["W_enc"].shape[1]
    if out["W_dec"].shape != (n, d) or out["b_enc"].shape[-1] != n or out["threshold"].shape[-1] != n:
        raise ValueError(
            "SAE shapes do not line up: "
            + ", ".join(f"{k} {tuple(v.shape)}" for k, v in out.items())
        )
    return out


class JumpReLUSAE(torch.nn.Module):
    """encode(x): which features fire, and how strongly.

    pre  = x @ W_enc + b_enc
    acts = relu(pre) where pre > threshold, else 0     (the "jump")
    decode(acts) = acts @ W_dec + b_dec                 (rebuilds x)
    Row W_dec[f] is feature f's direction in the model's activation space.
    """

    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.as_tensor(W_enc), requires_grad=False)
        self.W_dec = torch.nn.Parameter(torch.as_tensor(W_dec), requires_grad=False)
        self.b_enc = torch.nn.Parameter(torch.as_tensor(b_enc), requires_grad=False)
        self.b_dec = torch.nn.Parameter(torch.as_tensor(b_dec), requires_grad=False)
        self.threshold = torch.nn.Parameter(torch.as_tensor(threshold), requires_grad=False)

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.W_enc.device, dtype=self.W_enc.dtype)
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @classmethod
    def from_params(cls, raw: dict, device="cpu", dtype=torch.float32) -> "JumpReLUSAE":
        return cls(**normalize_params(raw)).to(device=device, dtype=dtype)

    @classmethod
    def from_npz(cls, path: str | Path, device="cpu", dtype=torch.float32) -> "JumpReLUSAE":
        params = np.load(path)
        return cls.from_params({k: params[k] for k in params.files}, device, dtype)

    @classmethod
    def from_safetensors(cls, path: str | Path, device="cpu", dtype=torch.float32) -> "JumpReLUSAE":
        from safetensors.torch import load_file

        return cls.from_params(load_file(str(path)), device, dtype)

    @classmethod
    def from_gemma_scope_1(cls, repo_id="google/gemma-scope-2b-pt-res", layer=20, width="16k", l0=71,
                           device="cpu", token=None) -> "JumpReLUSAE":
        """Gemma Scope 1 (Gemma 2): layer_{layer}/width_{width}/average_l0_{l0}/params.npz"""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz", token=token)
        return cls.from_npz(path, device=device)

    @classmethod
    def from_gemma_scope_2(cls, repo_id="google/gemma-scope-2-4b-it", layer=17, width="16k", l0="medium",
                           site="resid_post", device="cpu", token=None) -> "JumpReLUSAE":
        """Gemma Scope 2 (Gemma 3): {site}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors

        The main resid_post folder has 4 layers per model (about 25, 50, 65 and
        85 percent depth) with l0 small, medium or big. resid_post_all covers
        every layer with l0 small or big only.

        Only params.safetensors is downloaded, not the larger examples file
        next to it. Its tensor names (w_enc, w_dec, b_enc, b_dec, threshold) match
        SAELens's Gemma Scope 2 loader but have not been loaded from the real files here;
        normalize_params accepts common spellings and lists the actual names
        if none match.
        """
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id, f"{site}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors", token=token)
        return cls.from_safetensors(path, device=device)

    # Backwards-compatible name used by version 1.
    from_hub = from_gemma_scope_1
