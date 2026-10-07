"""A minimal JumpReLU sparse autoencoder, the architecture Gemma Scope uses.

Written out in full so you can read exactly what turns a layer's activations
into a short list of "concepts" (features). No SAELens dependency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class JumpReLUSAE(torch.nn.Module):
    """encode(x): which features fire, and how strongly.

    pre  = x @ W_enc + b_enc
    acts = relu(pre) where pre > threshold, else 0     (the "jump")
    decode(acts) = acts @ W_dec + b_dec                 (rebuilds x)
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
        x = x.to(self.W_enc.dtype)
        pre = x @ self.W_enc + self.b_enc
        return (pre > self.threshold) * torch.relu(pre)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @classmethod
    def from_npz(cls, path: str | Path, device="cpu", dtype=torch.float32) -> "JumpReLUSAE":
        params = np.load(path)
        missing = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"} - set(params.files)
        if missing:
            raise ValueError(f"{path} is missing SAE parameters: {sorted(missing)}")
        sae = cls(**{k: torch.from_numpy(params[k]) for k in params.files})
        return sae.to(device=device, dtype=dtype)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "google/gemma-scope-2b-pt-res",
        layer: int = 20,
        width: str = "16k",
        l0: int = 71,
        device="cpu",
        token: str | None = None,
    ) -> "JumpReLUSAE":
        """Download one Gemma Scope SAE.

        Files live at layer_{layer}/width_{width}/average_l0_{l0}/params.npz.
        Each layer/width has several L0 variants (how many features fire on
        average). Check the repo tree for the values that exist.
        """
        from huggingface_hub import hf_hub_download

        filename = f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return cls.from_npz(path, device=device)
