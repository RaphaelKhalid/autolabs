"""Matryoshka BatchTopK sparse autoencoder.

Encoder: W_enc (d_in x width), b_enc (width,).
Decoder: W_dec (width x d_in), rows unit-norm, b_dec (d_in,).

BatchTopK (Bussmann et al., 2024): pre-activations are ReLU'd, then only the
top `k * n_tokens` activations across the *whole batch* survive (not top-k
per token), which lets sparsity vary per token while keeping the batch
average at k.

Matryoshka (Nabeshima / Bussmann matryoshka SAEs): reconstruction loss is
computed at several nested prefix widths ("shells") of the same code vector
-- decoding with only the first `shell` features and the matching decoder
rows -- and summed with equal weight, so early features are forced to carry
a self-contained coarse reconstruction while later features refine it.

Dead-feature auxiliary loss follows the standard TopK-SAE recipe (Gao et al.
/ OpenAI "Scaling and evaluating sparse autoencoders"): the top-k_aux
activations among features that have not fired within a token window are
used to reconstruct the *residual* of the main reconstruction, weighted by
`aux_loss_coef`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as _safetensors_load_file
    from safetensors.torch import save_file as _safetensors_save_file
except ImportError:  # pragma: no cover - safetensors is a hard requirement on the pod
    _safetensors_load_file = None
    _safetensors_save_file = None


def batch_topk(acts: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top `k * n_tokens` activations across the whole
    batch. `acts` is (n_tokens, width) and assumed non-negative (post-ReLU).
    Ties are broken arbitrarily by torch.topk; this does not affect the
    count of surviving entries.
    """
    if acts.ndim != 2:
        raise ValueError("acts must be (n_tokens, width)")
    n_tokens, _width = acts.shape
    total_k = int(k) * n_tokens
    total_k = max(0, min(total_k, acts.numel()))
    if total_k == 0:
        return torch.zeros_like(acts)
    flat = acts.reshape(-1)
    _topk_vals, topk_idx = torch.topk(flat, total_k, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[topk_idx] = True
    mask = mask.view(acts.shape)
    return acts * mask


@dataclass
class SAELossOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    aux_loss: torch.Tensor
    per_shell_mse: Dict[int, torch.Tensor]
    codes: torch.Tensor  # post batch-topk codes, (n_tokens, width)
    fraction_variance_explained: torch.Tensor
    l0: torch.Tensor  # mean number of active features per token
    dead_fraction: torch.Tensor


class MatryoshkaBatchTopKSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        width: int,
        shells: List[int],
        k: int,
        aux_loss_coef: float = 1.0 / 32.0,
        seed: int = 0,
    ):
        super().__init__()
        if not shells:
            raise ValueError("shells must be non-empty")
        if shells[-1] > width:
            raise ValueError("largest shell cannot exceed width")
        if sorted(shells) != list(shells):
            raise ValueError("shells must be sorted ascending")
        self.d_in = d_in
        self.width = width
        self.shells = list(shells)
        self.k = k
        self.aux_loss_coef = aux_loss_coef

        gen = torch.Generator().manual_seed(seed)
        w_enc = torch.randn(d_in, width, generator=gen) / math.sqrt(d_in)
        w_dec = torch.randn(width, d_in, generator=gen) / math.sqrt(width)
        w_dec = w_dec / w_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)

        self.W_enc = nn.Parameter(w_enc)
        self.b_enc = nn.Parameter(torch.zeros(width))
        self.W_dec = nn.Parameter(w_dec)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Tokens seen since each feature last fired; used to define "dead".
        self.register_buffer("tokens_since_fired", torch.zeros(width, dtype=torch.long))

    def normalize_decoder_(self) -> None:
        """Re-project decoder rows to unit norm. Call after every optimizer
        step: without this the decoder norm can drift and make L0 / dose
        calibration meaningless."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(norms)

    def encode_preact(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x @ self.W_enc + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encode_preact -> batch_topk, i.e. the post-sparsity codes used
        everywhere else in this class."""
        return batch_topk(self.encode_preact(x), self.k)

    def decode(self, codes: torch.Tensor, n_features: Optional[int] = None) -> torch.Tensor:
        if n_features is None:
            return codes @ self.W_dec + self.b_dec
        return codes[:, :n_features] @ self.W_dec[:n_features, :] + self.b_dec

    def reconstruct(
        self,
        x: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
        n_features: Optional[int] = None,
    ) -> torch.Tensor:
        """The single reconstruction path shared by `forward_loss` (its main,
        full-shell reconstruction) and `checks.sae_replace_check`, so the two
        can never diverge. Defaults to the outermost matryoshka shell
        (`self.shells[-1]`, which may be less than `self.width` if the
        widest shell doesn't span the full SAE) -- pass `codes` to reuse an
        already-computed encoding instead of recomputing it from `x`."""
        if codes is None:
            if x is None:
                raise ValueError("reconstruct requires x or codes")
            codes = self.encode(x)
        if n_features is None:
            n_features = self.shells[-1]
        return self.decode(codes, n_features=n_features)

    def update_dead_stats(self, codes: torch.Tensor, dead_window_tokens: int) -> torch.Tensor:
        """codes: (n_tokens, width) post-topk. Returns a bool dead mask and
        updates the running tokens-since-fired counters."""
        n_tokens = codes.shape[0]
        fired = (codes > 0).any(dim=0)
        with torch.no_grad():
            self.tokens_since_fired += n_tokens
            self.tokens_since_fired[fired] = 0
        return self.tokens_since_fired >= dead_window_tokens

    def forward_loss(self, x: torch.Tensor, dead_window_tokens: int) -> SAELossOutput:
        preact = self.encode_preact(x)
        codes = batch_topk(preact, self.k)

        per_shell_mse: Dict[int, torch.Tensor] = {}
        recon_losses = []
        main_recon = None
        for shell in self.shells:
            recon = self.reconstruct(codes=codes, n_features=shell)
            mse = F.mse_loss(recon, x)
            per_shell_mse[shell] = mse
            recon_losses.append(mse)
            if shell == self.shells[-1]:
                main_recon = recon
        recon_loss = torch.stack(recon_losses).mean()

        dead_mask = self.update_dead_stats(codes, dead_window_tokens)
        aux_loss = x.new_zeros(())
        if main_recon is not None and bool(dead_mask.any()):
            residual = (x - main_recon).detach()
            dead_preact = preact * dead_mask.to(preact.dtype)
            k_aux = min(self.k, int(dead_mask.sum().item()))
            if k_aux > 0:
                dead_codes = batch_topk(dead_preact, k_aux)
                aux_recon = dead_codes @ self.W_dec
                aux_loss = F.mse_loss(aux_recon, residual)

        loss = recon_loss + self.aux_loss_coef * aux_loss

        with torch.no_grad():
            total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-8)
            resid = x - main_recon if main_recon is not None else x
            resid_var = resid.var(dim=0, unbiased=False).sum()
            fve = 1.0 - (resid_var / total_var)
            l0 = (codes > 0).float().sum(dim=1).mean()
            dead_fraction = dead_mask.float().mean()

        return SAELossOutput(
            loss=loss,
            recon_loss=recon_loss,
            aux_loss=aux_loss,
            per_shell_mse=per_shell_mse,
            codes=codes,
            fraction_variance_explained=fve,
            l0=l0,
            dead_fraction=dead_fraction,
        )

    # -- persistence ---------------------------------------------------
    def state_dict_safetensors(self) -> Dict[str, torch.Tensor]:
        return {
            "W_enc": self.W_enc.detach().cpu().contiguous(),
            "b_enc": self.b_enc.detach().cpu().contiguous(),
            "W_dec": self.W_dec.detach().cpu().contiguous(),
            "b_dec": self.b_dec.detach().cpu().contiguous(),
        }

    def save(self, path: "str | Path", feature_stats: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _safetensors_save_file is None:
            raise RuntimeError("safetensors is required to save the SAE")
        _safetensors_save_file(self.state_dict_safetensors(), str(path))
        if feature_stats is not None:
            stats_path = path.with_name("feature_stats.json")
            with stats_path.open("w", encoding="utf-8") as fh:
                json.dump(feature_stats, fh, indent=2, sort_keys=True)

    @classmethod
    def load(
        cls,
        path: "str | Path",
        shells: List[int],
        k: int,
        aux_loss_coef: float = 1.0 / 32.0,
    ) -> "MatryoshkaBatchTopKSAE":
        if _safetensors_load_file is None:
            raise RuntimeError("safetensors is required to load the SAE")
        tensors = _safetensors_load_file(str(path))
        d_in = tensors["W_enc"].shape[0]
        width = tensors["W_enc"].shape[1]
        sae = cls(d_in=d_in, width=width, shells=shells, k=k, aux_loss_coef=aux_loss_coef)
        with torch.no_grad():
            sae.W_enc.copy_(tensors["W_enc"])
            sae.b_enc.copy_(tensors["b_enc"])
            sae.W_dec.copy_(tensors["W_dec"])
            sae.b_dec.copy_(tensors["b_dec"])
        return sae


class FeatureStatsTracker:
    """Tracks per-feature max activation (over all of training) and firing
    density (over the last `warmup_fraction` of training, i.e. the "last
    20%" the spec asks for), for `feature_stats.json`."""

    def __init__(self, width: int, warmup_fraction: float = 0.8):
        self.width = width
        self.warmup_fraction = warmup_fraction
        self.max_act = torch.zeros(width)
        self.fire_count = torch.zeros(width, dtype=torch.long)
        self.token_count = 0

    def observe(self, codes: torch.Tensor, progress_fraction: float) -> None:
        codes = codes.detach()
        n_tokens = codes.shape[0]
        batch_max = codes.max(dim=0).values.cpu()
        self.max_act = torch.maximum(self.max_act, batch_max)
        if progress_fraction >= self.warmup_fraction:
            fired = (codes > 0).cpu().sum(dim=0)
            self.fire_count += fired
            self.token_count += n_tokens

    def to_dict(self) -> dict:
        denom = max(self.token_count, 1)
        density = (self.fire_count.float() / denom).tolist()
        return {
            "max_activation": self.max_act.tolist(),
            "firing_density": density,
            "tokens_observed_for_density": self.token_count,
        }
