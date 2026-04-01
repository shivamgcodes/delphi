"""
Cache SLICE / CrossSegment MoE `_last_routing_probs` for Delphi (same format as MoELatentCache).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel

from delphi import logger
from delphi.config import CacheConfig
from delphi.latents.cache import InMemoryCache

RoutingMode = Literal["expert_probs", "segment_expert"]

token_tensor_type = Int[Tensor, "batch sequence"]


def _routing_to_latents(
    raw: Tensor,
    mode: RoutingMode,
) -> Tensor:
    """
    raw: [B, T, H, M, N] or [B, T, M, N] (if H already reduced).
    Returns [B, T, L] with L = N (expert_probs) or M*N (segment_expert).
    """
    if raw.dim() == 5:
        g_sum = raw.sum(dim=2)
    elif raw.dim() == 4:
        g_sum = raw
    else:
        raise ValueError(f"Expected 4D or 5D routing tensor, got shape {tuple(raw.shape)}")

    if mode == "expert_probs":
        x = g_sum.mean(dim=2)
        denom = x.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = x / denom
        return x
    if mode == "segment_expert":
        b, t, m, n = g_sum.shape
        return g_sum.reshape(b, t, m * n)

    raise ValueError(f"Unknown routing mode: {mode}")


def _latent_width_for_layer(
    raw_shape: tuple[int, ...],
    mode: RoutingMode,
) -> int:
    if mode == "expert_probs":
        return raw_shape[-1]
    if mode == "segment_expert":
        if len(raw_shape) == 5:
            _, _, _, m, n = raw_shape
            return m * n
        _, _, m, n = raw_shape
        return m * n
    raise ValueError(mode)


class SliceLatentCache:
    """
    Cache router-derived activations from SLICE MoE layers into Delphi safetensors layout.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        moe_layers: list[torch.nn.Module],
        batch_size: int,
        routing_mode: RoutingMode = "expert_probs",
        top_k_only: bool = True,
        top_k_cap: int | None = None,
        log_path: Path | None = None,
    ):
        self.model = model
        self.moe_layers = moe_layers
        self.batch_size = batch_size
        self.routing_mode: RoutingMode = routing_mode
        self.top_k_only = top_k_only
        self.top_k_cap = top_k_cap
        self.cache = InMemoryCache(batch_size=batch_size)
        self.log_path = log_path
        self.width: int | None = None

    def _effective_top_k(self, moe_layer: torch.nn.Module, width: int) -> int:
        k = getattr(moe_layer, "top_k", width)
        if self.top_k_cap is not None:
            k = min(k, self.top_k_cap)
        return min(int(k), width)

    @staticmethod
    def _sparsify_topk(router_probs: Tensor, k: int) -> Tensor:
        topk_vals, topk_idx = router_probs.topk(k, dim=-1)
        sparse = torch.zeros_like(router_probs)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def load_token_batches(
        self, n_tokens: int, tokens: token_tensor_type
    ) -> list[token_tensor_type]:
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]
        n_mini_batches = len(tokens) // self.batch_size
        return [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

    def run(self, n_tokens: int, tokens: token_tensor_type):
        token_batches = self.load_token_batches(n_tokens, tokens)
        if not token_batches:
            logger.warning("No token batches to process (check n_tokens and batch_size).")
            return

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching SLICE router activations") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    self.model(batch.to(self.model.device))

                    for layer_idx, moe_layer in enumerate(self.moe_layers):
                        hookpoint = f"moe_layer_{layer_idx}"
                        raw = moe_layer._last_routing_probs
                        if raw is None:
                            logger.warning(
                                "No _last_routing_probs for %s; skipping batch slice.",
                                hookpoint,
                            )
                            continue

                        latents = _routing_to_latents(raw, self.routing_mode)
                        if latents.dim() != 3:
                            raise RuntimeError(
                                f"Expected [B,T,L] latents, got {tuple(latents.shape)}"
                            )

                        width = latents.shape[-1]
                        if self.width is None:
                            self.width = width
                        elif self.width != width:
                            raise ValueError(
                                f"Inconsistent latent width: {self.width} vs {width}"
                            )

                        if self.top_k_only:
                            k = self._effective_top_k(moe_layer, width)
                            if k < width:
                                latents = self._sparsify_topk(latents, k)

                        self.cache.add(latents, batch, batch_number, hookpoint)

                        moe_layer._last_routing_probs = None

                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def _generate_split_indices(self, n_splits: int) -> list[tuple[Tensor, Tensor]]:
        assert self.width is not None, "Width must be set before generating splits"
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        from safetensors.numpy import save_file
        import numpy as np

        split_indices = self._generate_split_indices(n_splits)

        for module_path in self.cache.latent_locations.keys():
            latent_locations = self.cache.latent_locations[module_path]
            latent_activations = self.cache.latent_activations[module_path]
            tokens = self.cache.tokens[module_path].numpy()

            latent_indices = latent_locations[:, 2]

            for start, end in split_indices:
                mask = (latent_indices >= start) & (latent_indices <= end)

                masked_activations = latent_activations[mask].half().numpy()
                masked_locations = latent_locations[mask].numpy()

                masked_locations[:, 2] = masked_locations[:, 2] - start.item()

                if (
                    masked_locations[:, 2].max() < 2**16
                    and masked_locations[:, 0].max() < 2**16
                ):
                    masked_locations = masked_locations.astype(np.uint16)
                else:
                    masked_locations = masked_locations.astype(np.uint32)
                    logger.warning(
                        "Increasing the number of splits might reduce the"
                        "memory usage of the cache."
                    )

                module_dir = save_dir / module_path
                module_dir.mkdir(parents=True, exist_ok=True)

                output_file = module_dir / f"{start}_{end}.safetensors"

                split_data = {
                    "locations": masked_locations,
                    "activations": masked_activations,
                }
                if save_tokens:
                    split_data["tokens"] = tokens

                save_file(split_data, output_file)

    def save_config(
        self,
        save_dir: Path,
        cfg: CacheConfig,
        model_name: str,
        *,
        routing_mode: str | None = None,
        slice_extra: dict | None = None,
    ):
        extra = slice_extra or {}
        if routing_mode is not None:
            extra["slice_routing_mode"] = routing_mode

        for module_path in self.cache.latent_locations.keys():
            config_file = save_dir / module_path / "config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                config_dict.update(extra)
                json.dump(config_dict, f, indent=4)
