import json
import sys
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedModel

from delphi import logger
from delphi.config import CacheConfig
from delphi.latents.cache import InMemoryCache

# Make slice package accessible (slice is at SPAR/slice, not SPAR/delphi/slice)
# slice uses "from src.expert_construction..." so we need /workspace/slice in path
slice_path = Path(__file__).parent.parent.parent.parent / "slice"
if str(slice_path) not in sys.path:
    sys.path.insert(0, str(slice_path))

from src.expert_construction.model_converter import ModelWrapper

token_tensor_type = Int[Tensor, "batch sequence"]


class MoELatentCache:
    """
    MoE-specific latent cache that stores router probabilities instead of SAE activations.
    Produces the same safetensors format as LatentCache for compatibility with downstream pipeline.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        wrapper: ModelWrapper,
        batch_size: int,
        top_k_only: bool = True,
        log_path: Path | None = None,
    ):
        """
        Initialize the MoELatentCache.

        Args:
            model: The model with MoE wrapper attached.
            wrapper: The ModelWrapper containing MoE layers.
            batch_size: Size of batches for processing.
            top_k_only: Whether to sparsify by keeping only top-k router probs.
            log_path: Path to save logging output.
        """
        self.model = model
        self.wrapper = wrapper
        self.batch_size = batch_size
        self.top_k_only = top_k_only
        self.width = wrapper.moe_layers[0].num_experts if len(wrapper.moe_layers) > 0 else None
        self.cache = InMemoryCache(batch_size=batch_size)
        self.log_path = log_path

    def _sparsify_topk(self, router_probs: Tensor, k: int) -> Tensor:
        """
        Zero out all but top-k entries per token position.

        Args:
            router_probs: Router probabilities [batch, seq, num_experts].
            k: Number of top experts to keep.

        Returns:
            Sparsified router probabilities with same shape.
        """
        topk_vals, topk_idx = router_probs.topk(k, dim=-1)
        sparse = torch.zeros_like(router_probs)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def load_token_batches(
        self, n_tokens: int, tokens: token_tensor_type
    ) -> list[token_tensor_type]:
        """
        Load and prepare token batches for processing.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.

        Returns:
            list[Tensor]: list of token batches.
        """
        max_batches = n_tokens // tokens.shape[1]
        tokens = tokens[:max_batches]

        n_mini_batches = len(tokens) // self.batch_size

        token_batches = [
            tokens[self.batch_size * i : self.batch_size * (i + 1), :]
            for i in range(n_mini_batches)
        ]

        return token_batches

    def run(self, n_tokens: int, tokens: token_tensor_type):
        """
        Run the MoE router probability caching process.

        Args:
            n_tokens: Total number of tokens to process.
            tokens: Input tokens.
        """
        token_batches = self.load_token_batches(n_tokens, tokens)

        total_tokens = 0
        total_batches = len(token_batches)
        tokens_per_batch = token_batches[0].numel()

        with tqdm(total=total_batches, desc="Caching MoE router probs") as pbar:
            for batch_number, batch in enumerate(token_batches):
                total_tokens += tokens_per_batch

                with torch.no_grad():
                    # Forward pass triggers MoE hooks which store _router_probs
                    self.model(batch.to(self.model.device))

                    # Collect router probs from each MoE layer
                    for layer_idx, moe_layer in enumerate(self.wrapper.moe_layers):
                        hookpoint = f"moe_layer_{layer_idx}"
                        router_probs = moe_layer._router_probs  # [batch, seq, num_experts]

                        # Optionally apply top-k sparsification
                        if self.top_k_only:
                            k = moe_layer.k
                            router_probs = self._sparsify_topk(router_probs, k)

                        # Add to cache (same interface as SAE latents)
                        self.cache.add(router_probs, batch, batch_number, hookpoint)

                        if self.width is None:
                            self.width = router_probs.shape[2]

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

        logger.info(f"Total tokens processed: {total_tokens:,}")
        self.cache.save()

    def _generate_split_indices(self, n_splits: int) -> list[tuple[Tensor, Tensor]]:
        """
        Generate indices for splitting the latent space.

        Args:
            n_splits: Number of splits to generate.

        Returns:
            list[tuple[int, int]]: list of start and end indices for each split.
        """
        assert self.width is not None, "Width must be set before generating splits"
        boundaries = torch.linspace(0, self.width, steps=n_splits + 1).long()

        # Adjust end by one
        return list(zip(boundaries[:-1], boundaries[1:] - 1))

    def save_splits(self, n_splits: int, save_dir: Path, save_tokens: bool = True):
        """
        Save the cached router probabilities in split safetensors files.

        Args:
            n_splits: Number of splits to generate.
            save_dir: Directory to save the splits.
            save_tokens: Whether to save the dataset tokens. Defaults to True.
        """
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

                # Optimization to reduce the max value to enable a smaller dtype
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

    def save_config(self, save_dir: Path, cfg: CacheConfig, model_name: str):
        """
        Save the configuration for the cached latents.

        Args:
            save_dir: Directory to save the configuration.
            cfg: Configuration object.
            model_name: Name of the model.
        """
        for module_path in self.cache.latent_locations.keys():
            config_file = save_dir / module_path / "config.json"
            with open(config_file, "w") as f:
                config_dict = cfg.to_dict()
                config_dict["model_name"] = model_name
                json.dump(config_dict, f, indent=4)
