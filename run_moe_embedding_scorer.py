#!/usr/bin/env python
"""
Script to run embedding scoring on MoE model experts.

This uses the EmbeddingScorer which doesn't require an LLM - it uses
sentence transformers to compute similarity between examples and explanations.

Usage:
    python run_moe_embedding_scorer.py \
        --model_name facebook/opt-1.3b \
        --wrapper_path /workspace/slice/models/exp2/kmeans_13b_wrapper.pt \
        --latents_path results/kmeans-moe/latents \
        --output_path results/kmeans-moe/scores/embedding \
        --n_tokens 1000000 \
        --num_experts 256
"""

import argparse
import asyncio
import sys
from functools import partial
from pathlib import Path

import orjson
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add slice to path
slice_path = Path(__file__).parent.parent / "slice"
if str(slice_path) not in sys.path:
    sys.path.insert(0, str(slice_path))

from src.expert_construction.model_converter import ModelWrapper

from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig, RunConfig
from delphi.explainers.explainer import ExplainerResult
from delphi.latents import LatentDataset
from delphi.latents.cache_moe import MoELatentCache
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import EmbeddingScorer, DetectionScorer
from delphi.clients import Offline
from delphi.utils import load_tokenized_data


def _detection_scorer_preprocess(result):
    """``LatentDataset`` yields ``LatentRecord``; full pipeline uses ``ExplainerResult``."""
    if isinstance(result, list):
        result = result[0]
    if isinstance(result, ExplainerResult):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active  # type: ignore[assignment]
        return record
    return result


def load_moe_model(model_name: str, wrapper_path: str, load_in_8bit: bool = False):
    """Load model and MoE wrapper."""
    if load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": "cuda"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=load_in_8bit) if load_in_8bit else None
        ),
        torch_dtype=dtype,
    )
    model.eval()

    # Load MoE wrapper
    wrapper_checkpoint = torch.load(wrapper_path, weights_only=False)
    wrapper = ModelWrapper(**wrapper_checkpoint["kwargs"])
    wrapper.load_state_dict(wrapper_checkpoint["state_dict"])
    wrapper.attach_layers(model)

    print(f"Loaded MoE wrapper from {wrapper_path}")
    print(f"Number of MoE layers: {len(wrapper.moe_layers)}")

    return model, wrapper


def cache_moe_activations(
    model,
    wrapper,
    tokenizer,
    output_path: Path,
    n_tokens: int = 10_000_000,
    batch_size: int = 32,
    ctx_len: int = 256,
    dataset_repo: str = "EleutherAI/SmolLM2-135M-10B",
    dataset_split: str = "train[:1%]",
    filter_bos: bool = True,
):
    """Cache MoE router activations."""
    output_path.mkdir(parents=True, exist_ok=True)

    cache_cfg = CacheConfig(
        dataset_repo=dataset_repo,
        dataset_split=dataset_split,
        cache_ctx_len=ctx_len,
        batch_size=batch_size,
        n_tokens=n_tokens,
    )

    tokens = load_tokenized_data(
        cache_cfg.cache_ctx_len,
        tokenizer,
        cache_cfg.dataset_repo,
        cache_cfg.dataset_split,
        cache_cfg.dataset_name,
        cache_cfg.dataset_column,
        seed=42,
    )

    if filter_bos:
        if tokenizer.bos_token_id is not None:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(flattened_tokens, torch.tensor([tokenizer.bos_token_id]))
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = MoELatentCache(
        model,
        wrapper,
        batch_size=batch_size,
        top_k_only=True,
    )
    cache.run(cache_cfg.n_tokens, tokens)
    cache.save_splits(n_splits=cache_cfg.n_splits, save_dir=output_path)
    cache.save_config(save_dir=output_path, cfg=cache_cfg, model_name=model_name)

    print(f"Cached activations saved to {output_path}")
    return [f"moe_layer_{i}" for i in range(len(wrapper.moe_layers))]


def run_embedding_scorer(
    latents_path: Path,
    output_path: Path,
    tokenizer,
    hookpoints: list[str],
    latent_range: torch.Tensor | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Run embedding scorer on cached activations."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Load embedding model
    print(f"Loading embedding model: {embedding_model}")
    emb_model = SentenceTransformer(embedding_model)

    # Create dataset
    if latent_range is not None:
        latent_dict = {hook: latent_range for hook in hookpoints}
    else:
        latent_dict = None

    sampler_cfg = SamplerConfig()
    constructor_cfg = ConstructorConfig(
        non_activating_source="random",
    )

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    # Create scorer
    scorer = EmbeddingScorer(model=emb_model, verbose=True)

    def scorer_postprocess(result, score_dir: Path):
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = process_wrapper(
        scorer,
        postprocess=partial(scorer_postprocess, score_dir=output_path),
    )

    # Run pipeline
    print("Running embedding scorer pipeline...")
    pipeline = Pipeline(dataset, scorer_pipe)
    asyncio.run(pipeline.run(max_concurrent=4))

    print(f"Scores saved to {output_path}")


def run_detection_scorer(
    latents_path: Path,
    output_path: Path,
    tokenizer,
    hookpoints: list[str],
    latent_range: torch.Tensor | None = None,
    explainer_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    num_gpus: int = 1,
    max_memory: float = 0.7,
):
    """Run detection scorer on cached activations (requires LLM)."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Load LLM client
    print(f"Loading explainer model: {explainer_model}")
    client = Offline(
        explainer_model,
        max_memory=max_memory,
        max_model_len=4096,
        num_gpus=num_gpus,
    )

    # Create dataset
    if latent_range is not None:
        latent_dict = {hook: latent_range for hook in hookpoints}
    else:
        latent_dict = None

    sampler_cfg = SamplerConfig()
    constructor_cfg = ConstructorConfig()

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    # Create scorer
    scorer = DetectionScorer(
        client=client,
        n_examples_shown=5,
        verbose=True,
    )

    def scorer_postprocess(result, score_dir: Path):
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = process_wrapper(
        scorer,
        preprocess=_detection_scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=output_path),
    )

    # Run pipeline
    print("Running detection scorer pipeline...")
    pipeline = Pipeline(dataset, scorer_pipe)
    asyncio.run(pipeline.run(max_concurrent=1))

    print(f"Scores saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run embedding/detection scoring on MoE experts")
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--wrapper_path", type=str, required=True)
    parser.add_argument("--latents_path", type=str, default=None,
                        help="Path to cached activations (if None, will cache)")
    parser.add_argument("--output_path", type=str, default="results/moe_scores")
    parser.add_argument("--n_tokens", type=int, default=10_000_000)
    parser.add_argument("--num_experts", type=int, default=256,
                        help="Number of experts per MoE layer")
    parser.add_argument("--max_experts", type=int, default=None,
                        help="Max experts to score (for testing)")
    parser.add_argument("--scorer", type=str, choices=["embedding", "detection", "both"],
                        default="embedding")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--explainer_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--skip_cache", action="store_true",
                        help="Skip caching if latents_path exists")

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Determine latents path
    if args.latents_path:
        latents_path = Path(args.latents_path)
    else:
        latents_path = Path(args.output_path) / "latents"

    # Cache activations if needed
    if not args.skip_cache and not latents_path.exists():
        print("\n=== Caching MoE Activations ===")
        model, wrapper = load_moe_model(
            args.model_name,
            args.wrapper_path,
            args.load_in_8bit
        )
        hookpoints = cache_moe_activations(
            model, wrapper, tokenizer,
            latents_path,
            n_tokens=args.n_tokens
        )
        del model, wrapper
        torch.cuda.empty_cache()
    else:
        # Infer hookpoints from existing cache
        hookpoints = [d.name for d in latents_path.iterdir() if d.is_dir()]
        print(f"Using existing cache at {latents_path}")
        print(f"Found hookpoints: {hookpoints}")

    # Determine expert range
    if args.max_experts:
        latent_range = torch.arange(args.max_experts)
    else:
        latent_range = None

    # Run scorers
    output_path = Path(args.output_path)

    if args.scorer in ["embedding", "both"]:
        print("\n=== Running Embedding Scorer ===")
        run_embedding_scorer(
            latents_path,
            output_path / "scores" / "embedding",
            tokenizer,
            hookpoints,
            latent_range,
            args.embedding_model,
        )

    if args.scorer in ["detection", "both"]:
        print("\n=== Running Detection Scorer ===")
        run_detection_scorer(
            latents_path,
            output_path / "scores" / "detection",
            tokenizer,
            hookpoints,
            latent_range,
            args.explainer_model,
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
