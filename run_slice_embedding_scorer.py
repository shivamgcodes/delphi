#!/usr/bin/env python
"""
Run Delphi embedding (or detection) scoring on SLICE ModelConverter checkpoints.

Loads training config + DeepSpeed ZeRO weights under ``checkpoint_final/pytorch_model``,
caches `_last_routing_probs` in two modes, then reuses the same pipeline as
``run_moe_embedding_scorer.py``.

Requires the **slice** repo: either flat ``src/architectures`` or packaged
``src/expert_construction``. Set ``DELPHI_SLICE_ROOT`` to the repo root if it is not a
sibling folder named ``slice`` next to the delphi repo.

Example (auto-download ``config.json`` + ``pytorch_model`` from the Hub if missing):

    python run_slice_embedding_scorer.py \\
        --routing_mode expert_probs \\
        --output_path results/slice_exp9 \\
        --n_tokens 500000

Override paths or use a config URL:

    python run_slice_embedding_scorer.py \\
        --config https://huggingface.co/.../config.json \\
        --hf_cache_dir ~/.cache/delphi/slice_hf

On RunPod-style hosts with ``/workspace``, this script sets (when unset) ``HF_HOME``,
``TMPDIR``, ``TORCHINDUCTOR_CACHE_DIR``, and ``TRITON_CACHE_DIR`` under
``/workspace/.cache/...`` so Hub data and vLLM/torch Inductor compiles do not fill the
small root ``/tmp``. Override with env vars if needed.

MMLU tokens (like slice's lm_eval ``mmlu_*_continuation`` style prompts):

    python run_slice_embedding_scorer.py \\
        --data_source mmlu \\
        --mmlu_subjects abstract_algebra,anatomy \\
        --mmlu_split test \\
        --routing_mode expert_probs \\
        --output_path results/slice_mmlu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from functools import partial
from pathlib import Path

import orjson
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from delphi.config import CacheConfig, ConstructorConfig, SamplerConfig
from delphi.latents import LatentDataset
from delphi.latents.cache_slice import RoutingMode, SliceLatentCache
from delphi.latents.slice_adapter import load_slice_model, load_training_config
from delphi.latents.slice_hf_download import (
    DEFAULT_HF_EXPERIMENT,
    DEFAULT_HF_REPO,
    resolve_config_and_weights,
)
from delphi.pipeline import Pipeline, process_wrapper
from delphi.scorers import DetectionScorer, EmbeddingScorer
from delphi.clients import Offline
from delphi.utils import load_tokenized_data
from delphi.latents.mmlu_tokens import MMLU_REPO, load_mmlu_tokenized_data


def _default_disk_base() -> Path:
    """Use ``/workspace`` when present (RunPod / vast.ai); else the user's home."""
    ws = Path("/workspace")
    return ws if ws.is_dir() else Path.home()


def _cache_inference_dtype(name: str) -> "torch.dtype":
    """Dtype for SLICE MoE forward during activation caching (after ZeRO merge)."""
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    # auto
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _configure_tmp_and_compiler_cache_for_workspace() -> None:
    """
    vLLM + torch.compile / Inductor + Triton write large temp files under ``/tmp``
    (e.g. ``/tmp/torchinductor_root/...``). On RunPod, ``/`` is often tiny and fills
    with errno 28 while ``/workspace`` is large. Redirect when ``/workspace`` exists
    and the user has not set these variables.
    """
    ws = Path("/workspace")
    if not ws.is_dir():
        return
    base = ws / ".cache"
    tmp = base / "tmp"
    inductor = base / "torchinductor"
    triton = base / "triton"
    for d in (tmp, inductor, triton):
        d.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("TMPDIR"):
        os.environ["TMPDIR"] = str(tmp)
        os.environ.setdefault("TEMP", str(tmp))
        os.environ.setdefault("TMP", str(tmp))
    if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)
    if not os.environ.get("TRITON_CACHE_DIR"):
        os.environ["TRITON_CACHE_DIR"] = str(triton)

    print(
        f"Compile/temp caches -> {base} (TMPDIR, TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR; "
        "avoids filling /tmp on small root disks)"
    )


def _configure_hf_home_for_workspace() -> None:
    """
    Dataset Hub blobs and ``from_pretrained`` caches default to ``HF_HOME`` (often
    ``/root/.cache/huggingface``), which fills the root FS on small RunPod images.
    If ``HF_HOME`` is unset/empty and ``/workspace`` exists, pin it under
    ``/workspace/.cache/huggingface``.
    """
    if os.environ.get("HF_HOME"):
        return
    ws = Path("/workspace")
    if not ws.is_dir():
        return
    hf_home = ws / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    print(
        f"HF_HOME -> {hf_home} (large Hub downloads; set HF_HOME yourself to override)"
    )


def _latent_hookpoint_dirs(latents_path: Path) -> list[str]:
    """Return sorted cache module names (subdirs, e.g. ``moe_layer_0``)."""
    if not latents_path.is_dir():
        return []
    return sorted(p.name for p in latents_path.iterdir() if p.is_dir())


def cache_slice_activations(
    model,
    moe_layers: list,
    tokenizer,
    output_path: Path,
    routing_mode: RoutingMode,
    n_tokens: int = 10_000_000,
    batch_size: int = 4,
    ctx_len: int = 256,
    dataset_repo: str = "EleutherAI/SmolLM2-135M-10B",
    dataset_split: str = "train[:10000]",
    data_source: str = "generic",
    mmlu_subjects: str = "all",
    mmlu_split: str = "test",
    filter_bos: bool = True,
    top_k_only: bool = True,
    top_k_cap: int | None = None,
    model_name_for_config: str = "",
    slice_extra: dict | None = None,
    datasets_cache_dir: str | None = None,
):
    output_path.mkdir(parents=True, exist_ok=True)

    merged_extra = dict(slice_extra or {})
    if data_source == "mmlu":
        merged_extra["cache_data_source"] = "mmlu"
        merged_extra["mmlu_subjects"] = mmlu_subjects
        merged_extra["mmlu_split"] = mmlu_split
        cache_cfg = CacheConfig(
            dataset_repo=MMLU_REPO,
            dataset_split=f"{mmlu_split}:{mmlu_subjects}",
            cache_ctx_len=ctx_len,
            batch_size=batch_size,
            n_tokens=n_tokens,
        )
        print(
            f"Loading MMLU tokens (subjects={mmlu_subjects!r}, split={mmlu_split!r}) …"
        )
        tokens = load_mmlu_tokenized_data(
            cache_cfg.cache_ctx_len,
            tokenizer,
            subjects=mmlu_subjects,
            split=mmlu_split,
            seed=42,
        )
    else:
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
            datasets_cache_dir=datasets_cache_dir,
        )

    if filter_bos:
        if tokenizer.bos_token_id is not None:
            flattened_tokens = tokens.flatten()
            mask = ~torch.isin(
                flattened_tokens, torch.tensor([tokenizer.bos_token_id])
            )
            masked_tokens = flattened_tokens[mask]
            truncated_tokens = masked_tokens[
                : len(masked_tokens) - (len(masked_tokens) % cache_cfg.cache_ctx_len)
            ]
            tokens = truncated_tokens.reshape(-1, cache_cfg.cache_ctx_len)

    cache = SliceLatentCache(
        model,
        moe_layers,
        batch_size=batch_size,
        routing_mode=routing_mode,
        top_k_only=top_k_only,
        top_k_cap=top_k_cap,
    )
    cache.run(cache_cfg.n_tokens, tokens)
    cache.save_splits(n_splits=cache_cfg.n_splits, save_dir=output_path)
    cache.save_config(
        save_dir=output_path,
        cfg=cache_cfg,
        model_name=model_name_for_config or "slice_moe",
        routing_mode=routing_mode,
        slice_extra=merged_extra,
    )

    print(f"Cached activations saved to {output_path}")
    return [f"moe_layer_{i}" for i in range(len(moe_layers))]


def run_embedding_scorer(
    latents_path: Path,
    output_path: Path,
    tokenizer,
    hookpoints: list[str],
    latent_range: torch.Tensor | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {embedding_model}")
    emb_model = SentenceTransformer(embedding_model)

    if latent_range is not None:
        latent_dict = {hook: latent_range for hook in hookpoints}
    else:
        latent_dict = None

    sampler_cfg = SamplerConfig()
    constructor_cfg = ConstructorConfig(non_activating_source="random")

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    scorer = EmbeddingScorer(model=emb_model, verbose=True)

    def scorer_postprocess(result, score_dir: Path):
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = process_wrapper(
        scorer,
        postprocess=partial(scorer_postprocess, score_dir=output_path),
    )

    print("Running embedding scorer pipeline...")
    pipeline = Pipeline(dataset, scorer_pipe)
    asyncio.run(pipeline.run(n_processes=4))

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
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading explainer model: {explainer_model}")
    client = Offline(
        explainer_model,
        max_memory=max_memory,
        max_model_len=4096,
        num_gpus=num_gpus,
    )

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

    scorer = DetectionScorer(
        client=client,
        n_examples_shown=5,
        verbose=True,
    )

    def scorer_preprocess(result):
        record = result.record
        record.explanation = result.explanation
        record.extra_examples = record.not_active
        return record

    def scorer_postprocess(result, score_dir: Path):
        safe_latent_name = str(result.record.latent).replace("/", "--")
        with open(score_dir / f"{safe_latent_name}.txt", "wb") as f:
            f.write(orjson.dumps(result.score))

    scorer_pipe = process_wrapper(
        scorer,
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=output_path),
    )

    print("Running detection scorer pipeline...")
    pipeline = Pipeline(dataset, scorer_pipe)
    asyncio.run(pipeline.run(n_processes=1))

    print(f"Scores saved to {output_path}")


def main():
    _configure_tmp_and_compiler_cache_for_workspace()
    _configure_hf_home_for_workspace()

    parser = argparse.ArgumentParser(
        description="Run embedding/detection scoring on SLICE MoE checkpoints"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path or URL to SLICE training config.json. "
        "If omitted, uses HF cache (see --hf_cache_dir / --hf_experiment).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Directory with ZeRO shards (.../checkpoint_final/pytorch_model). "
        "If omitted, uses HF cache; missing files are downloaded from --hf_repo.",
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=DEFAULT_HF_REPO,
        help="Hugging Face repo id for snapshot download",
    )
    parser.add_argument(
        "--hf_experiment",
        type=str,
        default=DEFAULT_HF_EXPERIMENT,
        help="Experiment folder name inside the repo (contains config.json)",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=str(_default_disk_base() / ".cache" / "delphi" / "slice_hf"),
        help="Local root where config.json and checkpoint_final/ are stored "
        "(default: under /workspace/.cache/... when /workspace exists)",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="Optional git revision (branch name, tag, or commit) for snapshot_download",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (default: env HF_TOKEN if set)",
    )
    parser.add_argument(
        "--routing_mode",
        type=str,
        choices=["expert_probs", "segment_expert"],
        default="expert_probs",
        help="expert_probs: sum heads, mean segments, renorm N. "
        "segment_expert: sum heads, flatten M*N.",
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        default=None,
        help="Existing cache dir; if missing and not --skip_cache, will run forward cache",
    )
    parser.add_argument("--output_path", type=str, default="results/slice_moe_scores")
    parser.add_argument("--n_tokens", type=int, default=10_000_000)
    parser.add_argument(
        "--num_experts",
        type=int,
        default=None,
        help="Override latent width; if omitted, read from cached config (slice_moe_*) or default N=192.",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=None,
        help="Max latent indices to score (subset of 0..num_experts-1 or 0..M*N-1)",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=["embedding", "detection", "both"],
        default="embedding",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--explainer_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Forward batch size during SLICE latent caching. Cross-segment MoE einsums "
        "scale ~linearly with this; try 2–8 if CUDA OOM (default 4).",
    )
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument(
        "--cache_inference_dtype",
        type=str,
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
        help="Model dtype while caching router activations only (after fp32 ZeRO merge). "
        "auto=bfloat16 on CUDA when supported (~half activation VRAM). float32 uses more memory.",
    )
    parser.add_argument(
        "--hf_datasets_cache",
        type=str,
        default=None,
        help="Passed as ``cache_dir`` to ``load_dataset``. Default: unset — then "
        "``datasets`` uses ``HF_HOME`` (see startup message when /workspace exists).",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="EleutherAI/SmolLM2-135M-10B",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:10000]",
        help="HF split slice. Default caps rows to limit parquet downloads (disk). "
        "Use e.g. train[:1%%] for more data if you have space.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["generic", "mmlu"],
        default="generic",
        help="generic: text dataset via --dataset_repo. "
        "mmlu: cais/mmlu with continuation-style prompts (cf. slice lm_eval mmlu_*_continuation).",
    )
    parser.add_argument(
        "--mmlu_subjects",
        type=str,
        default="all",
        help="Comma-separated cais/mmlu subject configs, or 'all' for all 57.",
    )
    parser.add_argument(
        "--mmlu_split",
        type=str,
        default="test",
        help="MMLU split passed to load_dataset (e.g. test, validation, dev).",
    )
    parser.add_argument(
        "--no_top_k_sparsify",
        action="store_true",
        help="Store full routing vectors (can be very large for segment_expert)",
    )
    parser.add_argument(
        "--top_k_cap",
        type=int,
        default=None,
        help="Cap top-k sparsification (per token, last dim)",
    )

    args = parser.parse_args()
    routing_mode: RoutingMode = args.routing_mode  # type: ignore[assignment]

    if args.hf_datasets_cache:
        datasets_cache_dir = str(Path(args.hf_datasets_cache).expanduser().resolve())
    elif os.environ.get("HF_HOME"):
        datasets_cache_dir = str(Path(os.environ["HF_HOME"]) / "datasets")
    else:
        datasets_cache_dir = None

    hf_token = args.hf_token if args.hf_token else os.environ.get("HF_TOKEN")
    if hf_token is not None and hf_token == "":
        hf_token = None

    config_src, weights_path = resolve_config_and_weights(
        config_arg=args.config,
        weights_arg=args.weights,
        hf_repo=args.hf_repo,
        hf_experiment=args.hf_experiment,
        hf_cache_dir=args.hf_cache_dir,
        hf_token=hf_token,
        hf_revision=args.hf_revision,
    )
    print(f"Using config: {config_src}")
    print(f"Using weights: {weights_path}")

    if args.latents_path:
        latents_path = Path(args.latents_path)
    else:
        latents_path = Path(args.output_path) / "latents"

    training_config = load_training_config(config_src)
    tokenizer_name = training_config.tokenizer_name or training_config.model_name

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model_name_label = training_config.model_name

    hookpoints_existing = _latent_hookpoint_dirs(latents_path)
    # Treat an empty or missing latents dir as "no cache" so we do not skip forward pass
    # just because ``latents/`` was created (e.g. by a failed run).
    need_slice_cache = not args.skip_cache and not hookpoints_existing

    if need_slice_cache:
        print("\n=== Caching SLICE router activations ===")
        model, moe_layers, train_cfg = load_slice_model(training_config, weights_path)
        cid = _cache_inference_dtype(args.cache_inference_dtype)
        if cid != torch.float32:
            print(f"Casting model to {cid} for caching (use --cache_inference_dtype float32 to disable)")
            model = model.to(dtype=cid)
        m_seg = train_cfg.moe.num_segments
        n_exp = train_cfg.moe.num_experts
        slice_extra = {
            "slice_moe_num_experts": n_exp,
            "slice_moe_num_segments": m_seg,
        }
        cache_slice_activations(
            model,
            moe_layers,
            tokenizer,
            latents_path,
            routing_mode=routing_mode,
            n_tokens=args.n_tokens,
            batch_size=args.batch_size,
            ctx_len=args.ctx_len,
            dataset_repo=args.dataset_repo,
            dataset_split=args.dataset_split,
            data_source=args.data_source,
            mmlu_subjects=args.mmlu_subjects,
            mmlu_split=args.mmlu_split,
            top_k_only=not args.no_top_k_sparsify,
            top_k_cap=args.top_k_cap,
            model_name_for_config=model_name_label,
            slice_extra=slice_extra,
            datasets_cache_dir=datasets_cache_dir,
        )
        del model, moe_layers
        torch.cuda.empty_cache()
    else:
        print(f"Using existing cache at {latents_path} (--skip_cache or cache exists)")

    hookpoints = _latent_hookpoint_dirs(latents_path)
    if not hookpoints:
        raise RuntimeError(
            f"No module subdirs (e.g. moe_layer_0) under {latents_path}. "
            "Delete or fix that folder, then re-run without --skip_cache so the SLICE forward "
            "cache is written, or pass --latents_path to a directory that already contains "
            "moe_layer_* subfolders."
        )

    def read_latent_width() -> int:
        cfg_json = latents_path / hookpoints[0] / "config.json"
        if cfg_json.is_file():
            with open(cfg_json, encoding="utf-8") as f:
                meta = json.load(f)
            mode = meta.get("slice_routing_mode", routing_mode)
            n_exp = meta.get("slice_moe_num_experts")
            m_seg = meta.get("slice_moe_num_segments")
            if mode == "segment_expert" and n_exp is not None and m_seg is not None:
                return int(m_seg) * int(n_exp)
            if n_exp is not None:
                return int(n_exp)
        if args.num_experts is not None:
            return int(args.num_experts)
        return 192

    width = read_latent_width()
    if args.num_experts is not None:
        width = int(args.num_experts)

    if args.max_experts is not None:
        latent_range = torch.arange(min(args.max_experts, width))
    else:
        latent_range = None

    output_path = Path(args.output_path)

    if args.scorer in ("embedding", "both"):
        print("\n=== Running Embedding Scorer ===")
        run_embedding_scorer(
            latents_path,
            output_path / "scores" / "embedding",
            tokenizer,
            hookpoints,
            latent_range,
            args.embedding_model,
        )

    if args.scorer in ("detection", "both"):
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
