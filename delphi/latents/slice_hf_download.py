"""
Download SLICE training config and DeepSpeed ZeRO shards from Hugging Face when missing.

Default repo: ``SharmaLlama12/smol_final_checkpoints``.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_HF_REPO = "SharmaLlama12/smol_final_checkpoints"
DEFAULT_HF_EXPERIMENT = "smol_fineweb_edu_streaming-2048-2-heads-72-segments-exp-9"


def pytorch_model_has_weights(pytorch_model_dir: Path) -> bool:
    """True if directory exists and contains at least one DeepSpeed model shard."""
    if not pytorch_model_dir.is_dir():
        return False
    return any(pytorch_model_dir.glob("mp_rank_*_model_states.pt"))


def default_config_path(cache_dir: Path, experiment: str) -> Path:
    return Path(cache_dir) / experiment / "config.json"


def default_pytorch_model_dir(cache_dir: Path, experiment: str) -> Path:
    return Path(cache_dir) / experiment / "checkpoint_final" / "pytorch_model"


def ensure_slice_hf_checkpoint(
    *,
    repo_id: str = DEFAULT_HF_REPO,
    experiment: str = DEFAULT_HF_EXPERIMENT,
    cache_dir: str | Path,
    token: str | bool | None = None,
    revision: str | None = None,
) -> tuple[Path, Path]:
    """
    Ensure ``{experiment}/config.json`` and ``{experiment}/checkpoint_final/pytorch_model/*``
    exist under ``cache_dir``; download from the Hub if anything is missing.

    Returns:
        ``(config_path, pytorch_model_dir)`` as concrete ``Path`` objects.
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_path = default_config_path(cache_dir, experiment)
    pytorch_dir = default_pytorch_model_dir(cache_dir, experiment)

    need_config = not config_path.is_file()
    need_weights = not pytorch_model_has_weights(pytorch_dir)

    if need_config or need_weights:
        if need_config:
            print(f"Missing {config_path.name} — downloading from {repo_id} …")
        if need_weights:
            print(f"Missing ZeRO shards under …/pytorch_model — downloading from {repo_id} …")

        allow_patterns = [
            f"{experiment}/config.json",
            f"{experiment}/checkpoint_final/pytorch_model/*",
        ]
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            token=token,
            revision=revision,
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected config after download: {config_path}\n"
            f"Check repo layout for {repo_id} and experiment {experiment!r}."
        )
    if not pytorch_model_has_weights(pytorch_dir):
        raise FileNotFoundError(
            f"Expected mp_rank_*_model_states.pt under {pytorch_dir} after download.\n"
            f"See: https://huggingface.co/{repo_id}/tree/main/{experiment}/checkpoint_final/pytorch_model"
        )

    return config_path, pytorch_dir


def resolve_config_and_weights(
    *,
    config_arg: str | None,
    weights_arg: str | None,
    hf_repo: str,
    hf_experiment: str,
    hf_cache_dir: str | Path,
    hf_token: str | bool | None = None,
    hf_revision: str | None = None,
) -> tuple[str, Path]:
    """
    Resolve config (filesystem path or URL string) and local ZeRO weights directory.

    Omitted ``config_arg`` / ``weights_arg`` resolve to paths under
    ``hf_cache_dir / hf_experiment /``. Missing files there are downloaded from the Hub.

    Custom ``--weights`` outside that layout must already contain shards; no automatic
    download into arbitrary directories.
    """
    cache_dir = Path(hf_cache_dir).expanduser().resolve()
    default_cfg = default_config_path(cache_dir, hf_experiment)
    default_w = default_pytorch_model_dir(cache_dir, hf_experiment)

    config_is_url = bool(
        config_arg
        and (
            config_arg.startswith("http://") or config_arg.startswith("https://")
        )
    )

    if config_is_url:
        weights_path = (
            Path(weights_arg).expanduser().resolve()
            if weights_arg
            else default_w
        )
        if not pytorch_model_has_weights(weights_path):
            if weights_arg and weights_path.resolve() != default_w.resolve():
                raise FileNotFoundError(
                    f"Missing ZeRO shards under {weights_path}. "
                    "Omit --weights to use the Hub cache path, or place "
                    "mp_rank_*_model_states.pt there."
                )
            ensure_slice_hf_checkpoint(
                repo_id=hf_repo,
                experiment=hf_experiment,
                cache_dir=cache_dir,
                token=hf_token,
                revision=hf_revision,
            )
            weights_path = default_w
        return config_arg, weights_path  # type: ignore[return-value]

    cfg_path = Path(config_arg).expanduser().resolve() if config_arg else default_cfg
    weights_path = Path(weights_arg).expanduser().resolve() if weights_arg else default_w

    need_fetch = (cfg_path.resolve() == default_cfg.resolve() and not cfg_path.is_file()) or (
        weights_path.resolve() == default_w.resolve()
        and not pytorch_model_has_weights(weights_path)
    )

    if need_fetch:
        ensure_slice_hf_checkpoint(
            repo_id=hf_repo,
            experiment=hf_experiment,
            cache_dir=cache_dir,
            token=hf_token,
            revision=hf_revision,
        )

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not pytorch_model_has_weights(weights_path):
        raise FileNotFoundError(
            f"No mp_rank_*_model_states.pt in {weights_path}. "
            "Use the default Hub cache (--hf_cache_dir + omit --weights) or download ZeRO files."
        )

    return str(cfg_path), weights_path
