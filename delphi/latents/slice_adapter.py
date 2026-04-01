"""
Load SLICE / ModelConverter MoE checkpoints (e.g. DeepSpeed ZeRO under .../pytorch_model).

SLICE Python modules live under the repo's ``src/`` directory (``architectures``, ``base``, …).

Resolution order for the slice **repo root** (the folder that contains ``src/``):

1. Environment variable ``DELPHI_SLICE_ROOT`` or ``SLICE_REPO`` (absolute path on disk).
2. Otherwise ``<parent-of-delphi-repo>/slice`` (sibling checkout, e.g. ``SPAR/slice``).

On RunPod/Docker, clone slice and set e.g. ``export DELPHI_SLICE_ROOT=/workspace/slice``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel


def _slice_repo_root() -> Path:
    env = os.environ.get("DELPHI_SLICE_ROOT") or os.environ.get("SLICE_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent.parent / "slice"


_SLICE_SRC = _slice_repo_root() / "src"
if not _SLICE_SRC.is_dir():
    raise ModuleNotFoundError(
        f"SLICE checkout not found: expected directory {_SLICE_SRC!s} "
        "(clone the slice repo so it contains src/architectures, or set "
        "DELPHI_SLICE_ROOT to the slice repo root)."
    )
if str(_SLICE_SRC) not in sys.path:
    sys.path.insert(0, str(_SLICE_SRC))

import architectures  # noqa: F401, E402 — register MoE architectures
import initialization  # noqa: F401, E402
import losses  # noqa: F401, E402
from base import BaseMoEArchitecture, ModelConverter  # noqa: E402
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint  # noqa: E402
from schemas import TrainingConfig  # noqa: E402


def load_training_config(config_path_or_url: str) -> TrainingConfig:
    """Load JSON training config from a local path or http(s) URL."""
    if urlparse(config_path_or_url).scheme in ("http", "https"):
        with urlopen(config_path_or_url) as resp:  # noqa: S310 — user-supplied eval URL
            data = json.loads(resp.read().decode("utf-8"))
    else:
        with open(config_path_or_url, encoding="utf-8") as f:
            data = json.load(f)
    return TrainingConfig.from_dict(data)


def collect_moe_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """
    Return MoE modules in deterministic module-name order (for stable hookpoint indices).
    """
    named: list[tuple[str, torch.nn.Module]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, BaseMoEArchitecture):
            named.append((name, mod))
    named.sort(key=lambda x: x[0])
    return [m for _, m in named]


def load_slice_model(
    config_or_url: str | TrainingConfig,
    weights_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
) -> tuple[PreTrainedModel, list[torch.nn.Module], TrainingConfig]:
    """
    Build dense LM, convert to MoE per config, load ZeRO / DeepSpeed checkpoint.

    Args:
        config_or_url: Path or URL to training ``config.json``, or a loaded ``TrainingConfig``.
        weights_dir: Directory containing ``mp_rank_*_model_states.pt`` (and optim shards).
        device: Target device; default CUDA if available else CPU.
        torch_dtype: Model dtype before checkpoint merge; default float32 for merge stability.

    Returns:
        (model, moe_layers, training_config)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    if torch_dtype is None:
        torch_dtype = torch.float32

    if isinstance(config_or_url, TrainingConfig):
        training_config = config_or_url
    else:
        training_config = load_training_config(config_or_url)
    weights_path = Path(weights_dir)
    if not weights_path.is_dir():
        raise FileNotFoundError(f"Weights directory not found: {weights_path}")

    print(f"Loading base model: {training_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        training_config.model_name,
        torch_dtype=torch_dtype,
    )
    model = ModelConverter.convert_to_moe(model, training_config)

    print(f"Loading ZeRO checkpoint from {weights_path}")
    model = load_state_dict_from_zero_checkpoint(model, str(weights_path))

    model.to(device_t)
    model.eval()

    moe_layers = collect_moe_layers(model)
    print(f"Found {len(moe_layers)} MoE layer(s)")
    return model, moe_layers, training_config
