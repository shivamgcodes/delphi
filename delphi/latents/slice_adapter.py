"""
Load SLICE / ModelConverter MoE checkpoints (e.g. DeepSpeed ZeRO under .../pytorch_model).

Two slice repo layouts are supported:

1. **Flat ``src/``** (original): ``src/architectures``, ``base.py``, ``schemas.py`` on ``sys.path``
   via ``<repo>/src``.
2. **Packaged ``src/expert_construction``** (refactor): repo root on ``sys.path``; imports use
   ``from src.expert_construction…`` (same convention as ``run_moe_embedding_scorer.py``).

Resolution order for the slice **repo root** (the folder that contains ``src/``):

1. Environment variable ``DELPHI_SLICE_ROOT`` or ``SLICE_REPO`` (absolute path on disk).
2. Otherwise ``<parent-of-delphi-repo>/slice`` (sibling checkout, e.g. ``SPAR/slice``).

On RunPod/Docker, clone slice and set e.g. ``export DELPHI_SLICE_ROOT=/workspace/slice``.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel


def _slice_repo_root() -> Path:
    env = os.environ.get("DELPHI_SLICE_ROOT") or os.environ.get("SLICE_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent.parent.parent / "slice"


def _duck_type_moe_layer(module: nn.Module) -> bool:
    """Detect SLICE MoE blocks when ``BaseMoEArchitecture`` is not importable (slim repos)."""
    return hasattr(module, "_last_routing_probs") and hasattr(module, "num_experts")


_moe_layer_predicate: Callable[[nn.Module], bool] = _duck_type_moe_layer


def _set_moe_layer_predicate(base_cls: type | None) -> None:
    global _moe_layer_predicate
    if base_cls is not None:

        def _isinstance_moe(m: nn.Module) -> bool:
            return isinstance(m, base_cls)

        _moe_layer_predicate = _isinstance_moe
    else:
        _moe_layer_predicate = _duck_type_moe_layer


def _slice_layout_hint(repo_root: Path) -> str:
    msg = (
        f"Set DELPHI_SLICE_ROOT to the slice repo root. Expected either "
        f"src/architectures (flat layout) or src/expert_construction/ (packaged layout). "
        f"Tried {repo_root!s}."
    )
    if repo_root.is_dir():
        msg += f" Top-level: {sorted(p.name for p in repo_root.iterdir())[:40]!r}."
        src = repo_root / "src"
        if src.is_dir():
            msg += f" src/: {sorted(p.name for p in src.iterdir())[:40]!r}."
    return msg


def _resolve_slice_stack() -> tuple[type, type, type]:
    """
    Configure ``sys.path`` and import MoE stack symbols.

    Returns:
        ``(BaseMoEArchitecture, ModelConverter, TrainingConfig)``
    """
    root = _slice_repo_root().resolve()
    src = root / "src"
    ec = src / "expert_construction"

    if src.is_dir() and (src / "architectures").is_dir():
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import architectures  # noqa: F401 — register MoE modules
        import initialization  # noqa: F401
        import losses  # noqa: F401
        from base import BaseMoEArchitecture, ModelConverter
        from schemas import TrainingConfig

        _set_moe_layer_predicate(BaseMoEArchitecture)
        return BaseMoEArchitecture, ModelConverter, TrainingConfig

    if ec.is_dir():
        return _resolve_expert_construction_stack(root)

    raise ModuleNotFoundError(_slice_layout_hint(root))


def _resolve_expert_construction_stack(
    root: Path,
) -> tuple[type | None, type, type]:
    """
    Slim ``src/expert_construction`` checkouts often omit ``base.py``; load symbols from
    wherever the fork places them and fall back to duck typing for MoE layer discovery.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for sub in ("architectures", "initialization", "losses"):
        with suppress(ImportError):
            importlib.import_module(f"src.expert_construction.{sub}")

    model_converter_mod = None
    ModelConverter = None
    for dotted in (
        "src.expert_construction.model_converter",
        "src.expert_construction.model_conversion",
        "src.expert_construction.converter",
    ):
        with suppress(ImportError):
            mc = importlib.import_module(dotted)
            cand = getattr(mc, "ModelConverter", None)
            if cand is not None:
                ModelConverter = cand
                model_converter_mod = mc
                break
    if ModelConverter is None:
        raise ModuleNotFoundError(
            "Could not import ModelConverter from src.expert_construction "
            "(tried model_converter, model_conversion, converter)."
        )

    BaseMoEArchitecture: type | None = getattr(
        model_converter_mod, "BaseMoEArchitecture", None
    )
    if BaseMoEArchitecture is None:
        for dotted in (
            "src.expert_construction.base",
            "src.expert_construction.moe_base",
            "src.expert_construction.core.base",
        ):
            with suppress(ImportError, AttributeError):
                mod = importlib.import_module(dotted)
                BaseMoEArchitecture = getattr(mod, "BaseMoEArchitecture", None)
                if BaseMoEArchitecture is not None:
                    break

    TrainingConfig = None
    for dotted in (
        "src.expert_construction.schemas",
        "src.expert_construction.config",
        "src.expert_construction.training_config",
    ):
        with suppress(ImportError, AttributeError):
            mod = importlib.import_module(dotted)
            TrainingConfig = getattr(mod, "TrainingConfig", None)
            if TrainingConfig is not None:
                break
    if TrainingConfig is None:
        raise ModuleNotFoundError(
            "Could not import TrainingConfig from src.expert_construction "
            "(tried schemas, config, training_config)."
        )

    _set_moe_layer_predicate(BaseMoEArchitecture)
    return BaseMoEArchitecture, ModelConverter, TrainingConfig


BaseMoEArchitecture, ModelConverter, TrainingConfig = _resolve_slice_stack()

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint


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
        if _moe_layer_predicate(mod):
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
