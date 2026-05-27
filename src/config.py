"""Load and merge VL-JEPA YAML training configuration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_max_steps(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Set training.max_steps from epochs/samples/batch_size when null."""
    training = cfg.setdefault("training", {})
    max_steps = training.get("max_steps")
    if max_steps is None:
        data = cfg.get("data", {})
        samples = data.get("samples")
        if samples is not None:
            epochs = training.get("epochs", 1)
            batch_size = max(1, training.get("batch_size", data.get("batch_size", 1)))
            training["max_steps"] = epochs * max(1, samples // batch_size)
    return cfg


def load_config(
    config_path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load YAML config, apply overrides (CLI wins), resolve derived fields."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return _resolve_max_steps(cfg)


def overrides_from_cli(**kwargs: Any) -> Dict[str, Any]:
    """Build nested config overrides from explicit CLI values (None = skip)."""
    model: Dict[str, Any] = {}
    training: Dict[str, Any] = {}
    loss: Dict[str, Any] = {}
    data: Dict[str, Any] = {}
    output: Dict[str, Any] = {}

    mapping = {
        "hidden_dim": (model, "hidden_dim"),
        "patch_size": (model, "patch_size"),
        "image_size": (model, "image_size"),
        "mask_ratio": (model, "mask_ratio"),
        "text_mask_ratio": (model, "text_mask_ratio"),
        "predictor_layers": (model, "predictor_layers"),
        "momentum_tau": (model, "momentum_tau"),
        "momentum_tau_end": (training, "momentum_tau_end"),
        "momentum_schedule_steps": (training, "momentum_schedule_steps"),
        "epochs": (training, "epochs"),
        "batch_size": (training, "batch_size"),
        "learning_rate": (training, "learning_rate"),
        "weight_decay": (training, "weight_decay"),
        "warmup_steps": (training, "warmup_steps"),
        "max_steps": (training, "max_steps"),
        "use_multi_crop": (training, "use_multi_crop"),
        "global_crop_size": (training, "global_crop_size"),
        "local_crop_size": (training, "local_crop_size"),
        "max_grad_norm": (training, "max_grad_norm"),
        "alpha": (loss, "alpha"),
        "beta": (loss, "beta"),
        "gamma": (loss, "gamma"),
        "samples": (data, "samples"),
        "seq_len": (data, "seq_len"),
        "data_dir": (data, "data_dir"),
        "output_dir": (output, "output_dir"),
        "log_interval": (output, "log_interval"),
        "checkpoint_interval": (output, "checkpoint_interval"),
    }

    for key, value in kwargs.items():
        if value is None:
            continue
        section, field = mapping[key]
        section[field] = value

    out: Dict[str, Any] = {}
    if model:
        out["model"] = model
    if training:
        out["training"] = training
    if loss:
        out["loss"] = loss
    if data:
        out["data"] = data
    if output:
        out["output"] = output
    return out


def print_config(cfg: Dict[str, Any], indent: int = 2) -> None:
    """Pretty-print merged configuration."""
    prefix = " " * indent

    def _print(section: str, d: Mapping[str, Any]) -> None:
        print(f"{prefix}{section}:")
        for k, v in d.items():
            print(f"{prefix}  {k}: {v}")

    for section in ("model", "training", "loss", "data", "output"):
        if section in cfg:
            _print(section, cfg[section])
