import os
import math
import random
from pathlib import Path
from collections.abc import Callable

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn.init import (
    _no_grad_normal_,
    _no_grad_uniform_,
    _calculate_fan_in_and_fan_out,
)


def xavier_normal_small_init_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Initialize a tensor with a modified Xavier normal distribution."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Initialize a tensor with a modified Xavier uniform distribution."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from std
    return _no_grad_uniform_(tensor, -a, a)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible Python, NumPy, and PyTorch results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_module_parameters(model: nn.Module) -> None:
    """Print total and trainable parameter counts for each model module."""
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(
            p.numel()
            for p in module.parameters(recurse=False)
            if p.requires_grad
        )
        if params > 0:
            print(
                f"{name:60s} "
                f"total={params:10,d} "
                f"trainable={trainable:10,d}"
            )


def get_layer_names_linear(
    model: nn.Module, 
    target_layer: str,
) -> list[str]:
    """Return names of Linear layers matching a target module name.

    The function searches all named submodules and returns the names of
    modules that are instances of ``nn.Linear`` and whose full module name
    contains ``target_layer`` as a substring.

    For example, with ``target_layer="self_attn"``, this can return names
    such as ``"encoder.layers.0.self_attn.linears.0"``.

    Parameters
    ----------
    model : nn.Module
        Model whose Linear layers should be searched.
    target_layer : str
        Substring used to identify the target module. Examples include
        ``"self_attn"``, ``"feed_forward"``, and ``"src_embed"``.

    Returns
    -------
    list[str]
        Full module names of the matching ``nn.Linear`` layers.
    """
    names_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and target_layer in name:
            names_list.append(name)
    return names_list


def get_layer_names(model: nn.Module, target_layer: str) -> list[str]:
    return [
        name
        for name, _ in model.named_modules()
        if name and target_layer in name
    ]


def replace_linear_layers(
    model: nn.Module,
    conversion_fn: Callable[[nn.Module], nn.Module],
    layer_names: list[str],
) -> nn.Module:
    """Replace selected (Linear) layers using the provided conversion_fn."""
    for name in layer_names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_layer = getattr(parent, parts[-1])
        new_layer = conversion_fn(old_layer)
        setattr(parent, parts[-1], new_layer)
    return model


def get_module(model: nn.Module, name: str) -> nn.Module:
    """Return a nested module from its dot-separated name."""
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def update_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].append(value)


def create_dir_if_not_exists(directory: str) -> os.PathLike:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
