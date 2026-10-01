from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .utils import get_module


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return torch.sqrt(torch.mean((pred - target) ** 2))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable number of parameter."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def count_parameter_categories(model: nn.Module) -> dict[str, int]:
    """Return trainable number of parameters per category."""
    categories = {
        k:0 for k in ["self_attn", "feed_forward", "src_embed", "other"]
    }
    for name, param in model.named_parameters():
        if not param.requires_grad: 
            continue
        name_lower = name.lower()
        if "self_attn" in name_lower:
            categories["self_attn"] += param.numel()
        elif "feed_forward" in name_lower:
            categories["feed_forward"] += param.numel()
        elif "src_embed" in name_lower:
            categories["src_embed"] += param.numel()
        else:
            categories["other"] += param.numel()
    return categories


def count_effective_parameters(
    model: nn.Module,
    layer_names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Count stored, trainable, and effective non-zero parameters.

    For pruned layers, PyTorch stores the original weights as
    ``weight_orig`` together with a binary ``weight_mask``. The effective
    weight used during the forward pass is

        weight = weight_orig * weight_mask

    Therefore, the number of effective non-zero parameters is computed from
    the masked weights rather than from ``weight_orig``.

    Parameters
    ----------
    model : nn.Module
        PyTorch model whose parameters should be counted.
    layer_names : Sequence[str] or None, optional
        Names of specific modules to include. If ``None``, parameters from
        the entire model are counted.

    Returns
    -------
    dict[str, int]
        Dictionary containing:
        - ``"total"``: number of stored parameters.
        - ``"trainable"``: number of parameters with ``requires_grad=True``.
        - ``"nonzero"``: number of effective non-zero parameters after
          applying pruning masks.

    Notes
    -----
    For a pruned model, ``total`` and ``trainable`` can remain unchanged
    after pruning because ``weight_orig`` is still stored and trainable.
    The reduction is reflected in ``nonzero``, which counts only parameters
    that are active in the forward pass.

    For example, with 50% unstructured pruning:
        total     = 100,000
        trainable = 100,000
        nonzero   = 50,000
    The corresponding sparsity is therefore 50%.
    """
    if layer_names is None:
        modules = list(model.modules())
    else:
        modules = [get_module(model, name) for name in layer_names]

    total = trainable = nonzero = 0
    counted_parameters = set()
    for module in modules:
        # Handle pruned weights:
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            weight = module.weight_orig
            total += weight.numel() # Stored params
            if weight.requires_grad: # Trainable params
                trainable += weight.numel()
            effective_weight = weight * module.weight_mask # Effective params
            nonzero += torch.count_nonzero(effective_weight).item()
            counted_parameters.add(id(weight))
        # Handle other weights:
        for _, param in module.named_parameters(recurse=False):
            if id(param) in counted_parameters: 
                continue
            counted_parameters.add(id(param))
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
            nonzero += torch.count_nonzero(param.detach()).item()
    return {"total": total, "trainable": trainable, "nonzero": nonzero}
