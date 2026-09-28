import torch
import torch.nn as nn
from torch import Tensor


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
