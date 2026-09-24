import torch


def count_parameters(model) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def rmse(pred, target):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    return torch.sqrt(torch.mean((pred - target) ** 2))
