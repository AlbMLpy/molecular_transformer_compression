from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .utils import get_module


def prune_linear_layers(
    model: nn.Module,
    layer_names: Sequence[str],
    amount: float,
) -> nn.Module:
    """Apply global magnitude pruning to selected Linear layer weights.

    PyTorch replaces each pruned ``weight`` parameter with two tensors:

    - ``weight_orig``: the original trainable weight parameter.
    - ``weight_mask``: a binary mask where 0 indicates a pruned weight and
      1 indicates a retained weight.

    During the forward pass, PyTorch applies the mask:

        weight = weight_orig * weight_mask

    Therefore, pruned weights contribute zero to the forward pass. During
    fine-tuning, the mask remains active, so pruned weights remain zero in
    the effective weight and do not receive an effective gradient.

    The original ``weight_orig`` parameters remain trainable, while the
    binary ``weight_mask`` determines which parameters are active.

    Parameters
    ----------
    model : nn.Module
        Model containing the layers to prune.
    layer_names : Sequence[str]
        Names of Linear layers to prune.
    amount : float
        Fraction of weights to prune globally, in [0, 1]. For example,
        ``amount=0.5`` removes approximately 50% of the selected weights
        based on their magnitude.

    Returns
    -------
    nn.Module
        The model with pruning masks applied.

    Notes
    -----
    The pruning does not reduce the dense tensor size. The original
    parameters are still stored in ``weight_orig``. To obtain actual
    memory or parameter-count reduction, a sparse representation or
    structured pruning method is required.

    The pruning mask can be made permanent with ``prune.remove()``,
    which removes the pruning reparameterization and leaves zeros in
    the weight tensor.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be between 0 and 1.")
    
    parameters_to_prune = []
    for name in layer_names:
        layer = get_module(model, name)

        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"Expected '{name}' to be nn.Linear, "
                f"got {type(layer).__name__}."
            )

        parameters_to_prune.append((layer, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def remove_pruning(
    model: nn.Module,
    layer_names: Sequence[str],
) -> nn.Module:
    """Make pruning masks permanent and remove pruning reparameterizations."""
    for name in layer_names:
        layer = get_module(model, name)
        if isinstance(layer, nn.Linear) and hasattr(layer, "weight_mask"):
            prune.remove(layer, "weight")
    return model


def get_sparsity(
    model: nn.Module,
    layer_names: Sequence[str],
) -> float:
    """Return the fraction of zero weights in selected layers."""
    total, zeros = 0, 0
    for name in layer_names:
        layer = get_module(model, name)
        weight = layer.weight
        total += weight.numel()
        zeros += torch.count_nonzero(weight == 0).item()
    if total == 0:
        return 0.0
    return zeros / total
