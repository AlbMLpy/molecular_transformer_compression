import torch
import tltorch
import torch.nn as nn
import torch.nn.functional as F


class TTLinear(nn.Module):
    """Linear layer with weights represented in Tensor-Train format.

    Uses TensorLy-Torch's factorized TT-matrix representation, avoiding
    reconstruction of the full dense weight matrix during the forward pass.
    """
    
    def __init__(
        self,
        in_features: tuple[int],
        out_features: tuple[int],
        rank: tuple[int],
        bias: bool = True,
        device=None, 
        dtype=None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.tt_linear = tltorch.FactorizedLinear(
            in_tensorized_features=in_features,
            out_tensorized_features=out_features,
            bias=bias,
            factorization="blocktt", # TTM representation
            rank=rank,
            implementation="factorized", # forward pass contracted core by core
            n_layers=1, 
            checkpointing=False,
            device=device,
            dtype=dtype
        )

    def forward(self, x):
        return self.tt_linear(x)


class LRLinear(nn.Module):
    """Low-rank linear layer with weight factorized into two matrices.

    Represents the weight as W ≈ B @ A, reducing the number of trainable
    parameters compared with a full dense linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.empty(out_features, rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, x):
        x = F.linear(x, self.A)
        x = F.linear(x, self.B, self.bias)
        return x

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        _str = (
            f"in_features={self.in_features}, " 
            + f"out_features={self.out_features}, "
            + f"rank={self.rank}, " 
            + f"bias={self.bias is not None}"
        )
        return _str


@torch.no_grad()
def dense_to_tt(
    dense_layer: nn.Linear,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    rank: tuple[int, ...],
) -> TTLinear:
    """Compress a (trained) nn.Linear into a TTLinear."""

    if len(input_shape) + 1 != len(rank):
        raise ValueError("rank must have len(input_shape) + 1 elements")

    if len(output_shape) != len(input_shape):
        raise ValueError("input/output_shape must have the same #modes")

    tt_layer = TTLinear(
        in_features=input_shape,
        out_features=output_shape,
        rank=rank,
        bias=dense_layer.bias is not None,
        device=dense_layer.weight.device,
        dtype=dense_layer.weight.dtype,
    )
    tt_layer.tt_linear = tltorch.FactorizedLinear.from_linear(
        dense_layer,
        rank=rank,
        auto_tensorize=False,
        in_tensorized_features=input_shape,
        out_tensorized_features=output_shape,
        bias=dense_layer.bias is not None,
        factorization="blocktt", # TTM representation
        implementation="factorized", # forward pass contracted core by core
    )
    return tt_layer


@torch.no_grad()
def dense_to_lr(dense_layer: nn.Linear, rank: int) -> LRLinear:
    """Compress a (trained) nn.Linear into a LRLinear."""
    W = dense_layer.weight.detach()
    device, dtype = W.device, W.dtype

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]
    lr_layer = LRLinear(
        in_features=dense_layer.in_features,
        out_features=dense_layer.out_features,
        rank=rank,
        bias=dense_layer.bias is not None,
    ).to(device=device, dtype=dtype)
    lr_layer.B.copy_(U_r * S_r.unsqueeze(0))
    lr_layer.A.copy_(Vh_r)
    
    if dense_layer.bias is not None:
        lr_layer.bias.copy_(dense_layer.bias)
    return lr_layer
