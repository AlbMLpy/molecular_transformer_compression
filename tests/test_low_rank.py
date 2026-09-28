import sys
sys.path.append('./')

import torch
import torch.nn as nn

from src.low_rank import dense_to_lr, dense_to_tt


def relative_error(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(a)).item()


# ---------------------------------------------------------------------------
# LR
# ---------------------------------------------------------------------------

def test_dense_to_lr_output():
    torch.manual_seed(0)
    dense = nn.Linear(8, 8)
    lr = dense_to_lr(dense, rank=4)
    x = torch.randn(16, 8)
    assert lr(x).shape == (16, 8)


def test_dense_to_lr_approximates_dense():
    torch.manual_seed(0)
    dense = nn.Linear(8, 8)
    lr = dense_to_lr(dense, rank=4)
    x = torch.randn(16, 8)
    with torch.no_grad():
        y_dense, y_lr = dense(x), lr(x)
    error = relative_error(y_dense, y_lr)
    assert error < 0.5


# ---------------------------------------------------------------------------
# TT
# ---------------------------------------------------------------------------

def test_dense_to_tt_output():
    torch.manual_seed(0)

    dense = nn.Linear(8, 8)

    tt = dense_to_tt(
        dense,
        input_shape=(2, 2, 2),
        output_shape=(2, 2, 2),
        rank=(1, 2, 2, 1),
    )

    x = torch.randn(16, 8)

    assert tt(x).shape == (16, 8)


def test_dense_to_tt_approximates_dense():
    torch.manual_seed(0)

    dense = nn.Linear(8, 8)

    tt = dense_to_tt(
        dense,
        input_shape=(2, 2, 2),
        output_shape=(2, 2, 2),
        rank=(1, 4, 4, 1),
    )

    x = torch.randn(16, 8)

    with torch.no_grad():
        y_dense = dense(x)
        y_tt = tt(x)

    error = relative_error(y_dense, y_tt)

    assert error < 0.5
