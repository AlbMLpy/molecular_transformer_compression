import torch.nn as nn

from src.utils import get_layer_names_linear, replace_linear_layers


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attn = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.output = nn.Linear(8, 1)


def test_get_layer_names_linear():
    model = SmallModel()
    names = get_layer_names_linear(model, "self_attn")
    assert names == ["self_attn.0", "self_attn.2"]


def test_replace_linear_layers():
    model = SmallModel()
    layer_names = get_layer_names_linear(model, "self_attn")
    def conversion_fn(layer):
        return nn.Linear(
            layer.in_features + 1,
            layer.out_features + 1,
            bias=layer.bias is not None,
        )

    replace_linear_layers(model, conversion_fn, layer_names)

    assert isinstance(model.self_attn[0], nn.Linear)
    assert isinstance(model.self_attn[2], nn.Linear)

    assert model.self_attn[0].in_features == 9
    assert model.self_attn[0].out_features == 9
    assert model.self_attn[2].in_features == 9
    assert model.self_attn[2].out_features == 9
