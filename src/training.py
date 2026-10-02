import copy
import math
import time
from collections import defaultdict

import torch
import pandas as pd
import torch.nn as nn

from .utils import get_layer_names, update_results_dict
from .metrics import count_effective_parameters, count_parameter_categories


def prepare_batch(batch, device):
    """
    Move a MAT batch to the target device and construct the atom mask.

    Expected batch: adjacency_matrix, node_features, distance_matrix, y.
    """
    adjacency_matrix, node_features, distance_matrix, y = batch
    node_features = node_features.to(device)
    adjacency_matrix = adjacency_matrix.to(device)
    distance_matrix = distance_matrix.to(device)
    y = y.to(device).float()
    # True for real atoms, False for padded atoms.
    batch_mask = (torch.sum(torch.abs(node_features), dim=-1) != 0)
    return (
        adjacency_matrix,
        node_features,
        distance_matrix,
        batch_mask,
        y,
    )


def forward_batch(model, batch, device):
    """Run one MAT batch through the model."""
    (
        adjacency_matrix,
        node_features,
        distance_matrix,
        batch_mask,
        y,
    ) = prepare_batch(batch, device)
    
    output = model(
        node_features,
        batch_mask,
        adjacency_matrix,
        distance_matrix,
        None,
    )
    output = output.squeeze(-1)
    y = y.reshape_as(output)
    return output, y


def compute_rmse(sum_squared_error, n_samples):
    """Compute RMSE from accumulated squared error."""
    if n_samples == 0:
        raise ValueError("Cannot compute RMSE with zero samples.")
    return math.sqrt(sum_squared_error / n_samples)


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
):
    """Train the model for one epoch and return its RMSE."""
    model.train()
    sq_error, n_samples = 0.0, 0
    for batch in data_loader:
        output, y = forward_batch(model, batch, device)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sq_error += torch.sum((output.detach() - y) ** 2).item()
        n_samples += y.numel()
    return compute_rmse(sq_error, n_samples)


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device,
):
    """
    Evaluate a model on a dataset.

    Returns:
        RMSE
    """
    model.eval()
    squared_error, n_samples = 0.0, 0
    for batch in data_loader:
        output, y = forward_batch(model, batch, device)
        squared_error += torch.sum((output - y) ** 2).item()
        n_samples += y.numel()
    return compute_rmse(squared_error, n_samples)


def save_best_state(model):
    """Create a CPU copy of the model's current state."""
    return copy.deepcopy(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
    )


def should_stop(
    epoch,
    best_epoch,
    patience,
):
    """Return True when validation patience has been exhausted."""
    return epoch - best_epoch >= patience


def train(
    model,
    train_loader,
    valid_loader,
    device,
    epochs=100,
    lr=1e-3,
    weight_decay=0.0,
    patience=20,
    verbose=True,
):
    """
    Train a MAT model using validation RMSE for model selection.

    The test set is intentionally not used here.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    history = {"train_rmse": [], "valid_rmse": []}
    best_valid_rmse, best_epoch, best_state = float("inf"), 0, None

    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_rmse = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        valid_rmse = evaluate_model(
            model=model,
            data_loader=valid_loader,
            device=device,
        )
        history["train_rmse"].append(train_rmse)
        history["valid_rmse"].append(valid_rmse)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            best_state = save_best_state(model)

        if verbose and (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:3d} | "
                f"train RMSE {train_rmse:.4f} | "
                f"valid RMSE {valid_rmse:.4f}"
            )

        if should_stop(
            epoch,
            best_epoch,
            patience,
        ):
            break

    training_time = (time.perf_counter() - start_time)
    if best_state is None:
        raise RuntimeError(
            "No best model state was saved."
        )
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_valid_rmse": best_valid_rmse,
        "best_epoch": best_epoch,
        "epochs_trained": len(history["train_rmse"]),
        "training_time": training_time,
        "history": history,
    }


def get_params_count_df(
    model: nn.Module, 
    target_layers: list[str] = ["all"]
) -> pd.DataFrame:
    params_count_dict = defaultdict(list)
    for target_layer in target_layers:
        if target_layer == "all":
            _counter = count_effective_parameters(model)
        else:
            _counter = count_effective_parameters(
                model, get_layer_names(model, target_layer)
            )
        update_results_dict(
            params_count_dict, target_layer=target_layer, **_counter,
        )

    pd.DataFrame(count_parameter_categories(model)).T
    return pd.DataFrame(params_count_dict)


def train_and_evaluate(
    model,
    train_loader,
    valid_loader,
    test_loader,
    device,
    epochs=100,
    lr=1e-3,
    weight_decay=0.0,
    patience=20,
    verbose=True,
):
    """
    Train using train/validation data and evaluate the
    selected model once on the held-out test set.
    """
    results = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=verbose,
    )

    model = results["model"]

    test_rmse = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    params_count_df = get_params_count_df(
        model, ["self_attn", "feed_forward", "src_embed", "all"],
    )
    results.update(
        {"test_rmse": test_rmse, "params_df": params_count_df}
    )
    return results
