from os import PathLike
from collections import defaultdict
from collections.abc import Callable

import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

from src.transformer import make_model
from src.data_process import get_loaders
from src.training import train_and_evaluate
from src.pruning import prune_linear_layers
from src.low_rank import dense_to_lr, dense_to_tt
from src.utils import (
    set_seed,
    get_layer_names,
    train_only_layers,
    update_results_dict,
    replace_linear_layers,
    get_layer_names_linear,
    create_dir_if_not_exists,
)


def compress_model(model: nn.Module, config: dict):
    """Apply LR, TT, or pruning compression to selected layers."""
    layer_names = get_layer_names_linear(model, config["target_layer"])
    if config["compress"] == "lr":
        conversion_fn = lambda layer: dense_to_lr(layer, rank=config["rank"])
        model = replace_linear_layers(model, conversion_fn, layer_names)
    elif config["compress"] == "tt": # Only for squared matrices 128 * 128!
        if config["model_config"]["d_model"] != 128:
            raise ValueError("TT and FT works only with d_model=128!")
        if config["target_layer"] not in ("self_attn", "feed_forward"):
            raise ValueError("'target_layer' = 'self_attn' or 'feed_forward'")
        rank = (1,) + (config["rank_c"],)*3 + (1,)
        in_out_shape = (4, 4, 2, 4) # d_model=128!
        conversion_fn = lambda layer: dense_to_tt(
            layer, in_out_shape, in_out_shape, rank,
        )
        model = replace_linear_layers(model, conversion_fn, layer_names)
    elif config["compress"] == "prune":
        model = prune_linear_layers(model, layer_names, config["prune_amount"])
    else:
        raise ValueError("Bad compression! Use: 'tt', 'lr', 'prune'.")
    # One might fine-tune only compressed layers or all the layers of the net
    if config["fine_tune_mode"] == "layer":
        model = train_only_layers(
            model, get_layer_names(model, config["target_layer"]),
        )
    elif config["fine_tune_mode"] == "all":
        pass
    else:
        raise ValueError("Bad fine-tune mode! Use: 'layer' or 'all'.")
    return model


def get_stats_several_trials(
    config: dict,
    load_data: Callable,
    device,
    res_dir: PathLike,
    ckp_dir: PathLike | None = None,
    tqdm_disable: bool = False,
    verbose: bool = False,
    save_model: bool = True,
) -> None:
    """Train and evaluate MAT across multiple random seeds.

    For each seed, the function creates the corresponding data split,
    trains a model, evaluates it on the validation and test sets, and
    saves the metrics, parameter counts, training history, and optionally
    the model checkpoint.
    """
    seeds = config['seed_options']
    res_dict = defaultdict(list)
    for trial, seed in tqdm(
        enumerate(seeds, start=1), 
        total=len(seeds), 
        disable=tqdm_disable,
    ):
        seed_dir = res_dir / f"seed_{seed}"
        create_dir_if_not_exists(seed_dir)
        set_seed(seed)
        train_loader, valid_loader, test_loader = get_loaders(
            config, load_data, seed,
        )
        d_atom = train_loader.dataset[0].node_features.shape[1]
        model = make_model(d_atom, **config['model_config'])
        # Compress the model:
        if ckp_dir is not None:
            # Download the appropriate dense model checkpoint:
            model.load_state_dict(
                torch.load(
                    ckp_dir / f"seed_{seed}/model.pt", 
                    weights_only=True, 
                    map_location=device
                )
            )
            # Model Compression:
            model = compress_model(model, config)
        # Train the model:
        result = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device,
            epochs=config['epochs'],
            lr=config['lr'],
            patience=config['patience'],
            verbose=verbose,
        )
        ### Save Metrics:
        seed_res = dict(
            trial=trial,
            time=result['training_time'], 
            test_rmse=result['test_rmse'],
            valid_rmse=result['best_valid_rmse']
        )
        update_results_dict(res_dict, **seed_res)
        pd.DataFrame(
            {k: [v,] for k, v in seed_res.items()}
        ).to_csv(seed_dir / 'metrics.csv')
        result['params_df'].to_csv(seed_dir / 'params_counter.csv')
        # Save training history:
        (
            pd.DataFrame(result['history'])
            .to_csv(seed_dir / 'training_history.csv')
        )
        # Save optimal model:
        if save_model:
            torch.save(result['model'].state_dict(), seed_dir / 'model.pt')
        # For testing/debugging purposes:
        with open(seed_dir / "model.txt", "w", encoding="utf-8") as f:
            f.write(str(result["model"]))
    pd.DataFrame(res_dict).to_csv(res_dir / 'metrics.csv')
