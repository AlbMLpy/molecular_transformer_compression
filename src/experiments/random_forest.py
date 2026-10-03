import time
from os import PathLike
from typing import Sequence
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

from ..utils import update_results_dict


def get_stats_several_trials(
    config: dict,
    load_data: Callable,
    res_dir: PathLike,
    tqdm_disable: bool = False,
) -> None:
    """Run several RF trials and save validation/test metrics to CSV."""
    seeds = config['seed_options']
    res_dict = defaultdict(list)
    for trial, seed in tqdm(
        enumerate(seeds, start=1), 
        total=len(seeds), 
        disable=tqdm_disable,
    ):
        train_split, valid_split, test_split = load_data(split_seed=seed)
        best_valid_rmse, best_n_est = hp_search(
            (train_split, valid_split), config,
        )
        start_time = time.perf_counter()
        rf = RandomForestRegressor(
            n_estimators=best_n_est, 
            random_state=config["model_seed"], 
            n_jobs=config["n_jobs"],
        )
        rf.fit(*train_split)
        training_time = (time.perf_counter() - start_time)
        X_test, y_test = test_split
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        update_results_dict(
            res_dict,
            trial=trial,
            time=training_time, 
            test_rmse=test_rmse,
            valid_rmse=best_valid_rmse,
            n_est=best_n_est,
        )
    pd.DataFrame(res_dict).to_csv(res_dir / 'metrics.csv')


def hp_search(
    train_valid_split: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict,
) -> tuple[float, int]:
    """Select the number of RF estimators using validation RMSE."""
    (X_train, y_train), (X_valid, y_valid) = train_valid_split
    best_valid_rmse, best_n_est = float("inf"), None
    for n_est in config["n_est_options"]:
        rf = RandomForestRegressor(
            n_estimators=n_est, 
            random_state=config["model_seed"], 
            n_jobs=config["n_jobs"],
        )
        rf.fit(X_train, y_train)
        y_valid_pred = rf.predict(X_valid)
        valid_rmse = root_mean_squared_error(
            y_valid, y_valid_pred,
        )
        if valid_rmse < best_valid_rmse:
            best_valid_rmse, best_n_est = valid_rmse, n_est
    return best_valid_rmse, best_n_est
