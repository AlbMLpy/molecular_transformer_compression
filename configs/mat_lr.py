import copy

from .mat_dense import MODEL_CONFIG
from .shared import (
    CONFIG_KEYS,
    SEED_OPTIONS, 
    TARGET_LAYERS_FS,
)


RANK_OPTIONS = (2, 4)


def update_mat_dense_config(target_layer: str, rank: int) -> dict:
    model_cfg = copy.deepcopy(MODEL_CONFIG)
    if rank <= 0:
        raise ValueError("rank must be a positive integer.")
    if target_layer not in TARGET_LAYERS_FS:
        raise ValueError(
            f"Invalid target_layer={target_layer!r}. "
            f"Expected one of {sorted(TARGET_LAYERS_FS)}."
        )
    config = dict(type="lr", rank=rank)
    if target_layer == "all":
        for config_key in CONFIG_KEYS.values():
            model_cfg[config_key] = config.copy()
    else:
        model_cfg[CONFIG_KEYS[target_layer]] = config.copy()
    return model_cfg


def get_config_mat_lr(mode: str, target_layer: str, rank: int) -> dict:
    if mode == "FS":
        model_config = update_mat_dense_config(target_layer, rank)
        return dict(
            model_config=model_config,
            seed_options=SEED_OPTIONS,
            batch_size=32, # 16 or 32
            epochs=100,
            lr=1e-3,
            patience=60,
        )
    elif mode == "FT":
        extra_config = dict(
            compress="lr",
            target_layer=target_layer,
            rank=rank,
            fine_tune_mode="layer",
        )
        config = dict(
            model_config=copy.deepcopy(MODEL_CONFIG),
            seed_options=SEED_OPTIONS,
            batch_size=32, # 16 or 32
            epochs=100,
            lr=1e-4, # smaller for fine tuning
            patience=60,
        )
        return config | extra_config
