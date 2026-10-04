import copy

from .shared import SEED_OPTIONS
from .mat_dense import MODEL_CONFIG


PRUNE_OPTIONS = (0.7, 0.5)


def get_config_mat_pruned(target_layer: str, prune_amount: float) -> dict:
        extra_config = dict(
            compress="prune",
            target_layer=target_layer,
            prune_amount=prune_amount,
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
