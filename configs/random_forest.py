from .shared import SEED_OPTIONS


def get_config_rf():
    return dict(
        seed_options=SEED_OPTIONS,
        n_est_options=(100, 300, 500, 700),
        n_jobs=-1,
        model_seed=42,
    )
