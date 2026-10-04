from functools import partial

from src.data_process import load_split_data
from src.utils import create_dir_if_not_exists
from src.experiments.mat import get_stats_several_trials
from src.experiments.random_forest import get_stats_several_trials as gsst_rf


from configs.mat_lr import get_config_mat_lr
from configs.mat_tt import get_config_mat_tt
from configs.random_forest import get_config_rf
from configs.mat_dense import get_config_mat_dense
from configs.mat_pruned import get_config_mat_pruned


def mat_dense_experiment(art_dir, data_path, device):
    print(f"Running MAT-Dense")
    res_dir = art_dir / f"mat_dense/"
    create_dir_if_not_exists(res_dir)

    config = get_config_mat_dense()
    load_data_f = partial(
        load_split_data, data_path=data_path, fingerprints=False,
    )
    get_stats_several_trials(
        config, load_data_f, device, res_dir, 
        ckp_dir=None, save_model=True,
    )


def mat_tt_experiment(
    art_dir, 
    data_path, 
    device, 
    mode: str, 
    target_layers, 
    rank_c_options,
):
    if mode == "FS":
        ckp_dir = None
    elif mode == "FT":
        ckp_dir = art_dir / f"mat_dense/"
    for target_layer in target_layers:
        for rank_c in rank_c_options:
            exp_str = (
                f"Running MAT-TT: mode={mode}|" 
                + f"layer={target_layer}|rank={rank_c}"
            )
            print(exp_str)
            res_dir = (
                art_dir 
                / f"mat_tt/mode_{mode}/layer_{target_layer}/rank_c_{rank_c}"
            )
            create_dir_if_not_exists(res_dir)

            config = get_config_mat_tt(mode, target_layer, rank_c)
            load_data_f = partial(
                load_split_data, data_path=data_path, fingerprints=False,
            )
            get_stats_several_trials(
                config, load_data_f, device, res_dir, 
                ckp_dir=ckp_dir, save_model=False,
            )


def mat_lr_experiment(
    art_dir, 
    data_path, 
    device,
    mode: str, 
    target_layers, 
    rank_options,
):
    if mode == "FS":
        ckp_dir = None
    elif mode == "FT":
        ckp_dir = art_dir / f"mat_dense/"
    for target_layer in target_layers:
        for rank in rank_options:
            exp_str = (
                f"Running MAT-LR: mode={mode}|" + 
                f"layer={target_layer}|rank={rank}"
            )
            print(exp_str)
            res_dir = (
                art_dir 
                / f"mat_lr/mode_{mode}/layer_{target_layer}/rank_{rank}"
            )
            create_dir_if_not_exists(res_dir)

            config = get_config_mat_lr(mode, target_layer, rank)
            load_data_f = partial(
                load_split_data, data_path=data_path, fingerprints=False,
            )
            get_stats_several_trials(
                config, load_data_f, device, res_dir, 
                ckp_dir=ckp_dir, save_model=False,
            )


def mat_pruned_experiment(
    art_dir, 
    data_path, 
    device,
    target_layers, 
    prune_amount_options,
):
    ckp_dir = art_dir / f"mat_dense/"
    for target_layer in target_layers:
        for prune_amount in prune_amount_options:
            exp_str = (
                f"Running MAT-Pruned: " + 
                f"|layer={target_layer}|amount={prune_amount}"
            )
            print(exp_str)
            res_dir = (
                art_dir 
                / f"mat_pruned/layer_{target_layer}/amount_{prune_amount}"
            )
            create_dir_if_not_exists(res_dir)

            config = get_config_mat_pruned(target_layer, prune_amount)
            load_data_f = partial(
                load_split_data, data_path=data_path, fingerprints=False,
            )
            get_stats_several_trials(
                config, load_data_f, device, res_dir, 
                ckp_dir=ckp_dir, save_model=False,
            ) 


def rf_experiment(art_dir, data_path):
    print(f"Running Random Forest")
    res_dir = art_dir / f"random_forest/"
    create_dir_if_not_exists(res_dir)

    config = get_config_rf()
    load_data_f = partial(
        load_split_data, data_path=data_path, fingerprints=True,
    )
    gsst_rf(config, load_data_f, res_dir)
