from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.utils import update_results_dict

from configs.mat_lr import RANK_OPTIONS
from configs.mat_tt import RANK_C_OPTIONS
from configs.mat_pruned import PRUNE_OPTIONS
from configs.shared import (
    TARGET_LAYERS_FS, 
    TARGET_LAYERS_FT,
)

sns.set_theme()


PARAMS = {
    'figure.figsize': (12, 5),
    'figure.constrained_layout.use': True,
    'figure.facecolor': 'white',
    'font.size': 10,
    'axes.labelsize': 16,
    'axes.grid': True,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'lines.linestyle': '-',
    'legend.fontsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 20,
    'figure.max_open_warning': 50,
}
LAYER2NAME = {"self_attn": "Attn", "feed_forward": "FF", "src_embed": "Emb"}


def get_metrics(metrics_path) -> dict:
    metrics = ["test_rmse", "valid_rmse", "time"]
    df = pd.read_csv(metrics_path, index_col=0)
    summary = df.groupby("trial")[metrics].mean().agg(["mean", "std"])
    result = dict()
    for m in metrics:
        result[f"{m}_m"] = float(summary.loc['mean', m])
        result[f"{m}_s"] = float(summary.loc['std', m])
    return result


def get_model_metrics(path, art_dir):
    _metrics = get_metrics(art_dir / f"{path}/metrics.csv")
    if "random_forest" in path:
        _params = dict(total="-", trainable="-", nonzero="-")
    else:
        _params = pd.read_csv(
            art_dir / f"{path}/seed_2/params_counter.csv", index_col=0,
        )
        _params = _params[_params["target_layer"] == "all"].iloc[0].to_dict()
        _params.pop("target_layer")
    return _metrics | _params


def get_metrics_baselines(art_dir):
    res_dict = defaultdict(list)
    # RF
    model_metrics = get_model_metrics("random_forest", art_dir)
    update_results_dict(
        res_dict, model="RF", best_config="-", **model_metrics,
    )
    # MAT-Dense
    model_metrics = get_model_metrics("mat_dense", art_dir)
    update_results_dict(
        res_dict, model="MAT-Dense", best_config="-", **model_metrics,
    )
    return pd.DataFrame(res_dict)


def get_metrics_mat_pruned(target_layers, prune_options, art_dir):
    res_dict = defaultdict(list)
    for target_layer in target_layers:
        for prune_amount in prune_options:
            model_name = f"MAT-Pruned"
            _path = f"mat_pruned/layer_{target_layer}/amount_{prune_amount}"
            best_config = (
                f"FT, {LAYER2NAME[target_layer]}, {prune_amount*100:.0f}%"
            )
            model_metrics = get_model_metrics(_path, art_dir)
            update_results_dict(
                res_dict, 
                model=model_name, 
                best_config=best_config, 
                **model_metrics,
            )
    return pd.DataFrame(res_dict)


def get_metrics_mat_lr(rank_options, art_dir):
    res_dict = defaultdict(list)
    for mode in ("FS", "FT"):
        target_layers = TARGET_LAYERS_FT if mode == "FT" else TARGET_LAYERS_FS
        for target_layer in target_layers:
            for rank in rank_options:
                model_name = f"MAT-LR"
                _path = f"mat_lr/mode_{mode}/layer_{target_layer}/rank_{rank}"
                best_config = (
                    f"{mode}, {LAYER2NAME[target_layer]}, rank {rank}"
                )
                model_metrics = get_model_metrics(_path, art_dir)
                update_results_dict(
                    res_dict, 
                    model=model_name, 
                    best_config=best_config, 
                    **model_metrics,
                )
    return pd.DataFrame(res_dict)


def get_metrics_mat_tt(rank_c_options, art_dir):
    res_dict = defaultdict(list)
    for mode in ("FS", "FT"):
        target_layers = TARGET_LAYERS_FT if mode == "FT" else TARGET_LAYERS_FS
        for target_layer in target_layers:
            for rank_c in rank_c_options:
                model_name = f"MAT-TT"
                _path = f"mat_tt/mode_{mode}/layer_{target_layer}/rank_c_{rank_c}"
                best_config = f"{mode}, {LAYER2NAME[target_layer]}, rank {rank_c}"
                model_metrics = get_model_metrics(_path, art_dir)
                update_results_dict(
                    res_dict, 
                    model=model_name, 
                    best_config=best_config, 
                    **model_metrics,
                )
    return pd.DataFrame(res_dict)


def prep_visual(df, prune_bool=False):
    df["layer"] = (
        df["best_config"]
        .str.extract(r"(Emb|Attn|FF)")[0]
        .map({
            "Emb": "Embedding",
            "Attn": "Attention",
            "FF": "Feed-forward",
        })
    )
    if prune_bool:
        df["sparsity"] = pd.to_numeric(
            df["best_config"].str.extract(r"(\d+)%")[0]
        )
    else:
        df["mode"] = df["best_config"].str.extract(r"^(FS|FT)")[0]
        df["rank"] = pd.to_numeric(
            df["best_config"].str.extract(r"rank (\d+)")[0]
        )
    return df


def prepare_best_results_table(
    art_dir,
    decimals: int = 3,
) -> pd.DataFrame:
    """Prepare the final best results table."""
    bl_df = get_metrics_baselines(art_dir)
    pruned_df = get_metrics_mat_pruned(
        TARGET_LAYERS_FT, PRUNE_OPTIONS, art_dir,
    )
    pruned_df = pruned_df.iloc[[pruned_df['valid_rmse_m'].argmin()]]

    lr_df = get_metrics_mat_lr(RANK_OPTIONS, art_dir)
    lr_df = lr_df.iloc[[lr_df['valid_rmse_m'].argmin()]]

    tt_df = get_metrics_mat_tt(RANK_C_OPTIONS, art_dir)
    tt_df = tt_df.iloc[[tt_df['valid_rmse_m'].argmin()]]

    df = pd.concat([bl_df, pruned_df, lr_df, tt_df])

    dense_params = df.loc[df["model"].eq("MAT-Dense"), "total"].iloc[0]
    result = pd.DataFrame({
        "Method": df["model"],
        "Configuration": df["best_config"],
        "Test RMSE ↓": (
            df["test_rmse_m"].map(lambda x: f"{x:.{decimals}f}")
            + " ± "
            + df["test_rmse_s"].map(lambda x: f"{x:.{decimals}f}")
        ),
        "Train Time ↓": (
            df["time_m"].map(lambda x: f"{x:.1f}")
            + " ± "
            + df["time_s"].map(lambda x: f"{x:.1f}")
        ),
        "Parameters ↓": df["total"].map(
            lambda x: "-" if pd.isna(x) or x == "-" else f"{int(x):,}"
        ),
        "Compression Ratio ↑": df["total"].apply(
            lambda x: (
                "-"
                if pd.isna(x) or x == "-"
                else f"{dense_params / x:.2f}×"
            )
        ),
    })
    return result


def prepare_ablations_table(
    df: pd.DataFrame, 
    decimals: int = 3,
    non_zero: bool = False,
) -> pd.DataFrame:
    dense_params = df.loc[df["model"].eq("MAT-Dense"), "total"].iloc[0]
    result = pd.DataFrame({
        "Method": df["model"],
        "Configuration": df["best_config"],
        "Test RMSE ↓": (
            df["test_rmse_m"].map(lambda x: f"{x:.{decimals}f}")
            + " ± "
            + df["test_rmse_s"].map(lambda x: f"{x:.{decimals}f}")
        ),
        "Validation RMSE ↓": (
            df["valid_rmse_m"].map(lambda x: f"{x:.{decimals}f}")
            + " ± "
            + df["valid_rmse_s"].map(lambda x: f"{x:.{decimals}f}")
        ),
        "Train Time ↓": (
            df["time_m"].map(lambda x: f"{x:.1f}")
            + " ± "
            + df["time_s"].map(lambda x: f"{x:.1f}")
        ),
        "Parameters (Total) ↓": df["total"].map(
            lambda x: "-" if pd.isna(x) or x == "-" else f"{int(x):,}"
        ),
        "Parameters (Trainable) ↓": df["trainable"].map(
            lambda x: "-" if pd.isna(x) or x == "-" else f"{int(x):,}"
        ),
        "Compression Ratio ↑": df["total"].apply(
            lambda x: (
                "-"
                if pd.isna(x) or x == "-"
                else f"{dense_params / x:.2f}×"
            )
        ),
    })
    if non_zero:
        result["Parameters (Nonzero) ↓"] = df["nonzero"].map(
            lambda x: "-" if pd.isna(x) or x == "-" else f"{int(x):,}"
        )
    return result


### Visualization ###


def add_bar_labels(ax, bars, values, decimals=3):
    """Add numeric values to the center of each bar."""
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{value:.{decimals}f}",
            ha="center",
            va="center",
            fontsize=10,
        )


def plot_compression_bar(
    df: pd.DataFrame,
    x_col: str,
    group_col: str,
    y_col: str = "test_rmse_m",
    yerr_col: str = "test_rmse_s",
    style_col: str | None = None,
    title: str = "Compression ablation",
    x_label: str = "Rank",
    y_label: str = "Test RMSE",
    group_order: list[str] | None = None,
    style_order: list[str] | None = None,
    reference_rmse: float | None = None,
    reference_std: float | None = None,
    reference_label: str = "MAT-Dense",
    figsize: tuple = (7, 5),
    show_plot: bool = True,
    dpi: int = 500,
    save_path: Optional[str] = None,
) -> None:
    """Plot grouped compression bars with an optional dense reference."""

    if group_order is None:
        group_order = list(df[group_col].dropna().unique())

    if style_col is None:
        style_values = [None]
    elif style_order is None:
        style_values = list(df[style_col].dropna().unique())
    else:
        style_values = style_order

    n_panels = len(style_values)

    with mpl.rc_context(PARAMS):
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(figsize[0] * n_panels, figsize[1]),
            sharey=True,
            squeeze=False,
        )
        axes = axes.ravel()
        x_values = sorted(df[x_col].dropna().unique())
        n_groups = len(group_order)
        bar_width = 0.8 / n_groups
        for ax, style_value in zip(axes, style_values):
            if style_col is None:
                plot_df = df.copy()
                panel_title = title
            else:
                plot_df = df[df[style_col] == style_value].copy()
                panel_title = f"{title} — {style_value}"
            # Add dense reference bar
            all_x_labels = ["Dense"] + [str(x) for x in x_values]
            x_positions = np.arange(len(all_x_labels))
            if reference_rmse is not None:
                bars = ax.bar(
                    x_positions[0],
                    reference_rmse,
                    width=0.3,
                    yerr=reference_std,
                    capsize=4,
                    label=reference_label,
                )
                add_bar_labels(ax, bars, [reference_rmse,])
            # Compression bars
            compression_positions = x_positions[1:]
            for i, group in enumerate(group_order):
                values, errors = [], []
                for x_value in x_values:
                    row = plot_df[
                        (plot_df[x_col] == x_value)
                        & (plot_df[group_col] == group)
                    ]
                    if row.empty:
                        values.append(np.nan)
                        errors.append(0.0)
                    else:
                        values.append(row[y_col].iloc[0])
                        errors.append(row[yerr_col].iloc[0])
                positions = (
                    compression_positions
                    + (i - (n_groups - 1) / 2) * bar_width
                )
                bars = ax.bar(
                    positions,
                    values,
                    width=bar_width,
                    yerr=errors,
                    capsize=4,
                    label=group,
                )
                add_bar_labels(ax, bars, values)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(all_x_labels)
            ax.set_xlabel(x_label)
            ax.set_title(panel_title)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(y_label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=n_groups + 1,
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
        )
        #fig.suptitle(title, y=1.02)
        fig.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()


def load_training_history(result_dir: str | Path) -> pd.DataFrame:
    """Load training history files from all seeds in a result directory."""
    result_dir = Path(result_dir)

    histories = []

    for seed_dir in sorted(result_dir.glob("seed_*")):
        history_file = seed_dir / "training_history.csv"

        if not history_file.exists():
            continue

        history = pd.read_csv(history_file, index_col=0)
        history["seed"] = seed_dir.name
        history["epoch"] = range(1, len(history) + 1)
        histories.append(history)

    if not histories:
        raise FileNotFoundError(
            f"No training_history.csv files found in {result_dir}"
        )

    return pd.concat(histories, ignore_index=True)

def summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation of RMSE across seeds."""
    return (
        history
        .groupby("epoch")["valid_rmse"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot_training_history(
    histories: dict,
    figsize: tuple = (15, 4.5),
    x_label: str = "Epoch",
    y_label: str = "Validation RMSE",
    title: str = "Training dynamics under TT compression",
    show_plot: bool = True,
    dpi: int = 500,
    save_path: Optional[str] = None,
) -> None:
    """Plot validation RMSE during training for TT-compressed layers.

    Parameters
    ----------
    histories : dict
        Nested dictionary of the form
        ``{layer: {model: history_dataframe}}``. Each history dataframe
        must contain ``epoch``, ``valid_rmse``, and ``seed`` columns.
    figsize : tuple, optional
        Figure size.
    title : str, optional
        Overall figure title.
    y_label : str, optional
        Y-axis label.
    """
    with mpl.rc_context(PARAMS):
        fig, axes = plt.subplots(
            1,
            len(histories),
            figsize=figsize,
            sharey=False,
        )
        # Handle the case of a single subplot.
        if len(histories) == 1:
            axes = [axes]
        for ax, (layer, models) in zip(axes, histories.items()):
            for model_name, history in models.items():
                summary = summarize_history(history)
                ax.plot(summary["epoch"], summary["mean"], label=model_name)
                ax.fill_between(
                    summary["epoch"],
                    summary["mean"] - summary["std"],
                    summary["mean"] + summary["std"],
                    alpha=0.2,
                )
            ax.set_title(layer)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(
                axis="y",
                linestyle="-",
                linewidth=0.8,
                alpha=0.8,
            )
            ax.legend(loc="best", frameon=True)

        fig.suptitle(title, fontweight="bold", size=20)
        fig.tight_layout()
        fig.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()


def prepare_history(model, mode, art_dir):
    layer2name = {
        "self_attn": "Attention", 
        "feed_forward": "Feed Forward", 
        "src_embed": "Embedding",
    }
    histories = dict()
    target_layers = TARGET_LAYERS_FT if mode == "FT" else TARGET_LAYERS_FS
    if model == "TT":
        for target_layer in target_layers:
            l_name = layer2name[target_layer]
            histories[l_name] = dict(
                Dense=load_training_history(art_dir / "mat_dense"),
            )
            for rank_c in RANK_C_OPTIONS:
                _path = (
                    f"mat_tt/mode_{mode}/layer_{target_layer}/rank_c_{rank_c}"
                )
                histories[l_name][f"TT Rank {rank_c}"] = load_training_history(
                    art_dir / _path,
                )
    elif model == "LR":
        for target_layer in target_layers:
            l_name = layer2name[target_layer]
            histories[l_name] = dict(
                Dense=load_training_history(art_dir / "mat_dense"),
            )
            for rank in RANK_OPTIONS:
                _path = (
                    f"mat_lr/mode_{mode}/layer_{target_layer}/rank_{rank}"
                )
                histories[l_name][f"Low-Rank {rank}"] = load_training_history(
                    art_dir / _path,
                )
    elif model == "Pruned":
        for target_layer in target_layers:
            l_name = layer2name[target_layer]
            histories[l_name] = dict(
                Dense=load_training_history(art_dir / "mat_dense"),
            )
            for prune_amount in PRUNE_OPTIONS:
                _path = (
                    f"mat_pruned/layer_{target_layer}/amount_{prune_amount}"
                )
                histories[l_name][f"Pruned {prune_amount*100:.0f}%"] = load_training_history(
                    art_dir / _path,
                )
    return histories
