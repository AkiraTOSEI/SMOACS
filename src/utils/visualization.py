import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def optimization_history_display_for_promising_candidates(
    gap_min: float,
    gap_max: float,
    npz_path: str,
    csv_path: str,
    model_name: str,
    num_display: int,
    history_img_path: str,
):
    """
    Visualize the bandgap and formation energy optimization histories for the most promising candidates.

    This function loads optimization histories from `.npz` and `.csv` files, selects the top-performing
    structures based on loss and formation energy, and plots their optimization trajectories. The plots are
    saved as an image and also displayed.

    Args:
        gap_min (float): Lower bound of the target bandgap range (eV).
        gap_max (float): Upper bound of the target bandgap range (eV).
        npz_path (str): Path to the `.npz` file containing optimization history arrays.
        csv_path (str): Path to the `.csv` file containing structure indices and losses.
        model_name (str): Name of the model used for optimization. Should be 'alignn', 'crystalformer', or 'Both'.
        num_display (int): Number of top-performing structures to visualize.
        history_img_path (str): Path to save the generated history plot image.

    Returns:
        None. The function displays the plot and saves it to `history_img_path`.
    """
    d = np.load(npz_path, allow_pickle=True)
    gap_history = d["gap_history"]
    if gap_history.ndim == 1:
        gap_history = gap_history[:, None]
    ef_history = d["ef_history"]
    if ef_history.ndim == 1:
        ef_history = ef_history[:, None]
    # formation energyが低く、指定のバンドギャップを満たすものから順
    sorted_index = (
        pd.read_csv(csv_path)
        .sort_values(["loss_onehot", "ef_onehot"])["lattice_index"]
        .values
    )
    num_display = min(num_display, len(sorted_index))

    title_font_size, label_font_size, legend_font_size, tick_font_size = (
        16,
        14,
        14,
        14,
    )  # 目盛りのフォントサイズを追加
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列、図のサイズは12x5インチ

    colors_list = (
        list(mcolors.TABLEAU_COLORS.values())
        + list(mcolors.BASE_COLORS.values())
        + list(mcolors.CSS4_COLORS.values())
    )
    # Bandgap optimization historyのプロット（最初のサブプロット）
    if model_name != "Both":
        axs[0].plot(gap_history[:, sorted_index[:num_display]])
        axs[0].hlines(
            gap_min,
            0,
            gap_history.shape[0],
            linestyles="dashed",
            colors="black",
            label="target area",
        )
        axs[0].hlines(
            gap_max,
            0,
            gap_history.shape[0],
            linestyles="dashed",
            colors="black",
        )
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[0].plot(
                d["gap_history_alignn"][:, sorted_index[i]], color=color, linestyle="-"
            )  # , label=f'alignn_{i}')
            axs[0].plot(
                d["gap_history_crystalformer"][:, sorted_index[i]],
                color=color,
                linestyle="-.",
            )
        axs[0].hlines(
            gap_min,
            0,
            d["gap_history_alignn"].shape[0],
            linestyles="dashed",
            colors="black",
            label="target area",
        )
        axs[0].hlines(
            gap_max,
            0,
            d["gap_history_alignn"].shape[0],
            linestyles="dashed",
            colors="black",
        )
        axs[0].plot([], color="gray", linestyle="-", label="alignn")
        axs[0].plot([], color="gray", linestyle="-.", label="crystalformer")

    axs[0].set_title(
        "Bandgap optimization history", fontsize=title_font_size
    )  # タイトルのフォントサイズ
    axs[0].set_xlabel("step", fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[0].set_ylabel(
        "bandgap (eV)", fontsize=label_font_size
    )  # y軸ラベルのフォントサイズ
    axs[0].tick_params(
        axis="both", labelsize=tick_font_size
    )  # x軸とy軸の目盛りのフォントサイズ

    axs[0].legend()  # 凡例の表示

    # Formation energy optimization historyのプロット（2番目のサブプロット）
    if model_name != "Both":
        axs[1].plot(ef_history[:, sorted_index[:num_display]])
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[1].plot(
                d["ef_history_alignn"][:, sorted_index[i]], color=color, linestyle="-"
            )
            axs[1].plot(
                d["ef_history_crystalformer"][:, sorted_index[i]],
                color=color,
                linestyle="-.",
            )
    axs[1].set_title(
        "Formation energy optimization history", fontsize=title_font_size
    )  # タイトルのフォントサイズ
    axs[1].set_xlabel("step", fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[1].set_ylabel(
        "formation energy (eV/at.)", fontsize=label_font_size
    )  # y軸ラベルのフォントサイズ
    axs[1].tick_params(
        axis="both", labelsize=tick_font_size
    )  # x軸とy軸の目盛りのフォントサイズ

    plt.tight_layout()  # レイアウトの調整
    if os.path.dirname(history_img_path) != "":
        os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plt.savefig(history_img_path)
    plt.close()
    figure = plt.figure(figsize=(10, 10))
    img = Image.open(history_img_path)
    plt.imshow(img)
    plt.show()


def optimization_history_display_for_superhard(
    npz_path: str,
    csv_path: str,
    num_display: int,
    history_img_path: str,
    k_min: float = 1.0,
    k_max: float = 1.5,
) -> None:
    """
    Visualize G/B optimization history and k=G/B trends for the superhard task.

    Uses gap_history/ef_history as G/B histories.
    """
    d = np.load(npz_path, allow_pickle=True)
    df = pd.read_csv(csv_path)
    if "loss_total" in df.columns:
        sorted_index = df.sort_values(["loss_total"])["lattice_index"].values
    else:
        sorted_index = df.sort_values(["loss_onehot", "ef_onehot"])["lattice_index"].values
    num_display = min(num_display, len(sorted_index))

    g_hist = d["gap_history"]
    if g_hist.ndim == 1:
        g_hist = g_hist[:, None]
    b_hist = d["ef_history"]
    if b_hist.ndim == 1:
        b_hist = b_hist[:, None]

    idx = sorted_index[:num_display]
    mean_g = g_hist[:, idx].mean(axis=1)
    mean_b = b_hist[:, idx].mean(axis=1)
    mean_k = mean_g / (mean_b + 1e-8)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(mean_g, label="mean G", color="tab:blue")
    ax1.plot(mean_b, label="mean B", color="tab:green")
    ax1.set_xlabel("step")
    ax1.set_ylabel("G, B (ALIGNN predictions)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(mean_k, label="mean k=G/B", color="tab:red")
    ax2.axhline(k_min, color="tab:red", linestyle="--", linewidth=1)
    ax2.axhline(k_max, color="tab:red", linestyle="--", linewidth=1)
    ax2.set_ylabel("k ratio")
    ax2.legend(loc="upper right")

    plt.title("Superhard optimization history (mean G, B, k)")
    plt.tight_layout()
    if os.path.dirname(history_img_path) != "":
        os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plt.savefig(history_img_path)
    plt.close()
