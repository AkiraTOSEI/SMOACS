# visualization and evaluation utilities
import ast
from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from jarvis.db.figshare import data as jarvis_data

from src.utils.save_optimization_result import save_optimization_results
from src.utils.schedule_utils import set_schedule
from src.utils.structure_neutrality_check import (
    check_neurality_bondlength_and_save_structure,
    perovskite_coordinate_check,
)
from src.utils.visualization import (
    optimization_history_display_for_promising_candidates,
)


def create_MEGNet_element_set():
    megnet_data = jarvis_data("megnet")
    megnet_elms_set = set()
    for mdata in megnet_data:
        elms = mdata["atoms"]["elements"]
        elms.sort()
        megnet_elms_set.add(str(elms))
    return megnet_elms_set


def load_and_prepare_dataframe(
    npz_path: str, num_candidate: int, e_form_criteria: float, tmp_csv_path: str
) -> pd.DataFrame:
    """
    Load .npz file and prepare the initial dataframe with lattice and basic optimization results.

    Args:
        npz_path (str): Path to the .npz result file.
        num_candidate (int): Expected number of candidates (for assertion check).
        e_form_criteria (float): Energy formation criteria for success evaluation.
        tmp_csv_path (str): Path to save the temporary CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with lattice info and basic results.
    """
    d = np.load(npz_path, allow_pickle=True)
    abc_strings = [
        f"[{abc[0]:.3f}, {abc[1]:.3f}, {abc[2]:.3f}]" for abc in d["batch_abc"]
    ]
    angle_strings = [
        f"[{angle[0]:.3f}, {angle[1]:.3f}, {angle[2]:.3f}]"
        for angle in d["batch_angle"]
    ]

    df = pd.DataFrame(
        {
            "lattice_index": np.arange(len(d["size"])),
            "bandgap_onehot": d["bandgap_onehot"],
            "ef_onehot": d["ef_onehot"],
            "loss_onehot": d["loss_onehot"],
            "num_atoms": d["num_atoms"],
            "original_fnames": d["original_fnames"],
            "tolerance": d["tolerance"],
            "batch_abc": abc_strings,
            "batch_angle": angle_strings,
        }
    )

    df = df.sort_values(["loss_onehot", "ef_onehot"]).reset_index(drop=True)
    df.set_index("lattice_index", inplace=True)

    df["Ef_success"] = df["ef_onehot"] < e_form_criteria
    df["bandgap_success"] = df["loss_onehot"] == 0
    df.to_csv(tmp_csv_path)

    # データ数が一致するか確認
    assert len(d["lattice_vectors"]) == num_candidate, (
        f"Expected {num_candidate} lattice vectors but got {len(d['lattice_vectors'])}"
    )

    return df


def merge_evaluation_results(
    df: pd.DataFrame, df2: pd.DataFrame, csv_path: str
) -> pd.DataFrame:
    try:
        df = pd.concat(
            [
                df.reset_index().set_index("original_fnames"),
                df2.set_index("original_fnames"),
            ],
            axis=1,
        )
    except KeyError:
        # All entries failed — save empty CSV and return
        df = pd.DataFrame([])
        df.to_csv(csv_path)
    return df


def evaluate_perovskite_properties(
    df: pd.DataFrame,
    d: dict,
    crystal_system: Optional[str],
    tolerance_range: tuple,
    limit_coords_displacement: Optional[float],
) -> pd.DataFrame:
    """
    Evaluate perovskite-specific criteria such as tolerance factor and coordinate displacement.
    For non-perovskite systems, NaN values are assigned to related columns.

    Args:
        df (pd.DataFrame): DataFrame containing structure information.
        d (dict): Loaded data from .npz file (includes init_coords, dir_coords, size).
        crystal_system (Optional[str]): Indicates whether the system is perovskite.
        tolerance_range (tuple): Acceptable tolerance factor range.
        limit_coords_displacement (Optional[float]): Threshold for coordinate displacement.

    Returns:
        pd.DataFrame: Updated DataFrame with evaluation results.
    """
    if crystal_system == "perovskite":
        tolerence_bool = (min(tolerance_range) <= df["tolerance"]) & (
            df["tolerance"] <= max(tolerance_range)
        )
        df["tolerance_success"] = tolerence_bool
        df["perov_coords"] = perovskite_coordinate_check(
            d["init_coords"],
            d["dir_coords"],
            d["size"],
            limit_of_displacement=limit_coords_displacement,
        )
        df["perov_success"] = df["perov_coords"]
    else:
        df["tolerance_success"] = [np.nan] * df.shape[0]
        df["tolerance"] = [np.nan] * df.shape[0]
        df["perov_coords"] = [np.nan] * df.shape[0]
        df["perov_success"] = [np.nan] * df.shape[0]

    return df


def assign_success_flags(
    df: pd.DataFrame,
    bg_loss_mode: str,
    crystal_system: Optional[str],
    perovskite_size: Optional[str],
) -> pd.DataFrame:
    """
    Assign final success flags based on target property constraints,
    crystal type, and optimization goal (e.g., maximization of bandgap).

    Args:
        df (pd.DataFrame): DataFrame containing property evaluation results.
        bg_loss_mode (str): 'maximization' or numerical target.
        crystal_system (Optional[str]): e.g., 'perovskite'.
        perovskite_size (Optional[str]): e.g., '3x3x3'.

    Returns:
        pd.DataFrame: DataFrame with 'success' column and possibly additional bandgap thresholds.
    """
    if bg_loss_mode == "maximization" and crystal_system is None:
        df["success"] = df["Ef_success"] & df["is_neutral_cmn"]
        for bg_above in [5.0, 10.0, 15.0, 20.0]:
            df["bandgap_above_{}".format(bg_above)] = df["bandgap_onehot"] > bg_above

    elif crystal_system == "perovskite" and perovskite_size in ["3x3x3"]:
        df["success"] = (
            df["tolerance_success"]
            & df["bandgap_success"]
            & df["perov_success"]
            & df["Ef_success"]
            # & df["is_neutral_cmn"] # remove neutrality check for 3x3x3
        )
    elif crystal_system == "perovskite":
        df["success"] = (
            df["tolerance_success"]
            & df["bandgap_success"]
            & df["perov_success"]
            & df["Ef_success"]
            & df["is_neutral_cmn"]
        )
    else:
        df["success"] = df["is_neutral_cmn"] & df["bandgap_success"] & df["Ef_success"]

    return df


def select_and_reorder_columns(
    df: pd.DataFrame, bg_loss_mode: str, crystal_system: Optional[str]
) -> pd.DataFrame:
    """
    Select and reorder the final columns to be saved in the CSV file.

    Args:
        df (pd.DataFrame): DataFrame after all evaluation.
        bg_loss_mode (str): Loss mode, e.g., 'maximization'.
        crystal_system (Optional[str]): Type of crystal system, e.g., 'perovskite'.

    Returns:
        pd.DataFrame: Filtered and reordered DataFrame.
    """
    columns = [
        "success",
        "bandgap_success",
        "tolerance_success",
        "perov_success",
        "is_neutral_cmn",
        "tolerance",
        "batch_abc",
        "batch_angle",
        "num_atoms",
        "bandgap_onehot",
        "ef_onehot",
        "tolerance",
        "minbond_less_than_0.5",
        "elements_cmn",
        "ox_states_cmn",
    ]

    if bg_loss_mode == "maximization" and crystal_system is None:
        columns.remove("bandgap_success")
        columns += [
            "bandgap_above_5.0",
            "bandgap_above_10.0",
            "bandgap_above_15.0",
            "bandgap_above_20.0",
        ]

    return df[columns]


def rename_bandgap_columns(df: pd.DataFrame, main_target_name: str) -> pd.DataFrame:
    """
    Rename 'bandgap' columns to the actual target name if different from 'bandgap'.

    Args:
        df (pd.DataFrame): DataFrame containing target-related columns.
        main_target_name (str): Actual target name (e.g., 'tc' or 'bulk_modulus').

    Returns:
        pd.DataFrame: DataFrame with renamed columns if necessary.
    """
    if main_target_name != "bandgap":
        for col in df.columns:
            if "bandgap" in col:
                new_col = col.replace("bandgap", main_target_name)
                df.rename(columns={col: new_col}, inplace=True)
                print(col, "→", new_col)
    return df


def sort_elms(elms):
    try:
        ast_elms = ast.literal_eval(elms)
        ast_elms.sort()
    except ValueError:
        ast_elms = "nan"
    return str(ast_elms)


def analysis_unique_crystals(each_df, megnet_elms_set, element_col_name="elements_cmn"):
    """
    最適化結果にどれだけのユニークな結晶が含まれているかを分析する。分析方法や元素の組み合せ。
    """

    unique_elms = set(each_df[element_col_name].apply(sort_elms).tolist())
    num_unique_all = len(unique_elms)
    unique_success_elms = set(
        each_df[each_df["success"]][element_col_name].apply(sort_elms).tolist()
    )

    unique_elms = unique_elms - {"nan"}
    unique_success_elms = unique_success_elms - {"nan"}

    num_unique_success = len(unique_success_elms)
    all_nuique_rate = num_unique_all / each_df.shape[0]
    success_nuique_rate = (
        num_unique_success / each_df[each_df["success"]].shape[0]
        if each_df[each_df["success"]].shape[0] != 0
        else np.nan
    )

    newly_megnet_all = len(unique_elms - (unique_elms & megnet_elms_set))
    newly_megnet_success = len(
        unique_success_elms - (unique_success_elms & megnet_elms_set)
    )
    assert newly_megnet_all >= 0 and newly_megnet_success >= 0
    success_unique_str = f"{num_unique_success}/{each_df[each_df['success']].shape[0]}"
    newly_megnet_success_str = (
        f"{newly_megnet_success}/{each_df[each_df['success']].shape[0]}"
    )

    newly_megnet_success_rate = (
        newly_megnet_success / each_df[each_df["success"]].shape[0]
        if each_df[each_df["success"]].shape[0] != 0
        else "-"
    )

    return (
        all_nuique_rate,
        success_nuique_rate,
        newly_megnet_all / each_df.shape[0],
        newly_megnet_success_rate,
        success_unique_str,
        newly_megnet_success_str,
    )


def summarize_results_and_novelty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform uniqueness and novelty analysis, and summarize success-related metrics.

    Args:
        df (pd.DataFrame): Evaluation results.

    Returns:
        pd.DataFrame: Summary dataframe containing success rates and novelty metrics.
    """

    megnet_elms_set = create_MEGNet_element_set()

    # Unique and novelty analysis
    (
        all_unique_rate,
        success_unique_rate,
        newly_megnet_all,
        newly_megnet_success,
        success_unique_str,
        newly_megnet_success_str,
    ) = analysis_unique_crystals(df, megnet_elms_set)

    # Score summary
    summary_dict = (df.loc[:, df.dtypes == bool].sum() / df.shape[0]).to_dict()
    summary_dict["success_nuique_rate"] = success_unique_rate
    summary_dict["newly_megnet_success"] = newly_megnet_success

    summary_df = pd.DataFrame([summary_dict])
    return summary_df


def evaluation_and_analysis(
    npz_path: str,
    bg_loss_mode: str,
    poscar_saved_dir: str,
    history_img_path: str,
    crystal_system: Optional[str],
    perovskite_size: Optional[str],
    gap_min_eval: float,
    gap_max_eval: float,
    model_name: str,
    num_candidate: int,
    tolerance_range: tuple,
    e_form_criteria: float,
    limit_coords_displacement: Optional[float],
    main_target_name: str,
    acceptable_margin: float = 1.0,
):
    # =============================
    # 1. Load optimization results and construct initial DataFrame
    # =============================
    csv_path = npz_path.replace(".npz", ".csv")
    df = load_and_prepare_dataframe(npz_path, num_candidate, e_form_criteria, csv_path)
    display(df)

    # =============================
    # 2. Save optimized structures and check charge neutrality & bond lengths
    # =============================
    df2 = check_neurality_bondlength_and_save_structure(
        npz_path,
        csv_path,
        poscar_saved_dir,
        crystal_system,
        perovskite_size,
        neutral_check=True,
        acceptable_margin=acceptable_margin,
    )
    clear_output()

    # =============================
    # 3. Visualize optimization history (top 20 candidates)
    # =============================
    optimization_history_display_for_promising_candidates(
        gap_min_eval,
        gap_max_eval,
        npz_path,
        csv_path,
        model_name,
        num_display=20,
        history_img_path=history_img_path,
    )

    # =============================
    # 4. Merge evaluation results
    # =============================
    df = merge_evaluation_results(df, df2, csv_path)
    if df.empty:
        return

    # =============================================================
    # 5. Evaluate perovskite-specific properties
    # =============================================================
    df = evaluate_perovskite_properties(
        df,
        np.load(npz_path, allow_pickle=True),
        crystal_system,
        tolerance_range,
        limit_coords_displacement,
    )

    # =============================================================
    # 6. Assign final success flags based on task objective and crystal constraints
    # =============================================================
    df = assign_success_flags(df, bg_loss_mode, crystal_system, perovskite_size)

    # =============================================================
    # 7. Perform uniqueness analysis and summarize success metrics
    # =============================================================
    summary_df = summarize_results_and_novelty(df)
    summary_df = rename_bandgap_columns(summary_df, main_target_name)

    # =============================================================
    # 8. Select and rename final output columns for CSV export
    # =============================================================
    df = select_and_reorder_columns(df, bg_loss_mode, crystal_system)
    df = rename_bandgap_columns(df, main_target_name)

    df.to_csv(csv_path)
    summary_df.to_csv(csv_path.replace(".csv", "_summary.csv"))
    print(summary_df)
