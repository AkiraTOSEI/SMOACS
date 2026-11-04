# 実験条件一覧

各Sectionのスクリプトを、そのSectionのディレクトリから `bash run_section_2_*.sh` で実行します。
新しい最適化の出力はリポジトリ直下の `results/<実験名>/` に保存します。
以下は通常実行の35条件（16 + 8 + 5 + 6）です。
Section 2.1は各条件をseed 42・43・44・45で4回実行するため、全体では83回
（16 × 4 + 8 + 5 + 6）の最適化を実行します。
`SMOACS_SMOKE_TEST=1` を付けると全35条件を候補数2・batch 2・10 stepsで各1回実行し、
実験名に `__smoke` を付けます。Section 2.1もseed 42のみを使用し、4回平均の集計を省略します。
mutationとhybrid gradient区間の縮小設定は
[README.md](README.md#minimal-execution-check)を参照してください。

## Section 2.1

Run script: [run_section_2_1.sh](section_2_1/run_section_2_1.sh)

Inputs: the first 128 of the 256 POSCARs listed in
`section_2_1/templates_masks/initial_structures/template_order.txt`
and their frozen per-template arrays in `section_2_1/templates_masks/masks/initial_masks.npz`.
Each condition uses Crystalformer, 128 candidates, batch size 128, and 200 steps.
Each condition is repeated four times with seeds 42, 43, 44, and 45 (64 runs total).
The target band gap is 2.5 eV, with an optimization half-width of 0.04 eV
and an evaluation half-width of 0.2 eV.
Atom, lattice, and coordinate learning rates are 0.007, 0.002, and 0.005.
Formation-energy optimization is enabled, with a success criterion of `< -0.5 eV/atom`.

| Initialization | Geometry | Oxidation-state mask | Methods |
| --- | --- | --- | --- |
| `random-random` | Randomized | Disabled | `grad`, `grad_sa`, `mutation`, `mutation_sa` |
| `random-AtomicMask` | Randomized | Enabled | `grad`, `grad_sa`, `mutation`, `mutation_sa` |
| `Struct-random` | Template | Disabled | `grad`, `grad_sa`, `mutation`, `mutation_sa` |
| `Struct-AtomicMask` | Template | Enabled | `grad`, `grad_sa`, `mutation`, `mutation_sa` |

All groups use uniform initial element distributions. Seeds are reset before each run;
for a given seed, randomized geometries are shared across the four methods
and retain the template atom counts and lattice scaling.
The experiment name is `section_2_1_<initialization>_<method>__seed<seed>`.
After all 64 runs finish, [summarize_runs.py](section_2_1/summarize_runs.py)
writes the arithmetic means of the run-level metrics across the four seeds for each condition
to `results/section_2_1_mean.csv` (16 rows).
Details of SA and mutation are given in [README.md](README.md).

## Section 2.2

実行スクリプト：[run_section_2_2.sh](section_2_2/run_section_2_2.sh)

入力は `section_2_2/templates_masks/historical_unit_cell/` と `historical_3x3x3/` の各128件です。
`template_order.txt` の順で読み込み、共用マスクは `templates_masks/masks/` に置きます。
実行時はリポジトリ直下の `data/intermidiate/` にある共用マスクとの一致を確認します。
各条件を200 steps実行し、学習率は実行スクリプトで次の値を明示指定します。

Table 2に合わせ、全8条件の形成エネルギー成功条件を `E_f < -0.5 eV/atom` とします。
Crystalformerの3×3×3は、原稿とSI Table B.2に合わせてbatch sizeを32、
mutationのstepを120・180（0始まり）、`top_ratio`を0.42、
`mutation_noise`を0.02、`group_C_use_rate`を0.27に設定します。
smoke実行ではbatch sizeを2、mutationのstepを6・9に縮小します。

| モデル / セル | 格子学習率 | 座標学習率 | 元素分布学習率 |
| --- | --- | --- | --- |
| ALIGNN / 単位胞 | 9.0 | 0.002 | 0.00005 |
| Crystalformer / 単位胞 | 0.002 | 0.005 | 0.007 |
| ALIGNN / 3×3×3 | 0.01 | 0.000667 | 0.00004 |
| Crystalformer / 3×3×3 | 0.6 | 0.00067 | 0.002 |

| モデル | セル | 目標band gap (eV) | 候補数 | batch | seed | 実験名 |
| --- | --- | --- | --- | --- | --- | --- |
| ALIGNN | 単位胞 | 0.5 | 128 | 128 | 42 | `section_2_2_ALIGNN_unit_cell_bg0p5` |
| ALIGNN | 単位胞 | 4.0 | 128 | 128 | 42 | `section_2_2_ALIGNN_unit_cell_bg4p0` |
| Crystalformer | 単位胞 | 0.5 | 128 | 128 | 42 | `section_2_2_Crystalformer_unit_cell_bg0p5` |
| Crystalformer | 単位胞 | 4.0 | 128 | 128 | 42 | `section_2_2_Crystalformer_unit_cell_bg4p0` |
| ALIGNN | 3×3×3 | 0.5 | 128 | 128 | 42 | `section_2_2_ALIGNN_3x3x3_bg0p5` |
| ALIGNN | 3×3×3 | 4.0 | 128 | 128 | 42 | `section_2_2_ALIGNN_3x3x3_bg4p0` |
| Crystalformer | 3×3×3 | 0.5 | 128 | 32 | 42 | `section_2_2_Crystalformer_3x3x3_bg0p5` |
| Crystalformer | 3×3×3 | 4.0 | 128 | 32 | 42 | `section_2_2_Crystalformer_3x3x3_bg4p0` |

## Section 2.3

実行スクリプト：[run_section_2_3.sh](section_2_3/run_section_2_3.sh)

入力は `section_2_3/templates_masks/initial_mp_nsites2_12__mask_dict_super_common.pkl` の先頭512件です。
共用マスクは `templates_masks/masks/mask_dict_super_common.npz` です。モデルはALIGNNです。
各条件を600 steps実行し、`grad_sa_hybrid` のgradient区間は200 stepsです。

| 目標band gap (eV) | 候補数 | batch | seed | 実験名 |
| --- | --- | --- | --- | --- |
| 2.5 | 512 | 128 | 42 | `section_2_3_bg2p5` |
| 3.0 | 512 | 128 | 42 | `section_2_3_bg3p0` |
| 3.5 | 512 | 128 | 42 | `section_2_3_bg3p5` |
| 4.0 | 512 | 128 | 42 | `section_2_3_bg4p0` |
| 4.5 | 512 | 128 | 42 | `section_2_3_bg4p5` |

## Section 2.4

実行スクリプト：[run_section_2_4.sh](section_2_4/run_section_2_4.sh)

入力は `section_2_4/templates_masks/initial_mp_nsites2_30__mask_dict_super_common_level5.pkl` の先頭8,192件です。
マスクはPKLに含まれています。モデルはALIGNNです。
末尾92件のbatchも処理します。
各条件を650 steps実行し、`grad_sa_hybrid` のgradient区間は300 stepsです。
エネルギー損失にはMatterSim 5M、磁気モーメント損失には `jv_magmom_alignn` を使用します。

| 元素数損失重み | 整数磁気モーメント重み | 候補数 | batch | seed | 実験名 |
| --- | --- | --- | --- | --- | --- |
| 100 | 1 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w100__level5__w-mag-int1` |
| 1000 | 1 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w1000__level5__w-mag-int1` |
| 100 | 10 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w100__level5__w-mag-int10` |
| 1000 | 10 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w1000__level5__w-mag-int10` |
| 100 | 100 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w100__level5__w-mag-int100` |
| 1000 | 100 | 8192 | 150 | 42 | `section_2_4_best_trial00010__elm-w1000__level5__w-mag-int100` |
