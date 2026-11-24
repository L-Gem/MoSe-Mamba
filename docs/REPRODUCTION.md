# Reproduction notes

Run commands from the repository root, on an authorized CUDA environment. These commands are reconstructed from archived scripts, configurations and filenames, not recovered verbatim from the final training logs. They have not been executed end-to-end.

## Data layout

```text
data/
  dvlog/
    labels.csv
    <sample_id>/
      <sample_id>_visual.npy
      <sample_id>_acoustic.npy
  lmvd/
    labels.csv
    visual/<sample_id>_visual.npy
    audio/<sample_id>.npy
```

See `datasets/dvlog.py` and `datasets/lmvd.py` for the expected label format and split handling. The label files are deliberately excluded.

## D-Vlog

The archived `main_pw08.py` runs one seed per invocation; use seeds 2024, 2025 and 2026. Example commands for seed 2024, using saved defaults (not independently confirmed launch arguments):

```bash
# Base
python main_pw08.py -ds dvlog --data_dir ./data --seed 2024 --pooling_mode mean
# Full
python main_pw08.py -ds dvlog --data_dir ./data --seed 2024 --pooling_mode motion_segment_attention --pooling_residual true --pooling_residual_alpha 0.1 --pooling_motion_emb_dim 32
# w/o MDCE cues
python main_no_mdce_pw08.py -ds dvlog --data_dir ./data --seed 2024 --pooling_mode motion_segment_attention --pooling_residual true --pooling_residual_alpha 0.1 --pooling_motion_emb_dim 32
# w/o RTA
python main_pw08.py -ds dvlog --data_dir ./data --seed 2024 --pooling_mode motion_segment_attention --pooling_residual false --pooling_residual_alpha 0.1 --pooling_motion_emb_dim 32
```

## LMVD

`main_motion_pool.py` loops over three seeds starting at the supplied seed. `pooling_motion_cue_mode` controls the ablation in this implementation. Examples (not verified historical commands):

```bash
python main_motion_pool.py -ds lmvd --data_dir ./data --seed 2024 --pooling_motion_cue_mode full --pooling_residual_alpha 0.1
python main_motion_pool.py -ds lmvd --data_dir ./data --seed 2024 --pooling_motion_cue_mode no_motion --pooling_residual_alpha 0.1
python main_motion_pool.py -ds lmvd --data_dir ./data --seed 2024 --pooling_motion_cue_mode no_residual --pooling_residual_alpha 0.1
```

## Segment visualizations

The archived `scripts/export_segment_analysis.py` and `scripts/evaluate_segment_masking.py` require authorized data and matching trained checkpoints. Use their `--help` and explicitly provide checkpoint paths and the corresponding configuration. Their default alpha is 0.2; the paper uses 0.1. The exact checkpoint and plotting invocation still need verification.

## Parameter and result verification required

- The D-Vlog matching result filenames record `emb32`, and `models/DepMamba.py` uses the supplied `pooling_motion_emb_dim` to build the motion encoder. The paper states 64 dimensions.
- The LMVD matching filenames also record `emb32`, but the archived `DepMamba_motion.py` constructs the motion MLP using `motion_hidden_size=64` by default; the training script passes `pooling_motion_emb_dim` in keyword arguments, which is not the same constructor parameter. Do not infer the actual embedding width solely from filenames.
- Both saved YAML configurations default to learning rate `8e-5`; the manuscript states `1e-4`. CLI overrides were possible. Confirm final launch commands or logs before changing the manuscript.
- The selected three LMVD no-motion records recompute Precision to 67.47 and Recall to 73.63 (two decimals), while the current paper shows 67.46 and 73.62. Confirm the exact original aggregation/rounding method. The selected full and no-residual LMVD records match the reported means.
- The LMVD controlled Base and the reproduced DepMamba results are not established by the selected files. Do not substitute nearby exploratory results for missing final records.
- Exact CUDA/PyTorch package versions, final training invocation, and checkpoints were not supplied.

Source Python files are retained without algorithmic changes. Only two YAML path settings were made portable. No training or clinical validation was performed during packaging.
