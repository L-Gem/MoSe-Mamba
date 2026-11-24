# MoSe-Mamba

**MoSe-Mamba: A Motion–Semantic Segment Aggregation Model for Multimodal Depression Detection**

Research code associated with the full paper accepted at IEEE SSCI 2027. This repository snapshot is organized from the authors' archived experiments. It is a release candidate, not an independently validated reproduction package.

## Overview

MoSe-Mamba investigates video-level temporal aggregation on top of the DepMamba backbone. The paper describes three components: Multimodal Dynamic Cue Encoder (MDCE), Motion–Semantic Segment Reweighting (MSSR), and Residual Temporal Aggregation (RTA). The archived D-Vlog and LMVD implementations are kept separately because they use different model entry points.

## Repository layout

| Path | Purpose |
| --- | --- |
| `models/DepMamba.py` | Model imported by the archived D-Vlog training scripts |
| `models/DepMamba_motion.py` | Model imported by the archived LMVD training script |
| `models/mamba/` | Supporting Mamba implementation; upstream attribution retained |
| `main_pw08.py` | D-Vlog training, including positive-class loss weight 0.8 |
| `main_no_mdce_pw08.py` | D-Vlog ablation with zeroed motion embedding |
| `main_motion_pool.py` | LMVD training and cue-mode ablations |
| `config/` | Archived configuration defaults, with local paths substituted |
| `datasets/` | Dataset loading code only |
| `scripts/` | Archived segment analysis and masking utilities |
| `reported_results/` | Selected archived aggregate run metrics and calculated summaries |
| `docs/` | Reproduction notes, provenance and release restrictions |

## Environment and data

The archived code targets Python 3.10-era PyTorch/CUDA environments. Install a compatible version of PyTorch, Mamba SSM and causal-conv1d (including the selective-scan CUDA extension), then the remaining dependencies in `requirements.txt`. Exact package versions have not been recovered from the supplied archive. See the [original DepMamba repository](https://github.com/Jiaxin-Ye/DepMamba) for upstream setup information.

D-Vlog and LMVD are third-party datasets. Obtain the data and features through their authorized distribution channels and comply with their access and licensing terms. This snapshot does not distribute participant IDs, labels, features, videos or checkpoints. Set `data_dir` in the YAML configurations or pass `--data_dir` to the training scripts. See `docs/REPRODUCTION.md` for the expected layout and example commands.

## Experimental records

`reported_results/per_run.csv` records selected run-level metrics directly from the supplied archive. `reported_results/summary.csv` recomputes means and both sample and population standard deviations. These are archived results, not newly trained experiments. Some paper rows (including the reproduced DepMamba baseline and LMVD controlled Base) are not established by the selected records. See `docs/REPRODUCTION.md` before claiming exact reproduction.

## Acknowledgments and licensing

This work builds on J. Ye, J. Zhang and H. Shan, *DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection*, ICASSP 2025. Upstream source: https://github.com/Jiaxin-Ye/DepMamba. Mamba-derived files retain their original copyright and attribution comments, including references to Vision Mamba and Tri Dao / Albert Gu.

No new license is granted by this packaging step. The authors must confirm rights to redistribute upstream and modified code and obtain co-author/institutional approval before public release. See `docs/RELEASE_CHECKLIST.md`.

## Citation

MoSe-Mamba: A Motion–Semantic Segment Aggregation Model for Multimodal Depression Detection. Accepted at IEEE SSCI 2027. Publication metadata and DOI will be added when available.

This is research software, not a clinical diagnostic tool.
