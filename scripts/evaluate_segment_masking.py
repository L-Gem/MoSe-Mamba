"""Evaluate segment-masking robustness for DepMamba checkpoints.

This script compares:
1. Original DepMamba / mean pooling.
2. Motion-aware segment pooling with alpha=0.2.

It loads existing checkpoints only. No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
import types
from importlib import import_module
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASKED_SEGMENTS = [0, 1, 2, 3, 4]


def _win_long_path(path: Path | str) -> str:
    path = os.path.abspath(os.fspath(path))
    if os.name != "nt" or path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def _display_path(path: Path | str) -> str:
    text = os.fspath(path)
    if text.startswith("\\\\?\\UNC\\"):
        return "\\\\" + text[8:]
    if text.startswith("\\\\?\\"):
        return text[4:]
    return text


def _path_exists(path: Path | str) -> bool:
    return os.path.exists(os.fspath(path)) or os.path.exists(_win_long_path(path))


def _iterdir(path: Path | str):
    root = Path(_win_long_path(path))
    if not root.exists():
        return []
    return list(root.iterdir())


def _read_text(path: Path | str) -> str:
    with open(_win_long_path(path), "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_yaml(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Run this script in the DepMamba evaluation "
            "environment, or install pyyaml there."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_runtime_dependencies():
    required = [
        "torch",
        "yaml",
        "speechbrain",
        "mamba_ssm",
        "einops",
        "causal_conv1d",
        "selective_scan_cuda",
    ]
    missing = []
    for name in required:
        try:
            import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies required for real checkpoint inference: "
            + ", ".join(missing)
            + ". Run this script in the same environment used to evaluate DepMamba."
        )


def _matplotlib_available():
    try:
        import_module("matplotlib")
    except ImportError:
        return False
    return True

def _parse_result_metrics(project_root: Path, result_subdir: str, result_stem: str):
    result_path = project_root / result_subdir / f"{result_stem}.txt"
    if not _path_exists(result_path):
        return None
    text = _read_text(result_path)
    metrics = {}
    for key in ("Accuracy", "Precision", "Recall", "F1", "Avg"):
        match = re.search(rf"{key}:([0-9.]+)", text)
        if match:
            metrics[key.lower() if key != "F1" else "f1"] = float(match.group(1))
    return metrics or None


def _metric_avg(metrics):
    return (
        metrics["accuracy"]
        + metrics["precision"]
        + metrics["recall"]
        + metrics["f1"]
    ) / 4.0


def _find_baseline_checkpoint(project_root: Path, run_index: int | None):
    candidates = []
    for output_subdir in ("outputs", "save_dir"):
        output_root = project_root / output_subdir
        for child in _iterdir(output_root):
            name = child.name
            match = re.fullmatch(r"dvlog_DepMamba_(\d+)", name)
            if not match:
                continue
            run = int(match.group(1))
            if run_index is not None and run != run_index:
                continue
            checkpoint = child / "checkpoints" / "best_model.pt"
            if not _path_exists(checkpoint):
                continue
            metrics = _parse_result_metrics(project_root, "results", name)
            avg = metrics.get("avg", _metric_avg(metrics)) if metrics else -1.0
            candidates.append((avg, run, output_subdir, checkpoint, name))
    if not candidates:
        raise FileNotFoundError(
            "No formal baseline checkpoint found in outputs/ or save_dir/. "
            "Pass --baseline-checkpoint explicitly."
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2] == "outputs"), reverse=True)
    return candidates[0][3], candidates[0][4], candidates[0][0]


def _find_motion_checkpoint(project_root: Path, alpha_token: str, run_index: int | None):
    output_root = project_root / "outputs_motion"
    candidates = []
    for child in _iterdir(output_root):
        name = child.name
        if f"_a{alpha_token}_" not in name or "cgcmal" in name.lower():
            continue
        match = re.search(r"_(\d+)$", name)
        run = int(match.group(1)) if match else -1
        if run_index is not None and run != run_index:
            continue
        checkpoint = child / "checkpoints" / "best_model.pt"
        if not _path_exists(checkpoint):
            continue
        metrics = _parse_result_metrics(project_root, "results_motion", name)
        avg = metrics.get("avg", _metric_avg(metrics)) if metrics else -1.0
        candidates.append((avg, run, checkpoint, name))
    if not candidates:
        raise FileNotFoundError(
            f"No formal motion checkpoint matched alpha token a{alpha_token}. "
            "Pass --motion-checkpoint explicitly."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3], candidates[0][0]


def _ensure_models_package(project_root: Path):
    package = types.ModuleType("models")
    package.__path__ = [str(project_root / "models")]
    sys.modules["models"] = package


def _load_class_from_file(project_root: Path, module_name: str, file_name: str):
    _ensure_models_package(project_root)
    spec = importlib.util.spec_from_file_location(
        module_name,
        project_root / "models" / file_name,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.DepMamba


def _load_dvlog_loader(project_root: Path):
    spec = importlib.util.spec_from_file_location(
        "depmamba_dvlog_loader",
        project_root / "datasets" / "dvlog.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["depmamba_dvlog_loader"] = module
    spec.loader.exec_module(module)
    return module.get_dvlog_dataloader


def _state_dict_without_module_prefix(state_dict):
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def _load_checkpoint(model, checkpoint, device):
    import torch

    with open(_win_long_path(checkpoint), "rb") as f:
        state = torch.load(f, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(_state_dict_without_module_prefix(state), strict=True)


def _common_model_kwargs(config):
    kwargs = dict(config["mmmamba"])
    for key in (
        "use_global_alignment",
        "pooling_mode",
        "pooling_num_segments",
        "pooling_scales",
        "pooling_scale_fusion",
        "pooling_dropout",
        "pooling_residual",
        "pooling_residual_alpha",
        "pooling_motion_emb_dim",
        "pooling_motion_cue_mode",
        "pooling_reliability_beta",
        "pooling_gate_emb_dim",
        "pooling_gamma_init",
        "pooling_gamma_max",
        "gate_cue_mode",
        "use_dynamic_temporal_refinement",
        "dynamic_refinement_layers",
        "dynamic_refinement_kernel_size",
        "dynamic_refinement_gate_activation",
        "dynamic_refinement_dyn_type",
        "dynamic_refinement_dropout",
        "dynamic_refinement_residual_scale",
    ):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


def _build_models(args, baseline_config, motion_config, device):
    baseline_cls = _load_class_from_file(
        args.project_root,
        "models.DepMamba_baseline_eval",
        "DepMamba.py",
    )
    motion_cls = _load_class_from_file(
        args.project_root,
        "models.DepMamba_eval",
        "DepMamba.py",
    )

    baseline_kwargs = _common_model_kwargs(baseline_config)
    baseline_kwargs["pooling_mode"] = "mean"

    motion_kwargs = _common_model_kwargs(motion_config)
    motion_kwargs["pooling_mode"] = "motion_segment_attention"
    motion_kwargs["pooling_num_segments"] = args.num_segments
    motion_kwargs["pooling_residual"] = True
    motion_kwargs["pooling_residual_alpha"] = args.motion_alpha
    motion_kwargs["pooling_motion_emb_dim"] = 32
    motion_kwargs["pooling_motion_cue_mode"] = "full"

    baseline_model = baseline_cls(**baseline_kwargs)
    motion_model = motion_cls(**motion_kwargs)
    _load_checkpoint(baseline_model, args.baseline_checkpoint, device)
    _load_checkpoint(motion_model, args.motion_checkpoint, device)
    baseline_model.to(device).eval()
    motion_model.to(device).eval()
    return baseline_model, motion_model


def _segment_bounds(valid_length: int, num_segments: int):
    return [
        (
            (segment_idx * valid_length) // num_segments,
            ((segment_idx + 1) * valid_length) // num_segments,
        )
        for segment_idx in range(num_segments)
    ]


def _random_drop_indices(sample_index: int, masked_segments: int, num_segments: int, seed: int, repeat_idx: int):
    if masked_segments <= 0:
        return []
    rng_seed = int(seed) + int(repeat_idx) * 1_000_003 + int(sample_index) * 9_176
    rng = np.random.default_rng(rng_seed)
    return sorted(int(v) for v in rng.choice(num_segments, size=masked_segments, replace=False))


def _importance_drop_indices(weights, valid_segments, masked_segments: int, num_segments: int):
    if masked_segments <= 0:
        return []
    scores = np.asarray(weights, dtype=float).copy()
    valid = np.asarray(valid_segments, dtype=bool)
    scores[~valid] = -np.inf
    ranked = list(np.argsort(-scores))
    selected = []
    for segment_idx in ranked:
        if np.isfinite(scores[segment_idx]):
            selected.append(int(segment_idx))
        if len(selected) == masked_segments:
            break
    if len(selected) < masked_segments:
        for segment_idx in ranked:
            if int(segment_idx) not in selected:
                selected.append(int(segment_idx))
            if len(selected) == masked_segments:
                break
    return sorted(selected)


def _compact_mask_segments(x, padding_mask, drop_indices_per_sample, num_segments: int):
    import torch

    masked_x = torch.zeros_like(x)
    masked_padding_mask = torch.zeros_like(padding_mask)
    batch_size = x.shape[0]
    for sample_idx in range(batch_size):
        valid_length = int(padding_mask[sample_idx].sum().item())
        if valid_length <= 0:
            raise ValueError("Encountered a sample with no valid frames before masking.")
        drop_set = set(int(v) for v in drop_indices_per_sample[sample_idx])
        chunks = []
        for segment_idx, (start, end) in enumerate(_segment_bounds(valid_length, num_segments)):
            if segment_idx in drop_set or end <= start:
                continue
            chunks.append(x[sample_idx, start:end, :])
        if not chunks:
            raise ValueError(
                "Masking removed all effective frames. Reduce --max-masked-segments "
                "or check very short test samples."
            )
        compacted = torch.cat(chunks, dim=0)
        new_length = compacted.shape[0]
        masked_x[sample_idx, :new_length, :] = compacted
        masked_padding_mask[sample_idx, :new_length] = 1
    return masked_x, masked_padding_mask


def _metrics_from_counts(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    metrics["avg"] = _metric_avg(metrics)
    return metrics


def _update_counts(counts, logits, labels):
    preds = (logits.squeeze(1) > 0.0).long()
    labels = labels.long()
    counts["tp"] += int(((preds == 1) & (labels == 1)).sum().item())
    counts["fp"] += int(((preds == 1) & (labels == 0)).sum().item())
    counts["tn"] += int(((preds == 0) & (labels == 0)).sum().item())
    counts["fn"] += int(((preds == 0) & (labels == 1)).sum().item())



def _forward_with_analysis(model, x, mask, labels=None):
    """Compatibility analysis forward for the final DepMamba implementation."""
    import torch

    raw_x = x
    z = model.temporal_feature_extractor(x, mask)

    h_mean = model.mean_pool_temporal_features(z, mask)

    segment_reprs, segment_valids = model._segment_representations(
        z,
        mask,
        num_segments=model.pooling_num_segments,
    )

    motion_cues = model._segment_motion_cues(
        raw_x,
        mask,
        num_segments=model.pooling_num_segments,
    )

    motion_emb = model.motion_encoder(motion_cues)
    score_input = torch.cat([segment_reprs, motion_emb], dim=-1)
    segment_scores = model.motion_segment_score(score_input).squeeze(-1)
    segment_scores = segment_scores.masked_fill(~segment_valids, -1e9)

    segment_weights = torch.softmax(segment_scores, dim=1)
    h_motion = torch.sum(
        segment_weights.unsqueeze(-1) * segment_reprs,
        dim=1,
    )

    if model.pooling_residual:
        alpha = float(model.pooling_residual_alpha)
        h_final = (1.0 - alpha) * h_mean + alpha * h_motion
    else:
        h_final = h_motion

    logits = model.classifier(h_final)

    analysis = {
        "segment_weights": segment_weights.detach(),
        "motion_cues": {
            "visual_motion_mean": motion_cues[:, :, 0].detach(),
            "visual_motion_std": motion_cues[:, :, 1].detach(),
            "audio_change_mean": motion_cues[:, :, 2].detach(),
            "audio_change_std": motion_cues[:, :, 3].detach(),
        },
        "motion_cues_tensor": motion_cues.detach(),
        "valid_segment_mask": segment_valids.detach(),
        "segment_representations": segment_reprs.detach(),
        "segment_scores": segment_scores.detach(),
        "z": z.detach(),
        "h_mean": h_mean.detach(),
        "h_motion": h_motion.detach(),
        "h_final": h_final.detach(),
        "motion_alpha": float(model.pooling_residual_alpha),
        "logits": logits.detach(),
        "predicted_probabilities": torch.sigmoid(logits).detach(),
    }

    if labels is not None:
        analysis["true_labels"] = labels.detach()

    return logits, analysis


def _evaluate_pair(
    baseline_model,
    motion_model,
    loader,
    device,
    strategy: str,
    masked_segments: int,
    num_segments: int,
    seed: int,
    repeat_idx: int,
):
    import torch

    counts = {
        "Original DepMamba / Mean Pooling": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "Motion-Aware Segment Pooling (alpha=0.2)": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }
    sample_offset = 0
    with torch.no_grad():
        for x, y, padding_mask in loader:
            x = x.to(device)
            y = y.to(device)
            padding_mask = padding_mask.to(device)
            batch_size = x.shape[0]

            if strategy == "random":
                drop_indices = [
                    _random_drop_indices(
                        sample_offset + sample_idx,
                        masked_segments,
                        num_segments,
                        seed,
                        repeat_idx,
                    )
                    for sample_idx in range(batch_size)
                ]
            elif strategy == "importance":
                if masked_segments == 0:
                    drop_indices = [[] for _ in range(batch_size)]
                else:
                    _, analysis = _forward_with_analysis(motion_model, x, padding_mask, labels=y)
                    weights = analysis["segment_weights"].detach().cpu().numpy()
                    valid_segments = analysis["valid_segment_mask"].detach().cpu().numpy()
                    drop_indices = [
                        _importance_drop_indices(
                            weights[sample_idx],
                            valid_segments[sample_idx],
                            masked_segments,
                            num_segments,
                        )
                        for sample_idx in range(batch_size)
                    ]
            else:
                raise ValueError(f"Unsupported masking strategy: {strategy}")

            masked_x, masked_padding_mask = _compact_mask_segments(
                x,
                padding_mask,
                drop_indices,
                num_segments,
            )
            baseline_logits = baseline_model(masked_x, masked_padding_mask)
            motion_logits = motion_model(masked_x, masked_padding_mask)
            _update_counts(
                counts["Original DepMamba / Mean Pooling"],
                baseline_logits,
                y,
            )
            _update_counts(
                counts["Motion-Aware Segment Pooling (alpha=0.2)"],
                motion_logits,
                y,
            )
            sample_offset += batch_size

    return {
        model_name: _metrics_from_counts(**model_counts)
        for model_name, model_counts in counts.items()
    }


def _aggregate_random_runs(run_metrics):
    values_by_model = {}
    for metrics_for_models in run_metrics:
        for model_name, metrics in metrics_for_models.items():
            values_by_model.setdefault(model_name, []).append(metrics)

    aggregated = {}
    for model_name, metrics_list in values_by_model.items():
        row = {}
        for key in ("accuracy", "precision", "recall", "f1", "avg"):
            vals = np.asarray([metrics[key] for metrics in metrics_list], dtype=float)
            row[key] = float(vals.mean())
            row[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        aggregated[model_name] = row
    return aggregated


def _result_rows(
    strategy,
    masked_segments,
    metrics_by_model,
    random_repeats,
    seed,
    baseline_checkpoint,
    motion_checkpoint,
):
    rows = []
    for model_name, metrics in metrics_by_model.items():
        checkpoint = (
            baseline_checkpoint
            if model_name.startswith("Original")
            else motion_checkpoint
        )
        row = {
            "strategy": strategy,
            "model": model_name,
            "masked_segments": masked_segments,
            "mask_ratio": f"{masked_segments}/8",
            "random_repeats": random_repeats,
            "random_seed": seed,
            "mask_layout": "compact_remaining_segments",
            "checkpoint": _display_path(checkpoint),
        }
        for key in ("accuracy", "precision", "recall", "f1", "avg"):
            row[key] = metrics[key]
            row[f"{key}_std"] = metrics.get(f"{key}_std", 0.0)
        rows.append(row)
    return rows


def _write_results_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "model",
        "masked_segments",
        "mask_ratio",
        "accuracy",
        "accuracy_std",
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "f1",
        "f1_std",
        "avg",
        "avg_std",
        "random_repeats",
        "random_seed",
        "mask_layout",
        "checkpoint",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_results(rows, svg_path: Path, pdf_path: Path, plot_metric: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    model_styles = {
        "Original DepMamba / Mean Pooling": {
            "color": "#555555",
            "marker": "o",
            "label": "Original / Mean",
        },
        "Motion-Aware Segment Pooling (alpha=0.2)": {
            "color": "#2B6CB0",
            "marker": "s",
            "label": "Motion-Aware, alpha=0.2",
        },
    }
    strategy_titles = {
        "random": "Random Masking",
        "importance": "Importance-guided Masking",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), sharey=True)
    for ax, strategy in zip(axes, ("random", "importance")):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        for model_name, style in model_styles.items():
            model_rows = sorted(
                [row for row in strategy_rows if row["model"] == model_name],
                key=lambda item: int(item["masked_segments"]),
            )
            xs = np.asarray([int(row["masked_segments"]) for row in model_rows], dtype=float)
            ys = np.asarray([float(row[plot_metric]) for row in model_rows], dtype=float)
            yerr = np.asarray(
                [float(row.get(f"{plot_metric}_std", 0.0)) for row in model_rows],
                dtype=float,
            )
            if strategy == "random":
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.2,
                    linewidth=1.15,
                    capsize=2.2,
                    label=style["label"],
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=3.2,
                    linewidth=1.15,
                    label=style["label"],
                )
        ax.set_title(strategy_titles[strategy])
        ax.set_xlabel("Masked segments out of 8")
        ax.set_xticks(MASKED_SEGMENTS)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(width=0.6, length=2.5)
    axes[0].set_ylabel(plot_metric.upper())
    axes[1].legend(loc="lower left", frameon=False)
    fig.text(
        0.5,
        0.01,
        "Random masking shows mean +/- std over repeated fixed-seed trials; importance-guided masks use Motion model segment weights.",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0], w_pad=1.2)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate time-segment masking robustness without retraining."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--baseline-config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--motion-config", type=Path, default=PROJECT_ROOT / "config" / "config_motion.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--motion-checkpoint", type=Path, default=None)
    parser.add_argument("--baseline-run-index", type=int, default=1)
    parser.add_argument("--motion-run-index", type=int, default=1)
    parser.add_argument("--motion-alpha", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-gender", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--max-masked-segments", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=2024)
    parser.add_argument("--random-repeats", type=int, default=10)
    parser.add_argument("--plot-metric", choices=["accuracy", "precision", "recall", "f1", "avg"], default="avg")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


def main():
    args = parse_args()
    args.project_root = args.project_root.resolve()
    baseline_config = _load_yaml(args.baseline_config)
    motion_config = _load_yaml(args.motion_config)
    data_dir = args.data_dir or Path(motion_config.get("data_dir", baseline_config["data_dir"]))
    if not _path_exists(data_dir / "labels.csv"):
        raise FileNotFoundError(
            f"D-Vlog labels.csv not found at {_display_path(data_dir / 'labels.csv')}. "
            "Use --data-dir to point to the D-Vlog feature directory."
        )

    if args.num_segments != 8:
        raise ValueError("This experiment is defined for exactly 8 segments.")
    if args.max_masked_segments >= args.num_segments:
        raise ValueError("max-masked-segments must leave at least one segment unmasked.")
    masked_segments_list = [k for k in MASKED_SEGMENTS if k <= args.max_masked_segments]

    alpha_token = str(args.motion_alpha).replace(".", "p")
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint, baseline_name, baseline_avg = _find_baseline_checkpoint(
            args.project_root,
            args.baseline_run_index,
        )
        print(
            f"Selected baseline checkpoint: {_display_path(args.baseline_checkpoint)} "
            f"(experiment={baseline_name}, Avg={baseline_avg:.6f})"
        )
    if args.motion_checkpoint is None:
        args.motion_checkpoint, motion_name, motion_avg = _find_motion_checkpoint(
            args.project_root,
            alpha_token,
            args.motion_run_index,
        )
        print(
            f"Selected motion checkpoint: {_display_path(args.motion_checkpoint)} "
            f"(experiment={motion_name}, Avg={motion_avg:.6f})"
        )

    _check_runtime_dependencies()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    get_dvlog_dataloader = _load_dvlog_loader(args.project_root)
    batch_size = args.batch_size or int(motion_config.get("batch_size", baseline_config.get("batch_size", 16)))
    test_gender = args.test_gender or motion_config.get("test_gender", baseline_config.get("test_gender", "both"))

    loader = get_dvlog_dataloader(
        data_dir,
        "test",
        batch_size,
        test_gender,
        aug=False,
    )
    baseline_model, motion_model = _build_models(args, baseline_config, motion_config, device)

    rows = []
    for masked_segments in masked_segments_list:
        random_runs = []
        for repeat_idx in range(args.random_repeats):
            random_runs.append(
                _evaluate_pair(
                    baseline_model,
                    motion_model,
                    loader,
                    device,
                    strategy="random",
                    masked_segments=masked_segments,
                    num_segments=args.num_segments,
                    seed=args.random_seed,
                    repeat_idx=repeat_idx,
                )
            )
        random_metrics = _aggregate_random_runs(random_runs)
        rows.extend(
            _result_rows(
                "random",
                masked_segments,
                random_metrics,
                args.random_repeats,
                args.random_seed,
                args.baseline_checkpoint,
                args.motion_checkpoint,
            )
        )

        importance_metrics = _evaluate_pair(
            baseline_model,
            motion_model,
            loader,
            device,
            strategy="importance",
            masked_segments=masked_segments,
            num_segments=args.num_segments,
            seed=args.random_seed,
            repeat_idx=0,
        )
        rows.extend(
            _result_rows(
                "importance",
                masked_segments,
                importance_metrics,
                1,
                args.random_seed,
                args.baseline_checkpoint,
                args.motion_checkpoint,
            )
        )

    csv_path = args.output_dir / "segment_masking_results.csv"
    svg_path = args.output_dir / "fig3_segment_masking.svg"
    pdf_path = args.output_dir / "fig3_segment_masking.pdf"
    _write_results_csv(rows, csv_path)
    print(f"Wrote {csv_path.resolve()}")

    if args.skip_plot:
        print("Skipping plot generation because --skip-plot was specified.")
        print("Fairness checks:")
        print("  - Both models use the same D-Vlog test loader and the same masked inputs.")
        print("  - Each sample is segmented using its original valid length before masking.")
        print("  - Masked visual and audio frames are removed together; remaining frames are compacted and trailing mask is set to 0.")
        print("  - Random masks are deterministic per sample/repeat with a fixed seed.")
        print("  - Importance-guided masks are selected from the Motion model weights on the unmasked input and then applied to both models.")
        return

    if not _matplotlib_available():
        print(
            "WARNING: matplotlib is not available; CSV was saved, "
            "and SVG/PDF plot generation was skipped."
        )
        print("Fairness checks:")
        print("  - Both models use the same D-Vlog test loader and the same masked inputs.")
        print("  - Each sample is segmented using its original valid length before masking.")
        print("  - Masked visual and audio frames are removed together; remaining frames are compacted and trailing mask is set to 0.")
        print("  - Random masks are deterministic per sample/repeat with a fixed seed.")
        print("  - Importance-guided masks are selected from the Motion model weights on the unmasked input and then applied to both models.")
        return

    _plot_results(rows, svg_path, pdf_path, args.plot_metric)
    print(f"Wrote {_display_path(svg_path)}")
    print(f"Wrote {_display_path(pdf_path)}")
    print("Fairness checks:")
    print("  - Both models use the same D-Vlog test loader and the same masked inputs.")
    print("  - Each sample is segmented using its original valid length before masking.")
    print("  - Masked visual and audio frames are removed together; remaining frames are compacted and trailing mask is set to 0.")
    print("  - Random masks are deterministic per sample/repeat with a fixed seed.")
    print("  - Importance-guided masks are selected from the Motion model weights on the unmasked input and then applied to both models.")


if __name__ == "__main__":
    main()
