"""Export real segment-level motion analysis for DepMamba.

The script loads a trained motion-aware DepMamba checkpoint, runs the D-Vlog
test set, exports true segment weights/cues to CSV, and plots the selected
samples for paper figures. It never synthesizes or randomizes analysis values.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
            "PyYAML is required. Run this script in the DepMamba training "
            "environment, or install pyyaml there."
        ) from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_runtime_dependencies():
    required = [
        "torch",
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
            + ". Run this script in the same environment used to train/evaluate DepMamba."
        )


def _matplotlib_available():
    try:
        import_module("matplotlib")
    except ImportError:
        return False
    return True

def _parse_result_avg(project_root: Path, experiment_name: str) -> float | None:
    result_path = project_root / "results_motion" / f"{experiment_name}.txt"
    if not _path_exists(result_path):
        return None
    match = re.search(r"Avg:([0-9.]+)", _read_text(result_path))
    return float(match.group(1)) if match else None


def _find_alpha_checkpoint(project_root: Path, alpha_token: str, run_index: int | None):
    output_root = project_root / "outputs_motion"
    candidates = []
    for child in _iterdir(output_root):
        name = child.name
        if f"_a{alpha_token}_" not in name:
            continue
        if "cgcmal" in name.lower():
            continue
        run_match = re.search(r"_(\d+)$", name)
        run = int(run_match.group(1)) if run_match else -1
        if run_index is not None and run != run_index:
            continue
        checkpoint = child / "checkpoints" / "best_model.pt"
        if not _path_exists(checkpoint):
            continue
        avg = _parse_result_avg(project_root, name)
        candidates.append((avg if avg is not None else -1.0, run, checkpoint, name))

    if not candidates:
        raise FileNotFoundError(
            f"No formal outputs_motion checkpoint matched alpha token a{alpha_token}. "
            "Pass --checkpoint explicitly if the checkpoint lives elsewhere."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2], candidates[0][3], candidates[0][0]


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
    state = _state_dict_without_module_prefix(state)
    model.load_state_dict(state, strict=True)


def _label_name(value: int) -> str:
    return "Depression" if int(value) == 1 else "Non-Depression"


def _read_dvlog_sample_ids(root: Path, fold: str, gender: str):
    labels_path = root / "labels.csv"
    if not _path_exists(labels_path):
        raise FileNotFoundError(
            f"D-Vlog labels.csv not found at {_display_path(labels_path)}. "
            "Use --data-dir to point to the local D-Vlog feature directory."
        )
    sample_ids = []
    with open(_win_long_path(labels_path), "r", encoding="utf-8") as f:
        for line in f:
            sample = line.strip().split(",")
            if len(sample) < 5:
                continue
            sample_gender, sample_fold = sample[3], sample[4]
            if gender == "both":
                keep = sample_fold == fold
            else:
                keep = sample_fold == fold and sample_gender == gender
            if keep:
                sample_ids.append(sample[0])
    return sample_ids


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _normalize(values):
    vals = list(float(v) for v in values)
    low = min(vals)
    high = max(vals)
    if high - low < 1e-12:
        return [0.0 for _ in vals]
    return [(v - low) / (high - low) for v in vals]


def _analysis_rows(sample):
    segments = range(1, len(sample["segment_weights"]) + 1)
    visual_norm = _normalize(sample["visual_motion_mean"])
    audio_norm = _normalize(sample["audio_change_mean"])
    rows = []
    for i, segment in enumerate(segments):
        rows.append(
            {
                "sample_role": sample["role"],
                "sample_index": sample["sample_index"],
                "sample_id": sample["sample_id"],
                "true_label": sample["true_label"],
                "true_label_name": _label_name(sample["true_label"]),
                "predicted_label": sample["predicted_label"],
                "predicted_label_name": _label_name(sample["predicted_label"]),
                "predicted_probability": sample["predicted_probability"],
                "logit": sample["logit"],
                "checkpoint": sample["checkpoint"],
                "segment": segment,
                "segment_weight": sample["segment_weights"][i],
                "valid_segment": sample["valid_segment_mask"][i],
                "visual_motion_mean": sample["visual_motion_mean"][i],
                "visual_motion_std": sample["visual_motion_std"][i],
                "audio_change_mean": sample["audio_change_mean"][i],
                "audio_change_std": sample["audio_change_std"][i],
                "visual_motion_mean_norm": visual_norm[i],
                "audio_change_mean_norm": audio_norm[i],
            }
        )
    return rows


def _write_csv(samples, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample in samples:
        rows.extend(_analysis_rows(sample))
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_samples(samples, svg_path: Path, pdf_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to draw fig2. Install matplotlib in the "
            "same environment used to run this script."
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 7.8,
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
    nrows = len(samples)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(7.16, max(1.85 * nrows, 2.1)),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    weight_color = "#B79AE2"
    visual_color = "#D9822B"
    audio_color = "#2B6CB0"
    edge_color = "#222222"
    grid_color = "#D8D8D8"
    segments = list(range(1, 9))
    segment_labels = [f"S{i}" for i in segments]

    for row_idx, sample in enumerate(samples):
        ax_w, ax_c = axes[row_idx]
        weights = [float(v) for v in sample["segment_weights"]]
        valid = [bool(v) for v in sample["valid_segment_mask"]]
        bar_colors = [weight_color if flag else "#EAEAEA" for flag in valid]
        ax_w.bar(segments, weights, color=bar_colors, edgecolor=edge_color, linewidth=0.45)
        ax_w.set_ylim(0.0, max(0.40, min(1.0, max(weights) * 1.25)))
        ax_w.set_ylabel("Weight")
        ax_w.set_xticks(segments, segment_labels)
        ax_w.set_xlabel("Segment")
        ax_w.grid(axis="y", color=grid_color, linewidth=0.45)
        ax_w.set_axisbelow(True)
        ax_w.set_title(
            (
                f"{sample['role']} | True: {_label_name(sample['true_label'])} | "
                f"Pred: {_label_name(sample['predicted_label'])} | "
                f"P(Dep.)={sample['predicted_probability']:.3f}"
            ),
            loc="left",
            pad=3,
        )

        visual_norm = _normalize(sample["visual_motion_mean"])
        audio_norm = _normalize(sample["audio_change_mean"])
        ax_c.plot(
            segments,
            visual_norm,
            marker="o",
            markersize=3.0,
            linewidth=1.15,
            color=visual_color,
            label="Visual motion mean",
        )
        ax_c.plot(
            segments,
            audio_norm,
            marker="s",
            markersize=3.0,
            linewidth=1.15,
            color=audio_color,
            label="Audio change mean",
        )
        ax_c.set_ylim(-0.05, 1.05)
        ax_c.set_ylabel("Normalized value")
        ax_c.set_xticks(segments, segment_labels)
        ax_c.set_xlabel("Segment")
        ax_c.grid(axis="y", color=grid_color, linewidth=0.45)
        ax_c.set_axisbelow(True)
        ax_c.legend(loc="upper right", frameon=False, handlelength=1.8)

        for ax in (ax_w, ax_c):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.6)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(width=0.6, length=2.5)

    fig.text(
        0.5,
        0.012,
        "Note: visual motion mean and audio change mean are independently normalized within each sample for visualization.",
        ha="center",
        va="bottom",
        fontsize=7.0,
    )
    fig.tight_layout(rect=[0.0, 0.045, 1.0, 1.0], h_pad=1.05, w_pad=1.2)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _make_model(project_root: Path, config, motion_alpha: float):
    sys.path.insert(0, str(project_root))
    models_pkg = sys.modules.get("models")
    if models_pkg is None:
        models_pkg = types.ModuleType("models")
        models_pkg.__path__ = [str(project_root / "models")]
        sys.modules["models"] = models_pkg
    spec = importlib.util.spec_from_file_location(
        "models.DepMamba",
        project_root / "models" / "DepMamba.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["models.DepMamba"] = module
    spec.loader.exec_module(module)
    DepMamba = module.DepMamba

    model_kwargs = dict(config["mmmamba"])
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
            model_kwargs[key] = config[key]
    model_kwargs["pooling_residual_alpha"] = motion_alpha
    return DepMamba(**model_kwargs)



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


def _select_samples(model, loader, sample_ids, device, checkpoint_display, include_error=True):
    import torch

    wanted = {
        "Correct Depression": None,
        "Correct Non-Depression": None,
    }
    if include_error:
        wanted["Misclassified"] = None

    seen = 0
    model.eval()
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            logits, analysis = _forward_with_analysis(model, x, mask, labels=y)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long()

            weights = _to_numpy(analysis["segment_weights"])
            valid_mask = _to_numpy(analysis["valid_segment_mask"]).astype(bool)
            cues = analysis["motion_cues"]
            visual_mean = _to_numpy(cues["visual_motion_mean"])
            visual_std = _to_numpy(cues["visual_motion_std"])
            audio_mean = _to_numpy(cues["audio_change_mean"])
            audio_std = _to_numpy(cues["audio_change_std"])
            logits_np = _to_numpy(logits.squeeze(1))
            probs_np = _to_numpy(probs)
            labels_np = _to_numpy(y).astype(int)
            preds_np = _to_numpy(preds).astype(int)

            for batch_idx in range(x.shape[0]):
                sample_index = seen + batch_idx
                true_label = int(labels_np[batch_idx])
                pred_label = int(preds_np[batch_idx])
                if true_label == 1 and pred_label == 1:
                    role = "Correct Depression"
                elif true_label == 0 and pred_label == 0:
                    role = "Correct Non-Depression"
                elif true_label != pred_label:
                    role = "Misclassified"
                else:
                    continue
                if role not in wanted or wanted[role] is not None:
                    continue
                sample_id = sample_ids[sample_index] if sample_index < len(sample_ids) else f"test_index_{sample_index}"
                wanted[role] = {
                    "role": role,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "predicted_probability": float(probs_np[batch_idx]),
                    "logit": float(logits_np[batch_idx]),
                    "checkpoint": checkpoint_display,
                    "segment_weights": [float(v) for v in weights[batch_idx]],
                    "valid_segment_mask": [bool(v) for v in valid_mask[batch_idx]],
                    "visual_motion_mean": [float(v) for v in visual_mean[batch_idx]],
                    "visual_motion_std": [float(v) for v in visual_std[batch_idx]],
                    "audio_change_mean": [float(v) for v in audio_mean[batch_idx]],
                    "audio_change_std": [float(v) for v in audio_std[batch_idx]],
                }
            if all(value is not None for value in wanted.values()):
                break
            seen += x.shape[0]

    required = ["Correct Depression", "Correct Non-Depression"]
    missing = [role for role in required if wanted[role] is None]
    if missing:
        raise RuntimeError(f"Could not find required samples: {', '.join(missing)}")
    return [wanted[role] for role in ("Correct Depression", "Correct Non-Depression", "Misclassified") if wanted.get(role)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export real DepMamba segment weights and motion cues for paper visualization."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config_motion.yaml")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-index", type=int, default=None)
    parser.add_argument("--motion-alpha", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--test-gender", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-error-sample", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_yaml(args.config)
    project_root = args.project_root.resolve()
    data_dir = args.data_dir or Path(config["data_dir"])
    batch_size = args.batch_size or int(config.get("batch_size", 16))
    test_gender = args.test_gender or config.get("test_gender", "both")
    sample_ids = _read_dvlog_sample_ids(data_dir, "test", test_gender)

    alpha_token = str(args.motion_alpha).replace(".", "p")
    if args.checkpoint is None:
        checkpoint, experiment_name, result_avg = _find_alpha_checkpoint(
            project_root, alpha_token, args.run_index
        )
        print(
            f"Selected checkpoint: {_display_path(checkpoint)} "
            f"(experiment={experiment_name}, Avg={result_avg:.6f})"
        )
    else:
        checkpoint = args.checkpoint
        print(f"Selected checkpoint: {_display_path(checkpoint)}")

    _check_runtime_dependencies()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(project_root))
    from datasets import get_dvlog_dataloader

    config = dict(config)
    config["pooling_mode"] = "motion_segment_attention"
    config["pooling_num_segments"] = 8
    config["pooling_residual"] = True
    config["pooling_residual_alpha"] = args.motion_alpha
    config["pooling_motion_emb_dim"] = 32
    config["pooling_motion_cue_mode"] = "full"
    model = _make_model(project_root, config, args.motion_alpha)
    _load_checkpoint(model, checkpoint, device)
    model = model.to(device)

    loader = get_dvlog_dataloader(
        data_dir,
        "test",
        batch_size,
        test_gender,
        aug=False,
    )
    samples = _select_samples(
        model,
        loader,
        sample_ids,
        device,
        _display_path(checkpoint),
        include_error=not args.no_error_sample,
    )

    csv_path = args.output_dir / "segment_analysis.csv"
    svg_path = args.output_dir / "fig2_segment_analysis.svg"
    pdf_path = args.output_dir / "fig2_segment_analysis.pdf"
    _write_csv(samples, csv_path)

    print(f"Wrote {_display_path(csv_path)}")
    print("Selected samples:")
    for sample in samples:
        print(
            f"  - {sample['role']}: id={sample['sample_id']}, "
            f"true={_label_name(sample['true_label'])}, "
            f"pred={_label_name(sample['predicted_label'])}, "
            f"P(Dep.)={sample['predicted_probability']:.4f}"
        )

    if args.skip_plot:
        print("Skipping plot generation because --skip-plot was specified.")
        return

    if not _matplotlib_available():
        print(
            "WARNING: matplotlib is not available; CSV was saved, "
            "and SVG/PDF plot generation was skipped."
        )
        return

    _plot_samples(samples, svg_path, pdf_path)
    print(f"Wrote {_display_path(svg_path)}")
    print(f"Wrote {_display_path(pdf_path)}")


if __name__ == "__main__":
    main()
