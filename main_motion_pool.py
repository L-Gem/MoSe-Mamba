import argparse
import os
import random
import yaml

import numpy as np
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.DepMamba_motion import DepMamba
from datasets import get_dvlog_dataloader, get_lmvd_dataloader

CONFIG_PATH = "./config/config_motion.yaml"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: True/False")
    
def parse_args():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Train and test a model."
    )
    # arguments whose default values are in config.yaml
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument("-m", "--model", type=str)
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("--use_global_alignment", type=str2bool)
    parser.add_argument("--lambda_global_alignment", type=float)
    parser.add_argument("--pooling_mode", type=str)
    parser.add_argument("--pooling_num_segments", type=int)
    parser.add_argument("--pooling_scales", type=str)
    parser.add_argument("--pooling_scale_fusion", type=str)
    parser.add_argument("--pooling_dropout", type=float)
    parser.add_argument("--pooling_residual", type=str2bool)
    parser.add_argument("--pooling_residual_alpha", type=float)
    parser.add_argument("--pooling_motion_emb_dim", type=int)
    parser.add_argument("--pooling_motion_cue_mode", type=str)
    parser.add_argument("--pooling_reliability_beta", type=float)
    parser.add_argument("--pooling_gate_emb_dim", type=int)
    parser.add_argument("--pooling_gamma_init", type=float)
    parser.add_argument("--pooling_gamma_max", type=float)
    parser.add_argument("--gate_cue_mode", type=str)
    parser.add_argument("--use_dynamic_temporal_refinement", type=str2bool)
    parser.add_argument("--dynamic_refinement_layers", type=int)
    parser.add_argument("--dynamic_refinement_kernel_size", type=int)
    parser.add_argument("--dynamic_refinement_gate_activation", type=str)
    parser.add_argument("--dynamic_refinement_dyn_type", type=str)
    parser.add_argument("--dynamic_refinement_dropout", type=float)
    parser.add_argument("--dynamic_refinement_residual_scale", type=float)
    parser.add_argument("--use_segment_consistency", type=str2bool)
    parser.add_argument("--segment_drop_lambda", type=float)
    parser.add_argument("--segment_consistency_lambda", type=float)
    parser.add_argument("--segment_drop_num_segments", type=int)
    parser.add_argument("--segment_drop_prob", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", type=str2bool)
    
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU id(s), e.g. '0' or '0,1' or 'cuda:0,1' or 'cpu'")
    
    parser.add_argument("-wdb", "--if_wandb", type=str2bool, default=False)
    parser.add_argument("-tqdm", "--tqdm_able", type=str2bool)
    parser.add_argument("-tr", "--train", type=str2bool)
    parser.set_defaults(**config)
    args = parser.parse_args()
    return args

# 解析 --gpu 字符串为绝对 GPU 编号列表；支持 cpu  # <<<
def _parse_gpu_arg(gpu_arg: str):
    s = str(gpu_arg).strip().lower()
    if s in ("cpu", "none", "-1", ""):
        return None  # 表示使用 CPU
    s = s.replace("cuda:", "")
    ids = [int(x) for x in s.split(",") if x != ""]
    return ids


def _unwrap_model(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(bool(deterministic))


def _format_for_path(value):
    text = str(value)
    return (
        text.replace("-", "m")
        .replace(".", "p")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _format_scales_for_path(value):
    if isinstance(value, (list, tuple)):
        text = "-".join(str(item) for item in value)
    else:
        text = str(value).replace(",", "-").replace(";", "-")
    return _format_for_path(text)


def build_experiment_name(args):
    global_tag = "global1" if args.use_global_alignment else "global0"
    lambda_global = _format_for_path(args.lambda_global_alignment)
    pooling_mode = _format_for_path(str(args.pooling_mode).lower())
    pooling_segments = _format_for_path(args.pooling_num_segments)
    pooling_scales = _format_scales_for_path(args.pooling_scales)
    pooling_scale_fusion = _format_for_path(str(args.pooling_scale_fusion).lower())
    pooling_dropout = _format_for_path(args.pooling_dropout)
    pooling_residual = "res1" if args.pooling_residual else "res0"
    pooling_alpha = _format_for_path(args.pooling_residual_alpha)
    pooling_motion_emb_dim = _format_for_path(args.pooling_motion_emb_dim)
    pooling_motion_cue_mode = _format_for_path(
        str(args.pooling_motion_cue_mode).lower()
    )
    pooling_reliability_beta = _format_for_path(args.pooling_reliability_beta)
    pooling_gate_emb_dim = _format_for_path(args.pooling_gate_emb_dim)
    pooling_gamma_init = _format_for_path(args.pooling_gamma_init)
    pooling_gamma_max = _format_for_path(args.pooling_gamma_max)
    gate_cue_mode = _format_for_path(str(args.gate_cue_mode).lower())
    dtr_tag = "dtr1" if args.use_dynamic_temporal_refinement else "dtr0"
    dtr_layers = _format_for_path(args.dynamic_refinement_layers)
    dtr_kernel = _format_for_path(args.dynamic_refinement_kernel_size)
    dtr_gate = _format_for_path(str(args.dynamic_refinement_gate_activation).lower())
    dtr_dyn = _format_for_path(str(args.dynamic_refinement_dyn_type).lower())
    dtr_dropout = _format_for_path(args.dynamic_refinement_dropout)
    dtr_residual_scale = _format_for_path(args.dynamic_refinement_residual_scale)
    consistency_tag = "cons1" if args.use_segment_consistency else "cons0"
    segment_drop_lambda = _format_for_path(args.segment_drop_lambda)
    segment_consistency_lambda = _format_for_path(args.segment_consistency_lambda)
    segment_drop_segments = _format_for_path(args.segment_drop_num_segments)
    segment_drop_prob = _format_for_path(args.segment_drop_prob)
    seed = _format_for_path(args.seed)
    return (
        f"{args.dataset}_{args.model}_"
        f"pool{pooling_mode}_k{pooling_segments}_sc{pooling_scales}_"
        f"sf{pooling_scale_fusion}_drop{pooling_dropout}_"
        f"{pooling_residual}_a{pooling_alpha}_emb{pooling_motion_emb_dim}_"
        f"cue{pooling_motion_cue_mode}_beta{pooling_reliability_beta}_"
        f"gemb{pooling_gate_emb_dim}_gcue{gate_cue_mode}_"
        f"gi{pooling_gamma_init}_gm{pooling_gamma_max}_"
        f"{dtr_tag}_l{dtr_layers}_k{dtr_kernel}_g{dtr_gate}_"
        f"dyn{dtr_dyn}_dd{dtr_dropout}_dr{dtr_residual_scale}_"
        f"{consistency_tag}_dl{segment_drop_lambda}_cl{segment_consistency_lambda}_"
        f"dk{segment_drop_segments}_dp{segment_drop_prob}_seed{seed}_"
        f"{global_tag}_lg{lambda_global}"
    )


def _metric_avg(results):
    return (
        results["acc"]
        + results["precision"]
        + results["recall"]
        + results["f1"]
    ) / 4.0


def _fmt_metric(value):
    return f"{value:.4f}"


def print_experiment_header(args, experiment_name, primary_device, dp_device_ids):
    width = 72
    print("=" * width)
    print("DepMamba Experiment")
    print("=" * width)
    print(f"{'Dataset':<13}: {args.dataset}")
    print(f"{'Model':<13}: {args.model}")
    print(f"{'Train/Test':<13}: {args.train_gender} -> {args.test_gender}")
    print(f"{'Epochs':<13}: {args.epochs}")
    print(f"{'Batch size':<13}: {args.batch_size}")
    print(f"{'LR':<13}: {args.learning_rate}")
    print(f"{'Seed':<13}: base={args.seed}, deterministic={args.deterministic}")
    print(f"{'Device':<13}: primary={primary_device}, data_parallel_ids={dp_device_ids}")
    print(
        f"{'Alignment':<13}: "
        f"global={args.use_global_alignment}"
    )
    print(
        f"{'Pooling':<13}: "
        f"{args.pooling_mode}, segments={args.pooling_num_segments}, "
        f"scales={args.pooling_scales}, "
        f"scale_fusion={args.pooling_scale_fusion}, "
        f"dropout={args.pooling_dropout}, residual={args.pooling_residual}, "
        f"alpha={args.pooling_residual_alpha}, "
        f"motion_emb={args.pooling_motion_emb_dim}, "
        f"cue={args.pooling_motion_cue_mode}, "
        f"beta={args.pooling_reliability_beta}, "
        f"gate_emb={args.pooling_gate_emb_dim}, "
        f"gate_cue={args.gate_cue_mode}, "
        f"gamma_init={args.pooling_gamma_init}, "
        f"gamma_max={args.pooling_gamma_max}"
    )
    print(
        f"{'DTR':<13}: "
        f"enabled={args.use_dynamic_temporal_refinement}, "
        f"layers={args.dynamic_refinement_layers}, "
        f"kernel={args.dynamic_refinement_kernel_size}, "
        f"gate={args.dynamic_refinement_gate_activation}, "
        f"dyn_type={args.dynamic_refinement_dyn_type}, "
        f"dropout={args.dynamic_refinement_dropout}, "
        f"res_scale={args.dynamic_refinement_residual_scale}"
    )
    print(
        f"{'Loss weights':<13}: "
        f"global={args.lambda_global_alignment}, "
        f"seg_cons={args.use_segment_consistency}, "
        f"drop_lambda={args.segment_drop_lambda}, "
        f"cons_lambda={args.segment_consistency_lambda}, "
        f"drop_k={args.segment_drop_num_segments}, "
        f"drop_prob={args.segment_drop_prob}"
    )
    print(f"{'Experiment':<13}: {experiment_name}")
    print("=" * width)


def print_run_header(i_iter, total_runs, run_seed, checkpoint_path, result_path):
    print("")
    print(f"[Run {i_iter + 1}/{total_runs}]")
    print(f"{'Seed':<11}: {run_seed}")
    print(f"{'Checkpoint':<11}: {checkpoint_path}")
    print(f"{'Result':<11}: {result_path}")


def print_test_results(results):
    print("")
    print("[Test Results]")
    print(f"{'Loss':<10}: {_fmt_metric(results['loss'])}")
    print(f"{'Accuracy':<10}: {_fmt_metric(results['acc'])}")
    print(f"{'Precision':<10}: {_fmt_metric(results['precision'])}")
    print(f"{'Recall':<10}: {_fmt_metric(results['recall'])}")
    print(f"{'F1':<10}: {_fmt_metric(results['f1'])}")
    print(f"{'Avg':<10}: {_fmt_metric(_metric_avg(results))}")


def print_final_summary(run_results):
    if not run_results:
        return
    print("")
    print("[Summary]")
    print(f"{'Run':<5}{'Acc':>9}{'Prec':>9}{'Recall':>9}{'F1':>9}{'Avg':>9}")
    for i, results in enumerate(run_results):
        print(
            f"{i:<5}"
            f"{_fmt_metric(results['acc']):>9}"
            f"{_fmt_metric(results['precision']):>9}"
            f"{_fmt_metric(results['recall']):>9}"
            f"{_fmt_metric(results['f1']):>9}"
            f"{_fmt_metric(_metric_avg(results)):>9}"
        )


def make_segment_drop_mask(
    padding_mask, num_segments=8, drop_prob=1.0
):
    drop_mask = padding_mask.clone()
    drop_mask = drop_mask.to(dtype=torch.bool, device=padding_mask.device)
    batch_size, steps = drop_mask.shape
    num_segments = max(int(num_segments), 1)
    drop_prob = float(drop_prob)
    if drop_prob <= 0.0:
        return drop_mask

    valid_lengths = drop_mask.sum(dim=1).clamp_min(1)
    drop_indices = torch.randint(
        low=0, high=num_segments, size=(batch_size,), device=drop_mask.device
    )
    should_drop = torch.rand(batch_size, device=drop_mask.device) < drop_prob

    for sample_idx in range(batch_size):
        if not should_drop[sample_idx]:
            continue
        valid_len = int(valid_lengths[sample_idx].item())
        segment_idx = int(drop_indices[sample_idx].item())
        start = (segment_idx * valid_len) // num_segments
        end = ((segment_idx + 1) * valid_len) // num_segments
        if end > start:
            drop_mask[sample_idx, start:end] = False

    return drop_mask


def forward_with_pooling_segment_consistency(net, x, mask, drop_mask):
    model = _unwrap_model(net)
    raw_input = x
    temporal_features = model.temporal_feature_extractor(x, mask)
    clean_pooled = model.pool_temporal_features(
        temporal_features, mask, raw_input=raw_input
    )
    drop_pooled = model.pool_temporal_features(
        temporal_features, drop_mask, raw_input=raw_input
    )
    clean_logits = model.classifier(clean_pooled)
    drop_logits = model.classifier(drop_pooled)
    return clean_logits, drop_logits


def train_epoch(
    net, train_loader, loss_fn, optimizer, device, 
    current_epoch, total_epochs, tqdm_able,
    lambda_global_alignment=0.0,
    use_segment_consistency=False,
    segment_drop_lambda=0.0,
    segment_consistency_lambda=0.0,
    segment_drop_num_segments=8,
    segment_drop_prob=1.0,
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    running_cls_loss = 0.
    running_drop_cls_loss = 0.
    running_consistency_loss = 0.
    running_global_align_loss = 0.
    running_weighted_global_align_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for x, y, mask in pbar:
            # print(x.shape,y.shape)
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            if use_segment_consistency:
                drop_mask = make_segment_drop_mask(
                    mask,
                    num_segments=segment_drop_num_segments,
                    drop_prob=segment_drop_prob,
                )
                y_pred, y_drop_pred = forward_with_pooling_segment_consistency(
                    net, x, mask, drop_mask
                )
            else:
                y_pred = net(x, mask)
                y_drop_pred = None

            cls_loss = loss_fn(y_pred, y.to(torch.float32))
            if y_drop_pred is not None:
                drop_cls_loss = loss_fn(y_drop_pred, y.to(torch.float32))
                consistency_loss = F.mse_loss(
                    torch.sigmoid(y_drop_pred),
                    torch.sigmoid(y_pred.detach()),
                )
            else:
                drop_cls_loss = cls_loss.new_tensor(0.0)
                consistency_loss = cls_loss.new_tensor(0.0)
            aux_losses = getattr(_unwrap_model(net), "aux_losses", {})
            global_align_loss = aux_losses.get("global_align_loss", cls_loss.new_tensor(0.0))
            weighted_global_align_loss = lambda_global_alignment * global_align_loss
            loss = (
                cls_loss
                + segment_drop_lambda * drop_cls_loss
                + segment_consistency_lambda * consistency_loss
                + weighted_global_align_loss
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            running_cls_loss += cls_loss.item() * x.shape[0]
            running_drop_cls_loss += drop_cls_loss.item() * x.shape[0]
            running_consistency_loss += consistency_loss.item() * x.shape[0]
            running_global_align_loss += global_align_loss.item() * x.shape[0]
            running_weighted_global_align_loss += weighted_global_align_loss.item() * x.shape[0]
            # binary classification with only one output neuron
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix({
                "total": running_loss / sample_count,
                "cls": running_cls_loss / sample_count,
                "drop_cls": running_drop_cls_loss / sample_count,
                "cons": running_consistency_loss / sample_count,
                "acc": correct_count / sample_count,
                "global": running_global_align_loss / sample_count,
                "w_global": running_weighted_global_align_loss / sample_count,
            })

    return {
        "loss": running_loss / sample_count,
        "cls_loss": running_cls_loss / sample_count,
        "drop_cls_loss": running_drop_cls_loss / sample_count,
        "consistency_loss": running_consistency_loss / sample_count,
        "acc": correct_count / sample_count,
        "global_align_loss": running_global_align_loss / sample_count,
        "weighted_global_align_loss": running_weighted_global_align_loss / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, tqdm_able
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                # print(x.shape,y.shape)
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }


def main():
    args = parse_args()

    gpu_ids = _parse_gpu_arg(args.gpu)
    if gpu_ids is None or not torch.cuda.is_available():
        primary_device = torch.device("cpu")
        dp_device_ids = None
    else:
        torch.cuda.set_device(gpu_ids[0])  # 设置默认 GPU
        primary_device = torch.device(f"cuda:{gpu_ids[0]}")
        dp_device_ids = gpu_ids if len(gpu_ids) > 1 else None

    args.data_dir = os.path.join(args.data_dir,args.dataset)
    experiment_name = build_experiment_name(args)
    print_experiment_header(args, experiment_name, primary_device, dp_device_ids)

    total_runs = 3
    run_results = []
    for i_iter in range(total_runs):
        run_seed = int(args.seed) + i_iter
        set_seed(run_seed, args.deterministic)

        if args.if_wandb:
            wandb_run_name = f"{experiment_name}-{args.train_gender}-{args.test_gender}-iter{i_iter}"
            wandb.init(
                project="mamnba_ad", config=args, name=wandb_run_name,
            )
            args = wandb.config
        
        # Build Save Dir
        experiment_dir = os.path.join(args.save_dir, f"{experiment_name}_{i_iter}")
        samples_dir = os.path.join(experiment_dir, "samples")
        checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoints_dir, "best_model.pt")
        result_path = os.path.join("./results_motion", f"{experiment_name}_{i_iter}.txt")
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        print_run_header(i_iter, total_runs, run_seed, checkpoint_path, result_path)

        # construct the model
        set_seed(run_seed, args.deterministic)
        if args.model == "DepMamba":
            if args.dataset=='lmvd':
                model_kwargs = dict(args.mmmamba_lmvd)
            elif args.dataset=='dvlog':
                model_kwargs = dict(args.mmmamba)
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")
            model_kwargs.update({
                "use_global_alignment": args.use_global_alignment,
                "pooling_mode": args.pooling_mode,
                "pooling_num_segments": args.pooling_num_segments,
                "pooling_scales": args.pooling_scales,
                "pooling_scale_fusion": args.pooling_scale_fusion,
                "pooling_dropout": args.pooling_dropout,
                "pooling_residual": args.pooling_residual,
                "pooling_residual_alpha": args.pooling_residual_alpha,
                "pooling_motion_emb_dim": args.pooling_motion_emb_dim,
                "pooling_motion_cue_mode": args.pooling_motion_cue_mode,
                "pooling_reliability_beta": args.pooling_reliability_beta,
                "pooling_gate_emb_dim": args.pooling_gate_emb_dim,
                "pooling_gamma_init": args.pooling_gamma_init,
                "pooling_gamma_max": args.pooling_gamma_max,
                "gate_cue_mode": args.gate_cue_mode,
                "use_dynamic_temporal_refinement": args.use_dynamic_temporal_refinement,
                "dynamic_refinement_layers": args.dynamic_refinement_layers,
                "dynamic_refinement_kernel_size": args.dynamic_refinement_kernel_size,
                "dynamic_refinement_gate_activation": args.dynamic_refinement_gate_activation,
                "dynamic_refinement_dyn_type": args.dynamic_refinement_dyn_type,
                "dynamic_refinement_dropout": args.dynamic_refinement_dropout,
                "dynamic_refinement_residual_scale": args.dynamic_refinement_residual_scale,
            })
            net = DepMamba(**model_kwargs)# mmmamba_lmvd mmmamba
        else:#if args.model == "MAMBA":
            raise NotImplementedError(f"The {args.model} method has not been implemented by this repo")
        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        # prepare the data
        set_seed(run_seed, args.deterministic)
        if args.dataset=='dvlog':
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset=='lmvd':
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )

        # set other training components
        if args.dataset == 'lmvd':
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([0.8], device=next(net.parameters()).device)
            )
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0
        best_test_acc = -1.0
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net, train_loader, loss_fn, optimizer, 
                    args.device[0], epoch, args.epochs, args.tqdm_able,
                    args.lambda_global_alignment,
                    args.use_segment_consistency,
                    args.segment_drop_lambda,
                    args.segment_consistency_lambda,
                    args.segment_drop_num_segments,
                    args.segment_drop_prob,
                )
                val_results = val(net, val_loader, loss_fn, args.device[0],args.tqdm_able)

                val_acc = (val_results["acc"] + val_results["precision"]+ val_results["recall"]+ val_results["f1"])/4.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(net.state_dict(), checkpoint_path)

                if args.if_wandb:
                    wandb.log({
                        "loss/train": train_results["loss"],
                        "loss/cls": train_results["cls_loss"],
                        "loss/drop_cls": train_results["drop_cls_loss"],
                        "loss/consistency": train_results["consistency_loss"],
                        "acc/train": train_results["acc"],
                        "loss/global_align": train_results["global_align_loss"],
                        "loss/weighted_global_align": train_results["weighted_global_align_loss"],
                        "loss/val": val_results["loss"],
                        "acc/val": val_results["acc"],
                        "precision/val": val_results["precision"],
                        "recall/val": val_results["recall"],
                        "f1/val": val_results["f1"]
                    })
            
        # upload the best model to wandb website
        # load the best model for testing
        with torch.no_grad():
            net.load_state_dict(
                torch.load(checkpoint_path, map_location=args.device[0])
            )
            net.eval()
            test_results = val(net, test_loader, loss_fn, args.device[0],args.tqdm_able)
            print_test_results(test_results)
            run_results.append(test_results)

            os.makedirs("./results_motion", exist_ok=True)
            with open(result_path,'w') as f:    
                test_result_str = f'Seed:{run_seed}, Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{_metric_avg(test_results)}'
                f.write(test_result_str)         

    print_final_summary(run_results)

    if args.if_wandb:
        artifact = wandb.Artifact("best_model", type="model")
        artifact.add_file(checkpoint_path)
        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.log_artifact(artifact)
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()


if __name__ == '__main__':
    main()

