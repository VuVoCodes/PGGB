#!/usr/bin/env python
"""
Training script for ASGML multimodal learning.

Supports multiple experimental conditions:
- Baseline: Standard joint training (all modalities update every step)
- Fixed frequency: Dominant modality updates every k steps
- Fixed staleness: Dominant modality uses τ-step-old gradients
- Adaptive ASGML: Probe-driven staleness/frequency adaptation
- MILES: Modality-Informed Learning Rate Scheduler (epoch-level LR adjustment)

Usage:
    # Baseline (no ASGML)
    python scripts/train.py --config configs/cremad.yaml --mode baseline

    # Fixed frequency (dominant updates every 2 steps)
    python scripts/train.py --config configs/cremad.yaml --mode frequency --fixed-ratio 2

    # Fixed staleness (dominant uses 2-step-old gradients)
    python scripts/train.py --config configs/cremad.yaml --mode staleness --fixed-staleness 2

    # Adaptive ASGML (full method)
    python scripts/train.py --config configs/cremad.yaml --mode adaptive

    # MILES (learning rate scheduling baseline)
    python scripts/train.py --config configs/cremad.yaml --mode miles --miles-threshold 0.2 --miles-reduction 0.5

    # OGM-GE + ASGML adaptive (combined)
    python scripts/train.py --config configs/cremad.yaml --mode adaptive --ogm-ge --alpha 0.8

    # OGM-GE alone (for comparison)
    python scripts/train.py --config configs/cremad.yaml --mode baseline --ogm-ge --alpha 0.8
"""

import argparse
import math
import sys
import yaml
import random
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import MultimodalModel, ProbeManager
from src.models.fusion import ConcatFusion
from src.datasets import CREMADDataset, AVEDataset, KineticsSoundsDataset, MOSEIDataset, CMUMOSIDataset, SarcasmDataset, TwitterDataset, Food101Dataset
from src.losses import (
    ASGMLLoss,
    ASGMLScheduler,
    compute_gradient_norms,
    apply_staleness_gradients,
)
from src.losses.cggm import CGGMModule
from src.utils import setup_logger, AverageMeter
from src.utils.metrics import MetricTracker


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset(config: dict, split: str):
    """Create dataset based on config."""
    name = config["dataset"]["name"]
    root = config["dataset"]["root"]

    if name == "cremad":
        fps = config["dataset"].get("fps", 1)
        num_frames = config["dataset"].get("num_frames", 1)
        return CREMADDataset(root=root, split=split, fps=fps, num_frames=num_frames)
    elif name == "ave":
        num_frames = config["dataset"].get("num_frames", 3)
        return AVEDataset(root=root, split=split, num_frames=num_frames)
    elif name == "kinetics_sounds":
        num_frames = config["dataset"].get("num_frames", 3)
        # KS uses 'val' instead of 'test' for evaluation
        ks_split = "val" if split == "test" else split
        return KineticsSoundsDataset(root=root, split=ks_split, num_frames=num_frames)
    elif name == "mosei":
        # MOSEI uses 'valid' instead of 'test' for validation, and 'test' for final eval
        mosei_split = split if split == "train" else "test"
        return MOSEIDataset(root=root, split=mosei_split)
    elif name == "mosi":
        mosi_split = "test" if split == "test" else split
        return CMUMOSIDataset(root=root, split=mosi_split)
    elif name == "sarcasm":
        # Sarcasm uses 'valid' instead of 'test' for validation
        sarcasm_split = split if split == "train" else ("valid" if split == "valid" else "test")
        return SarcasmDataset(root=root, split=sarcasm_split)
    elif name == "twitter":
        twitter_split = split if split == "train" else ("valid" if split == "valid" else "test")
        return TwitterDataset(root=root, split=twitter_split)
    elif name == "food101":
        # Food101 only has train/test (no validation split). Use test for both.
        food_split = "train" if split == "train" else "test"
        return Food101Dataset(root=root, split=food_split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_optimizer(
    model: nn.Module,
    config: dict,
    modality_lr_scales: dict = None,
):
    """
    Create optimizer based on config with optional per-modality learning rates.

    Args:
        model: The multimodal model
        config: Training configuration
        modality_lr_scales: Dict mapping modality names to LR scale factors.
            E.g., {"audio": 0.5, "visual": 1.0} means audio gets half the base LR.
            If None, all modalities use the same LR.
    """
    opt_name = config["training"]["optimizer"]
    base_lr = config["training"]["lr"]
    momentum = config["training"].get("momentum", 0.9)
    weight_decay = config["training"].get("weight_decay", 1e-4)

    # Build parameter groups for per-modality LRs
    if modality_lr_scales is not None:
        param_groups = []

        # Encoder parameters (per-modality LR)
        for modality in model.modalities:
            scale = modality_lr_scales.get(modality, 1.0)
            param_groups.append({
                'params': list(model.encoders[modality].parameters()),
                'lr': base_lr * scale,
                'name': f'encoder_{modality}',
            })

        # Fusion + classifier parameters (base LR)
        fusion_params = (
            list(model.fusion.parameters()) +
            list(model.classifier.parameters()) +
            list(model.unimodal_classifiers.parameters())
        )
        param_groups.append({
            'params': fusion_params,
            'lr': base_lr,
            'name': 'fusion',
        })

        params = param_groups
    else:
        params = model.parameters()

    if opt_name == "sgd":
        return optim.SGD(
            params,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adam":
        return optim.Adam(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        return optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config: dict):
    """Create LR scheduler based on config."""
    sched_name = config["training"].get("scheduler", "step")

    if sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("step_size", 40),
            gamma=config["training"].get("gamma", 0.1),
        )
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def get_miles_optimizer(model: nn.Module, config: dict):
    """
    Create optimizer with separate parameter groups for MILES.

    MILES requires per-modality learning rate control, so we create
    separate parameter groups for each encoder and the fusion module.
    """
    opt_name = config["training"].get("miles_optimizer", "adam")  # MILES paper uses Adam
    base_lr = config["training"]["lr"]
    weight_decay = config["training"].get("weight_decay", 1e-4)

    param_groups = []

    # Encoder parameters (per-modality LR)
    for modality in model.modalities:
        param_groups.append({
            'params': list(model.encoders[modality].parameters()),
            'lr': base_lr,
            'name': f'encoder_{modality}',
        })

    # Fusion + classifier parameters
    fusion_params = (
        list(model.fusion.parameters()) +
        list(model.classifier.parameters()) +
        list(model.unimodal_classifiers.parameters())
    )
    param_groups.append({
        'params': fusion_params,
        'lr': base_lr,
        'name': 'fusion',
    })

    if opt_name == "adam":
        return optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = config["training"].get("momentum", 0.9)
        return optim.SGD(param_groups, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer for MILES: {opt_name}")


def miles_adjust_learning_rates(
    optimizer: optim.Optimizer,
    acc_multimodal: float,
    acc_per_modality: dict,
    modalities: list,
    base_lr: float,
    threshold: float = 0.2,
    reduction: float = 0.5,
    logger=None,
):
    """
    MILES learning rate adjustment based on conditional utilization rates.

    Implements Algorithm 1 from the MILES paper:
    - Compute conditional utilization rate per modality: u_i = (M(ŷ_AB) - M(ŷ_j)) / M(ŷ_AB)
    - Compute δ_AB = |u_A - u_B|
    - If δ > threshold, reduce LR for dominant modality by factor μ

    Args:
        optimizer: Optimizer with per-modality parameter groups
        acc_multimodal: Multimodal fusion accuracy
        acc_per_modality: Dict mapping modality name to unimodal accuracy
        modalities: List of modality names
        base_lr: Base learning rate
        threshold: τ in MILES - threshold for triggering LR adjustment
        reduction: μ in MILES - factor to reduce dominant modality's LR
        logger: Optional logger

    Returns:
        dict with utilization metrics and adjusted LRs
    """
    # Compute conditional utilization rates (Eq. 4 in MILES paper)
    # u_A = (M(ŷ_AB) - M(ŷ_B)) / M(ŷ_AB)
    utilization = {}
    for m in modalities:
        other_modalities = [om for om in modalities if om != m]
        # For bimodal: u_A = (acc_multimodal - acc_B) / acc_multimodal
        # This measures how much modality A contributes beyond the other modality
        if len(other_modalities) == 1:
            other_m = other_modalities[0]
            if acc_multimodal > 0:
                utilization[m] = (acc_multimodal - acc_per_modality[other_m]) / acc_multimodal
            else:
                utilization[m] = 0.0
        else:
            # For N>2 modalities, use average of others
            avg_other = sum(acc_per_modality[om] for om in other_modalities) / len(other_modalities)
            if acc_multimodal > 0:
                utilization[m] = (acc_multimodal - avg_other) / acc_multimodal
            else:
                utilization[m] = 0.0

    # Compute δ_AB (Eq. 5 in MILES paper)
    # For bimodal case
    u_values = list(utilization.values())
    if len(u_values) == 2:
        delta = abs(u_values[0] - u_values[1])
    else:
        # For N>2, use max difference
        delta = max(u_values) - min(u_values)

    # Get param group name to index mapping
    group_indices = {}
    for i, group in enumerate(optimizer.param_groups):
        if 'name' in group:
            group_indices[group['name']] = i

    # MILES Algorithm 1 logic
    adjusted_lrs = {m: base_lr for m in modalities}
    adjusted_lrs['fusion'] = base_lr

    # Check if both modalities are underutilized (early training)
    all_negative = all(utilization[m] < 0 for m in modalities)

    if all_negative or delta <= threshold:
        # Case 1: No adjustment needed - keep base LR
        for m in modalities:
            group_name = f'encoder_{m}'
            if group_name in group_indices:
                optimizer.param_groups[group_indices[group_name]]['lr'] = base_lr
        if 'fusion' in group_indices:
            optimizer.param_groups[group_indices['fusion']]['lr'] = base_lr

        if logger:
            logger.info(f"MILES: No LR adjustment (δ={delta:.4f} <= τ={threshold} or all u<0)")
    else:
        # Determine dominant modality and adjust LR
        # Find modality with highest utilization (most dominant)
        dominant_m = max(utilization, key=utilization.get)

        # Check specific cases from MILES Algorithm 1
        m_list = list(modalities)
        if len(m_list) == 2:
            m_a, m_b = m_list[0], m_list[1]
            u_a, u_b = utilization[m_a], utilization[m_b]

            if u_a > 0 and u_b < 0 and delta > threshold:
                # m_a is dominant, reduce its LR
                dominant_m = m_a
            elif u_a < 0 and u_b > 0 and delta > threshold:
                # m_b is dominant, reduce its LR
                dominant_m = m_b
            elif u_a > 0 and u_b > 0 and delta > threshold:
                # Both positive but imbalanced - reduce the higher one
                dominant_m = m_a if u_a > u_b else m_b

        # Apply LR reduction to dominant modality
        for m in modalities:
            group_name = f'encoder_{m}'
            if group_name in group_indices:
                if m == dominant_m:
                    new_lr = base_lr * reduction
                    optimizer.param_groups[group_indices[group_name]]['lr'] = new_lr
                    adjusted_lrs[m] = new_lr
                else:
                    optimizer.param_groups[group_indices[group_name]]['lr'] = base_lr
                    adjusted_lrs[m] = base_lr

        # Fusion always gets base LR
        if 'fusion' in group_indices:
            optimizer.param_groups[group_indices['fusion']]['lr'] = base_lr

        if logger:
            logger.info(
                f"MILES: Adjusted LRs (δ={delta:.4f} > τ={threshold}), "
                f"dominant={dominant_m}, u={utilization}, "
                f"LRs={adjusted_lrs}"
            )

    return {
        'utilization': utilization,
        'delta': delta,
        'adjusted_lrs': adjusted_lrs,
        'dominant': max(utilization, key=utilization.get) if utilization else None,
    }


def train_epoch_miles(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
):
    """
    Train for one epoch in MILES mode.

    MILES uses standard training with auxiliary unimodal losses,
    but LR adjustment happens at epoch level (after validation).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter("Loss")
    loss_fusion_meter = AverageMeter("FusionLoss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    # Track unimodal metrics for MILES
    unimodal_trackers = {m: MetricTracker(['accuracy']) for m in modalities}

    gamma = config["asgml"].get("gamma", 1.0)  # Weight for unimodal losses

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits, unimodal_logits, features = model(inputs, return_features=True)

        # Compute losses (MILES uses multimodal + unimodal losses, Eq. 3 in paper)
        loss_fusion = criterion(logits, targets)
        loss_unimodal = {}
        for m in modalities:
            loss_unimodal[m] = criterion(unimodal_logits[m], targets)

        # Total loss: L = L_fusion + γ * Σ L_unimodal
        loss = loss_fusion + gamma * sum(loss_unimodal.values())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        loss_fusion_meter.update(loss_fusion.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # Track unimodal performance (for MILES algorithm)
        for m in modalities:
            unimodal_trackers[m].update(unimodal_logits[m], targets)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Logging
        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/fusion_loss", loss_fusion.item(), global_step)
            for m in modalities:
                writer.add_scalar(f"train/unimodal_loss_{m}", loss_unimodal[m].item(), global_step)

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fusion_loss"] = loss_fusion_meter.avg

    # Add unimodal accuracies (needed for MILES LR adjustment)
    for m in modalities:
        unimodal_metrics = unimodal_trackers[m].compute()
        metrics[f"unimodal_acc_{m}"] = unimodal_metrics['accuracy']

    return metrics


def train_epoch_inforeg(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    prev_epoch_weights: dict = None,
    fisher_traces: list = None,
    beta: float = 1.0,
    K: float = 0.05,
):
    """
    Train for one epoch in InfoReg mode (Huang et al., CVPR 2025).

    InfoReg slows information acquisition of dominant modalities during the
    prime learning window by adding a weight regularization term.

    Parameters
    ----------
    prev_epoch_weights : dict
        {modality: {name: tensor}} weights from start of previous epoch.
    fisher_traces : list
        List of dicts {modality: Tr(F)} for past epochs.
    beta : float
        Regulation strength (controls α = β * Δ_m).
    K : float
        Prime learning window threshold on relative Tr(F) change rate.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    # Track unimodal metrics
    unimodal_trackers = {m: MetricTracker(['accuracy']) for m in modalities}

    gamma = config["asgml"].get("gamma", 1.0)

    # Accumulate gradient norms for Fisher trace computation
    grad_norm_sq_accum = {m: 0.0 for m in modalities}
    num_batches = 0

    # Detect PLW: need at least 2 previous epochs of Fisher traces
    in_plw = {m: False for m in modalities}
    if fisher_traces is not None and len(fisher_traces) >= 2:
        for m in modalities:
            tr_prev = fisher_traces[-1].get(m, 0.0)
            tr_prev2 = fisher_traces[-2].get(m, 0.0)
            if tr_prev > 1e-10:
                plw_rate = (tr_prev - tr_prev2) / tr_prev
                in_plw[m] = plw_rate > K

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits, unimodal_logits, _ = model(inputs, return_features=True)

        # Compute losses (multimodal + unimodal, same as MILES)
        loss_fusion = criterion(logits, targets)
        loss_unimodal = {}
        for m in modalities:
            loss_unimodal[m] = criterion(unimodal_logits[m], targets)

        loss = loss_fusion + gamma * sum(loss_unimodal.values())

        # Compute performance scores and gap (Eq. 8-10 in InfoReg)
        scores = {}
        for m in modalities:
            probs = softmax(unimodal_logits[m].detach())
            scores[m] = -torch.log(probs[range(len(targets)), targets] + 1e-8).mean().item()

        # Lower score = better performance (it's negative log-likelihood)
        # InfoReg: dominant = lowest score (best performer)
        # Compute gap for dominant modality (Eq. 10)
        dominant_m = min(scores, key=scores.get)
        # Performance gap Δ_m for the dominant modality (Eq. 10)
        non_dominant_scores = [scores[m] for m in modalities if m != dominant_m]
        if non_dominant_scores:
            delta_dominant = (sum(non_dominant_scores) / len(non_dominant_scores)) - scores[dominant_m]
        else:
            delta_dominant = 0.0

        # Add regulation term for dominant modality in PLW (Eq. 12)
        regulation_loss = torch.tensor(0.0, device=device)
        if prev_epoch_weights is not None and delta_dominant > 0:
            if in_plw.get(dominant_m, False) or epoch <= 2:
                # α = β * Δ_m (Eq. 16)
                alpha_reg = beta * delta_dominant
                reg = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if f"encoders.{dominant_m}" in name and name in prev_epoch_weights.get(dominant_m, {}):
                        prev_w = prev_epoch_weights[dominant_m][name].to(device)
                        reg = reg + torch.sum((param - prev_w) ** 2)
                regulation_loss = (alpha_reg / 2.0) * reg
                loss = loss + regulation_loss

        # Backward pass
        loss.backward()

        # Accumulate gradient norms for Fisher trace
        for m in modalities:
            gnorm_sq = 0.0
            for name, param in model.named_parameters():
                if f"encoders.{m}" in name and param.grad is not None:
                    gnorm_sq += param.grad.data.norm(2).item() ** 2
            grad_norm_sq_accum[m] += gnorm_sq
        num_batches += 1

        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)
        for m in modalities:
            unimodal_trackers[m].update(unimodal_logits[m], targets)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Logging
        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/fusion_loss", loss_fusion.item(), global_step)
            writer.add_scalar("train/regulation_loss", regulation_loss.item(), global_step)
            for m in modalities:
                writer.add_scalar(f"train/unimodal_loss_{m}", loss_unimodal[m].item(), global_step)
                writer.add_scalar(f"inforeg/score_{m}", scores[m], global_step)
            writer.add_scalar("inforeg/in_plw_dominant", float(in_plw.get(dominant_m, False)), global_step)
            writer.add_scalar("inforeg/delta_dominant", delta_dominant, global_step)

    # Compute Fisher trace for this epoch
    epoch_fisher = {}
    for m in modalities:
        epoch_fisher[m] = grad_norm_sq_accum[m] / max(num_batches, 1)

    # Log Fisher traces
    if writer is not None:
        for m in modalities:
            writer.add_scalar(f"inforeg/fisher_trace_{m}", epoch_fisher[m], epoch)
            writer.add_scalar(f"inforeg/in_plw_{m}", float(in_plw[m]), epoch)

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fisher_traces"] = epoch_fisher
    for m in modalities:
        unimodal_metrics = unimodal_trackers[m].compute()
        metrics[f"unimodal_acc_{m}"] = unimodal_metrics['accuracy']

    # Log PLW status
    plw_str = ", ".join(f"{m}={'PLW' if in_plw[m] else 'no'}" for m in modalities)
    logger.info(f"InfoReg epoch {epoch}: {plw_str}, dominant={dominant_m}, Δ={delta_dominant:.4f}")

    return metrics


def train_epoch_arl(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    temperature: float = 8.0,
    gamma_arl: float = 4.0,
    arl_start: int = 5,
):
    """
    Train for one epoch in ARL mode (Wei et al., ICCV 2025).

    Asymmetric Representation Learning modulates per-modality gradients
    so that the optimization dependency ratio aligns with the inverse
    of each modality's prediction variance.

    Three components:
    1. Modality Analysis — measure variance via self-information entropy
    2. Asymmetric Learning — gradient modulation via GradScale (g + g * coeff)
    3. Unimodal Bias Regularization — unimodal CE losses to reduce bias

    Implementation follows the reference code at:
    https://github.com/shicaiwei123/ICCV2025-ARL

    Parameters
    ----------
    temperature : float
        T in Eq. 18 — controls modulation intensity. Paper uses T=8 for CREMA-D.
    gamma_arl : float
        γ in Eq. 22 — weight for unimodal bias regularization losses.
    arl_start : int
        Epoch after which to start applying asymmetric gradient modulation.
        Before this, use fixed coefficient 0.5 for warm-up (matching reference).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter("Loss")
    loss_fusion_meter = AverageMeter("FusionLoss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]
    m_list = list(modalities)

    # Track unimodal metrics
    unimodal_trackers = {m: MetricTracker(['accuracy']) for m in modalities}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # ===== Forward pass with ARL zero-out unimodal predictions =====
        # ARL uses ConcatFusion_AUXI: a single shared Linear(1024→C) layer
        # for both fused and unimodal predictions.  Unimodal predictions are
        # obtained by zeroing out the other modality's features and passing
        # through the SAME fusion layer.  This is structurally different from
        # using separate unimodal classifiers.
        # Reference: fusion_modules.py ConcatFusion_AUXI.forward()

        # Encode each modality
        features = {}
        for m in modalities:
            features[m] = model.encoders[m](inputs[m])

        # Fused prediction (all modalities)
        feature_list = [features[m] for m in m_list]
        fused = model.fusion(feature_list)
        logits = model.classifier(fused)  # Identity in ARL mode

        # Zero-out unimodal predictions through shared fusion layer
        unimodal_logits = {}
        for i, m in enumerate(m_list):
            zero_features = []
            for j, m2 in enumerate(m_list):
                if m2 == m:
                    zero_features.append(features[m2])
                else:
                    zero_features.append(torch.zeros_like(features[m2]))
            unimodal_logits[m] = model.classifier(model.fusion(zero_features))

        # ===== Component 1: Modality Analysis (Eq. 17) =====
        # Measure inverse variance q^m via self-information entropy
        # Reference: main_arl_variance.py lines 119-132
        entropy_raw = {}
        for m in modalities:
            probs_m = torch.softmax(unimodal_logits[m].detach(), dim=1)
            # H = -Σ p(j) * log(p(j))  averaged over batch
            entropy_raw[m] = -(probs_m * torch.log(probs_m + 1e-16)).sum(dim=1).mean().item()

        # Normalize entropies and clamp (reference lines 127-129)
        entropy_sum = sum(entropy_raw.values()) + 1e-8
        entropy_norm = {m: entropy_raw[m] / entropy_sum for m in modalities}
        entropy_clamped = {m: max(entropy_norm[m], 0.3) for m in modalities}

        # Inverse entropy = variance proxy weight (reference lines 131-141)
        inv_entropy = {m: 1.0 / entropy_clamped[m] for m in modalities}
        inv_sum = sum(inv_entropy.values())
        q_weights = {m: inv_entropy[m] / inv_sum for m in modalities}

        # ===== Component 2: Dependency via mean absolute logits (Eq. 4) =====
        # Reference: main_arl_variance.py lines 116-137
        d_raw = {}
        for m in modalities:
            # Mean absolute logit magnitude (NOT softmax scores)
            d_raw[m] = torch.mean(torch.abs(unimodal_logits[m].detach()), dim=0).sum().item()

        # Normalize dependency
        d_sum = sum(d_raw.values()) + 1e-8
        d_norm = {m: d_raw[m] / d_sum for m in modalities}

        # ===== Asymmetric Learning coefficients (Eq. 18) =====
        # Reference: main_arl_variance.py lines 150-157
        # weight = softmax([d_other * q_self * T, ...]) for each modality
        arl_coeffs = {}

        if epoch > arl_start and len(m_list) >= 2:
            # Build softmax input: for modality i, score = d_other * q_self * T
            # Reference for 2-modality: softmax([d_v * q_a * T, d_a * q_v * T])
            raw_scores = []
            for i, m in enumerate(m_list):
                others = [om for om in m_list if om != m]
                # d_other: average dependency of OTHER modalities
                d_other = sum(d_norm[om] for om in others) / len(others)
                # q_self: inverse variance weight of THIS modality
                q_self = q_weights[m]
                raw_scores.append(d_other * q_self * temperature)

            # Softmax to get coefficients
            raw_tensor = torch.tensor(raw_scores, dtype=torch.float32)
            arl_soft = torch.softmax(raw_tensor, dim=0)
            for i, m in enumerate(m_list):
                arl_coeffs[m] = arl_soft[i].item()
        else:
            # Warm-up: fixed coefficient (reference line 163-164)
            for m in m_list:
                arl_coeffs[m] = 0.5

        # ===== Component 3: Unimodal Bias Regularization (Eq. 21-22) =====
        loss_fusion = criterion(logits, targets)
        loss_unimodal = {}
        for m in modalities:
            loss_unimodal[m] = criterion(unimodal_logits[m], targets)

        # Total loss — matching reference code behavior exactly:
        # - Before arl_start: loss = loss_f + sum(unimodal) * 1.0  (gamma=1)
        # - After arl_start:  loss = loss_f + sum(unimodal) * gamma (gamma=4)
        # Reference lines 159-162: gamma only applied after start epoch
        effective_gamma = gamma_arl if epoch > arl_start else 1.0
        loss = loss_fusion + effective_gamma * sum(loss_unimodal.values())

        # Backward pass
        loss.backward()

        # ===== Apply ARL GradScale modulation (Eq. 19-20) =====
        # Reference backbone.py GradScale: grad = grad + grad * weight
        # IMPORTANT: In the reference code, the computed asymmetric softmax
        # weights are never assigned back to model.args.audio_weight/visual_weight
        # (they stay at the initialized 0.5).  So GradScale is always
        # grad * (1 + 0.5) = 1.5x for both modalities.  Only after arl_start.
        if epoch > arl_start:
            fixed_coeff = 0.5  # Matches reference code init value (never updated)
            for m in modalities:
                for name, param in model.named_parameters():
                    if f"encoders.{m}" in name and param.grad is not None:
                        param.grad.data = param.grad.data + param.grad.data * fixed_coeff

        # Gradient clipping (reference line 171: max_norm=40)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), targets.size(0))
        loss_fusion_meter.update(loss_fusion.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        for m in modalities:
            unimodal_trackers[m].update(unimodal_logits[m], targets)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # Logging
        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/fusion_loss", loss_fusion.item(), global_step)
            for m in modalities:
                writer.add_scalar(f"train/unimodal_loss_{m}", loss_unimodal[m].item(), global_step)
                writer.add_scalar(f"arl/entropy_{m}", entropy_raw[m], global_step)
                writer.add_scalar(f"arl/q_weight_{m}", q_weights[m], global_step)
                writer.add_scalar(f"arl/d_norm_{m}", d_norm[m], global_step)
                writer.add_scalar(f"arl/coeff_{m}", arl_coeffs[m], global_step)
            writer.add_scalar("arl/effective_gamma", effective_gamma, global_step)
            writer.add_scalar("arl/gradscale_active", float(epoch > arl_start), global_step)

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fusion_loss"] = loss_fusion_meter.avg

    for m in modalities:
        unimodal_metrics = unimodal_trackers[m].compute()
        metrics[f"unimodal_acc_{m}"] = unimodal_metrics['accuracy']

    return metrics


def compute_opm_drop_probabilities(
    model: nn.Module,
    features: dict,
    targets: torch.Tensor,
    modalities: list,
    q_base: float = 0.5,
    lam: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-modality drop probabilities for OPM (Wei et al., TPAMI 2024).

    Decomposes the shared fusion classifier weights to estimate per-modality
    discriminative scores, then computes adaptive drop probabilities based on
    the discrepancy ratio between modalities.

    Parameters
    ----------
    model : nn.Module
        Model with fusion.fc as Linear(sum_dims, num_classes).
    features : dict
        Per-modality feature tensors {modality: (B, D)}.
    targets : torch.Tensor
        Ground truth labels (B,).
    modalities : list
        List of modality names.
    q_base : float
        Base drop probability (Eq. 8 in paper).
    lam : float
        Lambda controlling adjustment degree (Eq. 8 in paper).

    Returns
    -------
    torch.Tensor
        Drop probability per modality, shape (M,).
    """
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)

    # Decompose shared classifier weights per modality
    w = model.fusion.fc.weight.clone().detach()  # (C, D_total)
    b = model.fusion.fc.bias.clone().detach()    # (C,)
    M = len(modalities)
    feature_dim = model.feature_dim

    # Compute unimodal discriminative scores (Eq. 6 in paper)
    performances = {}
    offset = 0
    for m in modalities:
        w_m = w[:, offset:offset + feature_dim]  # (C, D_m)
        # Unimodal prediction: W_m · φ_m + b/M
        t_m = torch.mm(features[m].detach(), w_m.t()) + b / M  # (B, C)
        # Sum of softmax probabilities for correct class across batch (Eq. 6)
        probs_m = softmax(t_m)
        performances[m] = sum(probs_m[i][targets[i]] for i in range(len(targets)))
        offset += feature_dim

    # Compute discrepancy ratios and drop probabilities (Eq. 7-8)
    m_list = list(modalities)
    q = torch.zeros(M)

    for i, m in enumerate(m_list):
        others = [om for om in m_list if om != m]
        # ρ_m = (1/(M-1)) * Σ_{j≠m} s_m / s_j  (Eq. 7)
        rho = sum(performances[m] / (performances[om] + 1e-8) for om in others) / len(others)

        if rho > 1:
            # q_m = q_base * (1 + λ * tanh(ρ - 1))  (Eq. 8)
            q[i] = q_base * (1 + lam * torch.tanh(relu(rho - 1)))
        else:
            q[i] = 0.0

    q = torch.clamp(q, 0.0, 1.0)
    return q, performances


def apply_opm_feature_drop(
    features: dict,
    modalities: list,
    q: torch.Tensor,
    p_exe: float = 0.7,
    device: torch.device = None,
) -> tuple:
    """
    Apply OPM feature dropout (Wei et al., TPAMI 2024).

    Drops dominant modality features with adaptive probability and rescales
    remaining features to maintain expected activation magnitude.

    Parameters
    ----------
    features : dict
        Per-modality feature tensors {modality: (B, D)}.
    modalities : list
        Ordered list of modality names.
    q : torch.Tensor
        Per-modality drop probabilities, shape (M,).
    p_exe : float
        Probability of executing the drop at all (not always applied).
    device : torch.device
        Device for tensors.

    Returns
    -------
    tuple
        (dropped_features, update_flag) where dropped_features is dict of
        (B, D) tensors and update_flag is (B,) bool tensor indicating
        which samples have at least one modality remaining.
    """
    m_list = list(modalities)
    M = len(m_list)
    B = features[m_list[0]].shape[0]
    D = features[m_list[0]].shape[1]

    # Decide whether to execute drop this step
    if np.random.rand() > p_exe:
        return features, torch.ones(B, dtype=torch.bool, device=device)

    # Compute rescaling factor: θ = Σ(D_m * q_m) / Σ(D_m)
    # Since all dims are equal: θ = mean(q)
    theta = q.mean().item()

    # Create Bernoulli masks: sample per (modality, batch) with prob (1-q)
    # mask shape: (M, B, 1) — broadcast over feature dim
    masks = []
    for i in range(M):
        mask = torch.bernoulli(
            torch.ones(B, 1, device=device) * (1 - q[i].item())
        )
        masks.append(mask)

    # Apply masks and rescale by 1/(1-θ) (inverted dropout)
    dropped_features = {}
    for i, m in enumerate(m_list):
        dropped_features[m] = features[m] * masks[i]
        if theta < 1.0:
            dropped_features[m] = dropped_features[m] / (1 - theta + 1e-8)

    # Flag samples where at least one modality survives
    mask_stack = torch.cat(masks, dim=1)  # (B, M)
    update_flag = mask_stack.sum(dim=1) > 0  # (B,)

    return dropped_features, update_flag


def train_epoch_opm(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    q_base: float = 0.5,
    lam: float = 0.5,
    p_exe: float = 0.7,
    warmup_epochs: int = 5,
    ogm_ge_config: dict = None,
):
    """
    Train for one epoch in OPM mode (Wei et al., TPAMI 2024).

    On-the-fly Prediction Modulation drops dominant modality features
    during the feed-forward stage with adaptive probability based on
    discriminative discrepancy. Optionally combined with OGM-GE for
    gradient modulation in the back-propagation stage (OPM+OGM).

    Parameters
    ----------
    q_base : float
        Base drop probability (paper default: 0.5).
    lam : float
        Lambda controlling adjustment degree (paper default: 0.5).
    p_exe : float
        Probability of executing the drop in each step (paper default: 0.7).
    warmup_epochs : int
        Number of warm-up epochs before OPM is applied (paper default: 5).
    ogm_ge_config : dict or None
        If not None, also apply OGM-GE gradient modulation (OPM+OGM combined).
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter("Loss")
    loss_fusion_meter = AverageMeter("FusionLoss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]
    m_list = list(modalities)

    # Track unimodal metrics
    unimodal_trackers = {m: MetricTracker(['accuracy']) for m in modalities}

    warm_up = epoch <= warmup_epochs

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # ===== Encode each modality =====
        features = {}
        for m in modalities:
            features[m] = model.encoders[m](inputs[m])

        # ===== Compute unimodal scores (for OPM and OGM) =====
        # Decompose shared classifier to get per-modality predictions
        w = model.fusion.fc.weight.clone().detach()
        b = model.fusion.fc.bias.clone().detach()
        M = len(m_list)
        unimodal_logits = {}
        offset = 0
        for m in m_list:
            w_m = w[:, offset:offset + model.feature_dim]
            unimodal_logits[m] = torch.mm(features[m].detach(), w_m.t()) + b / M
            offset += model.feature_dim

        # ===== Apply OPM feature dropout (feed-forward stage) =====
        if not warm_up:
            q, performances = compute_opm_drop_probabilities(
                model, features, targets, modalities,
                q_base=q_base, lam=lam,
            )
            dropped_features, update_flag = apply_opm_feature_drop(
                features, modalities, q, p_exe=p_exe, device=device,
            )
        else:
            dropped_features = features
            update_flag = torch.ones(targets.size(0), dtype=torch.bool, device=device)
            q = torch.zeros(len(m_list))

        # ===== Forward through fusion with (potentially dropped) features =====
        # Filter out samples where all modalities were dropped
        if not warm_up and not update_flag.all():
            # Only compute loss on samples with at least one modality
            select_mask = update_flag
            filtered_features = [dropped_features[m][select_mask] for m in m_list]
            filtered_targets = targets[select_mask]

            if len(filtered_targets) == 0:
                continue

            fused = model.fusion(filtered_features)
            logits = model.classifier(fused)
            loss = criterion(logits, filtered_targets)
        else:
            feature_list = [dropped_features[m] for m in m_list]
            fused = model.fusion(feature_list)
            logits = model.classifier(fused)
            loss = criterion(logits, targets)
            filtered_targets = targets

        # ===== Backward pass =====
        loss.backward()

        # ===== Apply OGM-GE gradient modulation (back-propagation stage) =====
        # OGM-GE is independent of OPM warm-up — always active when enabled
        if ogm_ge_config is not None and ogm_ge_config["enabled"]:
            ogm_coeffs, ogm_scores = apply_ogm_ge(
                model, unimodal_logits, targets, modalities,
                alpha=ogm_ge_config["alpha"],
                epoch=epoch,
                modulation_start=ogm_ge_config["modulation_start"],
                modulation_end=ogm_ge_config["modulation_end"],
            )
            if writer and batch_idx % config["logging"]["log_interval"] == 0:
                for m in modalities:
                    c = ogm_coeffs[m]
                    writer.add_scalar(
                        f"ogm_ge/coeff_{m}",
                        c.item() if isinstance(c, torch.Tensor) else c,
                        global_step,
                    )

        optimizer.step()

        # ===== Update metrics =====
        loss_meter.update(loss.item(), filtered_targets.size(0))
        loss_fusion_meter.update(loss.item(), filtered_targets.size(0))
        metric_tracker.update(logits, filtered_targets)

        for m in modalities:
            unimodal_trackers[m].update(unimodal_logits[m], targets)

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

        # ===== Logging =====
        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            for i, m in enumerate(m_list):
                writer.add_scalar(f"opm/q_{m}", q[i].item(), global_step)
            writer.add_scalar("opm/warm_up", float(warm_up), global_step)

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg
    metrics["fusion_loss"] = loss_fusion_meter.avg

    for m in modalities:
        unimodal_metrics = unimodal_trackers[m].compute()
        metrics[f"unimodal_acc_{m}"] = unimodal_metrics['accuracy']

    return metrics


def apply_ogm_ge(
    model: nn.Module,
    unimodal_logits: dict,
    targets: torch.Tensor,
    modalities: list,
    alpha: float,
    epoch: int,
    modulation_start: int = 0,
    modulation_end: int = 50,
) -> tuple:
    """
    Apply OGM-GE gradient modulation (Peng et al., NeurIPS 2022).

    Modulates encoder gradients based on per-modality softmax score ratios.
    The dominant modality's gradients are scaled down and injected with
    Gaussian noise.

    Supports N>=2 modalities: each modality's score is compared to the mean
    score across all modalities.  Modalities scoring above the mean are
    considered dominant and their gradients are scaled down.

    Applies to both Conv2d (4D) and Linear (2D) weight gradients, enabling
    use with MLP encoders (e.g., for pre-extracted features in MOSEI).

    Parameters
    ----------
    model : nn.Module
        The multimodal model with named parameters containing 'encoders.{m}'.
    unimodal_logits : dict
        Per-modality logits {modality_name: tensor of shape (B, num_classes)}.
    targets : torch.Tensor
        Ground truth labels of shape (B,).
    modalities : list
        List of modality names.
    alpha : float
        OGM-GE coefficient controlling modulation strength.
    epoch : int
        Current epoch number.
    modulation_start : int
        Epoch to start applying modulation.
    modulation_end : int
        Epoch to stop applying modulation.

    Returns
    -------
    tuple
        (coeffs, scores) where coeffs maps modality to scaling coefficient
        and scores maps modality to summed softmax score for true labels.
    """
    softmax = nn.Softmax(dim=1)
    tanh = nn.Tanh()
    relu = nn.ReLU(inplace=True)

    # Compute per-modality softmax scores for true labels
    scores = {}
    for m in modalities:
        probs = softmax(unimodal_logits[m])
        scores[m] = sum(probs[i][targets[i]] for i in range(len(targets)))

    # Compute ratios and coefficients (Eq. 10 in OGM-GE paper)
    # Generalized for N modalities: compare each modality's score to the mean.
    # Dominant modalities (score > mean) get scaled down; others stay at 1.0.
    m_list = list(modalities)
    coeffs = {m: 1.0 for m in modalities}

    if len(m_list) == 2:
        # Original 2-modality formulation (exact OGM-GE paper)
        ratio = scores[m_list[0]] / (scores[m_list[1]] + 1e-8)
        if ratio > 1:
            coeffs[m_list[0]] = 1 - tanh(alpha * relu(ratio))
        else:
            coeffs[m_list[1]] = 1 - tanh(alpha * relu(1.0 / (ratio + 1e-8)))
    elif len(m_list) > 2:
        # N-modality generalization: ratio = score_m / mean(scores)
        mean_score = sum(scores[m] for m in m_list) / len(m_list)
        for m in m_list:
            ratio = scores[m] / (mean_score + 1e-8)
            if ratio > 1:
                coeffs[m] = 1 - tanh(alpha * relu(ratio))

    # Apply modulation to encoder layers (Conv2d and Linear weights)
    if modulation_start <= epoch <= modulation_end:
        for m in modalities:
            coeff = coeffs[m]
            if isinstance(coeff, torch.Tensor):
                coeff_val = coeff
            else:
                continue  # coeff is 1.0, no modulation needed
            for name, param in model.named_parameters():
                if f"encoders.{m}" in name and param.grad is not None:
                    # Apply to Conv2d (4D) and Linear weight (2D) gradients
                    if len(param.grad.size()) >= 2:
                        param.grad = param.grad * coeff_val + \
                            torch.zeros_like(param.grad).normal_(
                                0, param.grad.std().item() + 1e-8
                            )

    return coeffs, scores


# =============================================================================
# MMPareto (Wei & Hu, ICML 2024)
# Pareto-optimal gradient weighting for multimodal + unimodal objectives
# Ported from: https://github.com/GeWu-Lab/MMPareto_ICML2024
# =============================================================================

class MinNormSolver:
    """Find minimum norm element in convex hull of gradient vectors."""
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        if v1v2 >= v1v1:
            return 0.999, v1v1
        if v1v2 >= v2v2:
            return 0.001, v2v2
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        dmin = 1e8
        sol = None
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = sum(
                        torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                        for k in range(len(vecs[i]))
                    )
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = sum(
                        torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                        for k in range(len(vecs[i]))
                    )
                if (j, j) not in dps:
                    dps[(j, j)] = sum(
                        torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                        for k in range(len(vecs[i]))
                    )
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))
        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        new_dps = {}
        new_init_sol = []
        for item in dps:
            new_dps[item] = dps[item].numpy() if torch.is_tensor(dps[item]) else dps[item]
        for item in init_sol:
            data = item.numpy() if torch.is_tensor(item) else item
            new_init_sol.append(data)
        dps = new_dps
        init_sol = new_init_sol

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        iter_count = 0
        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            v1v1, v1v2, v2v2 = 0.0, 0.0, 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
            iter_count += 1
        return sol_vec, nd


def apply_probe_boost_hook(
    model: nn.Module,
    modalities: list,
    probe_manager,
    scheduler,
    features: dict,
    targets,
    batch_idx: int,
    config: dict,
) -> None:
    """Post-backward plug-and-play hook: probe refresh + multiplicative boost
    on encoder gradients. Matches the continuous-mode boost used in Table 1
    Boost+OGM-GE exactly: split-batch probe protocol, EMA scale smoothing,
    same alpha / s_max / K / mu defaults.

    Must be called AFTER loss.backward() and any method-specific gradient
    modulation (OGM-GE throttle, AGM exp factor, MMPareto Pareto weights,
    CGGM magnitude+direction, G-Blend loss weighting), and BEFORE
    optimizer.step().
    """
    K = config["asgml"].get("continuous_eval_freq", 20)
    probe_steps = config["asgml"].get("continuous_probe_train_steps", 10)
    alpha = config["asgml"].get("continuous_alpha", 0.5)
    scale_max = config["asgml"].get("continuous_scale_max", 2.0)

    # Periodic probe refresh (split-batch protocol: train on first half, eval on second)
    if (batch_idx + 1) % K == 0 and features is not None and targets is not None:
        batch_size = targets.size(0)
        split = batch_size // 2
        if split > 0:
            train_feat = {m: f[:split] for m, f in features.items()}
            eval_feat = {m: f[split:] for m, f in features.items()}
            probe_manager.train_probes(train_feat, targets[:split], num_steps=probe_steps)
            probe_manager.evaluate_probes(eval_feat, targets[split:])
            util_scores = probe_manager.get_utilization_scores(use_ema=True)
            new_scales = scheduler.get_continuous_scales(
                utilization_scores=util_scores,
                alpha=alpha,
                scale_max=scale_max,
            )
            scheduler.update_continuous_scales(new_scales)

    # Apply EMA-smoothed boost scales to encoder gradients (every step)
    for m in modalities:
        scale = scheduler.current_continuous_scales.get(m, 1.0)
        if scale == 1.0:
            continue
        for name, param in model.named_parameters():
            if f"encoders.{m}" in name and param.grad is not None:
                param.grad.data.mul_(scale)


def train_epoch_mmpareto(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    gamma: float = 1.5,
    probe_manager=None,
    scheduler=None,
):
    """Train one epoch with MMPareto (Wei & Hu, ICML 2024).

    For each batch:
    1. Compute multimodal loss + per-modality unimodal losses
    2. Compute per-modality gradients for each loss separately
    3. Check gradient alignment (cosine similarity)
    4. If conflict: solve Pareto-optimal weights via MinNormSolver
    5. Apply weighted gradients with magnitude preservation
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    # Identify per-modality parameters (exclude fusion head)
    modality_params = {m: [] for m in modalities}
    for name, param in model.named_parameters():
        for m in modalities:
            if f"encoders.{m}" in name or f"unimodal_classifiers.{m}" in name:
                modality_params[m].append((name, param))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, unimodal_logits, features = model(inputs, return_features=True)

        # Compute losses
        loss_mm = criterion(logits, targets)
        unimodal_losses = {m: criterion(unimodal_logits[m], targets) for m in modalities}

        # Collect gradients per loss per modality
        all_losses = [loss_mm] + [unimodal_losses[m] for m in modalities]
        loss_labels = ["both"] + list(modalities)

        grads_per_modality = {m: {} for m in modalities}

        for idx, loss_label in enumerate(loss_labels):
            all_losses[idx].backward(retain_graph=True)
            for m in modalities:
                if loss_label == "both" or loss_label == m:
                    grads_per_modality[m][loss_label] = {}
                    for pname, param in modality_params[m]:
                        if param.grad is not None:
                            grads_per_modality[m][loss_label][pname] = param.grad.data.clone()
            optimizer.zero_grad()

        # Compute Pareto weights per modality
        pareto_weights = {}
        for m in modalities:
            if "both" not in grads_per_modality[m] or m not in grads_per_modality[m]:
                pareto_weights[m] = [0.5, 0.5]
                continue

            # Find parameters with gradients in BOTH loss types
            common_params = [
                pn for pn, _ in modality_params[m]
                if pn in grads_per_modality[m]["both"] and pn in grads_per_modality[m][m]
            ]
            if not common_params:
                pareto_weights[m] = [0.5, 0.5]
                continue

            # Concatenate gradients for cosine similarity
            grads_both = torch.cat([
                grads_per_modality[m]["both"][pn].flatten() for pn in common_params
            ])
            grads_uni = torch.cat([
                grads_per_modality[m][m][pn].flatten() for pn in common_params
            ])

            cos_sim = nn.functional.cosine_similarity(grads_both, grads_uni, dim=0)

            if cos_sim > 0:
                pareto_weights[m] = [0.5, 0.5]
            else:
                # Build gradient vectors for MinNormSolver
                grad_vecs = [
                    [grads_per_modality[m]["both"][pn] for pn in common_params],
                    [grads_per_modality[m][m][pn] for pn in common_params],
                ]
                sol, _ = MinNormSolver.find_min_norm_element(grad_vecs)
                pareto_weights[m] = sol.tolist()

        # Compute combined loss and backward for fusion head gradients
        total_loss = loss_mm + sum(unimodal_losses[m] for m in modalities)
        total_loss.backward()

        # Override encoder gradients with Pareto-weighted gradients
        for m in modalities:
            w_both, w_uni = pareto_weights[m]
            for pname, param in modality_params[m]:
                if param.grad is not None and pname in grads_per_modality[m].get("both", {}) and pname in grads_per_modality[m].get(m, {}):
                    three_norm = torch.norm(param.grad.data)
                    new_grad = 2 * w_both * grads_per_modality[m]["both"][pname] + \
                               2 * w_uni * grads_per_modality[m][m][pname]
                    new_norm = torch.norm(new_grad)
                    if new_norm > 0:
                        diff = three_norm / new_norm
                        if diff > 1:
                            param.grad = diff * new_grad * gamma
                        else:
                            param.grad = new_grad * gamma

        # Plug-and-play boost hook (matches Table 1 Boost+OGM-GE setup exactly)
        if probe_manager is not None and scheduler is not None:
            apply_probe_boost_hook(
                model, list(modalities), probe_manager, scheduler,
                features, targets, batch_idx, config,
            )

        optimizer.step()

        loss_meter.update(total_loss.item())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            metric_tracker.update(preds, targets)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if writer and batch_idx % config["logging"].get("log_interval", 20) == 0:
            writer.add_scalar("train/loss", total_loss.item(), global_step)
            writer.add_scalar("train/loss_mm", loss_mm.item(), global_step)
            for m in modalities:
                w = pareto_weights.get(m, [0.5, 0.5])
                writer.add_scalar(f"mmpareto/w_both_{m}", w[0], global_step)
                writer.add_scalar(f"mmpareto/w_uni_{m}", w[1], global_step)

    return loss_meter.avg, metric_tracker.compute()


# =============================================================================
# AGM (Li et al., 2023)
# Adaptive Gradient Modulation via loss-based exponential scaling
# Ported from: https://github.com/lihongcs/AGM
# =============================================================================

def train_epoch_agm(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    alpha: float = 1.0,
    modulation_start: int = 0,
    modulation_end: int = 50,
    running_scores: dict = None,
    probe_manager=None,
    scheduler=None,
):
    """Train one epoch with AGM (Li et al., 2023).

    For each batch:
    1. Compute per-modality cross-entropy loss scores
    2. Compute current batch ratios: exp(mean_score - score_m)
    3. Compute optimal ratios from running average scores
    4. Coefficient = exp(alpha * (optimal_ratio - current_ratio))
    5. Apply coefficients to encoder gradients via backward hooks
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]
    m_list = list(modalities)

    # Initialize running scores if not provided
    if running_scores is None:
        running_scores = {m: 0.0 for m in modalities}

    total_batch = len(dataloader)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * total_batch + batch_idx
        iteration = global_step + 1  # 1-indexed for running average

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, unimodal_logits, features = model(inputs, return_features=True)

        loss = criterion(logits, targets)

        # Compute per-modality cross-entropy scores (average per-sample log-loss)
        batch_scores = {}
        for m in m_list:
            probs = softmax(unimodal_logits[m])
            score = 0.0
            for k in range(probs.size(0)):
                p = probs[k][targets[k]]
                if p < 1e-8 or torch.isinf(torch.log(p)):
                    score += -torch.log(torch.tensor(1e-8, dtype=probs.dtype, device=device)).item()
                else:
                    score += -torch.log(p).item()
            batch_scores[m] = score / probs.size(0)

        # Compute coefficients
        coeffs = {m: 1.0 for m in m_list}
        if modulation_start <= epoch <= modulation_end:
            n_mod = len(m_list)
            mean_score = sum(batch_scores[m] for m in m_list) / n_mod
            mean_running = sum(running_scores[m] for m in m_list) / n_mod

            for m in m_list:
                # Current batch ratio
                ratio_m = math.exp(mean_score - batch_scores[m])
                # Optimal ratio from running average
                optimal_ratio_m = math.exp(mean_running - running_scores[m])
                # AGM coefficient
                coeffs[m] = math.exp(alpha * min(optimal_ratio_m - ratio_m, 10))

        # Update running average scores
        for m in m_list:
            running_scores[m] = running_scores[m] * (iteration - 1) / iteration + batch_scores[m] / iteration

        # Backward with gradient scaling
        loss.backward()

        if modulation_start <= epoch <= modulation_end:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    for m in m_list:
                        if f"encoders.{m}" in name:
                            param.grad.data = param.grad.data * coeffs[m]

        # Plug-and-play boost hook (matches Table 1 Boost+OGM-GE setup exactly)
        if probe_manager is not None and scheduler is not None:
            apply_probe_boost_hook(
                model, m_list, probe_manager, scheduler,
                features, targets, batch_idx, config,
            )

        optimizer.step()

        loss_meter.update(loss.item())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            metric_tracker.update(preds, targets)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if writer and batch_idx % config["logging"].get("log_interval", 20) == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            for m in m_list:
                writer.add_scalar(f"agm/coeff_{m}", coeffs[m], global_step)
                writer.add_scalar(f"agm/score_{m}", batch_scores[m], global_step)

    return loss_meter.avg, metric_tracker.compute(), running_scores


# =============================================================================
# Grad-Blending / G-Blend (Wang et al., CVPR 2020)
# Overfitting-to-generalization ratio weighting
# Implemented from paper description (no official repo available)
# =============================================================================

def train_epoch_gblend(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    blend_weights: dict = None,
    probe_manager=None,
    scheduler=None,
):
    """Train one epoch with Grad-Blending / G-Blend (Wang et al., CVPR 2020).

    G-Blend computes per-modality overfitting-to-generalization (OG) ratios
    and uses them to weight the loss contributions. The blend weights are
    computed at the END of each epoch and applied during the NEXT epoch.

    OG ratio for modality m:
        OG_m = (train_loss_m - val_loss_m) / val_loss_m
    Higher OG means more overfitting → lower weight.

    Weight for modality m:
        w_m = (1 / OG_m) / sum(1 / OG_j for all j)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]
    m_list = list(modalities)

    # Default to uniform weights if not provided
    if blend_weights is None:
        n = len(m_list)
        blend_weights = {m: 1.0 / n for m in m_list}

    # Track per-modality train losses for OG ratio computation
    train_loss_trackers = {m: AverageMeter(f"train_loss_{m}") for m in m_list}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits, unimodal_logits, features = model(inputs, return_features=True)

        # Weighted loss: fusion loss + weighted sum of unimodal losses
        fusion_loss = criterion(logits, targets)
        unimodal_losses = {}
        for m in m_list:
            unimodal_losses[m] = criterion(unimodal_logits[m], targets)
            train_loss_trackers[m].update(unimodal_losses[m].item())

        # Total loss = fusion + blend-weighted unimodal
        weighted_uni_loss = sum(blend_weights[m] * unimodal_losses[m] for m in m_list)
        total_loss = fusion_loss + weighted_uni_loss

        total_loss.backward()

        # Plug-and-play boost hook (matches Table 1 Boost+OGM-GE setup exactly)
        if probe_manager is not None and scheduler is not None:
            apply_probe_boost_hook(
                model, m_list, probe_manager, scheduler,
                features, targets, batch_idx, config,
            )

        optimizer.step()

        loss_meter.update(total_loss.item())
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            metric_tracker.update(preds, targets)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if writer and batch_idx % config["logging"].get("log_interval", 20) == 0:
            writer.add_scalar("train/loss", total_loss.item(), global_step)
            for m in m_list:
                writer.add_scalar(f"gblend/weight_{m}", blend_weights[m], global_step)
                writer.add_scalar(f"gblend/train_loss_{m}", train_loss_trackers[m].avg, global_step)

    train_losses = {m: train_loss_trackers[m].avg for m in m_list}
    return loss_meter.avg, metric_tracker.compute(), train_losses


def compute_gblend_weights(
    model: nn.Module,
    train_losses: dict,
    val_loader: DataLoader,
    device: torch.device,
    modalities: list,
    logger,
) -> dict:
    """Compute G-Blend weights from overfitting-to-generalization ratios.

    Called at the end of each epoch to compute weights for the next epoch.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    m_list = list(modalities)
    val_loss_trackers = {m: AverageMeter(f"val_loss_{m}") for m in m_list}

    with torch.no_grad():
        for batch in val_loader:
            inputs = {m: batch[m].to(device) for m in modalities}
            targets = batch["label"].to(device)
            _, unimodal_logits, _ = model(inputs, return_features=True)
            for m in m_list:
                val_loss = criterion(unimodal_logits[m], targets)
                val_loss_trackers[m].update(val_loss.item())

    # Compute OG ratios and weights
    og_ratios = {}
    for m in m_list:
        val_loss = val_loss_trackers[m].avg
        train_loss = train_losses[m]
        if val_loss > 1e-8:
            og_ratios[m] = max((train_loss - val_loss) / val_loss, 1e-4)
        else:
            og_ratios[m] = 1e-4

    # Inverse OG ratio as weight
    inv_og = {m: 1.0 / og_ratios[m] for m in m_list}
    total = sum(inv_og.values())
    weights = {m: inv_og[m] / total for m in m_list}

    logger.info(
        f"G-Blend OG ratios: {', '.join(f'{m}={og_ratios[m]:.4f}' for m in m_list)} → "
        f"weights: {', '.join(f'{m}={weights[m]:.4f}' for m in m_list)}"
    )

    return weights


def train_epoch_cggm(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    cggm: CGGMModule,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    ogm_ge_config: dict = None,
    probe_manager=None,
    scheduler=None,
):
    """Train one epoch with CGGM (Guo et al., NeurIPS 2024).

    CGGM modulates both gradient magnitude and direction:
    1. Forward main model → loss → backward
    2. Forward per-modality classifiers on detached features
    3. Compute accuracy-change coefficients (magnitude)
    4. Compute gradient direction loss L_gm
    5. Scale encoder gradients by coefficients * rou
    6. Optimizer step
    """
    model.train()
    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]
    num_classes = config["dataset"]["num_classes"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        # ========== Forward + Backward ==========
        optimizer.zero_grad()
        logits, unimodal_logits, features = model(inputs, return_features=True)

        # Main task loss
        main_loss = criterion(logits, targets)

        # Add L_gm from previous iteration (CGGM applies it one step delayed)
        l_gm_prev = cggm.get_l_gm_for_loss()
        if l_gm_prev is not None:
            main_loss = main_loss + cggm.lamda * l_gm_prev

        main_loss.backward()

        # ========== OGM-GE (optional, can combine with CGGM) ==========
        if ogm_ge_config is not None and ogm_ge_config["enabled"]:
            apply_ogm_ge(
                model, unimodal_logits, targets, modalities,
                alpha=ogm_ge_config["alpha"],
                epoch=epoch,
                modulation_start=ogm_ge_config["modulation_start"],
                modulation_end=ogm_ge_config["modulation_end"],
            )

        # ========== CGGM Step ==========
        l_gm_value, coeff = cggm.step(model, features, targets, criterion)

        # Plug-and-play boost hook (matches Table 1 Boost+OGM-GE setup exactly)
        if probe_manager is not None and scheduler is not None:
            apply_probe_boost_hook(
                model, list(modalities), probe_manager, scheduler,
                features, targets, batch_idx, config,
            )

        # ========== Gradient clipping + optimizer step ==========
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
        optimizer.step()

        # ========== Logging ==========
        loss_meter.update(main_loss.item())
        with torch.no_grad():
            if num_classes > 1:
                preds = logits.argmax(dim=-1)
            else:
                preds = logits.squeeze()
            metric_tracker.update(preds, targets)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", mask={m: 1 for m in modalities})

        if writer and batch_idx % config["logging"].get("log_interval", 20) == 0:
            writer.add_scalar("train/loss", main_loss.item(), global_step)
            writer.add_scalar("cggm/l_gm", l_gm_value, global_step)
            for m in modalities:
                writer.add_scalar(f"cggm/coeff_{m}", coeff[m], global_step)

    return loss_meter.avg, metric_tracker.compute()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: ASGMLLoss,
    scheduler: ASGMLScheduler,
    probe_manager: ProbeManager,
    device: torch.device,
    epoch: int,
    config: dict,
    logger,
    writer: SummaryWriter = None,
    scaler: GradScaler = None,
    use_amp: bool = False,
    ogm_ge_config: dict = None,
    probe_stability_log: list = None,
    probe_stability_loader: DataLoader = None,
):
    """
    Train for one epoch with ASGML.

    This implements the full ASGML training loop:
    1. Forward pass through model
    2. Compute loss
    3. Backward pass (compute gradients)
    4. Store gradients in staleness buffer (if staleness mode)
    5. Apply staleness/frequency mask to gradients
    6. Optimizer step (fusion head always updates)
    7. Periodically train and evaluate probes
    """
    model.train()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    eval_freq = config["asgml"].get("eval_freq", 100)
    probe_train_steps = config["asgml"].get("probe_train_steps", 50)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        global_step = (epoch - 1) * len(dataloader) + batch_idx

        # Move data to device (non_blocking for async transfer)
        inputs = {m: batch[m].to(device, non_blocking=True) for m in modalities}
        targets = batch["label"].to(device, non_blocking=True)

        # ========== Forward Pass ==========
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            logits, unimodal_logits, features = model(inputs, return_features=True)

            # Get update mask and staleness values from scheduler
            update_mask = scheduler.get_update_mask()

            # Diagnostic logging for debugging ASGML activation
            if batch_idx < 5 or batch_idx % 20 == 0:
                if scheduler.mode == "continuous":
                    logger.debug(
                        f"[Step {global_step}] mode=continuous "
                        f"scales={scheduler.current_continuous_scales} "
                        f"mask={update_mask}"
                    )
                else:
                    logger.debug(
                        f"[Step {global_step}] tau={scheduler.current_tau} "
                        f"mask={update_mask} "
                        f"speeds={scheduler.dynamics.compute_learning_speed()}"
                    )

            # ========== Compute Loss ==========
            loss, loss_dict = loss_fn(
                logits, unimodal_logits, targets, update_mask
            )

        # ========== Backward Pass ==========
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # CRITICAL FIX: Unscale gradients BEFORE storing/modifying
            # This prevents AMP scale factor mismatch when applying stale gradients
            scaler.unscale_(optimizer)

            # Check for inf/nan gradients after unscaling
            # scaler.unscale_() can produce inf/nan on overflow
            grads_valid = True
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grads_valid = False
                        break
        else:
            loss.backward()
            grads_valid = True

        # ========== Compute Gradient Norms (for tracking) ==========
        grad_norms = compute_gradient_norms(model, modalities) if grads_valid else {m: 0.0 for m in modalities}

        # ========== OGM-GE Gradient Modulation (optional) ==========
        if ogm_ge_config is not None and ogm_ge_config["enabled"] and grads_valid:
            ogm_coeffs, ogm_scores = apply_ogm_ge(
                model, unimodal_logits, targets, modalities,
                alpha=ogm_ge_config["alpha"],
                epoch=epoch,
                modulation_start=ogm_ge_config["modulation_start"],
                modulation_end=ogm_ge_config["modulation_end"],
            )
            # Log OGM-GE metrics
            if writer and batch_idx % config["logging"]["log_interval"] == 0:
                for m in modalities:
                    c = ogm_coeffs[m]
                    writer.add_scalar(
                        f"ogm_ge/coeff_{m}",
                        c.item() if isinstance(c, torch.Tensor) else c,
                        global_step,
                    )

        # ========== Store Gradients (for staleness mode) ==========
        # Skip storing if AMP overflow detected (grads_valid=False)
        # fp32 mode: grads_valid is always True (no overflow issues)
        if scheduler.mode == "staleness" and grads_valid:
            for m in modalities:
                encoder_params = {
                    name: param
                    for name, param in model.named_parameters()
                    if f"encoders.{m}" in name
                }
                scheduler.staleness_buffer.store_gradients(
                    m, encoder_params, global_step
                )

        # ========== Apply ASGML Gradient Modifications ==========
        # Only modify gradients if they are valid (no inf/nan from AMP overflow)
        if config["asgml"]["enabled"] and grads_valid:
            if scheduler.mode == "staleness":
                # Apply stale gradients for dominant modality
                staleness_vals = scheduler.get_staleness_values()
                grad_scales = scheduler.get_gradient_scales()

                for m in modalities:
                    if staleness_vals[m] > 0:
                        apply_staleness_gradients(
                            model,
                            scheduler.staleness_buffer,
                            m,
                            staleness_vals[m],
                            grad_scales[m],
                        )
            elif scheduler.mode == "continuous":
                # Continuous probe-guided gradient scaling
                # Apply EMA-smoothed scales to encoder gradients every step
                noise_sigma = config["asgml"].get("continuous_noise_sigma", 0.0)
                for m in modalities:
                    scale = scheduler.current_continuous_scales[m]
                    for name, param in model.named_parameters():
                        if f"encoders.{m}" in name and param.grad is not None:
                            param.grad.data.mul_(scale)
                            if noise_sigma > 0 and len(param.grad.size()) == 4:
                                noise = torch.zeros_like(param.grad).normal_(
                                    0, param.grad.std().item() * noise_sigma + 1e-8
                                )
                                param.grad.data.add_(noise)
            else:
                # Frequency mode: reduce encoder gradients for non-updating modalities
                hard_mask = config["asgml"].get("hard_frequency_mask", False)
                soft_scale = config["asgml"].get("soft_mask_scale", 0.1)

                for m in modalities:
                    if not update_mask[m]:
                        for name, param in model.named_parameters():
                            if f"encoders.{m}" in name and param.grad is not None:
                                if hard_mask:
                                    param.grad.data.zero_()
                                else:
                                    param.grad.data.mul_(soft_scale)

        # ========== Optimizer Step ==========
        # NOTE: Fusion head always updates regardless of modality schedules
        if use_amp and scaler is not None:
            # Gradients already unscaled above, so use step() directly
            # Check for inf/nan before stepping (scaler.step handles this)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ========== Update Scheduler ==========
        scheduler.dynamics.update(
            {m: loss_dict[f'unimodal_{m}'].item() for m in modalities},
            grad_norms,
        )
        scheduler.step()

        # ========== Update Metrics ==========
        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # ========== Probe Training & Evaluation (periodic) ==========
        # Continuous mode uses more frequent probe evaluations
        if scheduler.mode == "continuous":
            current_eval_freq = config["asgml"].get("continuous_eval_freq", 20)
            current_probe_steps = config["asgml"].get("continuous_probe_train_steps", 10)
        else:
            current_eval_freq = eval_freq
            current_probe_steps = probe_train_steps

        if (batch_idx + 1) % current_eval_freq == 0:
            # Split batch: train probes on first half, evaluate on second half
            # This prevents probe overfitting (e.g., audio probe hitting 100%
            # on training data while test accuracy is only ~56%)
            batch_size = targets.size(0)
            split = batch_size // 2
            train_features = {m: f[:split] for m, f in features.items()}
            train_targets = targets[:split]
            eval_features = {m: f[split:] for m, f in features.items()}
            eval_targets = targets[split:]

            # Train probes on first half (DETACHED - safe)
            probe_manager.train_probes(train_features, train_targets, num_steps=current_probe_steps)

            # Evaluate probes on second half (unseen by probe during training)
            probe_results = probe_manager.evaluate_probes(eval_features, eval_targets)
            utilization_gap = probe_manager.compute_utilization_gap()
            dominant = probe_manager.get_dominant_modality()

            # Probe-stability instrumentation: record batch-raw, EMA-smoothed, and
            # full-test-set probe accuracy for the estimator-stability appendix.
            if probe_stability_log is not None and probe_stability_loader is not None:
                import json as _json  # local to avoid polluting global namespace
                model.eval()
                full_feat = {m: [] for m in modalities}
                full_tgt = []
                with torch.no_grad():
                    for _b in probe_stability_loader:
                        _inp = {m: _b[m].to(device) for m in modalities}
                        _lbl = _b["label"].to(device)
                        _, _, _feat = model(_inp, return_features=True)
                        for m in modalities:
                            full_feat[m].append(_feat[m].detach().float())
                        full_tgt.append(_lbl)
                    full_feat = {m: torch.cat(full_feat[m], dim=0) for m in modalities}
                    full_tgt = torch.cat(full_tgt, dim=0)
                    full_results = probe_manager.evaluate_probes(full_feat, full_tgt)
                model.train()
                ema_scores = probe_manager.get_utilization_scores(use_ema=True)
                probe_stability_log.append({
                    "epoch": int(epoch),
                    "batch_idx": int(batch_idx),
                    "iteration": int((epoch - 1) * len(dataloader) + batch_idx),
                    "batch_raw": {m: float(probe_results[m]["accuracy"]) for m in modalities},
                    "ema_smoothed": {m: float(ema_scores[m]) for m in modalities},
                    "full_test": {m: float(full_results[m]["accuracy"]) for m in modalities},
                })

            # ========== Update Scheduler (Adaptive Mode) ==========
            if scheduler.adaptation == "adaptive":
                signal_source = config["asgml"].get("signal_source", "dual")

                if scheduler.mode == "continuous":
                    # Continuous mode: compute scales from probe utilization
                    util_scores = probe_manager.get_utilization_scores(use_ema=True)
                    new_scales = scheduler.get_continuous_scales(
                        utilization_scores=util_scores,
                        alpha=config["asgml"].get("continuous_alpha", 0.5),
                        scale_max=config["asgml"].get("continuous_scale_max", 2.0),
                    )
                    scheduler.update_continuous_scales(new_scales)
                    if util_scores:
                        scheduler.set_dominant_modality(
                            max(util_scores, key=util_scores.get)
                        )

                elif signal_source == "dual":
                    # Design doc method: S_i(t) = β·G_i(t) + (1-β)·D_i(t)
                    # Uses gradient magnitude + loss descent rate (no probes)
                    learning_speeds = scheduler.dynamics.compute_learning_speed()
                    dominant_from_dynamics = max(learning_speeds, key=learning_speeds.get)
                    scheduler.set_dominant_modality(dominant_from_dynamics)

                    # Update tau from learning speeds
                    for m in modalities:
                        raw_tau = scheduler.tau_base * learning_speeds[m]
                        scheduler.current_tau[m] = max(
                            scheduler.tau_min,
                            min(scheduler.tau_max, raw_tau)
                        )

                elif signal_source == "probe" and utilization_gap is not None:
                    # Probe-based method (original implementation)
                    scheduler.set_dominant_modality(dominant)
                    scheduler.update_from_utilization(
                        probe_manager.get_utilization_scores(),
                        utilization_gap,
                    )

                elif signal_source == "both" and utilization_gap is not None:
                    # Combine dual-signal and probe signals
                    learning_speeds = scheduler.dynamics.compute_learning_speed()
                    probe_scores = probe_manager.get_utilization_scores()

                    # Normalize probe scores to be comparable to learning speeds
                    max_probe = max(probe_scores.values()) if probe_scores.values() else 1.0
                    norm_probe = {m: v / max(max_probe, 1e-8) for m, v in probe_scores.items()}

                    alpha = config["asgml"].get("signal_blend", 0.5)  # Weight for dual-signal

                    for m in modalities:
                        combined = alpha * learning_speeds[m] + (1 - alpha) * norm_probe[m]
                        raw_tau = scheduler.tau_base * combined
                        scheduler.current_tau[m] = max(
                            scheduler.tau_min,
                            min(scheduler.tau_max, raw_tau)
                        )

                    # Set dominant from combined signal
                    combined_scores = {
                        m: alpha * learning_speeds[m] + (1 - alpha) * norm_probe[m]
                        for m in modalities
                    }
                    scheduler.set_dominant_modality(max(combined_scores, key=combined_scores.get))

            # Log probe metrics
            if writer is not None:
                for m in modalities:
                    writer.add_scalar(
                        f"probe/accuracy_{m}",
                        probe_results[m]['accuracy'],
                        global_step,
                    )
                if utilization_gap is not None:
                    writer.add_scalar("probe/utilization_gap", utilization_gap, global_step)

                # Log learning dynamics (dual-signal)
                learning_speeds = scheduler.dynamics.compute_learning_speed()
                for m in modalities:
                    writer.add_scalar(f"dynamics/learning_speed_{m}", learning_speeds[m], global_step)

                # Log continuous mode scales
                if scheduler.mode == "continuous":
                    for m in modalities:
                        writer.add_scalar(
                            f"continuous/scale_{m}",
                            scheduler.current_continuous_scales[m],
                            global_step,
                        )

        # ========== Logging ==========
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "mask": str({m: int(v) for m, v in update_mask.items()}),
        })

        if writer is not None and batch_idx % config["logging"]["log_interval"] == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/fusion_loss", loss_dict['fusion'].item(), global_step)

            for m in modalities:
                writer.add_scalar(f"train/grad_norm_{m}", grad_norms[m], global_step)
                writer.add_scalar(f"train/unimodal_loss_{m}", loss_dict[f'unimodal_{m}'].item(), global_step)
                writer.add_scalar(f"train/staleness_{m}", scheduler.current_tau[m], global_step)
                writer.add_scalar(f"train/update_{m}", int(update_mask[m]), global_step)

    # Compute epoch metrics
    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    probe_manager: ProbeManager = None,
):
    """Evaluate model on dataset."""
    model.eval()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    all_features = {m: [] for m in modalities}
    all_targets = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {m: batch[m].to(device) for m in modalities}
        targets = batch["label"].to(device)

        logits, unimodal_logits, features = model(inputs, return_features=True)
        loss = nn.functional.cross_entropy(logits, targets)

        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # Collect features for probe evaluation
        if probe_manager is not None:
            for m in modalities:
                all_features[m].append(features[m].cpu())
            all_targets.append(targets.cpu())

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    # Evaluate probes on full eval set
    if probe_manager is not None and all_targets:
        cat_features = {m: torch.cat(all_features[m], dim=0).to(device) for m in modalities}
        cat_targets = torch.cat(all_targets, dim=0).to(device)

        probe_results = probe_manager.evaluate_probes(cat_features, cat_targets)
        for m in modalities:
            metrics[f"probe_acc_{m}"] = probe_results[m]['accuracy']

        utilization_gap = probe_manager.compute_utilization_gap()
        if utilization_gap is not None:
            metrics["utilization_gap"] = utilization_gap

    return metrics


@torch.no_grad()
def evaluate_miles(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
):
    """
    Evaluate model for MILES mode.

    Returns both multimodal and unimodal accuracies needed for
    computing conditional utilization rates.
    """
    model.eval()

    loss_meter = AverageMeter("Loss")
    metric_tracker = MetricTracker(config["evaluation"]["metrics"])
    modalities = config["dataset"]["modalities"]

    # Track unimodal metrics
    unimodal_trackers = {m: MetricTracker(['accuracy']) for m in modalities}

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {m: batch[m].to(device) for m in modalities}
        targets = batch["label"].to(device)

        logits, unimodal_logits, features = model(inputs, return_features=True)
        loss = nn.functional.cross_entropy(logits, targets)

        loss_meter.update(loss.item(), targets.size(0))
        metric_tracker.update(logits, targets)

        # Track unimodal performance
        for m in modalities:
            unimodal_trackers[m].update(unimodal_logits[m], targets)

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.avg

    # Add unimodal accuracies
    for m in modalities:
        unimodal_metrics = unimodal_trackers[m].compute()
        metrics[f"unimodal_acc_{m}"] = unimodal_metrics['accuracy']

    # Compute utilization gap for logging
    acc_per_modality = {m: metrics[f"unimodal_acc_{m}"] for m in modalities}
    if len(modalities) == 2:
        m_a, m_b = list(modalities)
        if metrics['accuracy'] > 0:
            u_a = (metrics['accuracy'] - acc_per_modality[m_b]) / metrics['accuracy']
            u_b = (metrics['accuracy'] - acc_per_modality[m_a]) / metrics['accuracy']
            metrics['utilization_gap'] = abs(u_a - u_b)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ASGML model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "frequency", "staleness", "adaptive", "miles", "inforeg", "arl", "opm", "cggm", "mmpareto", "agm", "gblend"],
        help="Training mode",
    )
    parser.add_argument("--fixed-ratio", type=int, default=2, help="Fixed frequency ratio (for frequency mode)")
    parser.add_argument("--fixed-staleness", type=int, default=2, help="Fixed staleness τ (for staleness mode)")
    # MILES-specific arguments
    parser.add_argument("--miles-threshold", type=float, default=0.2,
                        help="MILES: τ threshold for triggering LR adjustment (δ > τ)")
    parser.add_argument("--miles-reduction", type=float, default=0.5,
                        help="MILES: μ factor to reduce dominant modality's LR")
    # InfoReg-specific arguments
    parser.add_argument("--inforeg-beta", type=float, default=0.9,
                        help="InfoReg: β regulation strength (α = β * Δ_m)")
    parser.add_argument("--inforeg-K", type=float, default=0.04,
                        help="InfoReg: K threshold for PLW detection on relative Tr(F) change")
    # OPM-specific arguments (Wei et al., TPAMI 2024)
    parser.add_argument("--opm-q-base", type=float, default=0.5,
                        help="OPM: base drop probability q_base (default: 0.5)")
    parser.add_argument("--opm-lam", type=float, default=0.5,
                        help="OPM: lambda controlling adjustment degree (default: 0.5)")
    parser.add_argument("--opm-p-exe", type=float, default=0.7,
                        help="OPM: probability of executing drop per step (default: 0.7)")
    parser.add_argument("--opm-warmup", type=int, default=5,
                        help="OPM: warm-up epochs before OPM activates (default: 5)")
    # ARL-specific arguments (Wei et al., ICCV 2025)
    parser.add_argument("--arl-temperature", type=float, default=8.0,
                        help="ARL: T temperature for asymmetric coefficient softmax (default: 8 for CREMA-D)")
    parser.add_argument("--arl-gamma", type=float, default=4.0,
                        help="ARL: γ weight for unimodal bias regularization losses (default: 4.0)")
    parser.add_argument("--arl-start", type=int, default=5,
                        help="ARL: epoch after which to start asymmetric modulation (default: 5)")
    # CGGM-specific arguments (Guo et al., NeurIPS 2024)
    parser.add_argument("--cggm-rou", type=float, default=1.3,
                        help="CGGM: ρ gradient scaling amplifier (default: 1.3)")
    parser.add_argument("--cggm-lamda", type=float, default=0.2,
                        help="CGGM: λ weight for gradient direction loss L_gm (default: 0.2)")
    parser.add_argument("--cggm-cls-lr", type=float, default=5e-4,
                        help="CGGM: classifier learning rate (default: 5e-4)")
    # AGM-specific arguments (Li et al., 2023)
    parser.add_argument("--agm-alpha", type=float, default=1.0,
                        help="AGM: α degree of gradient modulation (default: 1.0)")
    parser.add_argument("--agm-modulation-start", type=int, default=0,
                        help="AGM: epoch to start modulation (default: 0)")
    parser.add_argument("--agm-modulation-end", type=int, default=50,
                        help="AGM: epoch to stop modulation (default: 50)")
    # MMPareto-specific arguments (Wei & Hu, ICML 2024)
    parser.add_argument("--mmpareto-gamma", type=float, default=1.5,
                        help="MMPareto: γ gradient magnitude scaling (default: 1.5)")
    # Dataset overrides
    parser.add_argument("--fps", type=int, default=None,
                        help="Override dataset fps (frames per second extraction rate)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Override dataset num_frames (frames sampled per video)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training (AMP)")
    parser.add_argument(
        "--modality-lr",
        type=str,
        default=None,
        help="Per-modality LR scales as 'modality:scale,...' (e.g., 'audio:0.5,visual:1.0')"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Boost composition (plug-and-play, compose probe-guided boost with any method)
    parser.add_argument("--log-probe-stability", action="store_true",
                        help="Log per-iteration batch-raw, EMA-smoothed, and full-test probe "
                             "accuracy to probe_stability.json for the stability appendix.")
    parser.add_argument("--boost-compose", action="store_true",
                        help="Enable probe-guided boost composition with a baseline method "
                             "(--mode gblend | agm | mmpareto | cggm). Uses same setup as Table 1 Boost+OGM-GE.")
    # OGM-GE arguments (orthogonal to ASGML mode)
    parser.add_argument("--ogm-ge", action="store_true",
                        help="Enable OGM-GE gradient modulation (composable with any mode)")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="OGM-GE alpha coefficient (default: 0.8)")
    parser.add_argument("--modulation-start", type=int, default=0,
                        help="OGM-GE start epoch (default: 0)")
    parser.add_argument("--modulation-end", type=int, default=50,
                        help="OGM-GE end epoch (default: 50)")
    # ASGML hyperparameter overrides (for sweep)
    parser.add_argument("--tau-base", type=float, default=None,
                        help="Override ASGML tau_base (base staleness multiplier)")
    parser.add_argument("--tau-max", type=float, default=None,
                        help="Override ASGML tau_max (maximum staleness cap)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override ASGML beta (gradient vs loss signal weight)")
    parser.add_argument("--gamma-asgml", type=float, default=None,
                        help="Override ASGML gamma (unimodal regularization weight)")
    parser.add_argument("--lambda-comp", type=float, default=None,
                        help="Override ASGML lambda_comp (gradient compensation)")
    parser.add_argument("--threshold-delta", type=float, default=None,
                        help="Override ASGML threshold_delta (adaptation threshold)")
    parser.add_argument("--signal-source", type=str, default=None,
                        choices=["dual", "probe", "both"],
                        help="Override ASGML signal_source")
    parser.add_argument("--soft-mask-scale", type=float, default=None,
                        help="Override ASGML soft_mask_scale")
    parser.add_argument("--asgml-mode", type=str, default=None,
                        choices=["frequency", "staleness", "continuous"],
                        help="Override ASGML asgml_mode")
    parser.add_argument("--continuous-alpha", type=float, default=None,
                        help="Override continuous mode scaling strength (0-1)")
    parser.add_argument("--continuous-scale-max", type=float, default=None,
                        help="Override continuous mode maximum boost scale")
    parser.add_argument("--continuous-noise-sigma", type=float, default=None,
                        help="Override continuous mode Gaussian noise sigma")
    parser.add_argument("--continuous-eval-freq", type=int, default=None,
                        help="Override continuous mode probe eval frequency K (probe refresh every K batches)")
    parser.add_argument("--continuous-scale-ema", type=float, default=None,
                        help="Override continuous mode EMA coefficient for boost scale smoothing (default: 0.3)")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Override experiment name (default: auto-generated)")
    parser.add_argument("--single-layer-fusion", action="store_true",
                        help="Use single Linear(concat_dim→num_classes) fusion (OPM-style arch)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override dataset fps/num_frames from CLI
    if args.fps is not None:
        config["dataset"]["fps"] = args.fps
    if args.num_frames is not None:
        config["dataset"]["num_frames"] = args.num_frames

    # Configure ASGML based on mode
    if args.mode in ("baseline", "miles", "inforeg", "arl", "opm", "mmpareto", "agm", "gblend"):
        config["asgml"]["enabled"] = False
    else:
        config["asgml"]["enabled"] = True

    # Apply ASGML hyperparameter overrides from CLI
    asgml_overrides = {
        "tau_base": args.tau_base, "tau_max": args.tau_max,
        "beta": args.beta, "gamma": args.gamma_asgml,
        "lambda_comp": args.lambda_comp, "threshold_delta": args.threshold_delta,
        "signal_source": args.signal_source, "soft_mask_scale": args.soft_mask_scale,
        "asgml_mode": args.asgml_mode,
        "continuous_alpha": args.continuous_alpha,
        "continuous_scale_max": args.continuous_scale_max,
        "continuous_noise_sigma": args.continuous_noise_sigma,
        "continuous_eval_freq": args.continuous_eval_freq,
        "continuous_scale_ema": args.continuous_scale_ema,
    }
    for key, val in asgml_overrides.items():
        if val is not None:
            config["asgml"][key] = val

    # Override epochs if specified
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    # Override learning rate if specified
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    # Parse per-modality LR scales
    modality_lr_scales = None
    if args.modality_lr:
        modality_lr_scales = {}
        for item in args.modality_lr.split(","):
            modality, scale = item.split(":")
            modality_lr_scales[modality.strip()] = float(scale)

    # Set seed
    set_seed(args.seed)

    # Setup output directory
    if args.resume:
        # When resuming, use the same directory as the checkpoint
        output_dir = Path(args.resume).parent
        exp_name = output_dir.name
    else:
        if args.exp_name:
            exp_name = args.exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{config['dataset']['name']}_{args.mode}_{timestamp}"
        output_dir = Path(args.output_dir) / exp_name
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_copy = config.copy()
    config_copy["experiment"] = {
        "mode": args.mode,
        "fixed_ratio": args.fixed_ratio,
        "fixed_staleness": args.fixed_staleness,
        "seed": args.seed,
        "amp": args.amp,
        "modality_lr_scales": modality_lr_scales,
        "miles_threshold": args.miles_threshold,
        "miles_reduction": args.miles_reduction,
        "inforeg_beta": args.inforeg_beta,
        "inforeg_K": args.inforeg_K,
        "arl_temperature": args.arl_temperature,
        "arl_gamma": args.arl_gamma,
        "arl_start": args.arl_start,
        "opm_q_base": args.opm_q_base,
        "opm_lam": args.opm_lam,
        "opm_p_exe": args.opm_p_exe,
        "opm_warmup": args.opm_warmup,
        "agm_alpha": args.agm_alpha,
        "agm_modulation_start": args.agm_modulation_start,
        "agm_modulation_end": args.agm_modulation_end,
        "mmpareto_gamma": args.mmpareto_gamma,
    }
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config_copy, f)

    # Setup logger
    logger = setup_logger("asgml", log_file=str(output_dir / "train.log"))
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {config}")

    # Setup tensorboard
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = get_dataset(config, "train")
    test_dataset = get_dataset(config, "test")

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    modalities = config["dataset"]["modalities"]

    # For MLP backbone (pre-extracted features), detect actual feature dims from dataset
    if config["model"]["backbone"] == "mlp" and config["dataset"]["name"] == "mosei":
        config["dataset"]["text_dim"] = train_dataset.text_dim
        config["dataset"]["audio_dim"] = train_dataset.audio_dim
        config["dataset"]["visual_dim"] = train_dataset.visual_dim
        logger.info(
            f"MOSEI feature dims: text={train_dataset.text_dim}, "
            f"audio={train_dataset.audio_dim}, vision={train_dataset.visual_dim}"
        )
    elif config["model"]["backbone"] == "mlp" and config["dataset"]["name"] == "mosi":
        dims = train_dataset.get_dims()
        for k, v in dims.items():
            config["dataset"][f"{k}_dim"] = v
        logger.info(f"MOSI feature dims: {dims}")
    elif config["model"]["backbone"] == "mlp" and config["dataset"]["name"] in ("sarcasm", "twitter", "food101"):
        config["dataset"]["text_dim"] = train_dataset.text_dim
        config["dataset"]["image_dim"] = train_dataset.image_dim
        logger.info(
            f"{config['dataset']['name'].capitalize()} feature dims: "
            f"text={train_dataset.text_dim}, image={train_dataset.image_dim}"
        )

    # Create model
    # Build encoder config per modality
    encoder_config = {}
    backbone = config["model"]["backbone"]
    for m in modalities:
        enc_cfg = {
            "backbone": backbone,
            "pretrained": config["model"]["pretrained"],
        }
        # For MLP backbone (pre-extracted features), pass input dimensions
        if backbone == "mlp":
            dim_key_map = {"text": "text_dim", "audio": "audio_dim", "vision": "visual_dim", "visual": "visual_dim", "image": "image_dim"}
            dim_key = dim_key_map.get(m, f"{m}_dim")
            enc_cfg["input_dim"] = config["dataset"].get(dim_key, 300)
            enc_cfg["dropout"] = config["model"].get("dropout", 0.3)
        encoder_config[m] = enc_cfg

    model = MultimodalModel(
        modalities=modalities,
        num_classes=config["dataset"]["num_classes"],
        encoder_config=encoder_config,
        fusion_type=config["model"]["fusion_type"],
        feature_dim=config["model"]["feature_dim"],
        fusion_dim=config["model"]["fusion_dim"],
    ).to(device)

    # For ARL/OPM modes or --single-layer-fusion: override fusion layer to single
    # Linear(concat_dim → num_classes). ARL/OPM need to decompose classifier weights
    # per modality. --single-layer-fusion allows any mode to use this architecture
    # for fair cross-method comparison.
    if args.mode in ("arl", "opm") or args.single_layer_fusion:
        num_classes = config["dataset"]["num_classes"]
        feature_dims = [config["model"]["feature_dim"]] * len(modalities)
        model.fusion = ConcatFusion(feature_dims, num_classes)
        model.classifier = nn.Identity()
        model = model.to(device)
        logger.info(f"{args.mode.upper()}: Replaced fusion with direct Linear(concat_dim→num_classes), classifier=Identity")

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========== MILES Mode: Separate Training Loop ==========
    if args.mode == "miles":
        logger.info(f"MILES mode: τ={args.miles_threshold}, μ={args.miles_reduction}")

        # Create MILES optimizer with per-modality parameter groups
        optimizer = get_miles_optimizer(model, config)
        base_lr = config["training"]["lr"]

        # Resume from checkpoint if specified
        start_epoch = 1
        best_acc = 0.0
        if args.resume and Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")

        # MILES Training Loop
        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            # Train
            train_metrics = train_epoch_miles(
                model, train_loader, optimizer, device, epoch, config, logger, writer
            )

            # Evaluate (on validation/test set for MILES LR adjustment)
            test_metrics = evaluate_miles(model, test_loader, device, config)

            # ========== MILES LR Adjustment (epoch-level) ==========
            acc_per_modality = {m: test_metrics[f"unimodal_acc_{m}"] for m in modalities}
            miles_result = miles_adjust_learning_rates(
                optimizer=optimizer,
                acc_multimodal=test_metrics["accuracy"],
                acc_per_modality=acc_per_modality,
                modalities=modalities,
                base_lr=base_lr,
                threshold=args.miles_threshold,
                reduction=args.miles_reduction,
                logger=logger,
            )

            # Log
            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            log_str += f", δ={miles_result['delta']:.4f}"
            logger.info(log_str)

            # Tensorboard logging
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)
            if "utilization_gap" in test_metrics:
                writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
            writer.add_scalar("miles/delta", miles_result['delta'], epoch)
            for m in modalities:
                writer.add_scalar(f"test/unimodal_acc_{m}", test_metrics[f"unimodal_acc_{m}"], epoch)
                writer.add_scalar(f"miles/utilization_{m}", miles_result['utilization'][m], epoch)
                writer.add_scalar(f"miles/lr_{m}", miles_result['adjusted_lrs'][m], epoch)

            # Save checkpoint
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Periodic checkpoint
            if epoch % config["logging"]["save_interval"] == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Final summary
        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for MILES mode

    # ========== InfoReg Mode: Separate Training Loop ==========
    if args.mode == "inforeg":
        logger.info(f"InfoReg mode: β={args.inforeg_beta}, K={args.inforeg_K}")

        # InfoReg uses SGD with lr=0.002, StepLR(step=30, gamma=0.1) per paper
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            momentum=config["training"].get("momentum", 0.9),
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["training"].get("step_size", 30),
            gamma=config["training"].get("gamma", 0.1),
        )

        # Resume from checkpoint if specified
        start_epoch = 1
        best_acc = 0.0
        fisher_traces = []
        prev_epoch_weights = None
        if args.resume and Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            fisher_traces = checkpoint.get("fisher_traces", [])
            prev_epoch_weights = checkpoint.get("prev_epoch_weights", None)
            for _ in range(checkpoint["epoch"]):
                lr_scheduler.step()
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")

        # InfoReg Training Loop
        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            # Save current epoch weights for regulation term
            prev_epoch_weights = {}
            for m in modalities:
                prev_epoch_weights[m] = {
                    name: param.data.clone().cpu()
                    for name, param in model.named_parameters()
                    if f"encoders.{m}" in name
                }

            # Train
            train_metrics = train_epoch_inforeg(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                prev_epoch_weights=prev_epoch_weights,
                fisher_traces=fisher_traces,
                beta=args.inforeg_beta,
                K=args.inforeg_K,
            )

            # Update Fisher trace history
            if "fisher_traces" in train_metrics:
                fisher_traces.append(train_metrics["fisher_traces"])

            # Step LR scheduler
            lr_scheduler.step()

            # Evaluate
            test_metrics = evaluate_miles(model, test_loader, device, config)

            # Log
            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            for m in modalities:
                log_str += f", {m}_acc={test_metrics.get(f'unimodal_acc_{m}', 0):.4f}"
            logger.info(log_str)

            # Tensorboard logging
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)
            if "utilization_gap" in test_metrics:
                writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
            for m in modalities:
                writer.add_scalar(f"test/unimodal_acc_{m}", test_metrics.get(f"unimodal_acc_{m}", 0), epoch)

            # Save checkpoint
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "fisher_traces": fisher_traces,
                    "prev_epoch_weights": prev_epoch_weights,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Periodic checkpoint
            if epoch % config["logging"]["save_interval"] == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "fisher_traces": fisher_traces,
                    "prev_epoch_weights": prev_epoch_weights,
                    "config": config_copy,
                }
                torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Final summary
        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for InfoReg mode

    # ========== ARL Mode: Separate Training Loop ==========
    if args.mode == "arl":
        logger.info(f"ARL mode: T={args.arl_temperature}, γ={args.arl_gamma}, start={args.arl_start}")

        # ARL uses SGD with MultiStepLR, matching reference code
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            momentum=config["training"].get("momentum", 0.9),
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        lr_scheduler = get_scheduler(optimizer, config)

        # Resume from checkpoint if specified
        start_epoch = 1
        best_acc = 0.0
        if args.resume and Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            if lr_scheduler is not None:
                for _ in range(checkpoint["epoch"]):
                    lr_scheduler.step()
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")

        # ARL Training Loop
        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            # Train
            train_metrics = train_epoch_arl(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                temperature=args.arl_temperature,
                gamma_arl=args.arl_gamma,
                arl_start=args.arl_start,
            )

            # Step LR scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Evaluate
            test_metrics = evaluate_miles(model, test_loader, device, config)

            # Log
            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            # Tensorboard logging
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)
            if "utilization_gap" in test_metrics:
                writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
            for m in modalities:
                writer.add_scalar(f"test/unimodal_acc_{m}", test_metrics.get(f"unimodal_acc_{m}", 0), epoch)

            # Save checkpoint
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Periodic checkpoint
            if epoch % config["logging"]["save_interval"] == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Final summary
        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for ARL mode

    # ========== Boost composition setup for baseline methods ==========
    # When --boost-compose is set with mode in {cggm, mmpareto, agm, gblend},
    # instantiate a probe manager + continuous-mode scheduler matching the
    # Table 1 Boost+OGM-GE setup (alpha/s_max/K/mu from config). The setup is
    # plug-and-play: the hook multiplies encoder gradients by bar s_m just
    # before optimizer.step() inside each train_epoch_X.
    if args.boost_compose and args.mode in ("cggm", "mmpareto", "agm", "gblend"):
        probe_manager = ProbeManager(
            modalities=modalities,
            feature_dim=config["model"]["feature_dim"],
            num_classes=config["dataset"]["num_classes"],
            probe_type=config["asgml"].get("probe_type", "linear"),
            probe_lr=config["asgml"].get("probe_lr", 1e-3),
            device=device,
            ema_alpha=config["asgml"].get("probe_ema_alpha", 0.1),
        )
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="continuous",
            adaptation="adaptive",
            tau_base=1.0,
            tau_min=1.0,
            tau_max=1.0,
            threshold_delta=config["asgml"].get("threshold_delta", 0.1),
            beta=config["asgml"].get("beta", 0.5),
            lambda_comp=config["asgml"].get("lambda_comp", 0.1),
            continuous_scale_ema=config["asgml"].get("continuous_scale_ema", 0.3),
        )
        logger.info(
            f"Boost composition enabled with {args.mode.upper()}: "
            f"alpha={config['asgml'].get('continuous_alpha', 0.5)}, "
            f"s_max={config['asgml'].get('continuous_scale_max', 2.0)}, "
            f"K={config['asgml'].get('continuous_eval_freq', 20)}, "
            f"mu={config['asgml'].get('continuous_scale_ema', 0.3)}"
        )

    # ========== OPM Mode: Separate Training Loop ==========
    if args.mode == "cggm":
        logger.info(
            f"CGGM mode: ρ={args.cggm_rou}, λ={args.cggm_lamda}, "
            f"cls_lr={args.cggm_cls_lr}"
        )

        # CGGM uses SGD like other methods
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            momentum=config["training"].get("momentum", 0.9),
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        lr_scheduler = get_scheduler(optimizer, config)

        # Create CGGM module
        cggm = CGGMModule(
            modalities=modalities,
            feature_dim=config["model"]["feature_dim"],
            num_classes=config["dataset"]["num_classes"],
            rou=args.cggm_rou,
            lamda=args.cggm_lamda,
            cls_lr=args.cggm_cls_lr,
            device=device,
        )

        # OGM-GE config if combined
        ogm_ge_config = None
        if args.ogm_ge:
            logger.info(f"CGGM+OGM-GE combined: alpha={args.alpha}")
            ogm_ge_config = {
                "enabled": True,
                "alpha": args.alpha,
                "modulation_start": args.modulation_start,
                "modulation_end": args.modulation_end,
            }

        criterion = nn.CrossEntropyLoss()

        start_epoch = 1
        best_acc = 0.0

        # CGGM Training Loop
        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_loss, train_metrics_dict = train_epoch_cggm(
                model, train_loader, optimizer, criterion, cggm,
                device, epoch, config, logger,
                writer=writer,
                ogm_ge_config=ogm_ge_config,
                probe_manager=probe_manager if args.boost_compose else None,
                scheduler=asgml_scheduler if args.boost_compose else None,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            # Evaluate
            if args.boost_compose:
                test_metrics = evaluate(model, test_loader, device, config, probe_manager)
            else:
                test_metrics = evaluate_miles(model, test_loader, device, config)

            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train Acc={train_metrics_dict.get('accuracy', 0):.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            if epoch % config["logging"]["save_interval"] == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for CGGM mode

    if args.mode == "opm":
        logger.info(
            f"OPM mode: q_base={args.opm_q_base}, λ={args.opm_lam}, "
            f"p_exe={args.opm_p_exe}, warmup={args.opm_warmup}"
        )
        if args.ogm_ge:
            logger.info(f"OPM+OGM combined: alpha={args.alpha}")

        # OPM uses SGD (matching reference code for CREMA-D)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config["training"]["lr"],
            momentum=config["training"].get("momentum", 0.9),
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
        lr_scheduler = get_scheduler(optimizer, config)

        # OGM-GE config if combined
        ogm_ge_config = None
        if args.ogm_ge:
            ogm_ge_config = {
                "enabled": True,
                "alpha": args.alpha,
                "modulation_start": args.modulation_start,
                "modulation_end": args.modulation_end,
            }

        # Resume from checkpoint if specified
        start_epoch = 1
        best_acc = 0.0
        if args.resume and Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            if lr_scheduler is not None:
                for _ in range(checkpoint["epoch"]):
                    lr_scheduler.step()
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")

        # OPM Training Loop
        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            # Train
            train_metrics = train_epoch_opm(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                q_base=args.opm_q_base,
                lam=args.opm_lam,
                p_exe=args.opm_p_exe,
                warmup_epochs=args.opm_warmup,
                ogm_ge_config=ogm_ge_config,
            )

            # Step LR scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # Evaluate
            test_metrics = evaluate_miles(model, test_loader, device, config)

            # Log
            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train Acc={train_metrics['accuracy']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            # Tensorboard logging
            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)
            if "utilization_gap" in test_metrics:
                writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
            for m in modalities:
                writer.add_scalar(f"test/unimodal_acc_{m}", test_metrics.get(f"unimodal_acc_{m}", 0), epoch)

            # Save checkpoint
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Periodic checkpoint
            if epoch % config["logging"]["save_interval"] == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

        # Final summary
        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for OPM mode

    # ========== MMPareto Mode (Wei & Hu, ICML 2024) ==========
    if args.mode == "mmpareto":
        logger.info(f"MMPareto mode: γ={args.mmpareto_gamma}")

        optimizer = get_optimizer(model, config)
        lr_scheduler = get_scheduler(optimizer, config)

        start_epoch = 1
        best_acc = 0.0

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_loss, train_metrics_dict = train_epoch_mmpareto(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                gamma=args.mmpareto_gamma,
                probe_manager=probe_manager if args.boost_compose else None,
                scheduler=asgml_scheduler if args.boost_compose else None,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            if args.boost_compose:
                test_metrics = evaluate(model, test_loader, device, config, probe_manager)
            else:
                test_metrics = evaluate_miles(model, test_loader, device, config)

            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train Acc={train_metrics_dict.get('accuracy', 0):.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            if epoch % config["logging"]["save_interval"] == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }, output_dir / f"checkpoint_epoch{epoch}.pt")

        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for MMPareto mode

    # ========== AGM Mode (Li et al., 2023) ==========
    if args.mode == "agm":
        logger.info(
            f"AGM mode: α={args.agm_alpha}, "
            f"modulation=[{args.agm_modulation_start}, {args.agm_modulation_end}]"
        )

        optimizer = get_optimizer(model, config)
        lr_scheduler = get_scheduler(optimizer, config)

        start_epoch = 1
        best_acc = 0.0
        running_scores = {m: 0.0 for m in modalities}

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_loss, train_metrics_dict, running_scores = train_epoch_agm(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                alpha=args.agm_alpha,
                modulation_start=args.agm_modulation_start,
                modulation_end=args.agm_modulation_end,
                running_scores=running_scores,
                probe_manager=probe_manager if args.boost_compose else None,
                scheduler=asgml_scheduler if args.boost_compose else None,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            if args.boost_compose:
                test_metrics = evaluate(model, test_loader, device, config, probe_manager)
            else:
                test_metrics = evaluate_miles(model, test_loader, device, config)

            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train Acc={train_metrics_dict.get('accuracy', 0):.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            if epoch % config["logging"]["save_interval"] == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }, output_dir / f"checkpoint_epoch{epoch}.pt")

        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for AGM mode

    # ========== G-Blend Mode (Wang et al., CVPR 2020) ==========
    if args.mode == "gblend":
        logger.info("G-Blend mode: overfitting-to-generalization ratio weighting")

        optimizer = get_optimizer(model, config)
        lr_scheduler = get_scheduler(optimizer, config)

        start_epoch = 1
        best_acc = 0.0
        blend_weights = None  # Start with uniform weights

        for epoch in range(start_epoch, config["training"]["epochs"] + 1):
            train_loss, train_metrics_dict, train_losses = train_epoch_gblend(
                model, train_loader, optimizer, device, epoch, config, logger,
                writer=writer,
                blend_weights=blend_weights,
                probe_manager=probe_manager if args.boost_compose else None,
                scheduler=asgml_scheduler if args.boost_compose else None,
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            if args.boost_compose:
                test_metrics = evaluate(model, test_loader, device, config, probe_manager)
            else:
                test_metrics = evaluate_miles(model, test_loader, device, config)

            # Compute blend weights for next epoch from OG ratios
            blend_weights = compute_gblend_weights(
                model, train_losses, test_loader, device, modalities, logger,
            )

            log_str = (
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Train Acc={train_metrics_dict.get('accuracy', 0):.4f}, "
                f"Test Acc={test_metrics['accuracy']:.4f}, "
                f"Test F1={test_metrics['f1_macro']:.4f}"
            )
            if "utilization_gap" in test_metrics:
                log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
            logger.info(log_str)

            writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
            writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
            writer.add_scalar("test/loss", test_metrics["loss"], epoch)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }
                torch.save(best_checkpoint, output_dir / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            if epoch % config["logging"]["save_interval"] == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config_copy,
                }, output_dir / f"checkpoint_epoch{epoch}.pt")

        logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
        writer.close()
        return  # Exit early for G-Blend mode

    # ========== Standard ASGML Modes ==========
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config, modality_lr_scales)
    lr_scheduler = get_scheduler(optimizer, config)

    # Create AMP scaler if enabled
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled")
    if modality_lr_scales:
        logger.info(f"Per-modality LR scales: {modality_lr_scales}")

    # Create ASGML components
    loss_fn = ASGMLLoss(
        modalities=modalities,
        gamma=config["asgml"]["gamma"],
    )

    # Configure ASGML scheduler based on mode
    if args.mode == "baseline":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="frequency",
            adaptation="fixed",
            fixed_ratio=1,  # Update every step = baseline
        )
    elif args.mode == "frequency":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="frequency",
            adaptation="fixed",
            fixed_ratio=args.fixed_ratio,
        )
    elif args.mode == "staleness":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="staleness",
            adaptation="fixed",
            fixed_staleness=args.fixed_staleness,
        )
    elif args.mode == "adaptive":
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode=config["asgml"].get("asgml_mode", "frequency"),
            adaptation="adaptive",
            tau_base=config["asgml"]["tau_base"],
            tau_min=config["asgml"]["tau_min"],
            tau_max=config["asgml"]["tau_max"],
            threshold_delta=config["asgml"].get("threshold_delta", 0.1),
            beta=config["asgml"]["beta"],
            lambda_comp=config["asgml"]["lambda_comp"],
            max_staleness_ratio=config["asgml"].get("max_staleness_ratio", 3.0),
            continuous_scale_ema=config["asgml"].get("continuous_scale_ema", 0.3),
        )
    else:
        # Baseline method modes (gblend, agm, mmpareto, cggm, ...).
        # If --boost-compose is set, use a continuous-mode scheduler matching
        # the Table 1 Boost+OGM-GE setup (alpha/s_max/K/mu from config).
        asgml_scheduler = ASGMLScheduler(
            modalities=modalities,
            mode="continuous",
            adaptation="adaptive",
            tau_base=1.0,
            tau_min=1.0,
            tau_max=1.0,
            threshold_delta=config["asgml"].get("threshold_delta", 0.1),
            beta=config["asgml"].get("beta", 0.5),
            lambda_comp=config["asgml"].get("lambda_comp", 0.1),
            continuous_scale_ema=config["asgml"].get("continuous_scale_ema", 0.3),
        )

    # Create probe manager
    probe_manager = ProbeManager(
        modalities=modalities,
        feature_dim=config["model"]["feature_dim"],
        num_classes=config["dataset"]["num_classes"],
        probe_type=config["asgml"].get("probe_type", "linear"),
        probe_lr=config["asgml"].get("probe_lr", 1e-3),
        device=device,
        ema_alpha=config["asgml"].get("probe_ema_alpha", 0.1),  # EMA smoothing for stable adaptive control
    )

    # Set initial dominant modality (will be updated by probes)
    # For CREMA-D, audio is typically dominant
    if "audio" in modalities:
        asgml_scheduler.set_dominant_modality("audio")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        if Path(args.resume).exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_acc", 0.0)
            if "probe_manager_state" in checkpoint:
                probe_manager.load_state_dict(checkpoint["probe_manager_state"])
            if scaler is not None and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            # Update LR scheduler to correct epoch
            if lr_scheduler is not None:
                for _ in range(checkpoint["epoch"]):
                    lr_scheduler.step()
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best_acc={best_acc:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {args.resume}, starting from scratch")

    # OGM-GE config (orthogonal to ASGML mode)
    ogm_ge_config = None
    if args.ogm_ge:
        ogm_ge_config = {
            "enabled": True,
            "alpha": args.alpha,
            "modulation_start": args.modulation_start,
            "modulation_end": args.modulation_end,
        }
        logger.info(
            f"OGM-GE enabled: alpha={args.alpha}, "
            f"modulation=[{args.modulation_start}, {args.modulation_end}]"
        )

    # Probe-stability instrumentation (appendix B.6 validation experiment).
    # When --log-probe-stability is set, record batch-raw / EMA-smoothed /
    # full-test probe accuracy at each probe-eval checkpoint for later plotting.
    probe_stability_log = [] if args.log_probe_stability else None
    probe_stability_loader = test_loader if args.log_probe_stability else None

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        # Reset ASGML state at start of epoch (optional - can also persist)
        # asgml_scheduler.reset()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, asgml_scheduler,
            probe_manager, device, epoch, config, logger, writer,
            scaler=scaler, use_amp=use_amp, ogm_ge_config=ogm_ge_config,
            probe_stability_log=probe_stability_log,
            probe_stability_loader=probe_stability_loader,
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, device, config, probe_manager)

        # Update LR scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Log
        log_str = (
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.4f}, "
            f"Test Acc={test_metrics['accuracy']:.4f}, "
            f"Test F1={test_metrics['f1_macro']:.4f}"
        )
        if "utilization_gap" in test_metrics:
            log_str += f", Util Gap={test_metrics['utilization_gap']:.4f}"
        logger.info(log_str)

        # Tensorboard logging
        writer.add_scalar("test/accuracy", test_metrics["accuracy"], epoch)
        writer.add_scalar("test/f1_macro", test_metrics["f1_macro"], epoch)
        writer.add_scalar("test/loss", test_metrics["loss"], epoch)
        if "utilization_gap" in test_metrics:
            writer.add_scalar("test/utilization_gap", test_metrics["utilization_gap"], epoch)
        for m in modalities:
            if f"probe_acc_{m}" in test_metrics:
                writer.add_scalar(f"test/probe_acc_{m}", test_metrics[f"probe_acc_{m}"], epoch)

        # Save checkpoint
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_manager_state": probe_manager.state_dict(),
                "best_acc": best_acc,
                "config": config_copy,
            }
            if scaler is not None:
                best_checkpoint["scaler_state_dict"] = scaler.state_dict()
            torch.save(best_checkpoint, output_dir / "best_model.pt")
            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

        # Periodic checkpoint
        if epoch % config["logging"]["save_interval"] == 0:
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "probe_manager_state": probe_manager.state_dict(),
                "best_acc": best_acc,
                "config": config_copy,
            }
            if scaler is not None:
                checkpoint_data["scaler_state_dict"] = scaler.state_dict()
            torch.save(checkpoint_data, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Probe-stability log dump (appendix B.6 validation)
    if probe_stability_log is not None and len(probe_stability_log) > 0:
        import json as _json
        stab_path = output_dir / "probe_stability.json"
        with open(stab_path, "w") as _f:
            _json.dump(probe_stability_log, _f, indent=2)
        logger.info(f"Probe stability log written to {stab_path} ({len(probe_stability_log)} checkpoints)")

    # Final summary
    logger.info(f"Training complete. Best accuracy: {best_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
