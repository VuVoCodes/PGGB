#!/usr/bin/env python3
"""
BraTS 2021 training script with ASGML probe-guided boosting.

Adapts CGGM's DeepLab multi-input architecture for fair comparison.
Modes: baseline, ogm_ge, asgml_boost, cggm

Usage:
    python scripts/train_brats.py --mode baseline --seed 42
    python scripts/train_brats.py --mode asgml_boost --seed 42
    python scripts/train_brats.py --mode cggm --seed 42
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root and CGGM to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'Papers', 'CGGM'))

from Papers.CGGM.datasets.BratsDataset import BraTSData
from Papers.CGGM.models.segmodel import DeepLabMultiInput, SegClassifier
from Papers.CGGM.src.eval_metrics import cosine_scheduler, cal_cos
from src.models.probes import ProbeManager


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation."""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()


class SegLoss(nn.Module):
    """Combined Dice + CE loss matching CGGM."""
    def __init__(self, n_classes=4, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(n_classes)

    def forward(self, pred, target):
        return self.ce(pred, target.long()) + self.dice(pred, target)


def cal_dice(pred, target):
    """Calculate Dice scores for WT, TC, ET."""
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    # WT: labels 1,2,3 (whole tumor)
    wt_pred = (pred > 0).astype(np.float32)
    wt_true = (target > 0).astype(np.float32)
    wt_dice = (2 * (wt_pred * wt_true).sum() + 1e-5) / (wt_pred.sum() + wt_true.sum() + 1e-5)

    # TC: labels 1,3 (tumor core, note label 4->3 in dataset)
    tc_pred = ((pred == 1) | (pred == 3)).astype(np.float32)
    tc_true = ((target == 1) | (target == 3)).astype(np.float32)
    tc_dice = (2 * (tc_pred * tc_true).sum() + 1e-5) / (tc_pred.sum() + tc_true.sum() + 1e-5)

    # ET: label 3 (enhancing tumor, originally 4)
    et_pred = (pred == 3).astype(np.float32)
    et_true = (target == 3).astype(np.float32)
    et_dice = (2 * (et_pred * et_true).sum() + 1e-5) / (et_pred.sum() + et_true.sum() + 1e-5)

    return wt_dice, tc_dice, et_dice


class BraTSProbeAdapter:
    """Adapts the main ProbeManager for BraTS spatial features.

    BraTS ASPP features are spatial (B, 256, H, W). This adapter pools them
    to (B, 256) before passing to ProbeManager, and converts segmentation
    labels to binary (tumor present/absent) for probe training.

    Uses the same ProbeManager as all other datasets for consistency:
    - Split-batch protocol (train on first half, eval on second half)
    - EMA-smoothed probe accuracies
    - Separate optimizers (no encoder backprop)
    """

    def __init__(self, feature_dim, num_classes, device, eval_freq=20, scale_ema_mu=0.3):
        self.modalities = ['flair', 't1ce', 't1', 't2']
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eval_freq = eval_freq  # K: probe refresh every K batches
        self.scale_ema_mu = scale_ema_mu  # mu for scale EMA smoothing

        # Persistent EMA-smoothed boost scales (applied every batch)
        self.stored_scales = {m: 1.0 for m in self.modalities}

        # Binary classification probe: tumor present vs absent
        self.probe_mgr = ProbeManager(
            modalities=self.modalities,
            feature_dim=feature_dim,
            num_classes=2,
            probe_type='linear',
            probe_lr=1e-3,
            device=device,
            ema_alpha=0.1,
        )

    def train_and_eval(self, features, labels):
        """Train and evaluate probes with split-batch on pooled features.

        Args:
            features: list of 4 tensors, each (B, 256, H, W) from ASPP
            labels: segmentation labels (B, H, W)

        Returns:
            dict of per-modality probe accuracies (evaluated on held-out half)
        """
        batch_size = features[0].shape[0]
        split = batch_size // 2
        if split < 2:
            return {m: 0.5 for m in self.modalities}

        # Pool spatial features to (B, 256)
        pooled = {}
        for i, m in enumerate(self.modalities):
            pooled[m] = self.pool(features[i].detach()).flatten(1)  # (B, 256)

        # Convert segmentation labels to binary (tumor present/absent)
        spatial_size = features[0].shape[2:]
        labels_down = F.interpolate(
            labels.float().unsqueeze(1), size=spatial_size, mode='nearest'
        ).squeeze(1)
        global_label = (labels_down > 0).float().mean(dim=(1, 2))
        binary_labels = (global_label > 0.1).long()

        # Split batch: train on first half
        train_features = {m: pooled[m][:split] for m in self.modalities}
        train_targets = binary_labels[:split]
        self.probe_mgr.train_probes(train_features, train_targets, num_steps=1)

        # Evaluate on second half
        eval_features = {m: pooled[m][split:] for m in self.modalities}
        eval_targets = binary_labels[split:]
        results = self.probe_mgr.evaluate_probes(eval_features, eval_targets)

        return {m: results[m]['accuracy'] for m in self.modalities}

    def update_scales(self, accs, alpha=0.5):
        """Compute raw boost scales and apply EMA smoothing (mu=0.3).

        Called every K batches when probes are refreshed. Updates stored_scales
        which are applied to gradients on every batch.
        """
        # Use EMA-smoothed accuracies for stable scaling
        ema_accs = self.probe_mgr.accuracy_ema
        use_accs = ema_accs if all(v is not None for v in ema_accs.values()) else accs

        min_acc = min(use_accs.values())
        max_acc = max(use_accs.values())
        gap = max_acc - min_acc + 1e-8

        for m in self.modalities:
            rel_weakness = 1.0 - (use_accs[m] - min_acc) / gap
            raw_scale = min(1.0 + alpha * rel_weakness, 2.0)
            # EMA smoothing: bar_s = mu * s + (1-mu) * bar_s (Eq 7 in paper)
            self.stored_scales[m] = (
                self.scale_ema_mu * raw_scale +
                (1 - self.scale_ema_mu) * self.stored_scales[m]
            )

    def get_stored_scales(self):
        """Return current EMA-smoothed boost scales (applied every batch)."""
        return self.stored_scales.copy()

    def get_utilization_gap(self):
        """Get current utilization gap from ProbeManager."""
        return self.probe_mgr.compute_utilization_gap(use_ema=True)


def apply_ogm_ge_brats(model, epoch, alpha=0.8, start=0, end=50):
    """Apply OGM-GE to BraTS 4-modality model using gradient magnitude ratios.

    Computes gradient norms per backbone, identifies dominant modalities
    (norm > mean), and scales their gradients down with Gaussian noise.
    """
    if not (start <= epoch <= end):
        return {}

    backbone_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
    modality_names = ['flair', 't1ce', 't1', 't2']
    tanh = torch.nn.Tanh()
    relu = torch.nn.ReLU()

    # Compute per-backbone gradient norms
    grad_norms = {}
    for bname, mname in zip(backbone_names, modality_names):
        total_norm = 0.0
        count = 0
        for name, param in model.named_parameters():
            if bname in name and param.grad is not None and len(param.grad.shape) >= 2:
                total_norm += param.grad.data.norm().item() ** 2
                count += 1
        grad_norms[mname] = (total_norm ** 0.5) if count > 0 else 0.0

    mean_norm = sum(grad_norms.values()) / len(grad_norms)
    if mean_norm < 1e-8:
        return {m: 1.0 for m in modality_names}

    # Compute coefficients: dominant modalities (norm > mean) get scaled down
    coeffs = {}
    for mname in modality_names:
        ratio = grad_norms[mname] / (mean_norm + 1e-8)
        if ratio > 1.0:
            coeffs[mname] = 1 - tanh(alpha * relu(torch.tensor(ratio))).item()
        else:
            coeffs[mname] = 1.0

    # Apply modulation
    for bname, mname in zip(backbone_names, modality_names):
        if coeffs[mname] < 1.0:
            for name, param in model.named_parameters():
                if bname in name and param.grad is not None and len(param.grad.shape) >= 2:
                    param.grad.data = param.grad.data * coeffs[mname] + \
                        torch.zeros_like(param.grad).normal_(0, param.grad.std().item() + 1e-8)

    return coeffs


def train_epoch(model, loader, optimizer, criterion, scheduler, epoch, device, args,
                classifier=None, cls_optimizer=None, probe_mgr=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    acc1 = [0] * 4
    l_gm = None

    for i_batch, batch in enumerate(loader):
        it = len(loader) * (epoch - 1) + i_batch
        optimizer.param_groups[0]['lr'] = scheduler[it]

        flair, t1ce, t1, t2, labels = [x.cuda() for x in batch]

        model.zero_grad()
        preds, hf, lf = model(flair, t1ce, t1, t2)

        # Main loss
        raw_loss = criterion(preds, labels)

        # CGGM: add L_gm from previous iteration
        if args.mode == 'cggm' and l_gm is not None:
            raw_loss = raw_loss + args.cggm_lamda * l_gm

        raw_loss.backward()

        # ========== Mode-specific gradient modulation ==========

        if args.mode in ('ogm_ge', 'boost_ogm_ge'):
            apply_ogm_ge_brats(model, epoch, alpha=args.ogm_alpha,
                               start=args.ogm_start, end=args.ogm_end)

        if args.mode in ('boost_ogm_ge', 'asgml_boost') and probe_mgr is not None:
            # Every K batches: refresh probes and update EMA-smoothed scales
            if (i_batch + 1) % probe_mgr.eval_freq == 0:
                accs = probe_mgr.train_and_eval(hf, labels)
                probe_mgr.update_scales(accs, alpha=args.boost_alpha)

                # Log probe metrics
                gap = probe_mgr.get_utilization_gap() or 0.0
                scales = probe_mgr.get_stored_scales()
                acc_str = ', '.join(f'{m}={accs[m]:.3f}' for m in ['flair', 't1ce', 't1', 't2'])
                scale_str = ', '.join(f'{m}={scales[m]:.3f}' for m in ['flair', 't1ce', 't1', 't2'])
                print(f'  [batch {i_batch}] probe_acc: {acc_str} | gap={gap:.3f} | scales: {scale_str}', flush=True)

            # Every batch: apply stored EMA-smoothed scales to encoder gradients
            scales = probe_mgr.get_stored_scales()
            backbone_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
            modality_names = ['flair', 't1ce', 't1', 't2']
            for bname, mname in zip(backbone_names, modality_names):
                for name, param in model.named_parameters():
                    if bname in name and param.grad is not None:
                        param.grad *= scales[mname]

        elif args.mode == 'cggm' and classifier is not None:
            cls_optimizer.zero_grad()
            cls_res = classifier(hf, lf)

            # Get fusion gradient
            fusion_grad = None
            for name, para in model.named_parameters():
                if 'decoder.last_conv.7.weight' in name:
                    fusion_grad = para
                    break

            cls_loss = sum(criterion(cls_res[i], labels) for i in range(4))
            cls_loss.backward()

            # Get classifier gradients
            cls_grad = []
            for name, para in classifier.named_parameters():
                if 'last_conv.7.weight' in name:
                    cls_grad.append(para)

            # Cosine similarity
            if fusion_grad is not None and len(cls_grad) == 4:
                llist = cal_cos(cls_grad, fusion_grad)

                # Accuracy-based coefficients
                acc2 = []
                for r in cls_res:
                    pred_cls = torch.argmax(r, dim=1)
                    acc = (pred_cls == labels).float().mean().item()
                    acc2.append(acc)

                diff = [acc2[i] - acc1[i] for i in range(4)]
                diff_sum = sum(diff) + 1e-8
                coeff = [(diff_sum - d) / diff_sum for d in diff]
                acc1 = acc2

                l_gm_val = sum(abs(coeff[i]) - coeff[i] * llist[i] for i in range(4)) / 4
                l_gm = l_gm_val

                # Scale encoder gradients
                backbone_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4']
                for i, bname in enumerate(backbone_names):
                    for name, params in model.named_parameters():
                        if bname in name and params.grad is not None:
                            params.grad *= (coeff[i] * args.cggm_rou)

            cls_optimizer.step()

        optimizer.step()
        total_loss += raw_loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss and dice scores."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            flair, t1ce, t1, t2, labels = [x.cuda() for x in batch]
            preds, _, _ = model(flair, t1ce, t1, t2)
            total_loss += criterion(preds, labels).item()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    wt, tc, et = cal_dice(all_preds, all_labels)
    avg_dice = (wt + tc + et) / 3

    return total_loss / len(loader), wt, tc, et, avg_dice


def main():
    parser = argparse.ArgumentParser(description='BraTS training with ASGML')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'asgml_boost', 'cggm', 'ogm_ge', 'boost_ogm_ge'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--base-lr', type=float, default=0.01)
    parser.add_argument('--data-dir', type=str, default='data/BraTS/h5_data')
    parser.add_argument('--output-dir', type=str, default='outputs/sweep_brats')
    parser.add_argument('--exp-name', type=str, default=None)
    # ASGML
    parser.add_argument('--boost-alpha', type=float, default=0.5)
    # OGM-GE
    parser.add_argument('--ogm-alpha', type=float, default=0.8, help='OGM-GE alpha coefficient')
    parser.add_argument('--ogm-start', type=int, default=0, help='OGM-GE start epoch')
    parser.add_argument('--ogm-end', type=int, default=50, help='OGM-GE end epoch')
    # CGGM
    parser.add_argument('--cggm-rou', type=float, default=1.3)
    parser.add_argument('--cggm-lamda', type=float, default=0.2)
    parser.add_argument('--cggm-cls-lr', type=float, default=0.001)
    args = parser.parse_args()

    set_seed(args.seed)

    exp_name = args.exp_name or f'brats_{args.mode}_seed{args.seed}'
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / 'train.log'
    log_fh = open(log_file, 'w')

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + '\n')
        log_fh.flush()

    log(f'Mode: {args.mode}, Seed: {args.seed}')
    log(f'Data: {args.data_dir}')

    # Data
    train_data = BraTSData(args.data_dir, 'train')
    valid_data = BraTSData(args.data_dir, 'valid')
    test_data = BraTSData(args.data_dir, 'test')

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

    log(f'Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}')

    # Model
    device = torch.device('cuda')
    model = DeepLabMultiInput(output_stride=16, num_classes=4).to(device)
    log(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    # Loss
    criterion = SegLoss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25]).cuda())

    # Optimizer + scheduler
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=1e-4, momentum=0.9)
    warmup_ep = min(5, args.epochs // 2)
    scheduler = cosine_scheduler(
        base_value=args.base_lr, final_value=1e-6,
        epochs=args.epochs, niter_per_ep=len(train_loader),
        warmup_epochs=warmup_ep, start_warmup_value=1e-4
    )

    # Mode-specific setup
    classifier, cls_optimizer, probe_mgr = None, None, None

    if args.mode == 'cggm':
        classifier = SegClassifier(num_classes=4).to(device)
        cls_optimizer = optim.SGD(classifier.parameters(), lr=args.cggm_cls_lr,
                                  weight_decay=1e-4, momentum=0.9)
        log(f'CGGM: rou={args.cggm_rou}, lamda={args.cggm_lamda}')

    elif args.mode == 'ogm_ge':
        log(f'OGM-GE: alpha={args.ogm_alpha}, epochs=[{args.ogm_start}, {args.ogm_end}]')

    elif args.mode == 'boost_ogm_ge':
        probe_mgr = BraTSProbeAdapter(256, 2, device)
        log(f'Boost+OGM-GE: boost_alpha={args.boost_alpha}, ogm_alpha={args.ogm_alpha}, '
            f'ogm_epochs=[{args.ogm_start}, {args.ogm_end}]')

    elif args.mode == 'asgml_boost':
        probe_mgr = BraTSProbeAdapter(256, 2, device)
        log(f'ASGML boost: alpha={args.boost_alpha}')

    # Training
    best_dice = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, device, args,
            classifier=classifier, cls_optimizer=cls_optimizer, probe_mgr=probe_mgr,
        )
        val_loss, wt, tc, et, avg_dice = evaluate(model, valid_loader, criterion, device)
        elapsed = time.time() - t0

        log(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Dice={avg_dice:.4f} '
            f'(WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f}) [{elapsed:.1f}s]')

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            log(f'New best model saved with dice: {best_dice:.4f}')

    # Final test
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_loss, wt, tc, et, avg_dice = evaluate(model, test_loader, criterion, device)
    log(f'Test: Dice={avg_dice:.4f} (WT={wt:.4f}, TC={tc:.4f}, ET={et:.4f})')
    log(f'Training complete. Best val dice: {best_dice:.4f}')

    log_fh.close()


if __name__ == '__main__':
    main()
