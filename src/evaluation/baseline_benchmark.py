"""Helpers for honest baseline benchmark evaluation."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def fit_ridge_probe(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    alpha: float = 1e-3,
) -> torch.Tensor:
    """Fit a linear ridge probe with a bias term."""
    if train_inputs.ndim != 2:
        raise ValueError("train_inputs must have shape (N, D)")
    if train_targets.ndim != 2:
        raise ValueError("train_targets must have shape (N, K)")

    train_inputs = train_inputs.float()
    train_targets = train_targets.float()

    ones = torch.ones(train_inputs.size(0), 1, dtype=train_inputs.dtype)
    design = torch.cat([train_inputs, ones], dim=1)
    ridge = alpha * torch.eye(design.size(1), dtype=design.dtype)
    ridge[-1, -1] = 0.0  # Don't regularize the bias term.

    lhs = design.T @ design + ridge
    rhs = design.T @ train_targets
    return torch.linalg.solve(lhs, rhs)


def predict_ridge_probe(
    probe_weights: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply a fitted ridge probe."""
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (N, D)")

    inputs = inputs.float()
    ones = torch.ones(inputs.size(0), 1, dtype=inputs.dtype)
    design = torch.cat([inputs, ones], dim=1)
    return design @ probe_weights


def compute_trajectory_horizon_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    sampling_rate: float = 2.0,
    horizons: Iterable[float] = (1.0, 2.0, 3.0),
) -> Dict[str, float]:
    """Compute ADE/FDE metrics at requested time horizons."""
    predictions = predictions.float()
    targets = targets.float()
    valid_mask = valid_mask.bool()

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shape")
    if valid_mask.shape != predictions.shape[:2]:
        raise ValueError("valid_mask must have shape (B, T)")

    distances = torch.norm(predictions - targets, dim=-1)
    metrics = {}

    total_steps = predictions.size(1)
    for horizon in horizons:
        steps = max(1, min(total_steps, int(round(horizon * sampling_rate))))
        horizon_mask = valid_mask[:, :steps]
        horizon_distances = distances[:, :steps]

        if horizon_mask.any():
            ade = horizon_distances.masked_select(horizon_mask).mean().item()
        else:
            ade = float("nan")

        final_errors = []
        for batch_idx in range(predictions.size(0)):
            valid_steps = torch.nonzero(horizon_mask[batch_idx], as_tuple=False).flatten()
            if valid_steps.numel() == 0:
                continue
            final_idx = valid_steps[-1]
            final_errors.append(horizon_distances[batch_idx, final_idx])

        fde = torch.stack(final_errors).mean().item() if final_errors else float("nan")

        horizon_suffix = f"{int(horizon)}s" if float(horizon).is_integer() else f"{horizon:.1f}s"
        metrics[f"trajectory/ade_{horizon_suffix}"] = ade
        metrics[f"trajectory/fde_{horizon_suffix}"] = fde

    return metrics


def collect_probe_targets(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract trajectory targets from a collated evaluation batch."""
    if "trajectory_ego" not in batch or "trajectory_valid_mask" not in batch:
        raise ValueError("batch must contain trajectory_ego and trajectory_valid_mask")

    return batch["trajectory_ego"], batch["trajectory_valid_mask"]


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move a flat tensor batch to the target device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_bev_classification_metrics(
    predicted_classes: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Compute mIoU, precision, and recall for BEV segmentation."""
    predicted_classes = predicted_classes.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    iou_per_class = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for cls in range(num_classes):
        pred_mask = predicted_classes == cls
        target_mask = targets == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou_per_class.append(np.nan)
        else:
            iou_per_class.append(intersection / union)

    pred_binary = predicted_classes > 0
    target_binary = targets > 0
    total_tp = int(np.logical_and(pred_binary, target_binary).sum())
    total_fp = int(np.logical_and(pred_binary, ~target_binary).sum())
    total_fn = int(np.logical_and(~pred_binary, target_binary).sum())

    miou = float(np.nanmean(iou_per_class)) if iou_per_class else float("nan")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float("nan")

    return {
        "bev/miou": miou,
        "bev/precision": precision,
        "bev/recall": recall,
    }
