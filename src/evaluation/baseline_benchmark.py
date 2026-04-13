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
    """Compute mean ADE/FDE metrics at requested time horizons."""
    predictions = predictions.float()
    targets = targets.float()
    valid_mask = valid_mask.bool()

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shape")
    if valid_mask.shape != predictions.shape[:2]:
        raise ValueError("valid_mask must have shape (B, T)")

    distances = torch.norm(predictions - targets, dim=-1)
    horizon_errors = compute_trajectory_horizon_errors(
        predictions,
        targets,
        valid_mask,
        sampling_rate=sampling_rate,
        horizons=horizons,
    )

    metrics = {}
    total_steps = predictions.size(1)
    for horizon in horizons:
        steps = max(1, min(total_steps, int(round(horizon * sampling_rate))))
        horizon_mask = valid_mask[:, :steps]
        horizon_distances = distances[:, :steps]
        horizon_suffix = f"{int(horizon)}s" if float(horizon).is_integer() else f"{horizon:.1f}s"

        if horizon_mask.any():
            metrics[f"trajectory/ade_{horizon_suffix}"] = (
                horizon_distances.masked_select(horizon_mask).mean().item()
            )
        else:
            metrics[f"trajectory/ade_{horizon_suffix}"] = float("nan")

        fde_values = horizon_errors[f"trajectory/fde_{horizon_suffix}"]
        finite_fde = fde_values[torch.isfinite(fde_values)]
        metrics[f"trajectory/fde_{horizon_suffix}"] = (
            finite_fde.mean().item() if finite_fde.numel() > 0 else float("nan")
        )

    return metrics


def compute_trajectory_horizon_errors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    sampling_rate: float = 2.0,
    horizons: Iterable[float] = (1.0, 2.0, 3.0),
) -> Dict[str, torch.Tensor]:
    """Compute per-sample ADE/FDE errors at requested time horizons."""
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
        ade = torch.full(
            (predictions.size(0),),
            float("nan"),
            dtype=horizon_distances.dtype,
        )
        fde = torch.full(
            (predictions.size(0),),
            float("nan"),
            dtype=horizon_distances.dtype,
        )

        for batch_idx in range(predictions.size(0)):
            valid_steps = torch.nonzero(horizon_mask[batch_idx], as_tuple=False).flatten()
            if valid_steps.numel() == 0:
                continue
            ade[batch_idx] = horizon_distances[batch_idx, valid_steps].mean()
            fde[batch_idx] = horizon_distances[batch_idx, valid_steps[-1]]

        horizon_suffix = f"{int(horizon)}s" if float(horizon).is_integer() else f"{horizon:.1f}s"
        metrics[f"trajectory/ade_{horizon_suffix}"] = ade
        metrics[f"trajectory/fde_{horizon_suffix}"] = fde

    return metrics


def paired_sign_flip_test(
    metric_a: torch.Tensor,
    metric_b: torch.Tensor,
    num_permutations: int = 4096,
    seed: int = 0,
    chunk_size: int = 256,
) -> Tuple[float, float, int]:
    """Run a paired sign-flip permutation test on two aligned metric vectors."""
    metric_a = torch.as_tensor(metric_a, dtype=torch.float64).flatten()
    metric_b = torch.as_tensor(metric_b, dtype=torch.float64).flatten()

    if metric_a.shape != metric_b.shape:
        raise ValueError("paired metric vectors must have the same shape")

    finite_mask = torch.isfinite(metric_a) & torch.isfinite(metric_b)
    deltas = metric_a[finite_mask] - metric_b[finite_mask]
    num_pairs = int(deltas.numel())

    if num_pairs == 0:
        raise ValueError("paired test requires at least one finite paired observation")

    observed_delta = deltas.mean().item()
    if torch.allclose(deltas, torch.zeros_like(deltas)):
        return observed_delta, 1.0, num_pairs

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    extreme_count = 0
    permutations_remaining = num_permutations
    while permutations_remaining > 0:
        current_chunk = min(chunk_size, permutations_remaining)
        random_bits = torch.randint(
            0,
            2,
            (current_chunk, num_pairs),
            generator=generator,
            dtype=torch.int64,
        )
        signs = random_bits.mul(2).sub(1).to(torch.float64)
        permuted = (signs * deltas.unsqueeze(0)).mean(dim=1).abs()
        extreme_count += int((permuted >= abs(observed_delta)).sum().item())
        permutations_remaining -= current_chunk

    p_value = (extreme_count + 1) / (num_permutations + 1)
    return observed_delta, p_value, num_pairs


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
