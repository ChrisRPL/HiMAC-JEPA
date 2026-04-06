"""Checkpoint helpers for HiMAC-JEPA training."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_training_checkpoints(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_root: str | Path,
    experiment_name: str,
    epoch: int,
    avg_total_loss: float,
    config: Optional[Dict[str, Any]] = None,
    best_loss: Optional[float] = None,
) -> Dict[str, Path]:
    """Save latest checkpoint and refresh best checkpoint when loss improves."""
    checkpoint_root = Path(checkpoint_root)
    experiment_dir = checkpoint_root / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "avg_total_loss": float(avg_total_loss),
        "model_state_dict": model.state_dict(),
        "config": config or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    latest_path = experiment_dir / "last_model.pth"
    best_path = experiment_dir / "best_model.pth"
    latest_alias_path = checkpoint_root / "last_model.pth"
    best_alias_path = checkpoint_root / "best_model.pth"

    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, latest_alias_path)

    saved_best = False
    if best_loss is None or avg_total_loss <= best_loss:
        torch.save(checkpoint, best_path)
        torch.save(checkpoint, best_alias_path)
        saved_best = True

    return {
        "latest_path": latest_path,
        "latest_alias_path": latest_alias_path,
        "best_path": best_path,
        "best_alias_path": best_alias_path,
        "saved_best": saved_best,
    }
