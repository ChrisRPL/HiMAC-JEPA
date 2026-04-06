"""Batching helpers for evaluation datasets."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def _extract_ego_trajectory(labels: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectory label dict into a padded-ready tensor plus valid mask."""
    trajectory_dict = labels.get("trajectory_ego", {})
    if not trajectory_dict:
        return torch.zeros(0, 2, dtype=torch.float32), torch.zeros(0, dtype=torch.bool)

    max_horizon = max(trajectory_dict.keys())
    trajectory = np.asarray(trajectory_dict[max_horizon], dtype=np.float32)

    if trajectory.ndim != 2:
        trajectory = trajectory.reshape(-1, 2)

    trajectory_tensor = torch.from_numpy(trajectory)
    valid_mask = torch.ones(trajectory_tensor.size(0), dtype=torch.bool)
    return trajectory_tensor, valid_mask


def collate_evaluation_batch(samples: List[Dict]) -> Dict:
    """Collate nuScenes evaluation samples while normalizing label targets."""
    collated = default_collate(
        [{k: v for k, v in sample.items() if k != "labels"} for sample in samples]
    )

    if not samples or "labels" not in samples[0]:
        return collated

    trajectories = []
    valid_masks = []
    bev_labels = []

    for sample in samples:
        labels = sample.get("labels", {})
        trajectory, valid_mask = _extract_ego_trajectory(labels)
        trajectories.append(trajectory)
        valid_masks.append(valid_mask)

        if "bev" in labels:
            bev_labels.append(torch.as_tensor(labels["bev"], dtype=torch.long))

    if trajectories:
        max_steps = max((trajectory.size(0) for trajectory in trajectories), default=0)
        padded_trajectories = torch.zeros(len(trajectories), max_steps, 2, dtype=torch.float32)
        padded_masks = torch.zeros(len(valid_masks), max_steps, dtype=torch.bool)

        for idx, (trajectory, valid_mask) in enumerate(zip(trajectories, valid_masks)):
            num_steps = trajectory.size(0)
            if num_steps == 0:
                continue
            padded_trajectories[idx, :num_steps] = trajectory
            padded_masks[idx, :num_steps] = valid_mask

        collated["trajectory_ego"] = padded_trajectories
        collated["trajectory_valid_mask"] = padded_masks

    if bev_labels:
        collated["bev_label"] = torch.stack(bev_labels, dim=0)

    return collated
