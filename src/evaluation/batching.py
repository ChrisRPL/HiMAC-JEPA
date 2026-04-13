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


def _extract_motion_targets(labels: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert motion label dict into padded-ready tensors."""
    motion_dict = labels.get("motion", {})
    current_states = np.asarray(motion_dict.get("current_states", np.zeros((0, 4))), dtype=np.float32)
    future_trajectories = np.asarray(
        motion_dict.get("future_trajectories", np.zeros((0, 0, 2))),
        dtype=np.float32,
    )
    valid_masks = np.asarray(
        motion_dict.get("valid_masks", np.zeros((0, 0), dtype=bool)),
        dtype=bool,
    )

    if current_states.ndim != 2:
        current_states = current_states.reshape(-1, 4)
    if future_trajectories.ndim != 3:
        future_trajectories = future_trajectories.reshape(0, 0, 2)
    if valid_masks.ndim != 2:
        valid_masks = valid_masks.reshape(0, 0)

    return (
        torch.from_numpy(current_states),
        torch.from_numpy(future_trajectories),
        torch.from_numpy(valid_masks),
    )


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
    motion_current_states = []
    motion_future_trajectories = []
    motion_valid_masks = []

    for sample in samples:
        labels = sample.get("labels", {})
        trajectory, valid_mask = _extract_ego_trajectory(labels)
        trajectories.append(trajectory)
        valid_masks.append(valid_mask)

        if "bev" in labels:
            bev_labels.append(torch.as_tensor(labels["bev"], dtype=torch.long))

        motion_current, motion_future, motion_valid = _extract_motion_targets(labels)
        motion_current_states.append(motion_current)
        motion_future_trajectories.append(motion_future)
        motion_valid_masks.append(motion_valid)

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

    if motion_current_states:
        max_agents = max((states.size(0) for states in motion_current_states), default=0)
        max_motion_steps = max((traj.size(1) for traj in motion_future_trajectories), default=0)

        padded_current_states = torch.zeros(len(samples), max_agents, 4, dtype=torch.float32)
        padded_future_trajectories = torch.zeros(
            len(samples),
            max_agents,
            max_motion_steps,
            2,
            dtype=torch.float32,
        )
        padded_motion_masks = torch.zeros(
            len(samples),
            max_agents,
            max_motion_steps,
            dtype=torch.bool,
        )
        motion_agent_mask = torch.zeros(len(samples), max_agents, dtype=torch.bool)

        for idx, (current_states, future_trajectory, valid_mask) in enumerate(
            zip(motion_current_states, motion_future_trajectories, motion_valid_masks)
        ):
            num_agents = current_states.size(0)
            num_steps = future_trajectory.size(1) if future_trajectory.ndim == 3 else 0
            if num_agents == 0:
                continue

            padded_current_states[idx, :num_agents] = current_states
            motion_agent_mask[idx, :num_agents] = True

            if num_steps > 0:
                padded_future_trajectories[idx, :num_agents, :num_steps] = future_trajectory
                padded_motion_masks[idx, :num_agents, :num_steps] = valid_mask

        collated["motion_current_states"] = padded_current_states
        collated["motion_future_trajectories"] = padded_future_trajectories
        collated["motion_valid_mask"] = padded_motion_masks
        collated["motion_agent_mask"] = motion_agent_mask

    return collated
