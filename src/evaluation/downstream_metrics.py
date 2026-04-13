"""Downstream task evaluation metrics for autonomous driving."""
import torch
import numpy as np
from typing import Dict, Tuple, Optional


class DownstreamEvaluator:
    """Evaluate performance on downstream autonomous driving tasks."""

    def __init__(self, model, dataloader, device='cuda'):
        """
        Initialize downstream evaluator.

        Args:
            model: HiMAC-JEPA model with task heads
            dataloader: Test dataloader
            device: Device to run evaluation on
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    def _reshape_trajectory_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Reshape flattened trajectory outputs into (B, T, 2)."""
        if prediction.dim() == 3:
            return prediction

        if prediction.dim() != 2 or prediction.size(-1) % 2 != 0:
            raise ValueError(f"Unexpected trajectory prediction shape: {tuple(prediction.shape)}")

        return prediction.view(prediction.size(0), -1, 2)

    def _align_trajectory_targets(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align trajectory predictions with available ground-truth steps."""
        prediction = self._reshape_trajectory_prediction(prediction)
        target = target.to(prediction.device)

        max_steps = min(prediction.size(1), target.size(1))
        prediction = prediction[:, :max_steps]
        target = target[:, :max_steps]

        if valid_mask is None:
            valid_mask = torch.ones(
                prediction.shape[:2],
                device=prediction.device,
                dtype=torch.bool
            )
        else:
            valid_mask = valid_mask[:, :max_steps].to(prediction.device).bool()

        return prediction, target, valid_mask

    def _reshape_motion_prediction(
        self,
        prediction: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Reshape flattened motion outputs into (B, A, T, 2)."""
        if prediction.dim() == 4:
            return prediction

        if prediction.dim() != 2 or prediction.size(-1) % (num_steps * 2) != 0:
            raise ValueError(
                f"Unexpected motion prediction shape {tuple(prediction.shape)} for {num_steps} steps"
            )

        num_agents = prediction.size(-1) // (num_steps * 2)
        return prediction.view(prediction.size(0), num_agents, num_steps, 2)

    def _align_motion_targets(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align fixed-width motion predictions with padded closest-agent labels."""
        target = target.to(prediction.device)
        valid_mask = valid_mask.to(prediction.device).bool()

        if target.dim() != 4 or valid_mask.dim() != 3:
            raise ValueError("motion targets must have shape (B, A, T, 2) and masks (B, A, T)")

        prediction = self._reshape_motion_prediction(prediction, num_steps=target.size(2))
        max_agents = min(prediction.size(1), target.size(1))
        max_steps = min(prediction.size(2), target.size(2))

        prediction = prediction[:, :max_agents, :max_steps]
        target = target[:, :max_agents, :max_steps]
        valid_mask = valid_mask[:, :max_agents, :max_steps]

        if agent_mask is None:
            agent_mask = torch.ones(
                prediction.shape[:2],
                device=prediction.device,
                dtype=torch.bool,
            )
        else:
            agent_mask = agent_mask[:, :max_agents].to(prediction.device).bool()

        return prediction, target, valid_mask, agent_mask

    def trajectory_metrics(
        self,
        horizon=30,
        num_modes=3
    ) -> Dict[str, float]:
        """
        Compute trajectory prediction metrics.

        Evaluates ADE (Average Displacement Error) and FDE (Final Displacement Error)
        for predicted trajectories.

        Args:
            horizon: Prediction horizon (number of timesteps)
            num_modes: Number of trajectory modes to predict

        Returns:
            Dictionary with ADE, FDE, minADE, minFDE
        """
        total_ade = 0.0
        total_fde = 0.0
        total_min_ade = 0.0
        total_min_fde = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                # Get model predictions
                _, _, trajectory_pred, _, _ = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                gt_trajectory = batch.get('trajectory_ego')
                valid_mask = batch.get('trajectory_valid_mask')
                if gt_trajectory is None:
                    raise ValueError(
                        "trajectory_metrics requires 'trajectory_ego' labels in the evaluation batch"
                    )

                trajectory_pred, gt_trajectory, valid_mask = self._align_trajectory_targets(
                    trajectory_pred,
                    gt_trajectory,
                    valid_mask,
                )

                ade = self._compute_ade(trajectory_pred, gt_trajectory, valid_mask)
                fde = self._compute_fde(trajectory_pred, gt_trajectory, valid_mask)

                batch_size = trajectory_pred.size(0)
                total_ade += ade * batch_size
                total_fde += fde * batch_size
                total_min_ade += ade * batch_size
                total_min_fde += fde * batch_size
                num_samples += batch_size

        return {
            'downstream/trajectory_ade': total_ade / num_samples if num_samples > 0 else 0.0,
            'downstream/trajectory_fde': total_fde / num_samples if num_samples > 0 else 0.0,
            'downstream/trajectory_min_ade': total_min_ade / num_samples if num_samples > 0 else 0.0,
            'downstream/trajectory_min_fde': total_min_fde / num_samples if num_samples > 0 else 0.0,
        }

    def bev_segmentation_metrics(self, num_classes=10) -> Dict[str, float]:
        """
        Compute BEV segmentation metrics.

        Evaluates mIoU (mean Intersection over Union), precision, and recall
        for bird's-eye-view segmentation predictions.

        Args:
            num_classes: Number of segmentation classes

        Returns:
            Dictionary with mIoU, precision, recall
        """
        total_iou = np.zeros(num_classes)
        total_tp = 0  # True positives
        total_fp = 0  # False positives
        total_fn = 0  # False negatives
        num_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                # Get BEV segmentation predictions
                _, _, _, _, bev_pred = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                gt_bev = batch.get('bev_label')
                if gt_bev is None:
                    raise ValueError(
                        "bev_segmentation_metrics requires 'bev_label' targets in the evaluation batch"
                    )

                pred_classes = torch.argmax(bev_pred, dim=1)
                iou = self._compute_iou(pred_classes, gt_bev, num_classes)
                tp, fp, fn = self._compute_confusion_matrix(pred_classes, gt_bev)

                total_iou += iou
                total_tp += tp
                total_fp += fp
                total_fn += fn
                num_samples += pred_classes.size(0)

        # Compute mean metrics
        miou = np.mean(total_iou / num_samples) if num_samples > 0 else 0.0

        # Compute precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        return {
            'downstream/bev_miou': miou,
            'downstream/bev_precision': precision,
            'downstream/bev_recall': recall,
        }

    def motion_metrics(self) -> Dict[str, float]:
        """
        Compute multi-agent motion prediction metrics.

        Uses the closest padded agents from the evaluation batch and reports ADE/FDE
        over the fixed-width motion head outputs.
        """
        total_ade = 0.0
        total_fde = 0.0
        num_agents = 0

        with torch.no_grad():
            for batch in self.dataloader:
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                _, _, _, motion_pred, _ = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                gt_motion = batch.get('motion_future_trajectories')
                motion_valid_mask = batch.get('motion_valid_mask')
                motion_agent_mask = batch.get('motion_agent_mask')
                if gt_motion is None or motion_valid_mask is None:
                    raise ValueError(
                        "motion_metrics requires collated motion targets in the evaluation batch"
                    )

                motion_pred, gt_motion, motion_valid_mask, motion_agent_mask = self._align_motion_targets(
                    motion_pred,
                    gt_motion,
                    motion_valid_mask,
                    motion_agent_mask,
                )

                batch_ade = self._compute_motion_ade(
                    motion_pred,
                    gt_motion,
                    motion_valid_mask,
                    motion_agent_mask,
                )
                batch_fde = self._compute_motion_fde(
                    motion_pred,
                    gt_motion,
                    motion_valid_mask,
                    motion_agent_mask,
                )

                active_agents = int(motion_agent_mask.sum().item())
                total_ade += batch_ade * active_agents
                total_fde += batch_fde * active_agents
                num_agents += active_agents

        return {
            'downstream/motion_ade': total_ade / num_agents if num_agents > 0 else 0.0,
            'downstream/motion_fde': total_fde / num_agents if num_agents > 0 else 0.0,
        }

    def _compute_ade(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Average Displacement Error.

        ADE = (1/T) * Σ||pred_t - gt_t||₂

        Args:
            pred: Predicted trajectories of shape (B, T, 2)
            gt: Ground truth trajectories of shape (B, T, 2)

        Returns:
            Average displacement error
        """
        if pred.numel() == 0 or gt.numel() == 0:
            return 0.0

        distances = torch.norm(pred - gt, dim=-1)  # (B, T)

        if valid_mask is None:
            return torch.mean(distances).item()

        valid_mask = valid_mask.bool()
        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            return 0.0

        ade = distances.masked_select(valid_mask).mean()
        return ade.item()

    def _compute_fde(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Final Displacement Error.

        FDE = ||pred_T - gt_T||₂

        Args:
            pred: Predicted trajectories of shape (B, T, 2)
            gt: Ground truth trajectories of shape (B, T, 2)

        Returns:
            Final displacement error
        """
        if pred.numel() == 0 or gt.numel() == 0:
            return 0.0

        if valid_mask is None:
            final_distance = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)
            return torch.mean(final_distance).item()

        valid_mask = valid_mask.bool()
        if valid_mask.ndim != 2:
            raise ValueError("valid_mask must have shape (B, T)")

        batch_errors = []
        for batch_idx in range(pred.size(0)):
            valid_steps = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            if valid_steps.numel() == 0:
                continue
            final_idx = valid_steps[-1]
            batch_errors.append(
                torch.norm(pred[batch_idx, final_idx] - gt[batch_idx, final_idx])
            )

        if not batch_errors:
            return 0.0

        return torch.stack(batch_errors).mean().item()

    def _compute_iou(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        num_classes: int
    ) -> np.ndarray:
        """
        Compute Intersection over Union for each class.

        Args:
            pred: Predicted segmentation mask (B, H, W) with class indices
            gt: Ground truth segmentation mask (B, H, W) with class indices
            num_classes: Number of classes

        Returns:
            IoU per class as numpy array
        """
        iou_per_class = []

        # Convert to numpy if tensor
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy()

        for cls in range(num_classes):
            # Binary masks for this class
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)

            # Intersection and union
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union == 0:
                # No samples of this class in ground truth or predictions
                iou_per_class.append(float('nan'))
            else:
                iou_per_class.append(intersection / union)

        return np.array(iou_per_class)

    def _compute_motion_ade(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute ADE across valid agent trajectories."""
        if pred.numel() == 0 or gt.numel() == 0:
            return 0.0

        distances = torch.norm(pred - gt, dim=-1)
        valid_mask = valid_mask.bool()
        if agent_mask is not None:
            valid_mask = valid_mask & agent_mask.bool().unsqueeze(-1)

        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            return 0.0

        return distances.masked_select(valid_mask).mean().item()

    def _compute_motion_fde(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute FDE across valid agent trajectories."""
        if pred.numel() == 0 or gt.numel() == 0:
            return 0.0

        valid_mask = valid_mask.bool()
        if agent_mask is None:
            agent_mask = torch.ones(pred.shape[:2], device=pred.device, dtype=torch.bool)
        else:
            agent_mask = agent_mask.bool()

        final_errors = []
        for batch_idx in range(pred.size(0)):
            for agent_idx in range(pred.size(1)):
                if not agent_mask[batch_idx, agent_idx]:
                    continue
                agent_valid = torch.nonzero(valid_mask[batch_idx, agent_idx], as_tuple=False).flatten()
                if agent_valid.numel() == 0:
                    continue
                final_idx = agent_valid[-1]
                final_errors.append(
                    torch.norm(pred[batch_idx, agent_idx, final_idx] - gt[batch_idx, agent_idx, final_idx])
                )

        if not final_errors:
            return 0.0

        return torch.stack(final_errors).mean().item()

    def _compute_confusion_matrix(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor
    ) -> Tuple[int, int, int]:
        """
        Compute true positives, false positives, false negatives.

        Args:
            pred: Predicted segmentation mask
            gt: Ground truth segmentation mask

        Returns:
            Tuple of (TP, FP, FN)
        """
        # Convert to binary (foreground vs background)
        pred_binary = (pred > 0)
        gt_binary = (gt > 0)

        # Convert to numpy if tensor
        if isinstance(pred_binary, torch.Tensor):
            pred_binary = pred_binary.cpu().numpy()
        if isinstance(gt_binary, torch.Tensor):
            gt_binary = gt_binary.cpu().numpy()

        # Compute confusion matrix elements
        tp = np.logical_and(pred_binary, gt_binary).sum()
        fp = np.logical_and(pred_binary, ~gt_binary).sum()
        fn = np.logical_and(~pred_binary, gt_binary).sum()

        return int(tp), int(fp), int(fn)

    def run_all(self, config) -> Dict[str, float]:
        """
        Run all downstream evaluations specified in config.

        Args:
            config: Evaluation configuration object

        Returns:
            Dictionary of metric names and values
        """
        results = {}

        print("Running downstream evaluations...")

        # Trajectory metrics
        if any(m.startswith('trajectory') for m in config.metrics.downstream):
            print("  Computing trajectory metrics...")
            traj_results = self.trajectory_metrics(
                horizon=config.downstream.trajectory_horizon,
                num_modes=config.downstream.trajectory_num_modes
            )
            results.update(traj_results)

        # BEV segmentation metrics
        if any(m.startswith('bev') for m in config.metrics.downstream):
            print("  Computing BEV segmentation metrics...")
            bev_results = self.bev_segmentation_metrics(
                num_classes=config.downstream.bev_classes
            )
            results.update(bev_results)

        if any(m.startswith('motion') for m in config.metrics.downstream):
            print("  Computing motion prediction metrics...")
            motion_results = self.motion_metrics()
            results.update(motion_results)

        return results
