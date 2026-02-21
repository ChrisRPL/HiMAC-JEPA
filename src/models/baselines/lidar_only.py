"""LiDAR-only baseline with PointNet++ architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


from .base import BaselineModel


class PointNetSetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction module.

    Samples points, groups neighbors, and applies PointNet.
    """

    def __init__(
        self,
        npoint: Optional[int],
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool = False
    ):
        """
        Initialize Set Abstraction module.

        Args:
            npoint: Number of points to sample (None for group_all)
            radius: Radius for ball query
            nsample: Maximum number of points in each ball
            in_channel: Input channel dimension
            mlp: List of output dimensions for MLP layers
            group_all: Whether to group all points
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP layers
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            xyz: (B, N, 3) point coordinates
            points: (B, N, C) point features (optional)

        Returns:
            new_xyz: (B, npoint, 3) sampled point coordinates
            new_points: (B, C', npoint) aggregated features
        """
        B, N, _ = xyz.shape

        if self.group_all:
            # Group all points
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)

            if points is not None:
                grouped_points = points.unsqueeze(1)  # (B, 1, N, C)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            # Sample points using FPS (simplified - use random sampling)
            if self.npoint is not None and self.npoint < N:
                fps_idx = torch.randperm(N, device=xyz.device)[:self.npoint]
                fps_idx = fps_idx.unsqueeze(0).expand(B, -1)  # (B, npoint)
                new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_xyz = xyz
                fps_idx = torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)

            # Group points (simplified - use k-nearest neighbors)
            # For simplicity, we'll use a fixed neighborhood
            grouped_xyz = self._group_points(xyz, new_xyz, self.nsample)

            if points is not None:
                grouped_points = self._group_points(
                    points.transpose(1, 2).contiguous(),
                    new_xyz,
                    self.nsample
                ).transpose(2, 3)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        # grouped_points: (B, npoint, nsample, C)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # (B, C, npoint, nsample)

        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling
        new_points = torch.max(grouped_points, dim=-1)[0]  # (B, C', npoint)

        return new_xyz, new_points

    def _group_points(self, points: torch.Tensor, centroids: torch.Tensor, nsample: int) -> torch.Tensor:
        """
        Group points around centroids (simplified k-NN).

        Args:
            points: (B, N, C) points to group
            centroids: (B, M, 3) centroid coordinates
            nsample: Number of samples per group

        Returns:
            grouped: (B, M, nsample, C)
        """
        B, N, C = points.shape
        M = centroids.size(1)

        # Simplified: just take first nsample points for each centroid
        # In real implementation, use ball query or k-NN
        grouped = points[:, :nsample].unsqueeze(1).expand(-1, M, -1, -1)

        return grouped


class LiDAROnlyBaseline(BaselineModel):
    """
    LiDAR-only baseline model.

    Architecture:
    - Simplified PointNet++ encoder (3 SA modules)
    - Max pooling for temporal aggregation
    - MLP projection head

    Training objective:
    - Supervised: Predict future point cloud occupancy
    - Loss: Chamfer distance or occupancy prediction

    This baseline demonstrates LiDAR-only approaches and their
    limitations compared to multi-modal fusion.
    """

    def __init__(self, config: Dict):
        """
        Initialize LiDAR-only baseline.

        Args:
            config: Configuration dictionary
                Required:
                - latent_dim: Latent representation dimension
                - learning_rate: Learning rate
                Optional:
                - num_points: Number of points to sample (default: 2048)
                - temporal_enabled: Use temporal sequences (default: False)
                - temporal_pool: Pooling strategy ('max', 'mean', 'last') (default: 'max')
        """
        super().__init__(config)

        self.num_points = config.get('num_points', 2048)
        self.temporal_enabled = config.get('temporal_enabled', False)
        self.temporal_pool = config.get('temporal_pool', 'max')

        # PointNet++ encoder
        # SA module 1: 2048 -> 512 points
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,  # xyz coordinates only
            mlp=[64, 64, 128]
        )

        # SA module 2: 512 -> 128 points
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256]
        )

        # SA module 3: 128 -> 1 point (global feature)
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            mlp=[256, 512, 1024],
            group_all=True
        )

        self.feature_dim = 1024

        # Projection head to latent space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, self.latent_dim)
        )

        # Predictor for future point cloud prediction
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, self.feature_dim)
        )

    def extract_features(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract features from point cloud using PointNet++.

        Args:
            point_cloud: (B, N, 3) or (B, T, N, 3) if temporal

        Returns:
            features: (B, feature_dim) or (B, T, feature_dim) if temporal
        """
        is_temporal = point_cloud.ndim == 4

        if is_temporal:
            B, T, N, C = point_cloud.shape
            # Reshape to (B*T, N, C)
            point_cloud = point_cloud.view(B * T, N, C)

        # Ensure we have exactly num_points (sample or pad)
        B_effective, N, C = point_cloud.shape
        if N > self.num_points:
            # Random sampling
            idx = torch.randperm(N, device=point_cloud.device)[:self.num_points]
            point_cloud = point_cloud[:, idx]
        elif N < self.num_points:
            # Pad with zeros
            pad_size = self.num_points - N
            padding = torch.zeros(B_effective, pad_size, C, device=point_cloud.device)
            point_cloud = torch.cat([point_cloud, padding], dim=1)

        # PointNet++ forward
        xyz = point_cloud[:, :, :3]  # (B, N, 3)

        # SA module 1
        xyz1, points1 = self.sa1(xyz, None)

        # SA module 2
        xyz2, points2 = self.sa2(xyz1, points1.transpose(1, 2).contiguous())

        # SA module 3 (global feature)
        _, points3 = self.sa3(xyz2, points2.transpose(1, 2).contiguous())

        # points3: (B, 1024, 1)
        features = points3.squeeze(-1)  # (B, 1024)

        if is_temporal:
            # Reshape back to (B, T, feature_dim)
            features = features.view(B, T, -1)

        return features

    def aggregate_temporal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features.

        Args:
            features: (B, T, feature_dim)

        Returns:
            aggregated: (B, feature_dim)
        """
        if self.temporal_pool == 'max':
            aggregated = torch.max(features, dim=1)[0]
        elif self.temporal_pool == 'mean':
            aggregated = torch.mean(features, dim=1)
        elif self.temporal_pool == 'last':
            aggregated = features[:, -1]
        else:
            raise ValueError(f"Unknown temporal_pool: {self.temporal_pool}")

        return aggregated

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing 'lidar' key
                lidar: (B, N, 3) or (B, T, N, 3) if temporal

        Returns:
            latent: (B, latent_dim)
        """
        point_cloud = batch['lidar']

        # Extract features
        features = self.extract_features(point_cloud)

        # Temporal aggregation if needed
        if features.ndim == 3:  # (B, T, feature_dim)
            features = self.aggregate_temporal(features)

        # Project to latent space
        latent = self.projection(features)

        return latent

    def get_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract latent representations for downstream tasks.

        Args:
            batch: Dictionary containing 'lidar' key

        Returns:
            latent: (B, latent_dim)
        """
        return self.forward(batch)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Training objective: Predict future point cloud features

        Args:
            batch: Dictionary containing point clouds and optional future clouds
            outputs: Latent representations from forward()

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        # Predict future point cloud features
        predicted_features = self.predictor(outputs)

        # Get target features
        if 'lidar_future' in batch:
            # Use actual future point cloud
            with torch.no_grad():
                target_cloud = batch['lidar_future']
                if target_cloud.ndim == 4:
                    # Take last future frame
                    target_cloud = target_cloud[:, -1]
                target_features = self.extract_features(target_cloud)
        else:
            # Use current point cloud features as target
            point_cloud = batch['lidar']
            if point_cloud.ndim == 4:
                # Take last frame
                point_cloud = point_cloud[:, -1]

            with torch.no_grad():
                target_features = self.extract_features(point_cloud)

        # MSE loss
        reconstruction_loss = F.mse_loss(predicted_features, target_features)

        # Optional: Add regularization on latent space
        latent_reg = 0.01 * torch.mean(outputs ** 2)

        # Total loss
        loss = reconstruction_loss + latent_reg

        # Metrics
        metrics = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'latent_reg': latent_reg.item(),
            'latent_mean': outputs.mean().item(),
            'latent_std': outputs.std().item(),
        }

        return loss, metrics

    def __repr__(self) -> str:
        """String representation."""
        base_repr = super().__repr__()
        return (
            f"LiDAROnlyBaseline(\n"
            f"  num_points={self.num_points},\n"
            f"  temporal_enabled={self.temporal_enabled},\n"
            f"  temporal_pool={self.temporal_pool},\n"
            f"  {base_repr.split('(', 1)[1]}"
        )
