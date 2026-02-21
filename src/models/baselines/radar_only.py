"""Radar-only baseline with 3D CNN architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .base import BaselineModel


class RadarOnlyBaseline(BaselineModel):
    """
    Radar-only baseline model.

    Architecture:
    - 3D CNN for radar tensors (spatial + range dimensions)
    - Global average pooling for aggregation
    - MLP projection head

    Training objective:
    - Supervised: Predict future radar occupancy
    - Loss: MSE between predicted and actual future features

    This baseline demonstrates radar-only approaches and their
    limitations compared to multi-modal fusion (sparse data, low resolution).
    """

    def __init__(self, config: Dict):
        """
        Initialize radar-only baseline.

        Args:
            config: Configuration dictionary
                Required:
                - latent_dim: Latent representation dimension
                - learning_rate: Learning rate
                Optional:
                - temporal_enabled: Use temporal sequences (default: False)
                - temporal_pool: Pooling strategy ('max', 'mean', 'last') (default: 'max')
        """
        super().__init__(config)

        self.temporal_enabled = config.get('temporal_enabled', False)
        self.temporal_pool = config.get('temporal_pool', 'max')

        # 3D CNN encoder for radar tensors
        # Assuming radar input: (C, H, W, D) where D is range dimension
        # Simplified: treat as (C, H, W) for 2D radar

        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),

            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.feature_dim = 256

        # Projection head to latent space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, self.latent_dim)
        )

        # Predictor for future radar prediction
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(128, self.feature_dim)
        )

    def extract_features(self, radar: torch.Tensor) -> torch.Tensor:
        """
        Extract features from radar tensor using 3D CNN.

        Args:
            radar: (B, C, H, W) or (B, T, C, H, W) if temporal

        Returns:
            features: (B, feature_dim) or (B, T, feature_dim) if temporal
        """
        is_temporal = radar.ndim == 5

        if is_temporal:
            B, T, C, H, W = radar.shape
            # Reshape to (B*T, C, H, W)
            radar = radar.view(B * T, C, H, W)

        # CNN forward
        features = self.encoder(radar)
        features = features.view(features.size(0), -1)  # Flatten: (B, 256)

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
            batch: Dictionary containing 'radar' key
                radar: (B, C, H, W) or (B, T, C, H, W) if temporal

        Returns:
            latent: (B, latent_dim)
        """
        radar = batch['radar']

        # Extract features
        features = self.extract_features(radar)

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
            batch: Dictionary containing 'radar' key

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

        Training objective: Predict future radar features

        Args:
            batch: Dictionary containing radar tensors and optional future tensors
            outputs: Latent representations from forward()

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        # Predict future radar features
        predicted_features = self.predictor(outputs)

        # Get target features
        if 'radar_future' in batch:
            # Use actual future radar
            with torch.no_grad():
                target_radar = batch['radar_future']
                if target_radar.ndim == 5:
                    # Take last future frame
                    target_radar = target_radar[:, -1]
                target_features = self.extract_features(target_radar)
        else:
            # Use current radar features as target
            radar = batch['radar']
            if radar.ndim == 5:
                # Take last frame
                radar = radar[:, -1]

            with torch.no_grad():
                target_features = self.extract_features(radar)

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
            f"RadarOnlyBaseline(\n"
            f"  temporal_enabled={self.temporal_enabled},\n"
            f"  temporal_pool={self.temporal_pool},\n"
            f"  {base_repr.split('(', 1)[1]}"
        )
