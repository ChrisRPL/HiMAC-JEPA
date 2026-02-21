"""V-JEPA baseline adapted for multi-modal driving (no action conditioning)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .base import BaselineModel


class VJEPABaseline(BaselineModel):
    """
    V-JEPA baseline adapted for multi-modal autonomous driving.

    Architecture:
    - Multi-modal encoders (camera, LiDAR, radar)
    - Temporal masking (predict future from past)
    - Simple fusion via concatenation (no hierarchical structure)
    - Predictor network
    - EMA for target encoders

    Training objective:
    - JEPA: Predict future latent from past latent
    - VICReg regularization

    Differences from HiMAC-JEPA:
    - No action conditioning
    - Simple fusion (concatenate features)
    - No hierarchical action structure
    - Temporal prediction only (no spatial masking)
    """

    def __init__(self, config: Dict):
        """
        Initialize V-JEPA baseline.

        Args:
            config: Configuration dictionary
                Required:
                - latent_dim: Latent representation dimension
                - learning_rate: Learning rate
                Optional:
                - embed_dim: Embedding dimension for each modality (default: 256)
                - camera_enabled: Use camera modality (default: True)
                - lidar_enabled: Use LiDAR modality (default: True)
                - radar_enabled: Use radar modality (default: True)
                - ema_decay: EMA decay rate (default: 0.996)
                - vicreg_lambda: VICReg variance weight (default: 25.0)
                - vicreg_mu: VICReg covariance weight (default: 25.0)
                - pred_horizon: Number of future steps to predict (default: 1)
        """
        super().__init__(config)

        # Modality settings
        self.embed_dim = config.get('embed_dim', 256)
        self.camera_enabled = config.get('camera_enabled', True)
        self.lidar_enabled = config.get('lidar_enabled', True)
        self.radar_enabled = config.get('radar_enabled', True)

        # JEPA settings
        self.ema_decay = config.get('ema_decay', 0.996)
        self.vicreg_lambda = config.get('vicreg_lambda', 25.0)
        self.vicreg_mu = config.get('vicreg_mu', 25.0)
        self.pred_horizon = config.get('pred_horizon', 1)

        # Count active modalities
        self.num_modalities = (
            int(self.camera_enabled) +
            int(self.lidar_enabled) +
            int(self.radar_enabled)
        )

        assert self.num_modalities > 0, "At least one modality must be enabled"

        # Context encoders (encode past frames)
        if self.camera_enabled:
            self.camera_context_encoder = self._create_camera_encoder()
            self.camera_target_encoder = self._create_camera_encoder()
            self._init_target_encoder(self.camera_context_encoder, self.camera_target_encoder)

        if self.lidar_enabled:
            self.lidar_context_encoder = self._create_lidar_encoder()
            self.lidar_target_encoder = self._create_lidar_encoder()
            self._init_target_encoder(self.lidar_context_encoder, self.lidar_target_encoder)

        if self.radar_enabled:
            self.radar_context_encoder = self._create_radar_encoder()
            self.radar_target_encoder = self._create_radar_encoder()
            self._init_target_encoder(self.radar_context_encoder, self.radar_target_encoder)

        # Fusion layer (simple concatenation)
        fusion_dim = self.num_modalities * self.embed_dim

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # Projection to latent space
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

    def _create_camera_encoder(self) -> nn.Module:
        """Create camera encoder (simplified ResNet)."""
        from torchvision import models

        resnet = models.resnet18(pretrained=False)
        encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Add projection to embed_dim
        return nn.Sequential(
            encoder,
            nn.Flatten(),
            nn.Linear(512, self.embed_dim)
        )

    def _create_lidar_encoder(self) -> nn.Module:
        """Create LiDAR encoder (simplified PointNet)."""
        return nn.Sequential(
            # Simplified: just use MLP on flattened points
            # In real implementation, use proper PointNet
            nn.Linear(2048 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.embed_dim)
        )

    def _create_radar_encoder(self) -> nn.Module:
        """Create radar encoder (simple CNN)."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.embed_dim)
        )

    def _init_target_encoder(self, context_encoder: nn.Module, target_encoder: nn.Module):
        """Initialize target encoder with context encoder weights and freeze."""
        target_encoder.load_state_dict(context_encoder.state_dict())
        for param in target_encoder.parameters():
            param.requires_grad = False

    def encode_modality(
        self,
        data: torch.Tensor,
        modality: str,
        use_target: bool = False
    ) -> torch.Tensor:
        """
        Encode a single modality.

        Args:
            data: Input tensor
            modality: 'camera', 'lidar', or 'radar'
            use_target: Use target encoder (EMA) instead of context encoder

        Returns:
            features: (B, embed_dim)
        """
        if modality == 'camera':
            encoder = self.camera_target_encoder if use_target else self.camera_context_encoder
        elif modality == 'lidar':
            # Flatten LiDAR points for simplified encoder
            if data.ndim == 3:  # (B, N, 3)
                data = data.view(data.size(0), -1)
            encoder = self.lidar_target_encoder if use_target else self.lidar_context_encoder
        elif modality == 'radar':
            encoder = self.radar_target_encoder if use_target else self.radar_context_encoder
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return encoder(data)

    def fuse_modalities(self, batch: Dict[str, torch.Tensor], use_target: bool = False) -> torch.Tensor:
        """
        Encode and fuse all available modalities.

        Args:
            batch: Dictionary with modality keys
            use_target: Use target encoders

        Returns:
            fused: (B, fusion_dim) concatenated features
        """
        features = []

        if self.camera_enabled and 'camera' in batch:
            camera_feat = self.encode_modality(batch['camera'], 'camera', use_target)
            features.append(camera_feat)

        if self.lidar_enabled and 'lidar' in batch:
            lidar_feat = self.encode_modality(batch['lidar'], 'lidar', use_target)
            features.append(lidar_feat)

        if self.radar_enabled and 'radar' in batch:
            radar_feat = self.encode_modality(batch['radar'], 'radar', use_target)
            features.append(radar_feat)

        # Concatenate all modalities
        fused = torch.cat(features, dim=-1)

        return fused

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass (inference mode).

        Args:
            batch: Dictionary containing modality keys
                camera: (B, C, H, W) optional
                lidar: (B, N, 3) optional
                radar: (B, C, H, W) optional

        Returns:
            latent: (B, latent_dim)
        """
        # Fuse modalities with context encoder
        fused = self.fuse_modalities(batch, use_target=False)

        # Project to latent space
        latent = self.projection(fused)

        return latent

    def get_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract latent representations for downstream tasks.

        Args:
            batch: Dictionary containing modality keys

        Returns:
            latent: (B, latent_dim)
        """
        return self.forward(batch)

    def vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg regularization loss (same as I-JEPA)."""
        B, D = z1.shape

        # Invariance loss
        invariance_loss = F.mse_loss(z1, z2)

        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        variance_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)

        cov_loss = (
            (cov_z1.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten() ** 2).sum() +
            (cov_z2.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten() ** 2).sum()
        ) / D

        # Total loss
        loss = invariance_loss + self.vicreg_lambda * variance_loss + self.vicreg_mu * cov_loss

        metrics = {
            'invariance_loss': invariance_loss.item(),
            'variance_loss': variance_loss.item(),
            'covariance_loss': cov_loss.item()
        }

        return loss, metrics

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute V-JEPA training loss.

        Assumes batch contains both past and future frames:
        - camera: (B, T, C, H, W) where T >= 2
        - lidar: (B, T, N, 3) where T >= 2
        - radar: (B, T, C, H, W) where T >= 2

        Args:
            batch: Dictionary containing temporal sequences
            outputs: Latent representations (not used)

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        # Split batch into past and future
        batch_past = {}
        batch_future = {}

        for key in ['camera', 'lidar', 'radar']:
            if key in batch:
                data = batch[key]
                if data.ndim >= 4:  # Temporal data
                    # Take last frame of past as current
                    batch_past[key] = data[:, -2] if data.size(1) > 1 else data[:, 0]
                    # Take last frame as future
                    batch_future[key] = data[:, -1]
                else:
                    # Single frame - use same for both
                    batch_past[key] = data
                    batch_future[key] = data

        # Encode past with context encoder
        context_features = self.fuse_modalities(batch_past, use_target=False)

        # Encode future with target encoder (no gradient)
        with torch.no_grad():
            target_features = self.fuse_modalities(batch_future, use_target=True)

        # Predict future from past
        predicted_future = self.predictor(context_features)

        # VICReg loss
        loss, vicreg_metrics = self.vicreg_loss(predicted_future, target_features)

        # Metrics
        metrics = {
            'loss': loss.item(),
            **vicreg_metrics,
            'context_mean': context_features.mean().item(),
            'target_mean': target_features.mean().item(),
        }

        return loss, metrics

    def update_target_encoders(self):
        """Update all target encoders via EMA."""
        with torch.no_grad():
            if self.camera_enabled:
                for p_ctx, p_tgt in zip(
                    self.camera_context_encoder.parameters(),
                    self.camera_target_encoder.parameters()
                ):
                    p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1 - self.ema_decay)

            if self.lidar_enabled:
                for p_ctx, p_tgt in zip(
                    self.lidar_context_encoder.parameters(),
                    self.lidar_target_encoder.parameters()
                ):
                    p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1 - self.ema_decay)

            if self.radar_enabled:
                for p_ctx, p_tgt in zip(
                    self.radar_context_encoder.parameters(),
                    self.radar_target_encoder.parameters()
                ):
                    p_tgt.data.mul_(self.ema_decay).add_(p_ctx.data, alpha=1 - self.ema_decay)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Training step with EMA updates."""
        metrics = super().train_step(batch, optimizer)
        self.update_target_encoders()
        return metrics

    def __repr__(self) -> str:
        """String representation."""
        base_repr = super().__repr__()
        return (
            f"VJEPABaseline(\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_modalities={self.num_modalities},\n"
            f"  camera_enabled={self.camera_enabled},\n"
            f"  lidar_enabled={self.lidar_enabled},\n"
            f"  radar_enabled={self.radar_enabled},\n"
            f"  ema_decay={self.ema_decay},\n"
            f"  {base_repr.split('(', 1)[1]}"
        )
