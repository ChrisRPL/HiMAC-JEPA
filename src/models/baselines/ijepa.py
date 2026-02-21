"""I-JEPA baseline adapted for autonomous driving (camera-only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from .base import BaselineModel


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer encoder for I-JEPA.

    Simplified ViT implementation for image encoding.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        """
        Initialize ViT encoder.

        Args:
            img_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Input channels (3 for RGB)
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            dropout: Dropout probability
        """
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input images
            mask: (B, num_patches) optional boolean mask (True = keep, False = mask)

        Returns:
            features: (B, num_patches, embed_dim) or (B, num_visible, embed_dim) if masked
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply mask if provided
        if mask is not None:
            x = x[mask.unsqueeze(-1).expand_as(x)].view(B, -1, self.embed_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class IJEPABaseline(BaselineModel):
    """
    I-JEPA baseline adapted for autonomous driving.

    Architecture:
    - ViT context encoder (encodes visible patches)
    - ViT target encoder (encodes target patches, EMA updated)
    - Predictor network (predicts target from context)

    Training objective:
    - JEPA: Predict masked regions in latent space
    - VICReg regularization for collapse prevention

    Differences from HiMAC-JEPA:
    - Single modality (camera only)
    - No action conditioning
    - Spatial masking only (no temporal)
    """

    def __init__(self, config: Dict):
        """
        Initialize I-JEPA baseline.

        Args:
            config: Configuration dictionary
                Required:
                - latent_dim: Latent representation dimension
                - learning_rate: Learning rate
                Optional:
                - img_size: Image size (default: 224)
                - patch_size: Patch size (default: 16)
                - embed_dim: ViT embedding dim (default: 384)
                - depth: Number of ViT blocks (default: 12)
                - num_heads: Number of attention heads (default: 6)
                - mask_ratio: Ratio of patches to mask (default: 0.75)
                - ema_decay: EMA decay rate for target encoder (default: 0.996)
                - vicreg_lambda: VICReg variance weight (default: 25.0)
                - vicreg_mu: VICReg covariance weight (default: 25.0)
        """
        super().__init__(config)

        # ViT parameters
        self.img_size = config.get('img_size', 224)
        self.patch_size = config.get('patch_size', 16)
        self.embed_dim = config.get('embed_dim', 384)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 6)

        # Masking parameters
        self.mask_ratio = config.get('mask_ratio', 0.75)

        # EMA parameters
        self.ema_decay = config.get('ema_decay', 0.996)

        # VICReg parameters
        self.vicreg_lambda = config.get('vicreg_lambda', 25.0)
        self.vicreg_mu = config.get('vicreg_mu', 25.0)

        # Context encoder (encodes visible patches)
        self.context_encoder = VisionTransformerEncoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            dropout=self.dropout_p
        )

        # Target encoder (encodes target patches, EMA updated)
        self.target_encoder = VisionTransformerEncoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            dropout=0.0  # No dropout for target encoder
        )

        # Initialize target encoder with same weights as context encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        # Freeze target encoder (will be updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor network
        self.predictor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Projection head to latent space
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass (inference mode - no masking).

        Args:
            batch: Dictionary containing 'camera' key
                camera: (B, C, H, W)

        Returns:
            latent: (B, latent_dim)
        """
        images = batch['camera']

        # Encode all patches with context encoder
        features = self.context_encoder(images, mask=None)

        # Global average pooling over patches
        features = features.mean(dim=1)  # (B, embed_dim)

        # Project to latent space
        latent = self.projection(features)

        return latent

    def get_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract latent representations for downstream tasks.

        Args:
            batch: Dictionary containing 'camera' key

        Returns:
            latent: (B, latent_dim)
        """
        return self.forward(batch)

    def generate_mask(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random block-wise mask for I-JEPA.

        Args:
            batch_size: Batch size
            device: Device for tensors

        Returns:
            context_mask: (B, num_patches) boolean mask for context (True = visible)
            target_mask: (B, num_patches) boolean mask for target (True = predict)
        """
        num_patches = self.context_encoder.num_patches
        num_masked = int(num_patches * self.mask_ratio)

        # Random masking for simplicity
        # In real I-JEPA, use block-wise masking
        context_mask = torch.ones(batch_size, num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for i in range(batch_size):
            # Randomly select patches to mask
            masked_indices = torch.randperm(num_patches)[:num_masked]
            context_mask[i, masked_indices] = False
            target_mask[i, masked_indices] = True

        return context_mask, target_mask

    def vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        VICReg regularization loss.

        Args:
            z1: (B, D) predicted features
            z2: (B, D) target features

        Returns:
            loss: Total VICReg loss
            metrics: Dict with variance and covariance components
        """
        B, D = z1.shape

        # Invariance loss (MSE)
        invariance_loss = F.mse_loss(z1, z2)

        # Variance loss (encourage std > 1)
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        variance_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance loss (encourage decorrelation)
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)

        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)

        # Off-diagonal elements should be zero
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
        Compute I-JEPA training loss.

        Args:
            batch: Dictionary containing camera images
            outputs: Latent representations (not used in JEPA training)

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        images = batch['camera']
        B = images.shape[0]

        # Generate masks
        context_mask, target_mask = self.generate_mask(B, images.device)

        # Encode context patches
        context_features = self.context_encoder(images, mask=context_mask)

        # Encode target patches (no gradient)
        with torch.no_grad():
            target_features = self.target_encoder(images, mask=target_mask)

        # Predict target from context
        # Average context features
        context_avg = context_features.mean(dim=1)  # (B, embed_dim)

        # Predict target patches
        predicted_target = self.predictor(context_avg)  # (B, embed_dim)

        # Average target features
        target_avg = target_features.mean(dim=1)  # (B, embed_dim)

        # VICReg loss
        loss, vicreg_metrics = self.vicreg_loss(predicted_target, target_avg)

        # Metrics
        metrics = {
            'loss': loss.item(),
            **vicreg_metrics,
            'context_mean': context_features.mean().item(),
            'target_mean': target_features.mean().item(),
        }

        return loss, metrics

    def update_target_encoder(self):
        """Update target encoder via EMA."""
        with torch.no_grad():
            for param_context, param_target in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_target.data.mul_(self.ema_decay).add_(
                    param_context.data, alpha=1 - self.ema_decay
                )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step with EMA update.

        Args:
            batch: Dictionary containing input tensors
            optimizer: Optimizer for parameter updates

        Returns:
            metrics: Dictionary of training metrics
        """
        # Standard training step
        metrics = super().train_step(batch, optimizer)

        # Update target encoder via EMA
        self.update_target_encoder()

        return metrics

    def __repr__(self) -> str:
        """String representation."""
        base_repr = super().__repr__()
        return (
            f"IJEPABaseline(\n"
            f"  img_size={self.img_size},\n"
            f"  patch_size={self.patch_size},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  depth={self.depth},\n"
            f"  mask_ratio={self.mask_ratio},\n"
            f"  ema_decay={self.ema_decay},\n"
            f"  {base_repr.split('(', 1)[1]}"
        )
