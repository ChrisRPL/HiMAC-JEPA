"""Camera-only baseline with ResNet + LSTM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torchvision import models

from .base import BaselineModel


class CameraOnlyBaseline(BaselineModel):
    """
    Camera-only baseline model.

    Architecture:
    - ResNet18 encoder (pretrained on ImageNet)
    - LSTM for temporal aggregation
    - Projection head for latent space

    Training objective:
    - Supervised: Predict future frame features
    - Loss: MSE between predicted and actual future features

    This baseline demonstrates the limitations of single-modal
    (camera-only) approaches compared to multi-modal fusion.
    """

    def __init__(self, config: Dict):
        """
        Initialize camera-only baseline.

        Args:
            config: Configuration dictionary
                Required:
                - latent_dim: Latent representation dimension
                - learning_rate: Learning rate
                Optional:
                - pretrained: Use ImageNet pretrained ResNet (default: True)
                - lstm_hidden_dim: LSTM hidden dimension (default: 512)
                - lstm_num_layers: Number of LSTM layers (default: 2)
                - temporal_enabled: Use temporal sequences (default: False)
        """
        super().__init__(config)

        self.pretrained = config.get('pretrained', True)
        self.lstm_hidden_dim = config.get('lstm_hidden_dim', 512)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)
        self.temporal_enabled = config.get('temporal_enabled', False)

        # ResNet18 encoder (remove final FC layer)
        resnet = models.resnet18(pretrained=self.pretrained)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet18 output dimension

        # Freeze early layers (optional)
        if self.pretrained:
            for param in list(self.feature_extractor.parameters())[:-20]:
                param.requires_grad = False

        # Temporal aggregation with LSTM
        if self.temporal_enabled:
            self.lstm = nn.LSTM(
                input_size=self.feature_dim,
                hidden_size=self.lstm_hidden_dim,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                dropout=self.dropout_p if self.lstm_num_layers > 1 else 0
            )
            projection_input_dim = self.lstm_hidden_dim
        else:
            self.lstm = None
            projection_input_dim = self.feature_dim

        # Projection head to latent space
        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, projection_input_dim // 2),
            nn.BatchNorm1d(projection_input_dim // 2) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(projection_input_dim // 2, self.latent_dim)
        )

        # Predictor for future frame prediction (supervised training)
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.latent_dim, self.feature_dim)
        )

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images using ResNet.

        Args:
            images: (B, C, H, W) or (B, T, C, H, W) if temporal

        Returns:
            features: (B, feature_dim) or (B, T, feature_dim) if temporal
        """
        is_temporal = images.ndim == 5

        if is_temporal:
            B, T, C, H, W = images.shape
            # Reshape to (B*T, C, H, W)
            images = images.view(B * T, C, H, W)

        # Extract features
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)  # Flatten

        if is_temporal:
            # Reshape back to (B, T, feature_dim)
            features = features.view(B, T, -1)

        return features

    def aggregate_temporal(self, features: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal features using LSTM.

        Args:
            features: (B, T, feature_dim)

        Returns:
            aggregated: (B, lstm_hidden_dim) - last hidden state
        """
        if self.lstm is None:
            # No temporal aggregation - just use last frame
            return features[:, -1]

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(features)

        # Use last hidden state
        # h_n shape: (num_layers, B, hidden_dim)
        aggregated = h_n[-1]  # Last layer: (B, hidden_dim)

        return aggregated

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing 'camera' key
                camera: (B, C, H, W) or (B, T, C, H, W) if temporal

        Returns:
            latent: (B, latent_dim)
        """
        images = batch['camera']

        # Extract features
        features = self.extract_features(images)

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
            batch: Dictionary containing 'camera' key

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

        Training objective: Predict future frame features from current latent

        Args:
            batch: Dictionary containing camera images and optional future frames
            outputs: Latent representations from forward()

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics for logging
        """
        # Predict future frame features
        predicted_features = self.predictor(outputs)

        # Get target features
        if 'camera_future' in batch:
            # Use actual future frame
            with torch.no_grad():
                target_images = batch['camera_future']
                if target_images.ndim == 5:
                    # Take last future frame
                    target_images = target_images[:, -1]
                target_features = self.extract_features(target_images)
        else:
            # Use current frame features as target (autoencoder-style)
            images = batch['camera']
            if images.ndim == 5:
                # Take last frame
                images = images[:, -1]

            with torch.no_grad():
                target_features = self.extract_features(images)

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

    def predict_future(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 1
    ) -> torch.Tensor:
        """
        Predict future frame features.

        Args:
            batch: Dictionary containing camera images
            num_steps: Number of future steps to predict

        Returns:
            predictions: (B, num_steps, feature_dim)
        """
        self.eval()

        with torch.no_grad():
            # Get current latent
            latent = self.forward(batch)

            predictions = []
            for _ in range(num_steps):
                # Predict next frame features
                pred_features = self.predictor(latent)
                predictions.append(pred_features)

                # Optional: Use predicted features to update latent
                # (for multi-step prediction)
                # latent = self.projection(pred_features)

            predictions = torch.stack(predictions, dim=1)

        return predictions

    def __repr__(self) -> str:
        """String representation."""
        base_repr = super().__repr__()
        return (
            f"CameraOnlyBaseline(\n"
            f"  pretrained={self.pretrained},\n"
            f"  temporal_enabled={self.temporal_enabled},\n"
            f"  lstm_hidden_dim={self.lstm_hidden_dim},\n"
            f"  {base_repr.split('(', 1)[1]}"
        )
