"""Base class for baseline models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaselineModel(nn.Module, ABC):
    """
    Abstract base class for all baseline models.

    This provides a common interface for training, evaluation, and comparison
    of different baseline approaches against HiMAC-JEPA.

    All baseline models should:
    1. Inherit from this class
    2. Implement the abstract methods
    3. Follow the same training protocol for fair comparison
    4. Return latent representations in a consistent format
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize baseline model.

        Args:
            config: Configuration dictionary with model parameters
                Required keys:
                - latent_dim: Dimension of latent representation
                - learning_rate: Learning rate for optimizer
                Optional keys:
                - dropout: Dropout probability (default: 0.1)
                - batch_norm: Whether to use batch normalization (default: True)
        """
        super().__init__()
        self.config = config
        self.latent_dim = config.get('latent_dim', 256)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.dropout_p = config.get('dropout', 0.1)
        self.use_batch_norm = config.get('batch_norm', True)

        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing input tensors
                Expected keys depend on specific baseline:
                - 'camera': (B, T, C, H, W) or (B, C, H, W)
                - 'lidar': (B, T, N, 3) or (B, N, 3)
                - 'radar': (B, T, C, H, W) or (B, C, H, W)

        Returns:
            latent: Latent representation tensor (B, latent_dim)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def get_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract latent representations for downstream tasks.

        This method is used during evaluation to extract features
        that will be fed to downstream task heads.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            latent: Latent representation tensor (B, latent_dim)
        """
        raise NotImplementedError("Subclasses must implement get_latent()")

    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss.

        Args:
            batch: Dictionary containing input tensors and labels
            outputs: Model outputs from forward()

        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary containing input tensors
            optimizer: Optimizer for parameter updates

        Returns:
            metrics: Dictionary of training metrics
        """
        self.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = self.forward(batch)

        # Compute loss
        loss, metrics = self.compute_loss(batch, outputs)

        # Backward pass
        loss.backward()

        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update parameters
        optimizer.step()

        # Update tracked metrics
        self.train_metrics = metrics

        return metrics

    def val_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single validation step.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            metrics: Dictionary of validation metrics
        """
        self.eval()

        with torch.no_grad():
            # Forward pass
            outputs = self.forward(batch)

            # Compute loss
            loss, metrics = self.compute_loss(batch, outputs)

        # Update tracked metrics
        self.val_metrics = metrics

        return metrics

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters.

        Returns:
            num_params: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self) -> float:
        """
        Get model size in megabytes.

        Returns:
            size_mb: Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb

    def get_inference_time(
        self,
        batch: Dict[str, torch.Tensor],
        num_iterations: int = 100
    ) -> float:
        """
        Measure average inference time.

        Args:
            batch: Sample batch for inference
            num_iterations: Number of iterations to average over

        Returns:
            avg_time_ms: Average inference time in milliseconds
        """
        import time

        self.eval()
        device = next(self.parameters()).device

        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(batch)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.forward(batch)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
        return avg_time_ms

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            optimizer: Optional optimizer state to save
            **kwargs: Additional items to save in checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            **kwargs
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into

        Returns:
            checkpoint: Dictionary containing checkpoint data
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore metrics
        self.train_metrics = checkpoint.get('train_metrics', {})
        self.val_metrics = checkpoint.get('val_metrics', {})

        return checkpoint

    def __repr__(self) -> str:
        """String representation of the model."""
        num_params = self.get_num_parameters()
        size_mb = self.get_model_size_mb()

        return (
            f"{self.__class__.__name__}(\n"
            f"  latent_dim={self.latent_dim},\n"
            f"  num_parameters={num_params:,},\n"
            f"  size_mb={size_mb:.2f}\n"
            f")"
        )
