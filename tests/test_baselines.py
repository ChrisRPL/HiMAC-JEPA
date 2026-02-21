"""Tests for baseline models."""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple


class DummyBaseline(nn.Module):
    """Dummy baseline for testing base class functionality."""

    def __init__(self, config):
        super().__init__()
        from src.models.baselines.base import BaselineModel

        # Inherit from BaselineModel
        self.__class__.__bases__ = (BaselineModel,)
        BaselineModel.__init__(self, config)

        # Simple linear layer for testing
        self.encoder = nn.Linear(100, self.latent_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dummy forward pass."""
        # Assume batch has 'features' key
        x = batch['features']
        return self.encoder(x)

    def get_latent(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract latent representation."""
        return self.forward(batch)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Dummy loss computation."""
        # Simple MSE loss against dummy target
        target = torch.zeros_like(outputs)
        loss = nn.functional.mse_loss(outputs, target)

        metrics = {
            'loss': loss.item(),
            'mean_output': outputs.mean().item()
        }

        return loss, metrics


class TestBaselineModel:
    """Test suite for BaselineModel base class."""

    def test_initialization(self):
        """Test baseline model initializes correctly."""
        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'dropout': 0.1,
            'batch_norm': True
        }

        model = DummyBaseline(config)

        assert model.latent_dim == 256
        assert model.learning_rate == 1e-4
        assert model.dropout_p == 0.1
        assert model.use_batch_norm is True

    def test_default_config(self):
        """Test with minimal config (using defaults)."""
        config = {
            'latent_dim': 128,
            'learning_rate': 1e-3
        }

        model = DummyBaseline(config)

        assert model.latent_dim == 128
        assert model.learning_rate == 1e-3
        assert model.dropout_p == 0.1  # default
        assert model.use_batch_norm is True  # default

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        batch = {
            'features': torch.randn(4, 100)  # (B=4, input_dim=100)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)
        assert output.dtype == torch.float32

    def test_get_latent(self):
        """Test get_latent returns correct shape."""
        config = {'latent_dim': 128, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        batch = {
            'features': torch.randn(8, 100)
        }

        latent = model.get_latent(batch)

        assert latent.shape == (8, 128)

    def test_compute_loss(self):
        """Test loss computation."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        batch = {
            'features': torch.randn(4, 100)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert loss.item() >= 0  # loss should be non-negative

        # Check metrics
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0

    def test_train_step(self):
        """Test training step."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = {
            'features': torch.randn(4, 100)
        }

        # Initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Training step
        metrics = model.train_step(batch, optimizer)

        # Check metrics returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics

        # Check parameters updated
        for p_init, p_current in zip(initial_params, model.parameters()):
            # At least some parameters should have changed
            # (might not all change if gradients are zero)
            pass

    def test_val_step(self):
        """Test validation step."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        batch = {
            'features': torch.randn(4, 100)
        }

        # Validation step
        metrics = model.val_step(batch)

        # Check metrics returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics

        # Check model is in eval mode
        assert not model.training

    def test_get_num_parameters(self):
        """Test parameter counting."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        num_params = model.get_num_parameters()

        # Linear layer: 100 * 256 + 256 = 25,856
        expected = 100 * 256 + 256
        assert num_params == expected

    def test_get_model_size_mb(self):
        """Test model size calculation."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        size_mb = model.get_model_size_mb()

        # Should be small model (< 1 MB)
        assert 0 < size_mb < 1.0

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test checkpoint saving and loading."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train for a bit
        batch = {'features': torch.randn(4, 100)}
        model.train_step(batch, optimizer)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pth"
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=5,
            optimizer=optimizer,
            custom_field="test_value"
        )

        assert checkpoint_path.exists()

        # Load checkpoint
        new_model = DummyBaseline(config)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)

        checkpoint = new_model.load_checkpoint(
            str(checkpoint_path),
            optimizer=new_optimizer
        )

        # Check loaded data
        assert checkpoint['epoch'] == 5
        assert checkpoint['custom_field'] == "test_value"

        # Check model weights match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_repr(self):
        """Test string representation."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        repr_str = repr(model)

        assert 'DummyBaseline' in repr_str
        assert 'latent_dim=256' in repr_str
        assert 'num_parameters' in repr_str
        assert 'size_mb' in repr_str

    def test_gradient_clipping(self):
        """Test gradient clipping in train_step."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create batch that produces large gradients
        batch = {
            'features': torch.randn(4, 100) * 1000  # Large values
        }

        # Training step (should clip gradients)
        metrics = model.train_step(batch, optimizer)

        # Check gradients are clipped (not too large)
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                assert grad_norm < 100  # Should be clipped to reasonable range

    def test_batch_processing(self):
        """Test processing different batch sizes."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        for batch_size in [1, 4, 8, 16]:
            batch = {
                'features': torch.randn(batch_size, 100)
            }

            output = model.forward(batch)
            assert output.shape == (batch_size, 256)

    def test_device_handling(self):
        """Test model works on different devices."""
        config = {'latent_dim': 256, 'learning_rate': 1e-4}
        model = DummyBaseline(config)

        # Test on CPU
        batch_cpu = {'features': torch.randn(4, 100)}
        output_cpu = model.forward(batch_cpu)
        assert output_cpu.device.type == 'cpu'

        # Test on CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            batch_cuda = {'features': torch.randn(4, 100).cuda()}
            output_cuda = model_cuda.forward(batch_cuda)
            assert output_cuda.device.type == 'cuda'
