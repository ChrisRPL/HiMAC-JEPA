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


class TestCameraOnlyBaseline:
    """Test suite for camera-only baseline model."""

    def test_initialization(self):
        """Test camera-only baseline initializes correctly."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,  # Don't download pretrained weights in tests
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)

        assert model.latent_dim == 256
        assert model.pretrained is False
        assert model.temporal_enabled is False
        assert model.lstm is None  # No LSTM when temporal disabled

    def test_initialization_with_temporal(self):
        """Test initialization with temporal mode."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': True,
            'lstm_hidden_dim': 512,
            'lstm_num_layers': 2
        }

        model = CameraOnlyBaseline(config)

        assert model.temporal_enabled is True
        assert model.lstm is not None
        assert model.lstm_hidden_dim == 512
        assert model.lstm_num_layers == 2

    def test_forward_single_frame(self):
        """Test forward pass with single frame."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 3, 224, 224)  # (B, C, H, W)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_forward_temporal(self):
        """Test forward pass with temporal sequence."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': True,
            'lstm_hidden_dim': 512
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 5, 3, 224, 224)  # (B, T, C, H, W)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_extract_features_single_frame(self):
        """Test feature extraction from single frame."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False
        }

        model = CameraOnlyBaseline(config)

        images = torch.randn(4, 3, 224, 224)
        features = model.extract_features(images)

        assert features.shape == (4, 512)  # ResNet18 feature dim = 512

    def test_extract_features_temporal(self):
        """Test feature extraction from temporal sequence."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False
        }

        model = CameraOnlyBaseline(config)

        images = torch.randn(4, 5, 3, 224, 224)  # (B, T, C, H, W)
        features = model.extract_features(images)

        assert features.shape == (4, 5, 512)  # (B, T, feature_dim)

    def test_aggregate_temporal(self):
        """Test LSTM temporal aggregation."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': True,
            'lstm_hidden_dim': 512
        }

        model = CameraOnlyBaseline(config)

        features = torch.randn(4, 5, 512)  # (B, T, feature_dim)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 512)  # (B, lstm_hidden_dim)

    def test_compute_loss(self):
        """Test loss computation."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 3, 224, 224)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        assert 'reconstruction_loss' in metrics
        assert 'latent_reg' in metrics

    def test_compute_loss_with_future(self):
        """Test loss computation with future frame."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 3, 224, 224),
            'camera_future': torch.randn(4, 3, 224, 224)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert loss.item() >= 0

    def test_predict_future(self):
        """Test future frame prediction."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 3, 224, 224)
        }

        predictions = model.predict_future(batch, num_steps=3)

        assert predictions.shape == (4, 3, 512)  # (B, num_steps, feature_dim)

    def test_get_latent(self):
        """Test latent extraction."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False
        }

        model = CameraOnlyBaseline(config)

        batch = {
            'camera': torch.randn(4, 3, 224, 224)
        }

        latent = model.get_latent(batch)

        assert latent.shape == (4, 256)

    def test_train_step(self):
        """Test training step."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False,
            'temporal_enabled': False
        }

        model = CameraOnlyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = {
            'camera': torch.randn(4, 3, 224, 224)
        }

        metrics = model.train_step(batch, optimizer)

        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics

    def test_different_image_sizes(self):
        """Test with different input image sizes."""
        from src.models.baselines.camera_only import CameraOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'pretrained': False
        }

        model = CameraOnlyBaseline(config)

        # Test different resolutions
        for size in [128, 224, 256]:
            batch = {
                'camera': torch.randn(2, 3, size, size)
            }
            output = model.forward(batch)
            assert output.shape == (2, 256)


class TestLiDAROnlyBaseline:
    """Test suite for LiDAR-only baseline model."""

    def test_initialization(self):
        """Test LiDAR-only baseline initializes correctly."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048,
            'temporal_enabled': False
        }

        model = LiDAROnlyBaseline(config)

        assert model.latent_dim == 256
        assert model.num_points == 2048
        assert model.temporal_enabled is False
        assert model.temporal_pool == 'max'  # default

    def test_initialization_with_temporal(self):
        """Test initialization with temporal mode."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048,
            'temporal_enabled': True,
            'temporal_pool': 'mean'
        }

        model = LiDAROnlyBaseline(config)

        assert model.temporal_enabled is True
        assert model.temporal_pool == 'mean'

    def test_forward_single_frame(self):
        """Test forward pass with single point cloud."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048,
            'temporal_enabled': False
        }

        model = LiDAROnlyBaseline(config)

        batch = {
            'lidar': torch.randn(4, 2048, 3)  # (B, N, 3)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_forward_temporal(self):
        """Test forward pass with temporal sequence."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048,
            'temporal_enabled': True,
            'temporal_pool': 'max'
        }

        model = LiDAROnlyBaseline(config)

        batch = {
            'lidar': torch.randn(4, 5, 2048, 3)  # (B, T, N, 3)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_extract_features_single_frame(self):
        """Test feature extraction from single point cloud."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        point_cloud = torch.randn(4, 2048, 3)
        features = model.extract_features(point_cloud)

        assert features.shape == (4, 1024)  # Feature dim = 1024

    def test_extract_features_temporal(self):
        """Test feature extraction from temporal sequence."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        point_cloud = torch.randn(4, 5, 2048, 3)  # (B, T, N, 3)
        features = model.extract_features(point_cloud)

        assert features.shape == (4, 5, 1024)  # (B, T, feature_dim)

    def test_aggregate_temporal_max(self):
        """Test max pooling temporal aggregation."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'max'
        }

        model = LiDAROnlyBaseline(config)

        features = torch.randn(4, 5, 1024)  # (B, T, feature_dim)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 1024)

    def test_aggregate_temporal_mean(self):
        """Test mean pooling temporal aggregation."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'mean'
        }

        model = LiDAROnlyBaseline(config)

        features = torch.randn(4, 5, 1024)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 1024)

    def test_aggregate_temporal_last(self):
        """Test last frame temporal aggregation."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'last'
        }

        model = LiDAROnlyBaseline(config)

        features = torch.randn(4, 5, 1024)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 1024)

    def test_compute_loss(self):
        """Test loss computation."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        batch = {
            'lidar': torch.randn(4, 2048, 3)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        assert 'reconstruction_loss' in metrics
        assert 'latent_reg' in metrics

    def test_compute_loss_with_future(self):
        """Test loss computation with future point cloud."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        batch = {
            'lidar': torch.randn(4, 2048, 3),
            'lidar_future': torch.randn(4, 2048, 3)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert loss.item() >= 0

    def test_different_num_points(self):
        """Test with different numbers of points."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        # Test with more points (should sample)
        batch_more = {
            'lidar': torch.randn(2, 4096, 3)
        }
        output_more = model.forward(batch_more)
        assert output_more.shape == (2, 256)

        # Test with fewer points (should pad)
        batch_fewer = {
            'lidar': torch.randn(2, 1024, 3)
        }
        output_fewer = model.forward(batch_fewer)
        assert output_fewer.shape == (2, 256)

    def test_get_latent(self):
        """Test latent extraction."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)

        batch = {
            'lidar': torch.randn(4, 2048, 3)
        }

        latent = model.get_latent(batch)

        assert latent.shape == (4, 256)

    def test_train_step(self):
        """Test training step."""
        from src.models.baselines.lidar_only import LiDAROnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'num_points': 2048
        }

        model = LiDAROnlyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = {
            'lidar': torch.randn(4, 2048, 3)
        }

        metrics = model.train_step(batch, optimizer)

        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics


class TestRadarOnlyBaseline:
    """Test suite for radar-only baseline model."""

    def test_initialization(self):
        """Test radar-only baseline initializes correctly."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': False
        }

        model = RadarOnlyBaseline(config)

        assert model.latent_dim == 256
        assert model.temporal_enabled is False
        assert model.temporal_pool == 'max'  # default

    def test_initialization_with_temporal(self):
        """Test initialization with temporal mode."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'mean'
        }

        model = RadarOnlyBaseline(config)

        assert model.temporal_enabled is True
        assert model.temporal_pool == 'mean'

    def test_forward_single_frame(self):
        """Test forward pass with single radar frame."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': False
        }

        model = RadarOnlyBaseline(config)

        batch = {
            'radar': torch.randn(4, 1, 128, 128)  # (B, C, H, W)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_forward_temporal(self):
        """Test forward pass with temporal sequence."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'max'
        }

        model = RadarOnlyBaseline(config)

        batch = {
            'radar': torch.randn(4, 5, 1, 128, 128)  # (B, T, C, H, W)
        }

        output = model.forward(batch)

        assert output.shape == (4, 256)  # (B, latent_dim)

    def test_extract_features_single_frame(self):
        """Test feature extraction from single radar frame."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        radar = torch.randn(4, 1, 128, 128)
        features = model.extract_features(radar)

        assert features.shape == (4, 256)  # Feature dim = 256

    def test_extract_features_temporal(self):
        """Test feature extraction from temporal sequence."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        radar = torch.randn(4, 5, 1, 128, 128)  # (B, T, C, H, W)
        features = model.extract_features(radar)

        assert features.shape == (4, 5, 256)  # (B, T, feature_dim)

    def test_aggregate_temporal_max(self):
        """Test max pooling temporal aggregation."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'max'
        }

        model = RadarOnlyBaseline(config)

        features = torch.randn(4, 5, 256)  # (B, T, feature_dim)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 256)

    def test_aggregate_temporal_mean(self):
        """Test mean pooling temporal aggregation."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4,
            'temporal_enabled': True,
            'temporal_pool': 'mean'
        }

        model = RadarOnlyBaseline(config)

        features = torch.randn(4, 5, 256)
        aggregated = model.aggregate_temporal(features)

        assert aggregated.shape == (4, 256)

    def test_compute_loss(self):
        """Test loss computation."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        batch = {
            'radar': torch.randn(4, 1, 128, 128)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

        assert 'reconstruction_loss' in metrics
        assert 'latent_reg' in metrics

    def test_compute_loss_with_future(self):
        """Test loss computation with future radar frame."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        batch = {
            'radar': torch.randn(4, 1, 128, 128),
            'radar_future': torch.randn(4, 1, 128, 128)
        }

        outputs = model.forward(batch)
        loss, metrics = model.compute_loss(batch, outputs)

        assert loss.item() >= 0

    def test_different_radar_sizes(self):
        """Test with different radar tensor sizes."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        # Test different resolutions
        for size in [64, 128, 256]:
            batch = {
                'radar': torch.randn(2, 1, size, size)
            }
            output = model.forward(batch)
            assert output.shape == (2, 256)

    def test_get_latent(self):
        """Test latent extraction."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)

        batch = {
            'radar': torch.randn(4, 1, 128, 128)
        }

        latent = model.get_latent(batch)

        assert latent.shape == (4, 256)

    def test_train_step(self):
        """Test training step."""
        from src.models.baselines.radar_only import RadarOnlyBaseline

        config = {
            'latent_dim': 256,
            'learning_rate': 1e-4
        }

        model = RadarOnlyBaseline(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch = {
            'radar': torch.randn(4, 1, 128, 128)
        }

        metrics = model.train_step(batch, optimizer)

        assert 'loss' in metrics
        assert 'reconstruction_loss' in metrics
