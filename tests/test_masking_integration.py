import torch
import pytest
from src.masking.spatio_temporal_masking import SpatioTemporalMasking
from src.models.himac_jepa import HiMACJEPA


class TestMaskingIntegration:
    """Test suite for masking integration with HiMAC-JEPA."""

    @pytest.fixture
    def masker(self):
        """Create a masking module for testing."""
        return SpatioTemporalMasking(
            mask_ratio_spatial=0.75,
            mask_ratio_temporal=0.5,
            patch_size_camera=(16, 16),
            num_temporal_steps=5
        )

    @pytest.fixture
    def model_config(self):
        """Create a minimal model config for testing."""
        return {
            "model": {
                "latent_dim": 128,
                "action_encoder": {
                    "strategic_vocab_size": 10,
                    "tactical_dim": 3,
                    "latent_dim": 128,
                    "num_heads": 8,
                    "depth": 2,
                    "dropout": 0.1
                }
            },
            "trajectory_head": {"output_dim": 30},
            "motion_prediction_head": {"output_dim": 60},
            "bev_segmentation_head": {
                "bev_h": 20,
                "bev_w": 20,
                "num_classes": 5
            }
        }

    @pytest.fixture
    def model(self, model_config):
        """Create a model for testing."""
        return HiMACJEPA(model_config)

    def test_batch_mask_generation(self, masker):
        """Test that batch masks are generated correctly."""
        batch_size = 4
        camera_shape = (3, 224, 224)
        lidar_shape = (1024, 3)
        radar_shape = (1, 64, 64)

        masks = masker.generate_joint_mask(camera_shape, lidar_shape, radar_shape, batch_size)

        assert 'camera' in masks
        assert 'lidar' in masks
        assert 'radar' in masks
        assert 'temporal' in masks

        # Check shapes
        assert masks['camera'].shape[0] == batch_size
        assert masks['lidar'].shape == (batch_size, lidar_shape[0])
        assert masks['radar'].shape[0] == batch_size
        assert masks['temporal'].shape == (batch_size, 5)

    def test_mask_application_lidar(self, model):
        """Test that LiDAR masks are applied correctly."""
        batch_size = 2
        num_points = 1024

        lidar = torch.randn(batch_size, num_points, 3)
        mask = torch.zeros(batch_size, num_points, dtype=torch.bool)

        # Mask first 100 points in each batch
        mask[:, :100] = True

        masked_lidar = model.apply_lidar_mask(lidar, mask)

        # Check that masked points are zero
        assert torch.all(masked_lidar[:, :100] == 0.0)
        # Check that unmasked points are unchanged
        assert torch.allclose(masked_lidar[:, 100:], lidar[:, 100:])

    def test_forward_with_masks(self, model):
        """Test forward pass with masking enabled."""
        batch_size = 2

        camera = torch.randn(batch_size, 3, 224, 224)
        lidar = torch.randn(batch_size, 1024, 3)
        radar = torch.randn(batch_size, 1, 64, 64)
        strategic_action = torch.randint(0, 10, (batch_size,))
        tactical_action = torch.randn(batch_size, 3)

        # Create dummy masks
        masks = {
            'camera': torch.rand(batch_size, 14, 14) > 0.5,
            'lidar': torch.rand(batch_size, 1024) > 0.5,
            'radar': torch.rand(batch_size, 64, 64) > 0.5,
            'temporal': torch.rand(batch_size, 5) > 0.5
        }

        model.eval()
        with torch.no_grad():
            mu, log_var, trajectory, motion, bev = model(
                camera, lidar, radar, strategic_action, tactical_action, masks
            )

        # Check output shapes
        assert mu.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)

    def test_forward_without_masks(self, model):
        """Test forward pass without masking (downstream tasks)."""
        batch_size = 2

        camera = torch.randn(batch_size, 3, 224, 224)
        lidar = torch.randn(batch_size, 1024, 3)
        radar = torch.randn(batch_size, 1, 64, 64)
        strategic_action = torch.randint(0, 10, (batch_size,))
        tactical_action = torch.randn(batch_size, 3)

        model.eval()
        with torch.no_grad():
            mu, log_var, trajectory, motion, bev = model(
                camera, lidar, radar, strategic_action, tactical_action, None
            )

        # Check output shapes
        assert mu.shape == (batch_size, 128)
        assert log_var.shape == (batch_size, 128)

    def test_mask_ratio_affects_output(self, masker):
        """Test that different mask ratios produce different numbers of masked patches."""
        batch_size = 1
        camera_shape = (3, 224, 224)
        lidar_shape = (1024, 3)
        radar_shape = (1, 64, 64)

        # Test with high mask ratio
        masker_high = SpatioTemporalMasking(mask_ratio_spatial=0.9, mask_ratio_temporal=0.9)
        masks_high = masker_high.generate_joint_mask(camera_shape, lidar_shape, radar_shape, batch_size)

        # Test with low mask ratio
        masker_low = SpatioTemporalMasking(mask_ratio_spatial=0.1, mask_ratio_temporal=0.1)
        masks_low = masker_low.generate_joint_mask(camera_shape, lidar_shape, radar_shape, batch_size)

        # High ratio should mask more
        assert masks_high['lidar'].sum() > masks_low['lidar'].sum()

    def test_different_masks_per_batch(self, masker):
        """Test that each batch element gets a different mask."""
        batch_size = 4
        camera_shape = (3, 224, 224)
        lidar_shape = (1024, 3)
        radar_shape = (1, 64, 64)

        masks = masker.generate_joint_mask(camera_shape, lidar_shape, radar_shape, batch_size)

        # Check that masks are different for different batch elements
        # (statistically very unlikely to be identical)
        for i in range(batch_size - 1):
            assert not torch.all(masks['lidar'][i] == masks['lidar'][i + 1])

    def test_gradient_flow_with_masking(self, model):
        """Test that gradients flow correctly through masked forward pass."""
        model.train()
        batch_size = 2

        camera = torch.randn(batch_size, 3, 224, 224)
        lidar = torch.randn(batch_size, 1024, 3)
        radar = torch.randn(batch_size, 1, 64, 64)
        strategic_action = torch.randint(0, 10, (batch_size,))
        tactical_action = torch.randn(batch_size, 3)

        masks = {
            'camera': torch.rand(batch_size, 14, 14) > 0.5,
            'lidar': torch.rand(batch_size, 1024) > 0.5,
            'radar': torch.rand(batch_size, 64, 64) > 0.5,
            'temporal': torch.rand(batch_size, 5) > 0.5
        }

        mu, log_var, _, _, _ = model(camera, lidar, radar, strategic_action, tactical_action, masks)
        loss = mu.sum()
        loss.backward()

        # Check that gradients exist
        assert model.camera_encoder.proj.weight.grad is not None
        assert model.lidar_encoder.mlp[0].weight.grad is not None
        assert model.action_encoder.strategic_embedding.weight.grad is not None
