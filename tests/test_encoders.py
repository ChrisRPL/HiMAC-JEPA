import torch
import pytest
from src.models.himac_jepa import CameraEncoder, LiDAREncoder, RadarEncoder


class TestCameraEncoder:
    """Test suite for CameraEncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create a camera encoder with default parameters."""
        return CameraEncoder(embed_dim=768, patch_size=16, depth=12, num_heads=12)

    def test_initialization(self, encoder):
        """Test that the encoder initializes correctly."""
        assert encoder.embed_dim == 768
        assert encoder.patch_size == 16
        assert encoder.depth == 12
        assert len(encoder.blocks) == 12  # Verify 12 layers created

    def test_forward_shape(self, encoder):
        """Test forward pass output shape."""
        batch_size = 4
        camera_input = torch.randn(batch_size, 3, 224, 224)

        encoder.eval()
        with torch.no_grad():
            output = encoder(camera_input)

        assert output.shape == (batch_size, 768), f"Expected shape ({batch_size}, 768), got {output.shape}"

    def test_configurable_depth(self):
        """Test that depth parameter works correctly."""
        for depth in [4, 8, 12, 16]:
            encoder = CameraEncoder(depth=depth)
            assert len(encoder.blocks) == depth

    def test_configurable_embed_dim(self):
        """Test that embed_dim parameter works correctly."""
        for embed_dim in [256, 512, 768, 1024]:
            encoder = CameraEncoder(embed_dim=embed_dim)
            batch_size = 2
            camera_input = torch.randn(batch_size, 3, 224, 224)

            encoder.eval()
            with torch.no_grad():
                output = encoder(camera_input)

            assert output.shape == (batch_size, embed_dim)

    def test_gradient_flow(self, encoder):
        """Test that gradients flow correctly through the encoder."""
        encoder.train()
        batch_size = 2
        camera_input = torch.randn(batch_size, 3, 224, 224)

        output = encoder(camera_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        assert encoder.proj.weight.grad is not None
        assert encoder.cls_token.grad is not None
        assert encoder.norm.weight.grad is not None

    def test_output_not_nan(self, encoder):
        """Test that output does not contain NaN values."""
        encoder.eval()
        batch_size = 2
        camera_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = encoder(camera_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_independence(self, encoder):
        """Test that batch elements are processed independently."""
        encoder.eval()
        camera_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output_single = encoder(camera_input)
            output_batch = encoder(camera_input.repeat(2, 1, 1, 1))

        # First element of batch should match single input
        assert torch.allclose(output_single[0], output_batch[0], atol=1e-5)


class TestLiDAREncoder:
    """Test suite for LiDAREncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create a LiDAR encoder with default parameters."""
        return LiDAREncoder(out_channels=512, dropout=0.1)

    def test_initialization(self, encoder):
        """Test that the encoder initializes correctly."""
        assert encoder.out_channels == 512
        assert isinstance(encoder.mlp1, torch.nn.Sequential)
        assert isinstance(encoder.mlp_global, torch.nn.Sequential)

    def test_forward_shape(self, encoder):
        """Test forward pass output shape."""
        batch_size = 4
        num_points = 1024
        lidar_input = torch.randn(batch_size, num_points, 3)

        encoder.eval()
        with torch.no_grad():
            output = encoder(lidar_input)

        assert output.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {output.shape}"

    def test_different_num_points(self, encoder):
        """Test encoder with different numbers of points."""
        encoder.eval()
        batch_size = 2

        for num_points in [256, 512, 1024, 2048]:
            lidar_input = torch.randn(batch_size, num_points, 3)

            with torch.no_grad():
                output = encoder(lidar_input)

            assert output.shape == (batch_size, 512)

    def test_configurable_output_channels(self):
        """Test that out_channels parameter works correctly."""
        for out_channels in [256, 512, 1024]:
            encoder = LiDAREncoder(out_channels=out_channels)
            batch_size = 2
            lidar_input = torch.randn(batch_size, 1024, 3)

            encoder.eval()
            with torch.no_grad():
                output = encoder(lidar_input)

            assert output.shape == (batch_size, out_channels)

    def test_gradient_flow(self, encoder):
        """Test that gradients flow correctly through the encoder."""
        encoder.train()
        batch_size = 2
        lidar_input = torch.randn(batch_size, 1024, 3)

        output = encoder(lidar_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        assert encoder.mlp1[0].weight.grad is not None  # First Linear layer
        assert encoder.mlp_global[0].weight.grad is not None

    def test_output_not_nan(self, encoder):
        """Test that output does not contain NaN values."""
        encoder.eval()
        batch_size = 2
        lidar_input = torch.randn(batch_size, 1024, 3)

        with torch.no_grad():
            output = encoder(lidar_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_max_pooling_invariance(self, encoder):
        """Test that max pooling provides permutation invariance to some degree."""
        encoder.eval()
        batch_size = 1
        num_points = 100

        # Create point cloud
        lidar_input = torch.randn(batch_size, num_points, 3)

        # Permute points
        perm_idx = torch.randperm(num_points)
        lidar_permuted = lidar_input[:, perm_idx, :]

        with torch.no_grad():
            output1 = encoder(lidar_input)
            output2 = encoder(lidar_permuted)

        # Outputs should be similar (not exact due to batch norm in training mode)
        assert torch.allclose(output1, output2, atol=1e-3)


class TestRadarEncoder:
    """Test suite for RadarEncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create a radar encoder with default parameters."""
        return RadarEncoder(out_channels=256, input_channels=1)

    def test_initialization(self, encoder):
        """Test that the encoder initializes correctly."""
        assert encoder.out_channels == 256
        assert encoder.input_channels == 1
        assert isinstance(encoder.spatial_branch, torch.nn.Sequential)

    def test_forward_shape(self, encoder):
        """Test forward pass output shape."""
        batch_size = 4
        radar_input = torch.randn(batch_size, 1, 64, 64)

        encoder.eval()
        with torch.no_grad():
            output = encoder(radar_input)

        assert output.shape == (batch_size, 256), f"Expected shape ({batch_size}, 256), got {output.shape}"

    def test_multi_channel_input(self):
        """Test encoder with multiple input channels (velocity-aware)."""
        encoder = RadarEncoder(out_channels=256, input_channels=4)
        batch_size = 2
        # 4 channels: range, doppler, azimuth, velocity
        radar_input = torch.randn(batch_size, 4, 64, 64)

        encoder.eval()
        with torch.no_grad():
            output = encoder(radar_input)

        assert output.shape == (batch_size, 256)

    def test_configurable_output_channels(self):
        """Test that out_channels parameter works correctly."""
        for out_channels in [128, 256, 512]:
            encoder = RadarEncoder(out_channels=out_channels)
            batch_size = 2
            radar_input = torch.randn(batch_size, 1, 64, 64)

            encoder.eval()
            with torch.no_grad():
                output = encoder(radar_input)

            assert output.shape == (batch_size, out_channels)

    def test_different_spatial_sizes(self, encoder):
        """Test encoder with different spatial input sizes."""
        encoder.eval()
        batch_size = 2

        for size in [32, 64, 128]:
            radar_input = torch.randn(batch_size, 1, size, size)

            with torch.no_grad():
                output = encoder(radar_input)

            assert output.shape == (batch_size, 256)

    def test_gradient_flow(self, encoder):
        """Test that gradients flow correctly through the encoder."""
        encoder.train()
        batch_size = 2
        radar_input = torch.randn(batch_size, 1, 64, 64)

        output = encoder(radar_input)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for key parameters
        assert encoder.spatial_branch[0].weight.grad is not None  # First conv layer
        assert encoder.fc[0].weight.grad is not None  # First fc layer

    def test_output_not_nan(self, encoder):
        """Test that output does not contain NaN values."""
        encoder.eval()
        batch_size = 2
        radar_input = torch.randn(batch_size, 1, 64, 64)

        with torch.no_grad():
            output = encoder(radar_input)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_batch_independence(self, encoder):
        """Test that batch elements are processed independently."""
        encoder.eval()
        radar_input = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output_single = encoder(radar_input)
            output_batch = encoder(radar_input.repeat(2, 1, 1, 1))

        # First element of batch should match single input
        assert torch.allclose(output_single[0], output_batch[0], atol=1e-5)
