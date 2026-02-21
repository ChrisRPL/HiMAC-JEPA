"""Tests for nuScenes dataset loader and preprocessing."""
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf


# Skip all tests if nuScenes dataset is not available
pytestmark = pytest.mark.skipif(
    not Path('/data/nuscenes/v1.0-mini').exists(),
    reason="nuScenes mini dataset not available at /data/nuscenes"
)


@pytest.fixture
def nuscenes_config():
    """Create test configuration for nuScenes dataset."""
    return OmegaConf.create({
        'dataset': 'nuscenes',
        'data_root': '/data/nuscenes',
        'version': 'v1.0-mini',
        'batch_size': 2,
        'num_workers': 0,  # Single-threaded for tests
        'image_size': [224, 224],
        'num_points': 1024,
        'radar_size': [64, 64],
        'normalize_images': True,
        'imagenet_mean': [0.485, 0.456, 0.406],
        'imagenet_std': [0.229, 0.224, 0.225],
        'augmentation': {
            'enabled': False,  # Disable for deterministic tests
            'random_crop': False,
            'color_jitter': False,
            'horizontal_flip': False
        },
        'strategic_action_classes': 10,
        'tactical_action_dim': 3,
        'action_normalization': True,
        'cameras': ['CAM_FRONT'],
        'lidar_sensors': ['LIDAR_TOP'],
        'radar_sensors': ['RADAR_FRONT'],
        'sequence_length': 1
    })


class TestNuScenesDataset:
    """Test suite for nuScenes dataset loader."""

    def test_dataset_initialization(self, nuscenes_config):
        """Test that dataset initializes without errors."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        assert len(dataset) > 0, "Dataset should have samples"
        assert hasattr(dataset, 'nusc'), "Dataset should have nuScenes instance"

    def test_sample_loading(self, nuscenes_config):
        """Test loading a single sample."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        # Check all required keys are present
        assert 'camera' in sample, "Sample should contain camera data"
        assert 'lidar' in sample, "Sample should contain lidar data"
        assert 'radar' in sample, "Sample should contain radar data"
        assert 'strategic_action' in sample, "Sample should contain strategic action"
        assert 'tactical_action' in sample, "Sample should contain tactical action"

    def test_camera_shape(self, nuscenes_config):
        """Test that camera images have correct shape."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        camera = sample['camera']
        assert camera.shape == (3, 224, 224), \
            f"Camera shape should be (3, 224, 224), got {camera.shape}"
        assert camera.dtype == torch.float32, "Camera should be float32 tensor"

    def test_lidar_shape(self, nuscenes_config):
        """Test that LiDAR point clouds have correct shape."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        lidar = sample['lidar']
        assert lidar.shape == (1024, 3), \
            f"LiDAR shape should be (1024, 3), got {lidar.shape}"
        assert lidar.dtype == torch.float32, "LiDAR should be float32 tensor"

    def test_radar_shape(self, nuscenes_config):
        """Test that radar data has correct shape."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        radar = sample['radar']
        assert radar.shape == (1, 64, 64), \
            f"Radar shape should be (1, 64, 64), got {radar.shape}"
        assert radar.dtype == torch.float32, "Radar should be float32 tensor"

    def test_strategic_action(self, nuscenes_config):
        """Test strategic action extraction and range."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        strategic = sample['strategic_action']
        assert strategic.dtype == torch.long, "Strategic action should be long tensor"
        assert strategic.dim() == 0, "Strategic action should be scalar"
        assert 0 <= strategic < 10, \
            f"Strategic action should be in [0, 10), got {strategic}"

    def test_tactical_action(self, nuscenes_config):
        """Test tactical action extraction and normalization."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        sample = dataset[0]

        tactical = sample['tactical_action']
        assert tactical.shape == (3,), f"Tactical action shape should be (3,), got {tactical.shape}"
        assert tactical.dtype == torch.float32, "Tactical action should be float32 tensor"
        assert torch.all(tactical >= -1), "Tactical action should be >= -1"
        assert torch.all(tactical <= 1), "Tactical action should be <= 1"

    def test_multiple_samples(self, nuscenes_config):
        """Test loading multiple samples."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')

        # Load first 3 samples
        num_samples = min(3, len(dataset))
        for idx in range(num_samples):
            sample = dataset[idx]
            assert 'camera' in sample
            assert 'lidar' in sample
            assert 'radar' in sample

    def test_data_loader_integration(self, nuscenes_config):
        """Test that dataset works with PyTorch DataLoader."""
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset
        from torch.utils.data import DataLoader

        dataset = NuScenesMultiModalDataset(nuscenes_config, split='train')
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        # Get one batch
        batch = next(iter(dataloader))

        # Check batch shapes
        assert batch['camera'].shape[0] == 2, "Batch size should be 2"
        assert batch['lidar'].shape[0] == 2, "Batch size should be 2"
        assert batch['radar'].shape[0] == 2, "Batch size should be 2"
        assert batch['strategic_action'].shape[0] == 2, "Batch size should be 2"
        assert batch['tactical_action'].shape[0] == 2, "Batch size should be 2"


class TestActionExtractor:
    """Test suite for action extraction utilities."""

    def test_action_extractor_initialization(self):
        """Test ActionExtractor initializes correctly."""
        from src.data.action_extraction import ActionExtractor

        extractor = ActionExtractor(strategic_classes=10, tactical_dim=3)
        assert extractor.strategic_classes == 10
        assert extractor.tactical_dim == 3
        assert len(extractor.strategic_map) == 10

    def test_tactical_action_normalization(self):
        """Test tactical action normalization."""
        from src.data.action_extraction import ActionExtractor

        extractor = ActionExtractor()

        # Test with sample CAN bus data
        can_bus = {
            'steering': 0.5,  # radians
            'accel': 2.5,     # m/s^2
            'velocity': 15.0  # m/s
        }

        tactical = extractor.extract_tactical(can_bus)

        assert tactical.shape == (3,)
        assert np.all(tactical >= -1) and np.all(tactical <= 1), \
            "Tactical actions should be normalized to [-1, 1]"

    def test_tactical_action_zero_defaults(self):
        """Test that missing CAN bus data defaults to zero."""
        from src.data.action_extraction import ActionExtractor

        extractor = ActionExtractor()

        # Empty CAN bus data
        tactical = extractor.extract_tactical({})

        assert tactical.shape == (3,)
        assert np.allclose(tactical, [0, 0, 0]), \
            "Missing CAN bus data should default to zeros"


class TestPreprocessors:
    """Test suite for preprocessing utilities."""

    def test_camera_preprocessor(self, nuscenes_config):
        """Test camera preprocessing."""
        from src.data.preprocessing import CameraPreprocessor
        from PIL import Image

        preprocessor = CameraPreprocessor(nuscenes_config)

        # Create a dummy image
        dummy_img = Image.new('RGB', (800, 600), color=(128, 128, 128))
        dummy_path = '/tmp/test_camera.jpg'
        dummy_img.save(dummy_path)

        # Preprocess
        result = preprocessor(dummy_path, augment=False)

        assert result.shape == (3, 224, 224), "Should resize to (3, 224, 224)"
        assert result.dtype == torch.float32

    def test_lidar_preprocessor(self, nuscenes_config):
        """Test LiDAR preprocessing."""
        from src.data.preprocessing import LiDARPreprocessor

        preprocessor = LiDARPreprocessor(nuscenes_config)

        # Create dummy point cloud with 2000 points
        dummy_pc = np.random.randn(2000, 5).astype(np.float32) * 10  # x,y,z,i,r

        # Preprocess
        result = preprocessor(dummy_pc)

        assert result.shape == (1024, 3), "Should sample to (1024, 3)"
        assert result.dtype == torch.float32

    def test_radar_preprocessor(self, nuscenes_config):
        """Test radar preprocessing."""
        from src.data.preprocessing import RadarPreprocessor

        preprocessor = RadarPreprocessor(nuscenes_config)

        # Create dummy radar point cloud
        dummy_radar = np.random.randn(100, 18).astype(np.float32)

        # Preprocess
        result = preprocessor(dummy_radar)

        assert result.shape == (1, 64, 64), "Should convert to (1, 64, 64) image"
        assert result.dtype == torch.float32
