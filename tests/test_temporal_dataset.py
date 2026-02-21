"""Tests for temporal sequence dataset and utilities."""
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
def temporal_config():
    """Create test configuration for temporal nuScenes dataset."""
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
            'enabled': False,
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
        'temporal': {
            'enabled': True,
            'seq_length': 3,  # Shorter for tests
            'pred_horizon': 2,
            'frame_skip': 1,
            'min_scene_length': 10
        }
    })


@pytest.fixture
def base_dataset(temporal_config):
    """Create base nuScenes dataset."""
    from src.data.nuscenes_dataset import NuScenesMultiModalDataset
    return NuScenesMultiModalDataset(temporal_config, split='train')


class TestTemporalSequenceBuilder:
    """Test suite for temporal sequence builder utilities."""

    def test_builder_initialization(self, base_dataset, temporal_config):
        """Test that sequence builder initializes correctly."""
        from src.data.temporal_utils import TemporalSequenceBuilder

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=3,
            pred_horizon=2,
            frame_skip=1
        )

        assert builder.seq_length == 3
        assert builder.pred_horizon == 2
        assert builder.frame_skip == 1
        assert builder.total_length == 5

    def test_get_scene_samples(self, base_dataset):
        """Test getting samples from a scene."""
        from src.data.temporal_utils import TemporalSequenceBuilder

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=3,
            pred_horizon=2
        )

        # Get first scene
        scene = base_dataset.nusc.scene[0]
        samples = builder.get_scene_samples(scene['token'])

        assert len(samples) > 0, "Scene should have samples"
        assert all(isinstance(token, str) for token in samples), \
            "All sample tokens should be strings"

    def test_build_sequences(self, base_dataset):
        """Test building temporal sequences."""
        from src.data.temporal_utils import TemporalSequenceBuilder

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=3,
            pred_horizon=2,
            frame_skip=1
        )

        sequences = builder.build_sequences(split='train')

        assert len(sequences) > 0, "Should build at least one sequence"

        # Check sequence structure
        seq = sequences[0]
        assert 'context_tokens' in seq
        assert 'target_tokens' in seq
        assert 'scene_token' in seq
        assert 'scene_name' in seq

        # Check lengths
        assert len(seq['context_tokens']) == 3, "Context should have 3 frames"
        assert len(seq['target_tokens']) == 2, "Target should have 2 frames"

    def test_sequence_validation(self, base_dataset):
        """Test sequence continuity validation."""
        from src.data.temporal_utils import TemporalSequenceBuilder

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=3,
            pred_horizon=2
        )

        # Get continuous sequence
        scene = base_dataset.nusc.scene[0]
        samples = builder.get_scene_samples(scene['token'])

        if len(samples) >= 5:
            continuous = samples[:5]
            assert builder.validate_sequence(continuous), \
                "Continuous sequence should be valid"

    def test_frame_skip(self, base_dataset):
        """Test frame skipping functionality."""
        from src.data.temporal_utils import TemporalSequenceBuilder

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=2,
            pred_horizon=1,
            frame_skip=2  # Skip every other frame
        )

        sequences = builder.build_sequences(split='train')

        # Frame skip should reduce number of sequences but maintain validity
        assert len(sequences) > 0, "Should still build sequences with frame skip"


class TestTemporalNuScenesDataset:
    """Test suite for temporal sequence dataset wrapper."""

    def test_dataset_initialization(self, base_dataset, temporal_config):
        """Test that temporal dataset initializes correctly."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        assert len(dataset) > 0, "Dataset should have sequences"
        assert dataset.seq_length == 3
        assert dataset.pred_horizon == 2
        assert hasattr(dataset, 'sequences')
        assert hasattr(dataset, 'token_to_idx')

    def test_sequence_loading(self, base_dataset, temporal_config):
        """Test loading a single temporal sequence."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]

        # Check top-level keys
        assert 'context' in sequence
        assert 'target' in sequence
        assert 'scene_name' in sequence
        assert 'scene_token' in sequence

    def test_context_data_structure(self, base_dataset, temporal_config):
        """Test context data has correct structure and shapes."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]
        context = sequence['context']

        # Check all required keys
        assert 'camera' in context
        assert 'lidar' in context
        assert 'radar' in context
        assert 'strategic_action' in context
        assert 'tactical_action' in context

        # Check temporal dimensions (T, ...)
        assert context['camera'].shape[0] == 3, "Should have 3 context frames"
        assert context['lidar'].shape[0] == 3
        assert context['radar'].shape[0] == 3
        assert context['strategic_action'].shape[0] == 3
        assert context['tactical_action'].shape[0] == 3

    def test_target_data_structure(self, base_dataset, temporal_config):
        """Test target data has correct structure and shapes."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]
        target = sequence['target']

        # Check all required keys
        assert 'camera' in target
        assert 'lidar' in target
        assert 'radar' in target
        assert 'strategic_action' in target
        assert 'tactical_action' in target

        # Check temporal dimensions (T, ...)
        assert target['camera'].shape[0] == 2, "Should have 2 target frames"
        assert target['lidar'].shape[0] == 2
        assert target['radar'].shape[0] == 2
        assert target['strategic_action'].shape[0] == 2
        assert target['tactical_action'].shape[0] == 2

    def test_camera_temporal_shape(self, base_dataset, temporal_config):
        """Test camera data has correct temporal shape."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]

        # Context camera: (T=3, C=3, H=224, W=224)
        context_camera = sequence['context']['camera']
        assert context_camera.shape == (3, 3, 224, 224), \
            f"Context camera shape should be (3, 3, 224, 224), got {context_camera.shape}"
        assert context_camera.dtype == torch.float32

        # Target camera: (T=2, C=3, H=224, W=224)
        target_camera = sequence['target']['camera']
        assert target_camera.shape == (2, 3, 224, 224), \
            f"Target camera shape should be (2, 3, 224, 224), got {target_camera.shape}"
        assert target_camera.dtype == torch.float32

    def test_lidar_temporal_shape(self, base_dataset, temporal_config):
        """Test LiDAR data has correct temporal shape."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]

        # Context LiDAR: (T=3, N=1024, D=3)
        context_lidar = sequence['context']['lidar']
        assert context_lidar.shape == (3, 1024, 3), \
            f"Context LiDAR shape should be (3, 1024, 3), got {context_lidar.shape}"
        assert context_lidar.dtype == torch.float32

        # Target LiDAR: (T=2, N=1024, D=3)
        target_lidar = sequence['target']['lidar']
        assert target_lidar.shape == (2, 1024, 3), \
            f"Target LiDAR shape should be (2, 1024, 3), got {target_lidar.shape}"
        assert target_lidar.dtype == torch.float32

    def test_radar_temporal_shape(self, base_dataset, temporal_config):
        """Test radar data has correct temporal shape."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]

        # Context radar: (T=3, C=1, H=64, W=64)
        context_radar = sequence['context']['radar']
        assert context_radar.shape == (3, 1, 64, 64), \
            f"Context radar shape should be (3, 1, 64, 64), got {context_radar.shape}"
        assert context_radar.dtype == torch.float32

        # Target radar: (T=2, C=1, H=64, W=64)
        target_radar = sequence['target']['radar']
        assert target_radar.shape == (2, 1, 64, 64), \
            f"Target radar shape should be (2, 1, 64, 64), got {target_radar.shape}"
        assert target_radar.dtype == torch.float32

    def test_action_temporal_shapes(self, base_dataset, temporal_config):
        """Test action data has correct temporal shapes."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        sequence = dataset[0]

        # Strategic actions: (T,)
        context_strategic = sequence['context']['strategic_action']
        assert context_strategic.shape == (3,), \
            f"Context strategic shape should be (3,), got {context_strategic.shape}"
        assert context_strategic.dtype == torch.long

        target_strategic = sequence['target']['strategic_action']
        assert target_strategic.shape == (2,), \
            f"Target strategic shape should be (2,), got {target_strategic.shape}"

        # Tactical actions: (T, 3)
        context_tactical = sequence['context']['tactical_action']
        assert context_tactical.shape == (3, 3), \
            f"Context tactical shape should be (3, 3), got {context_tactical.shape}"
        assert context_tactical.dtype == torch.float32

        target_tactical = sequence['target']['tactical_action']
        assert target_tactical.shape == (2, 3), \
            f"Target tactical shape should be (2, 3), got {target_tactical.shape}"

    def test_multiple_sequences(self, base_dataset, temporal_config):
        """Test loading multiple sequences."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        # Load first 3 sequences
        num_sequences = min(3, len(dataset))
        for idx in range(num_sequences):
            sequence = dataset[idx]
            assert 'context' in sequence
            assert 'target' in sequence
            assert sequence['context']['camera'].shape[0] == 3
            assert sequence['target']['camera'].shape[0] == 2

    def test_dataset_statistics(self, base_dataset, temporal_config):
        """Test dataset statistics method."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        stats = dataset.get_statistics()

        assert 'num_sequences' in stats
        assert 'seq_length' in stats
        assert 'pred_horizon' in stats
        assert 'frame_skip' in stats
        assert 'num_scenes' in stats
        assert 'sequences_per_scene' in stats

        assert stats['seq_length'] == 3
        assert stats['pred_horizon'] == 2
        assert stats['num_sequences'] == len(dataset)

    def test_token_index_mapping(self, base_dataset, temporal_config):
        """Test that token to index mapping is built correctly."""
        from src.data.temporal_dataset import TemporalNuScenesDataset

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        # Check that token_to_idx has entries
        assert len(dataset.token_to_idx) > 0, "Should have token mappings"

        # Verify mapping works
        first_sample = base_dataset.samples[0]
        assert first_sample['token'] in dataset.token_to_idx
        assert dataset.token_to_idx[first_sample['token']] == 0

    def test_data_loader_integration(self, base_dataset, temporal_config):
        """Test that temporal dataset works with PyTorch DataLoader."""
        from src.data.temporal_dataset import TemporalNuScenesDataset
        from torch.utils.data import DataLoader

        dataset = TemporalNuScenesDataset(
            base_dataset=base_dataset,
            config=temporal_config.temporal,
            split='train'
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        # Get one batch
        batch = next(iter(dataloader))

        # Check batch structure
        assert 'context' in batch
        assert 'target' in batch

        # Check context batch shapes: (B, T, ...)
        assert batch['context']['camera'].shape[0] == 2, "Batch size should be 2"
        assert batch['context']['camera'].shape[1] == 3, "Should have 3 context frames"

        # Check target batch shapes: (B, T, ...)
        assert batch['target']['camera'].shape[0] == 2
        assert batch['target']['camera'].shape[1] == 2, "Should have 2 target frames"
