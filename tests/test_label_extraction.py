"""Tests for BEV and motion label extraction."""
import pytest
import torch
import numpy as np
from pathlib import Path


# Skip all tests if nuScenes dataset is not available
pytestmark = pytest.mark.skipif(
    not Path('/data/nuscenes/v1.0-mini').exists(),
    reason="nuScenes mini dataset not available at /data/nuscenes"
)


@pytest.fixture
def nusc():
    """Create nuScenes instance for testing."""
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version='v1.0-mini', dataroot='/data/nuscenes', verbose=False)


@pytest.fixture
def sample_token(nusc):
    """Get a sample token for testing."""
    return nusc.sample[0]['token']


class TestBEVLabelExtractor:
    """Test suite for BEV semantic segmentation label extraction."""

    def test_initialization(self, nusc):
        """Test BEV extractor initializes correctly."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        extractor = BEVLabelExtractor(
            nusc,
            bev_size=(200, 200),
            bev_range=50.0
        )

        assert extractor.nusc == nusc
        assert extractor.bev_size == (200, 200)
        assert extractor.bev_range == 50.0
        assert extractor.num_classes == 6
        assert extractor.resolution == (2 * 50.0) / 200

    def test_default_classes(self, nusc):
        """Test default class mapping."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        extractor = BEVLabelExtractor(nusc)

        expected_classes = {
            0: 'background',
            1: 'drivable_area',
            2: 'lane_divider',
            3: 'pedestrian_crossing',
            4: 'vehicle',
            5: 'pedestrian'
        }

        assert extractor.classes == expected_classes

    def test_extract_bev_labels(self, nusc, sample_token):
        """Test BEV label extraction."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        extractor = BEVLabelExtractor(nusc, bev_size=(200, 200))
        bev_mask = extractor.extract_bev_labels(sample_token)

        # Should return segmentation mask
        assert isinstance(bev_mask, np.ndarray)
        assert bev_mask.shape == (200, 200)
        assert bev_mask.dtype == np.uint8

    def test_bev_class_indices(self, nusc, sample_token):
        """Test that BEV mask contains valid class indices."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        extractor = BEVLabelExtractor(nusc)
        bev_mask = extractor.extract_bev_labels(sample_token)

        # All values should be valid class indices
        unique_classes = np.unique(bev_mask)
        assert np.all(unique_classes < 6)  # 6 classes (0-5)
        assert np.all(unique_classes >= 0)

    def test_different_bev_sizes(self, nusc, sample_token):
        """Test with different BEV sizes."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        for size in [(100, 100), (200, 200), (256, 256)]:
            extractor = BEVLabelExtractor(nusc, bev_size=size)
            bev_mask = extractor.extract_bev_labels(sample_token)

            assert bev_mask.shape == size

    def test_different_bev_ranges(self, nusc, sample_token):
        """Test with different BEV ranges."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        for bev_range in [25.0, 50.0, 75.0]:
            extractor = BEVLabelExtractor(nusc, bev_range=bev_range)
            bev_mask = extractor.extract_bev_labels(sample_token)

            # Should still return valid mask
            assert bev_mask.shape == (200, 200)
            assert bev_mask.dtype == np.uint8

    def test_world_to_pixel_conversion(self, nusc):
        """Test world to pixel coordinate conversion."""
        from src.data.labels.bev_extractor import BEVLabelExtractor

        extractor = BEVLabelExtractor(nusc, bev_size=(200, 200), bev_range=50.0)

        # Test center point (should map to center of image)
        pixel = extractor._world_to_pixel(np.array([0.0, 0.0]))
        assert pixel is not None
        assert abs(pixel[0] - 100) < 2  # Allow small rounding error
        assert abs(pixel[1] - 100) < 2

        # Test out of bounds point
        pixel_oob = extractor._world_to_pixel(np.array([100.0, 0.0]))
        # Should be out of bounds (>50m range)
        # Depending on implementation, might be None or clipped


class TestMotionLabelExtractor:
    """Test suite for motion prediction label extraction."""

    def test_initialization(self, nusc):
        """Test motion extractor initializes correctly."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(
            nusc,
            pred_horizon=3.0,
            max_distance=50.0,
            min_visibility=0.5
        )

        assert extractor.nusc == nusc
        assert extractor.pred_horizon == 3.0
        assert extractor.max_distance == 50.0
        assert extractor.min_visibility == 0.5
        assert extractor.sampling_rate == 2.0

    def test_extract_motion_labels(self, nusc, sample_token):
        """Test motion label extraction."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc, pred_horizon=3.0)
        labels = extractor.extract_motion_labels(sample_token)

        # Should return dict with required keys
        assert isinstance(labels, dict)
        assert 'agent_ids' in labels
        assert 'agent_classes' in labels
        assert 'current_states' in labels
        assert 'future_trajectories' in labels
        assert 'valid_masks' in labels

    def test_motion_label_shapes(self, nusc, sample_token):
        """Test motion label output shapes."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc, pred_horizon=3.0)
        labels = extractor.extract_motion_labels(sample_token)

        N = len(labels['agent_ids'])

        # Check shapes
        assert labels['current_states'].shape == (N, 4)  # [x, y, vx, vy]
        assert labels['future_trajectories'].ndim == 3
        assert labels['future_trajectories'].shape[0] == N
        assert labels['future_trajectories'].shape[2] == 2  # (x, y)
        assert labels['valid_masks'].shape[0] == N

    def test_current_states_format(self, nusc, sample_token):
        """Test current state format [x, y, vx, vy]."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc)
        labels = extractor.extract_motion_labels(sample_token)

        if len(labels['agent_ids']) > 0:
            states = labels['current_states']

            # Should be finite
            assert np.all(np.isfinite(states))

            # Position and velocity in reasonable range
            positions = states[:, :2]
            velocities = states[:, 2:4]

            # Positions should be within max_distance
            distances = np.linalg.norm(positions, axis=1)
            assert np.all(distances <= extractor.max_distance + 1.0)  # +1 for tolerance

    def test_future_trajectories_temporal_dim(self, nusc, sample_token):
        """Test future trajectories have correct temporal dimension."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc, pred_horizon=3.0)
        labels = extractor.extract_motion_labels(sample_token)

        if len(labels['agent_ids']) > 0:
            trajs = labels['future_trajectories']

            # Temporal dimension should match prediction horizon
            # 3.0s at 2Hz = 6 timesteps
            expected_T = int(3.0 / 0.5)
            assert trajs.shape[1] == expected_T

    def test_valid_masks(self, nusc, sample_token):
        """Test valid masks indicate agent presence."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc, pred_horizon=2.0)
        labels = extractor.extract_motion_labels(sample_token)

        if len(labels['agent_ids']) > 0:
            masks = labels['valid_masks']

            # Should be boolean
            assert masks.dtype == bool

            # Should have same shape as trajectories (N, T)
            assert masks.shape == labels['future_trajectories'].shape[:2]

    def test_distance_filtering(self, nusc, sample_token):
        """Test agents filtered by distance."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        # Extract with small distance
        extractor_near = MotionLabelExtractor(nusc, max_distance=20.0)
        labels_near = extractor_near.extract_motion_labels(sample_token)

        # Extract with large distance
        extractor_far = MotionLabelExtractor(nusc, max_distance=100.0)
        labels_far = extractor_far.extract_motion_labels(sample_token)

        # Should have at least as many agents with larger distance
        assert len(labels_far['agent_ids']) >= len(labels_near['agent_ids'])

    def test_agent_statistics(self, nusc, sample_token):
        """Test get_agent_statistics method."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc)
        labels = extractor.extract_motion_labels(sample_token)

        stats = extractor.get_agent_statistics(labels)

        # Should have required keys
        assert 'num_agents' in stats
        assert 'num_vehicles' in stats
        assert 'num_pedestrians' in stats
        assert 'avg_valid_timesteps' in stats
        assert 'avg_distance' in stats
        assert 'avg_speed' in stats

        # Values should be non-negative
        assert stats['num_agents'] >= 0
        assert stats['avg_valid_timesteps'] >= 0
        assert stats['avg_distance'] >= 0
        assert stats['avg_speed'] >= 0

    def test_empty_scene(self, nusc):
        """Test handling of scenes with no agents."""
        from src.data.labels.motion_extractor import MotionLabelExtractor

        extractor = MotionLabelExtractor(nusc, max_distance=1.0)  # Very small distance

        # Try to find a sample with no nearby agents
        for sample in nusc.sample[:5]:
            labels = extractor.extract_motion_labels(sample['token'])

            # Should return empty arrays with correct structure
            assert isinstance(labels['agent_ids'], list)
            assert isinstance(labels['current_states'], np.ndarray)
            assert labels['current_states'].shape == (0, 4)


class TestLabelCache:
    """Test suite for label caching system."""

    def test_initialization(self):
        """Test label cache initializes correctly."""
        from src.data.labels.label_cache import LabelCache

        cache = LabelCache(cache_dir='./test_cache')

        assert cache.cache_dir == Path('./test_cache')
        assert cache.cache_dir.exists()

        # Cleanup
        import shutil
        shutil.rmtree('./test_cache', ignore_errors=True)

    def test_save_and_load(self):
        """Test saving and loading labels."""
        from src.data.labels.label_cache import LabelCache

        cache = LabelCache(cache_dir='./test_cache')

        # Create test labels
        test_labels = {
            'trajectory': {'ego': np.array([[1.0, 2.0], [3.0, 4.0]])},
            'bev': np.zeros((100, 100)),
            'motion': {'agent_ids': ['agent1', 'agent2']}
        }

        # Save
        cache.save_labels('test_token', test_labels, split='train')

        # Load
        loaded = cache.load_labels('test_token', split='train')

        assert loaded is not None
        assert 'trajectory' in loaded
        assert 'bev' in loaded
        assert 'motion' in loaded

        # Cleanup
        import shutil
        shutil.rmtree('./test_cache', ignore_errors=True)

    def test_check_cache(self):
        """Test checking if labels are cached."""
        from src.data.labels.label_cache import LabelCache

        cache = LabelCache(cache_dir='./test_cache')

        # Not cached initially
        assert not cache.check_cache('test_token', 'train')

        # Save labels
        cache.save_labels('test_token', {'test': 'data'}, 'train')

        # Should be cached now
        assert cache.check_cache('test_token', 'train')

        # Cleanup
        import shutil
        shutil.rmtree('./test_cache', ignore_errors=True)

    def test_cache_stats(self):
        """Test getting cache statistics."""
        from src.data.labels.label_cache import LabelCache

        cache = LabelCache(cache_dir='./test_cache')

        # Save a few labels
        for i in range(3):
            cache.save_labels(f'token_{i}', {'test': i}, 'train')

        stats = cache.get_cache_stats('train')

        assert stats['num_cached'] == 3
        assert stats['total_size_mb'] > 0

        # Cleanup
        import shutil
        shutil.rmtree('./test_cache', ignore_errors=True)
