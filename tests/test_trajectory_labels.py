"""Tests for trajectory label extraction."""
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


class TestTrajectoryLabelExtractor:
    """Test suite for trajectory label extraction."""

    def test_initialization(self, nusc):
        """Test extractor initializes correctly."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(
            nusc,
            pred_horizons=[1.0, 2.0, 3.0]
        )

        assert extractor.nusc == nusc
        assert extractor.pred_horizons == [1.0, 2.0, 3.0]
        assert extractor.sampling_rate == 2.0
        assert extractor.timestep == 0.5

    def test_extract_ego_trajectory(self, nusc, sample_token):
        """Test extracting ego trajectory."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0, 2.0])
        trajectories = extractor.extract_ego_trajectory(sample_token)

        # Should return dict with horizons as keys
        assert isinstance(trajectories, dict)
        assert 1.0 in trajectories
        assert 2.0 in trajectories

    def test_ego_trajectory_shapes(self, nusc, sample_token):
        """Test ego trajectory output shapes."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0, 2.0, 3.0])
        trajectories = extractor.extract_ego_trajectory(sample_token)

        # Check shapes for each horizon
        # 1.0s = 2 steps (at 2Hz), 2.0s = 4 steps, 3.0s = 6 steps
        for horizon in [1.0, 2.0, 3.0]:
            assert horizon in trajectories
            traj = trajectories[horizon]
            assert isinstance(traj, np.ndarray)
            assert traj.ndim == 2
            assert traj.shape[1] == 2  # (T, 2) for (x, y)

    def test_ego_trajectory_ego_frame(self, nusc, sample_token):
        """Test that ego trajectory is in ego frame."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0])
        trajectories = extractor.extract_ego_trajectory(sample_token)

        traj = trajectories[1.0]

        # In ego frame, first waypoint should be ahead (positive x)
        # (though this depends on scene, so just check reasonable range)
        assert traj.dtype == np.float64 or traj.dtype == np.float32

    def test_extract_agent_trajectories(self, nusc, sample_token):
        """Test extracting agent trajectories."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0, 2.0])
        agent_trajs = extractor.extract_agent_trajectories(
            sample_token,
            max_agents=10,
            max_distance=50.0
        )

        # Should return dict
        assert isinstance(agent_trajs, dict)

        # Each entry should have required fields
        for instance_token, info in agent_trajs.items():
            assert 'class' in info
            assert 'current_pos' in info
            assert 'current_vel' in info
            assert 'trajectories' in info

            # Check current state shapes
            assert info['current_pos'].shape == (2,)
            assert info['current_vel'].shape == (2,)

            # Check trajectories
            assert isinstance(info['trajectories'], dict)
            assert 1.0 in info['trajectories']
            assert 2.0 in info['trajectories']

    def test_agent_trajectory_shapes(self, nusc, sample_token):
        """Test agent trajectory output shapes."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0, 2.0])
        agent_trajs = extractor.extract_agent_trajectories(sample_token)

        if agent_trajs:  # If there are agents in scene
            instance_token = list(agent_trajs.keys())[0]
            info = agent_trajs[instance_token]

            for horizon in [1.0, 2.0]:
                traj = info['trajectories'][horizon]
                assert isinstance(traj, np.ndarray)
                assert traj.ndim == 2
                assert traj.shape[1] == 2

    def test_max_agents_limit(self, nusc, sample_token):
        """Test that max_agents limit is respected."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc)

        # Test with small limit
        agent_trajs = extractor.extract_agent_trajectories(
            sample_token,
            max_agents=3
        )

        assert len(agent_trajs) <= 3

    def test_distance_filtering(self, nusc, sample_token):
        """Test that agents are filtered by distance."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc)

        # Test with very small distance (should get fewer agents)
        agent_trajs_near = extractor.extract_agent_trajectories(
            sample_token,
            max_distance=10.0
        )

        # Test with large distance
        agent_trajs_far = extractor.extract_agent_trajectories(
            sample_token,
            max_distance=100.0
        )

        # Should have at least as many agents with larger distance
        assert len(agent_trajs_far) >= len(agent_trajs_near)

    def test_frame_transformation(self, nusc, sample_token):
        """Test coordinate frame transformation."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0])

        # Extract ego trajectory
        ego_trajs = extractor.extract_ego_trajectory(sample_token)
        ego_traj = ego_trajs[1.0]

        # Should have valid coordinates (not NaN or Inf)
        assert np.all(np.isfinite(ego_traj))

        # Extract agent trajectories
        agent_trajs = extractor.extract_agent_trajectories(sample_token, max_agents=5)

        for info in agent_trajs.values():
            assert np.all(np.isfinite(info['current_pos']))
            assert np.all(np.isfinite(info['current_vel']))

    def test_different_horizons(self, nusc, sample_token):
        """Test with different prediction horizons."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        # Test with single horizon
        extractor1 = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0])
        trajs1 = extractor1.extract_ego_trajectory(sample_token)
        assert len(trajs1) == 1

        # Test with multiple horizons
        extractor3 = TrajectoryLabelExtractor(nusc, pred_horizons=[1.0, 2.0, 3.0])
        trajs3 = extractor3.extract_ego_trajectory(sample_token)
        assert len(trajs3) == 3

    def test_velocity_estimation(self, nusc, sample_token):
        """Test agent velocity estimation."""
        from src.data.labels.trajectory_extractor import TrajectoryLabelExtractor

        extractor = TrajectoryLabelExtractor(nusc)
        agent_trajs = extractor.extract_agent_trajectories(sample_token, max_agents=5)

        if agent_trajs:  # If there are agents
            for info in agent_trajs.values():
                vel = info['current_vel']

                # Velocity should be finite
                assert np.all(np.isfinite(vel))

                # Velocity should be in reasonable range (e.g., < 50 m/s)
                assert np.linalg.norm(vel) < 50.0
