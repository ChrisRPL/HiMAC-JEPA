"""Action extraction utilities for nuScenes dataset."""
import numpy as np
from typing import Dict, Tuple, Optional


class ActionExtractor:
    """Extract hierarchical actions from nuScenes ego vehicle data."""

    def __init__(self, strategic_classes=10, tactical_dim=3):
        """
        Initialize action extractor.

        Args:
            strategic_classes: Number of strategic action classes
            tactical_dim: Dimension of tactical action vector
        """
        self.strategic_classes = strategic_classes
        self.tactical_dim = tactical_dim

        # Strategic action mapping
        self.strategic_map = {
            'keep_lane': 0,
            'change_lane_left': 1,
            'change_lane_right': 2,
            'turn_left': 3,
            'turn_right': 4,
            'stop': 5,
            'start': 6,
            'accelerate': 7,
            'decelerate': 8,
            'cruise': 9
        }

    def extract_strategic(self, ego_pose: Dict, next_ego_pose: Optional[Dict] = None) -> int:
        """
        Infer strategic action from trajectory analysis.

        Args:
            ego_pose: Current ego vehicle pose
            next_ego_pose: Next ego vehicle pose (optional, for trajectory analysis)

        Returns:
            Strategic action class index
        """
        # TODO: Analyze trajectory to classify strategic action
        # For now, return 'keep_lane' as placeholder
        # Future enhancement: analyze heading change, lateral displacement, etc.
        return self.strategic_map['keep_lane']

    def extract_tactical(self, can_bus_data: Dict) -> np.ndarray:
        """
        Extract and normalize tactical actions from CAN bus data.

        Args:
            can_bus_data: Dictionary containing CAN bus measurements

        Returns:
            Normalized tactical action vector [steering, accel, velocity]
        """
        # Extract raw values with defaults
        steering = can_bus_data.get('steering', 0.0)  # radians
        accel = can_bus_data.get('accel', 0.0)  # m/s^2
        velocity = can_bus_data.get('velocity', 0.0)  # m/s

        # Normalize to [-1, 1] range
        # Steering: ±π radians max
        steering_norm = np.clip(steering / np.pi, -1, 1)

        # Acceleration: ±5 m/s^2 max (typical for autonomous vehicles)
        accel_norm = np.clip(accel / 5.0, -1, 1)

        # Velocity: 30 m/s (~108 km/h) max
        velocity_norm = np.clip(velocity / 30.0, -1, 1)

        return np.array([steering_norm, accel_norm, velocity_norm], dtype=np.float32)

    def __call__(self, sample_data: Dict) -> Tuple[int, np.ndarray]:
        """
        Extract both strategic and tactical actions from sample.

        Args:
            sample_data: Dictionary containing 'ego_pose' and optional 'can_bus'

        Returns:
            Tuple of (strategic_action, tactical_action)
        """
        ego_pose = sample_data['ego_pose']
        can_bus = sample_data.get('can_bus', {})
        next_ego_pose = sample_data.get('next_ego_pose', None)

        strategic = self.extract_strategic(ego_pose, next_ego_pose)
        tactical = self.extract_tactical(can_bus)

        return strategic, tactical
