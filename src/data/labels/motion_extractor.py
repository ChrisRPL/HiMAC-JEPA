"""Motion prediction label extraction for multi-agent forecasting."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pyquaternion import Quaternion


class MotionLabelExtractor:
    """Extract motion prediction labels for agent forecasting."""

    def __init__(
        self,
        nusc,
        pred_horizon: float = 3.0,
        max_distance: float = 50.0,
        min_visibility: float = 0.5,
        sampling_rate: float = 2.0
    ):
        """
        Initialize motion label extractor.

        Args:
            nusc: NuScenes instance
            pred_horizon: Prediction horizon in seconds
            max_distance: Maximum distance from ego to track agents (meters)
            min_visibility: Minimum visibility score for agents (0-4 scale)
            sampling_rate: nuScenes sampling rate in Hz (default: 2.0 Hz)
        """
        self.nusc = nusc
        self.pred_horizon = pred_horizon
        self.max_distance = max_distance
        self.min_visibility = min_visibility
        self.sampling_rate = sampling_rate
        self.timestep = 1.0 / sampling_rate  # 0.5 seconds between samples

    def extract_motion_labels(self, sample_token: str) -> Dict:
        """
        Extract motion prediction labels for all trackable agents.

        Args:
            sample_token: Sample token to extract from

        Returns:
            Dictionary containing:
                - agent_ids: List of agent instance tokens
                - agent_classes: List of agent category names
                - current_states: (N, 4) array of [x, y, vx, vy] in ego frame
                - future_trajectories: (N, T, 2) array of future positions
                - valid_masks: (N, T) boolean array indicating valid timesteps
        """
        sample = self.nusc.get('sample', sample_token)

        # Get ego pose for frame transformation
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose_token = lidar_data['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)

        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])

        # Collect agents to track
        agents_to_track = self._get_trackable_agents(
            sample,
            ego_translation,
            ego_rotation
        )

        if not agents_to_track:
            # No agents to track, return empty arrays
            return {
                'agent_ids': [],
                'agent_classes': [],
                'current_states': np.zeros((0, 4)),
                'future_trajectories': np.zeros((0, 0, 2)),
                'valid_masks': np.zeros((0, 0), dtype=bool)
            }

        # Extract current states and future trajectories
        agent_ids = []
        agent_classes = []
        current_states = []
        future_trajectories = []
        valid_masks = []

        for ann in agents_to_track:
            instance_token = ann['instance_token']
            agent_ids.append(instance_token)
            agent_classes.append(ann['category_name'])

            # Current state
            current_state = self._get_current_state(
                ann,
                ego_translation,
                ego_rotation
            )
            current_states.append(current_state)

            # Future trajectory
            future_traj, valid_mask = self._get_future_trajectory(
                instance_token,
                sample_token,
                ego_translation,
                ego_rotation
            )
            future_trajectories.append(future_traj)
            valid_masks.append(valid_mask)

        # Convert to numpy arrays
        current_states = np.array(current_states)  # (N, 4)
        future_trajectories = np.array(future_trajectories)  # (N, T, 2)
        valid_masks = np.array(valid_masks)  # (N, T)

        return {
            'agent_ids': agent_ids,
            'agent_classes': agent_classes,
            'current_states': current_states,
            'future_trajectories': future_trajectories,
            'valid_masks': valid_masks
        }

    def _get_trackable_agents(
        self,
        sample: Dict,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> List[Dict]:
        """
        Get list of agents that should be tracked for motion prediction.

        Args:
            sample: Sample record
            ego_translation: Ego position
            ego_rotation: Ego rotation

        Returns:
            List of agent annotation dictionaries
        """
        trackable_agents = []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Filter by category (only vehicles and pedestrians)
            category = ann['category_name'].lower()
            if not ('vehicle' in category or 'pedestrian' in category or 'human' in category):
                continue

            # Filter by distance
            agent_pos = np.array(ann['translation'])
            distance = np.linalg.norm(agent_pos[:2] - ego_translation[:2])
            if distance > self.max_distance:
                continue

            # Filter by visibility
            visibility = ann.get('visibility_token', None)
            if visibility is not None:
                visibility_level = self.nusc.get('visibility', visibility)
                # nuScenes visibility: 0-4 scale (0=0-40%, 1=40-60%, 2=60-80%, 3=80-100%, 4=100%)
                if visibility_level['level'] < self.min_visibility:
                    continue

            trackable_agents.append(ann)

        # Sort by distance (closest first)
        trackable_agents.sort(
            key=lambda a: np.linalg.norm(
                np.array(a['translation'][:2]) - ego_translation[:2]
            )
        )

        return trackable_agents

    def _get_current_state(
        self,
        annotation: Dict,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Get current state of agent in ego frame.

        Args:
            annotation: Agent annotation
            ego_translation: Ego position
            ego_rotation: Ego rotation

        Returns:
            State vector [x, y, vx, vy] in ego frame
        """
        # Position
        global_pos = np.array(annotation['translation'][:2])
        ego_pos = self._transform_to_ego_frame(
            global_pos,
            ego_translation[:2],
            ego_rotation
        )

        # Velocity (estimate from next position)
        velocity = self._estimate_velocity(
            annotation,
            ego_rotation
        )

        return np.array([ego_pos[0], ego_pos[1], velocity[0], velocity[1]])

    def _estimate_velocity(
        self,
        annotation: Dict,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Estimate agent velocity from consecutive positions.

        Args:
            annotation: Current annotation
            ego_rotation: Ego rotation for frame transformation

        Returns:
            Velocity vector [vx, vy] in ego frame (m/s)
        """
        instance_token = annotation['instance_token']
        sample_token = annotation['sample_token']

        sample = self.nusc.get('sample', sample_token)
        next_token = sample.get('next', '')

        if not next_token:
            return np.zeros(2)

        # Find next annotation for this instance
        next_sample = self.nusc.get('sample', next_token)
        next_ann = None

        for ann_token in next_sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if ann['instance_token'] == instance_token:
                next_ann = ann
                break

        if next_ann is None:
            return np.zeros(2)

        # Calculate velocity in global frame
        current_pos = np.array(annotation['translation'][:2])
        next_pos = np.array(next_ann['translation'][:2])
        velocity_global = (next_pos - current_pos) / self.timestep

        # Transform to ego frame
        velocity_3d = np.array([velocity_global[0], velocity_global[1], 0.0])
        velocity_ego_3d = ego_rotation.inverse.rotate(velocity_3d)

        return velocity_ego_3d[:2]

    def _get_future_trajectory(
        self,
        instance_token: str,
        sample_token: str,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get future trajectory for an agent instance.

        Args:
            instance_token: Agent instance token
            sample_token: Starting sample token
            ego_translation: Ego position for frame transformation
            ego_rotation: Ego rotation for frame transformation

        Returns:
            Tuple of (trajectory, valid_mask)
                - trajectory: (T, 2) array of future positions
                - valid_mask: (T,) boolean array indicating valid timesteps
        """
        # Calculate number of future steps
        num_steps = int(self.pred_horizon / self.timestep)

        # Initialize arrays
        trajectory = np.zeros((num_steps, 2))
        valid_mask = np.zeros(num_steps, dtype=bool)

        # Traverse future samples
        current_token = sample_token

        for step in range(num_steps):
            sample = self.nusc.get('sample', current_token)
            next_token = sample.get('next', '')

            if not next_token:
                # No more future samples
                break

            # Find annotation for this instance in next sample
            next_sample = self.nusc.get('sample', next_token)
            found = False

            for ann_token in next_sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                if ann['instance_token'] == instance_token:
                    # Transform position to ego frame
                    global_pos = np.array(ann['translation'][:2])
                    ego_pos = self._transform_to_ego_frame(
                        global_pos,
                        ego_translation[:2],
                        ego_rotation
                    )

                    trajectory[step] = ego_pos
                    valid_mask[step] = True
                    found = True
                    break

            if not found:
                # Agent disappeared from scene
                break

            current_token = next_token

        return trajectory, valid_mask

    def _transform_to_ego_frame(
        self,
        global_pos: np.ndarray,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Transform position from global frame to ego frame.

        Args:
            global_pos: Position in global frame (2,)
            ego_translation: Ego position in global frame (2,)
            ego_rotation: Ego rotation quaternion

        Returns:
            Position in ego frame (2,)
        """
        # Translate to ego origin
        relative_pos = global_pos - ego_translation

        # Rotate to ego frame (inverse rotation)
        relative_pos_3d = np.array([relative_pos[0], relative_pos[1], 0.0])
        rotated_3d = ego_rotation.inverse.rotate(relative_pos_3d)

        return rotated_3d[:2]

    def get_agent_statistics(self, labels: Dict) -> Dict:
        """
        Get statistics about extracted motion labels.

        Args:
            labels: Motion labels dictionary

        Returns:
            Dictionary with statistics
        """
        num_agents = len(labels['agent_ids'])

        if num_agents == 0:
            return {
                'num_agents': 0,
                'num_vehicles': 0,
                'num_pedestrians': 0,
                'avg_valid_timesteps': 0.0,
                'avg_distance': 0.0,
                'avg_speed': 0.0
            }

        # Count by category
        num_vehicles = sum(
            1 for c in labels['agent_classes']
            if 'vehicle' in c.lower()
        )
        num_pedestrians = num_agents - num_vehicles

        # Valid timesteps
        valid_counts = labels['valid_masks'].sum(axis=1)
        avg_valid_timesteps = valid_counts.mean() if num_agents > 0 else 0.0

        # Distance from ego
        distances = np.linalg.norm(labels['current_states'][:, :2], axis=1)
        avg_distance = distances.mean() if num_agents > 0 else 0.0

        # Speed
        speeds = np.linalg.norm(labels['current_states'][:, 2:4], axis=1)
        avg_speed = speeds.mean() if num_agents > 0 else 0.0

        return {
            'num_agents': num_agents,
            'num_vehicles': num_vehicles,
            'num_pedestrians': num_pedestrians,
            'avg_valid_timesteps': float(avg_valid_timesteps),
            'avg_distance': float(avg_distance),
            'avg_speed': float(avg_speed)
        }
