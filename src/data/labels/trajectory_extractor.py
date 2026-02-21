"""Trajectory label extraction for future waypoint prediction."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from pyquaternion import Quaternion


class TrajectoryLabelExtractor:
    """Extract future trajectory waypoints from nuScenes data."""

    def __init__(self, nusc, pred_horizons=[1.0, 2.0, 3.0], sampling_rate=2.0):
        """
        Initialize trajectory label extractor.

        Args:
            nusc: NuScenes instance
            pred_horizons: List of prediction horizons in seconds (e.g., [1.0, 2.0, 3.0])
            sampling_rate: nuScenes sampling rate in Hz (default: 2.0 Hz)
        """
        self.nusc = nusc
        self.pred_horizons = pred_horizons
        self.sampling_rate = sampling_rate
        self.timestep = 1.0 / sampling_rate  # 0.5 seconds between samples

    def extract_ego_trajectory(self, sample_token: str) -> Dict[float, np.ndarray]:
        """
        Extract future ego vehicle waypoints at specified horizons.

        Args:
            sample_token: Sample token to extract trajectory from

        Returns:
            Dictionary mapping horizon -> waypoints array (T, 2) in ego frame
            Example: {1.0: array([[x1, y1], ...]), 2.0: array([...]), 3.0: array([...])}
        """
        sample = self.nusc.get('sample', sample_token)

        # Get current ego pose
        ego_pose_token = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token']
        current_pose = self.nusc.get('ego_pose', ego_pose_token)
        current_translation = np.array(current_pose['translation'][:2])  # x, y only
        current_rotation = Quaternion(current_pose['rotation'])

        # Collect future ego poses
        future_poses = self._get_future_ego_poses(sample_token)

        # Convert to ego frame and organize by horizon
        trajectories = {}

        for horizon in self.pred_horizons:
            # Calculate number of future steps needed
            num_steps = int(horizon / self.timestep)

            # Get waypoints up to this horizon
            waypoints = []
            for i, pose in enumerate(future_poses[:num_steps]):
                # Transform to current ego frame
                global_pos = np.array(pose['translation'][:2])
                local_pos = self._transform_to_ego_frame(
                    global_pos,
                    current_translation,
                    current_rotation
                )
                waypoints.append(local_pos)

            if waypoints:
                trajectories[horizon] = np.array(waypoints)  # (T, 2)
            else:
                # Return zeros if no future data available
                trajectories[horizon] = np.zeros((num_steps, 2))

        return trajectories

    def extract_agent_trajectories(
        self,
        sample_token: str,
        max_agents: int = 20,
        max_distance: float = 50.0
    ) -> Dict[str, Dict]:
        """
        Extract future agent trajectories.

        Args:
            sample_token: Sample token to extract from
            max_agents: Maximum number of agents to track
            max_distance: Maximum distance from ego to consider (meters)

        Returns:
            Dictionary mapping agent_instance_token -> trajectory info
            Each entry contains:
                - 'class': agent class name
                - 'current_pos': (2,) current position in ego frame
                - 'current_vel': (2,) current velocity in ego frame
                - 'trajectories': dict mapping horizon -> (T, 2) waypoints
        """
        sample = self.nusc.get('sample', sample_token)

        # Get current ego pose for frame transformation
        ego_pose_token = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token']
        current_ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        ego_translation = np.array(current_ego_pose['translation'][:2])
        ego_rotation = Quaternion(current_ego_pose['rotation'])

        # Get all agent annotations in current sample
        agent_annotations = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Filter by distance from ego
            agent_pos = np.array(ann['translation'][:2])
            distance = np.linalg.norm(agent_pos - ego_translation)

            if distance <= max_distance:
                agent_annotations.append(ann)

        # Sort by distance and take top max_agents
        agent_annotations.sort(
            key=lambda a: np.linalg.norm(
                np.array(a['translation'][:2]) - ego_translation
            )
        )
        agent_annotations = agent_annotations[:max_agents]

        # Extract trajectories for each agent
        agent_trajectories = {}

        for ann in agent_annotations:
            instance_token = ann['instance_token']

            # Get current state in ego frame
            current_pos_global = np.array(ann['translation'][:2])
            current_pos_ego = self._transform_to_ego_frame(
                current_pos_global, ego_translation, ego_rotation
            )

            # Estimate velocity from future positions
            current_vel_ego = self._estimate_agent_velocity(
                ann, ego_translation, ego_rotation
            )

            # Extract future trajectories
            future_traj = self._get_agent_future_trajectory(
                instance_token,
                sample_token,
                ego_translation,
                ego_rotation
            )

            agent_trajectories[instance_token] = {
                'class': ann['category_name'],
                'current_pos': current_pos_ego,
                'current_vel': current_vel_ego,
                'trajectories': future_traj
            }

        return agent_trajectories

    def _get_future_ego_poses(self, sample_token: str, max_steps: int = 10) -> List[Dict]:
        """
        Get future ego poses by traversing sample chain.

        Args:
            sample_token: Starting sample token
            max_steps: Maximum number of future steps to collect

        Returns:
            List of ego_pose dictionaries
        """
        poses = []
        current_token = sample_token

        for _ in range(max_steps):
            sample = self.nusc.get('sample', current_token)
            next_token = sample.get('next', '')

            if not next_token:
                break

            # Get ego pose for next sample
            next_sample = self.nusc.get('sample', next_token)
            lidar_token = next_sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            ego_pose_token = lidar_data['ego_pose_token']
            ego_pose = self.nusc.get('ego_pose', ego_pose_token)

            poses.append(ego_pose)
            current_token = next_token

        return poses

    def _get_agent_future_trajectory(
        self,
        instance_token: str,
        sample_token: str,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> Dict[float, np.ndarray]:
        """
        Get future trajectory for a specific agent instance.

        Args:
            instance_token: Agent instance token
            sample_token: Starting sample token
            ego_translation: Current ego position for frame transformation
            ego_rotation: Current ego rotation for frame transformation

        Returns:
            Dictionary mapping horizon -> waypoints (T, 2)
        """
        # Get instance record
        instance = self.nusc.get('instance', instance_token)

        # Collect future annotations for this instance
        future_annotations = []
        current_token = sample_token

        max_steps = int(max(self.pred_horizons) / self.timestep) + 1

        for _ in range(max_steps):
            sample = self.nusc.get('sample', current_token)
            next_token = sample.get('next', '')

            if not next_token:
                break

            # Find annotation for this instance in next sample
            next_sample = self.nusc.get('sample', next_token)

            found = False
            for ann_token in next_sample['anns']:
                ann = self.nusc.get('sample_annotation', ann_token)
                if ann['instance_token'] == instance_token:
                    future_annotations.append(ann)
                    found = True
                    break

            if not found:
                # Agent disappeared from scene
                break

            current_token = next_token

        # Organize by horizon
        trajectories = {}

        for horizon in self.pred_horizons:
            num_steps = int(horizon / self.timestep)

            waypoints = []
            for ann in future_annotations[:num_steps]:
                global_pos = np.array(ann['translation'][:2])
                local_pos = self._transform_to_ego_frame(
                    global_pos, ego_translation, ego_rotation
                )
                waypoints.append(local_pos)

            if waypoints:
                trajectories[horizon] = np.array(waypoints)
            else:
                trajectories[horizon] = np.zeros((num_steps, 2))

        return trajectories

    def _estimate_agent_velocity(
        self,
        annotation: Dict,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Estimate agent velocity from current and next position.

        Args:
            annotation: Current annotation
            ego_translation: Ego position for frame transformation
            ego_rotation: Ego rotation for frame transformation

        Returns:
            Velocity vector (2,) in ego frame [vx, vy] in m/s
        """
        # Check if agent has velocity attribute (some datasets provide this)
        if 'velocity' in annotation and annotation['velocity'] is not None:
            # nuScenes doesn't provide velocity, but keeping for compatibility
            pass

        # Estimate from next position
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

        # Calculate velocity
        current_pos = np.array(annotation['translation'][:2])
        next_pos = np.array(next_ann['translation'][:2])

        velocity_global = (next_pos - current_pos) / self.timestep

        # Transform to ego frame
        velocity_ego = self._transform_velocity_to_ego_frame(
            velocity_global, ego_rotation
        )

        return velocity_ego

    def _transform_to_ego_frame(
        self,
        global_pos: np.ndarray,
        ego_translation: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Transform position from global frame to ego vehicle frame.

        Args:
            global_pos: Position in global frame (2,)
            ego_translation: Ego vehicle position in global frame (2,)
            ego_rotation: Ego vehicle rotation quaternion

        Returns:
            Position in ego frame (2,)
        """
        # Translate to ego origin
        relative_pos = global_pos - ego_translation

        # Rotate to ego frame (inverse rotation)
        # For 2D, extract yaw from quaternion and rotate
        relative_pos_3d = np.array([relative_pos[0], relative_pos[1], 0.0])
        rotated_3d = ego_rotation.inverse.rotate(relative_pos_3d)

        return rotated_3d[:2]

    def _transform_velocity_to_ego_frame(
        self,
        velocity_global: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Transform velocity from global frame to ego frame.

        Args:
            velocity_global: Velocity in global frame (2,)
            ego_rotation: Ego vehicle rotation quaternion

        Returns:
            Velocity in ego frame (2,)
        """
        velocity_3d = np.array([velocity_global[0], velocity_global[1], 0.0])
        rotated_3d = ego_rotation.inverse.rotate(velocity_3d)
        return rotated_3d[:2]
