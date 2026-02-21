"""nuScenes dataset loader with synchronized multi-modal data."""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


class NuScenesMultiModalDataset(Dataset):
    """nuScenes dataset with synchronized camera, LiDAR, and radar data."""

    def __init__(self, config, split='train'):
        """
        Initialize nuScenes dataset.

        Args:
            config: Hydra configuration object with dataset parameters
            split: Dataset split - 'train', 'val', or 'test'
        """
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            raise ImportError(
                "nuScenes devkit not installed. Install with: pip install nuscenes-devkit"
            )

        self.config = config
        self.split = split

        # Initialize nuScenes SDK
        print(f"Loading nuScenes {config.version} from {config.data_root}...")
        self.nusc = NuScenes(
            version=config.version,
            dataroot=config.data_root,
            verbose=True
        )

        # Get samples for this split
        self.samples = self._get_samples(split)
        print(f"Loaded {len(self.samples)} samples for {split} split")

        # Initialize preprocessors
        from .preprocessing import CameraPreprocessor, LiDARPreprocessor, RadarPreprocessor
        self.camera_prep = CameraPreprocessor(config)
        self.lidar_prep = LiDARPreprocessor(config)
        self.radar_prep = RadarPreprocessor(config)

        # Initialize action extractor
        from .action_extraction import ActionExtractor
        self.action_extractor = ActionExtractor(
            strategic_classes=config.strategic_action_classes,
            tactical_dim=config.tactical_action_dim
        )

        # Initialize label extraction (optional)
        self.use_labels = config.get('labels', {}).get('enabled', False)
        if self.use_labels:
            print("Initializing label extractors...")
            from .labels import TrajectoryLabelExtractor, BEVLabelExtractor, MotionLabelExtractor, LabelCache

            label_config = config.labels

            # Initialize extractors
            self.traj_extractor = TrajectoryLabelExtractor(
                self.nusc,
                pred_horizons=label_config.trajectory.get('pred_horizons', [1.0, 2.0, 3.0])
            )

            self.bev_extractor = BEVLabelExtractor(
                self.nusc,
                bev_size=tuple(label_config.bev.get('size', [200, 200])),
                bev_range=label_config.bev.get('range', 50.0)
            )

            self.motion_extractor = MotionLabelExtractor(
                self.nusc,
                pred_horizon=label_config.motion.get('pred_horizon', 3.0),
                max_distance=label_config.motion.get('max_distance', 50.0),
                min_visibility=label_config.motion.get('min_visibility', 0.5)
            )

            # Initialize cache
            cache_config = label_config.get('cache', {})
            self.label_cache = LabelCache(
                cache_dir=cache_config.get('cache_dir', './cache/labels')
            )
            self.force_recompute = cache_config.get('force_recompute', False)

            print(f"Label extraction enabled for {split} split")

            # Print cache stats
            cache_stats = self.label_cache.get_cache_stats(split)
            if cache_stats['num_cached'] > 0:
                print(f"Found {cache_stats['num_cached']} cached labels ({cache_stats['total_size_mb']:.1f} MB)")
        else:
            self.label_cache = None

    def _get_samples(self, split: str) -> List[Dict]:
        """
        Get sample tokens for the specified split.

        Args:
            split: 'train', 'val', or 'test'

        Returns:
            List of sample dictionaries
        """
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError:
            # Fallback for older versions
            return self._get_all_samples()

        # Get scene names for this split
        splits = create_splits_scenes()

        # Handle test split (use val for now if test not available)
        if split == 'test' and split not in splits:
            split = 'val'

        scene_names = set(splits.get(split, []))

        if not scene_names:
            # Fallback: use all scenes
            print(f"Warning: Split '{split}' not found, using all scenes")
            return self._get_all_samples()

        # Collect samples from scenes in this split
        samples = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                # Get first sample in scene
                sample_token = scene['first_sample_token']

                # Iterate through all samples in scene
                while sample_token:
                    sample = self.nusc.get('sample', sample_token)
                    samples.append(sample)
                    sample_token = sample['next']

        return samples

    def _get_all_samples(self) -> List[Dict]:
        """Fallback: get all samples in dataset."""
        return list(self.nusc.sample)

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get synchronized multi-modal sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - camera: Camera image tensor (3, H, W)
                - lidar: LiDAR point cloud (num_points, 3)
                - radar: Radar image (1, H, W)
                - strategic_action: Strategic action class (scalar)
                - tactical_action: Tactical action vector (3,)
        """
        sample = self.samples[idx]

        # Get camera data (front camera)
        cam_sensor = self.config.cameras[0]  # Default: CAM_FRONT
        cam_token = sample['data'][cam_sensor]
        cam_path = self.nusc.get_sample_data_path(cam_token)

        # Preprocess camera image
        camera = self.camera_prep(
            cam_path,
            augment=(self.split == 'train' and self.config.augmentation.enabled)
        )

        # Get LiDAR data (top lidar)
        lidar_sensor = self.config.lidar_sensors[0]  # Default: LIDAR_TOP
        lidar_token = sample['data'][lidar_sensor]
        lidar_path = self.nusc.get_sample_data_path(lidar_token)

        # Load point cloud (nuScenes format: x, y, z, intensity, ring)
        pointcloud = np.fromfile(str(lidar_path), dtype=np.float32)
        pointcloud = pointcloud.reshape(-1, 5)  # 5 channels

        # Preprocess LiDAR
        lidar = self.lidar_prep(pointcloud)

        # Get radar data (front radar)
        radar_sensor = self.config.radar_sensors[0]  # Default: RADAR_FRONT
        radar_token = sample['data'][radar_sensor]
        radar_path = self.nusc.get_sample_data_path(radar_token)

        # Load radar point cloud (nuScenes radar has 18 channels)
        try:
            radar_pc = np.fromfile(str(radar_path), dtype=np.float32)
            radar_pc = radar_pc.reshape(-1, 18)  # 18 channels in nuScenes radar
        except Exception as e:
            # Handle missing or corrupted radar data
            print(f"Warning: Could not load radar data for sample {idx}: {e}")
            radar_pc = np.zeros((0, 18), dtype=np.float32)

        # Preprocess radar
        radar = self.radar_prep(radar_pc)

        # Extract actions from ego vehicle data
        # Get ego pose for this sample (use lidar timestamp as reference)
        ego_pose_token = self.nusc.get('sample_data', lidar_token)['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)

        # Try to get next sample for trajectory analysis
        next_sample_token = sample.get('next', '')
        next_ego_pose = None
        if next_sample_token:
            try:
                next_sample = self.nusc.get('sample', next_sample_token)
                next_lidar_token = next_sample['data'][lidar_sensor]
                next_ego_token = self.nusc.get('sample_data', next_lidar_token)['ego_pose_token']
                next_ego_pose = self.nusc.get('ego_pose', next_ego_token)
            except:
                pass

        # Extract actions
        # Note: nuScenes v1.0-mini doesn't have CAN bus data,
        # so we use placeholder zeros for tactical actions
        strategic_action, tactical_action = self.action_extractor({
            'ego_pose': ego_pose,
            'next_ego_pose': next_ego_pose,
            'can_bus': {}  # Not available in mini split
        })

        result = {
            'camera': camera,
            'lidar': lidar,
            'radar': radar,
            'strategic_action': torch.tensor(strategic_action, dtype=torch.long),
            'tactical_action': torch.from_numpy(tactical_action).float()
        }

        # Add labels if enabled
        if self.use_labels:
            sample_token = sample['token']
            labels = self._get_labels(sample_token)
            result['labels'] = labels

        return result

    def _get_labels(self, sample_token: str) -> Dict:
        """
        Get labels from cache or extract.

        Args:
            sample_token: Sample token

        Returns:
            Dictionary with extracted labels
        """
        # Check cache first (unless force_recompute is enabled)
        if not self.force_recompute:
            cached = self.label_cache.load_labels(sample_token, self.split)
            if cached is not None:
                return cached

        # Extract labels
        try:
            labels = {}

            # Trajectory labels
            if self.config.labels.trajectory.get('include_ego', True):
                labels['trajectory_ego'] = self.traj_extractor.extract_ego_trajectory(sample_token)

            if self.config.labels.trajectory.get('include_agents', True):
                labels['trajectory_agents'] = self.traj_extractor.extract_agent_trajectories(
                    sample_token,
                    max_agents=self.config.labels.trajectory.get('max_agents', 20)
                )

            # BEV labels
            labels['bev'] = self.bev_extractor.extract_bev_labels(sample_token)

            # Motion prediction labels
            labels['motion'] = self.motion_extractor.extract_motion_labels(sample_token)

        except Exception as e:
            # If extraction fails, return empty labels
            print(f"Warning: Label extraction failed for {sample_token}: {e}")
            labels = {
                'trajectory_ego': {},
                'trajectory_agents': {},
                'bev': np.zeros((200, 200), dtype=np.uint8),
                'motion': {
                    'agent_ids': [],
                    'agent_classes': [],
                    'current_states': np.zeros((0, 4)),
                    'future_trajectories': np.zeros((0, 0, 2)),
                    'valid_masks': np.zeros((0, 0), dtype=bool)
                }
            }

        # Save to cache
        if self.label_cache is not None:
            self.label_cache.save_labels(sample_token, labels, self.split)

        return labels
