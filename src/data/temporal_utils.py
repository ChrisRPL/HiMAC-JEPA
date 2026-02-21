"""Utilities for temporal sequence construction from nuScenes."""
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


class TemporalSequenceBuilder:
    """Build valid temporal sequences from nuScenes samples."""

    def __init__(self, nusc, seq_length=5, pred_horizon=3, frame_skip=1,
                 max_time_gap=0.6, validate_sensors=True):
        """
        Initialize sequence builder.

        Args:
            nusc: NuScenes instance
            seq_length: Number of context frames (past)
            pred_horizon: Number of future frames (prediction target)
            frame_skip: Sample every Nth frame (1 = all frames, 2 = half)
            max_time_gap: Maximum allowed time gap between frames (seconds)
            validate_sensors: Whether to validate sensor data availability
        """
        self.nusc = nusc
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.frame_skip = frame_skip
        self.total_length = seq_length + pred_horizon
        self.max_time_gap = max_time_gap
        self.validate_sensors = validate_sensors

        # Statistics for validation
        self.validation_stats = {
            'total_candidates': 0,
            'gaps_detected': 0,
            'timestamp_violations': 0,
            'sensor_missing': 0,
            'valid_sequences': 0
        }

    def get_scene_samples(self, scene_token: str) -> List[str]:
        """
        Get all sample tokens in a scene in temporal order.

        Args:
            scene_token: Scene identifier

        Returns:
            List of sample tokens in temporal order
        """
        scene = self.nusc.get('scene', scene_token)
        samples = []

        # Traverse sample chain
        sample_token = scene['first_sample_token']
        while sample_token:
            samples.append(sample_token)
            sample = self.nusc.get('sample', sample_token)
            sample_token = sample.get('next', '')

        return samples

    def build_sequences(self, split='train') -> List[Dict]:
        """
        Build all valid temporal sequences for given split.

        Args:
            split: Dataset split ('train', 'val', 'test')

        Returns:
            List of sequence dictionaries with context/target tokens
        """
        from nuscenes.utils.splits import create_splits_scenes

        # Get scene names for this split
        splits = create_splits_scenes()
        if split == 'test' and split not in splits:
            split = 'val'  # Fallback

        scene_names = set(splits.get(split, []))
        sequences = []

        for scene in self.nusc.scene:
            if scene['name'] not in scene_names:
                continue

            # Get all samples in scene
            sample_tokens = self.get_scene_samples(scene['token'])

            # Apply frame skip
            if self.frame_skip > 1:
                sample_tokens = sample_tokens[::self.frame_skip]

            # Skip scenes that are too short
            if len(sample_tokens) < self.total_length:
                continue

            # Create sliding window sequences
            for i in range(len(sample_tokens) - self.total_length + 1):
                context_tokens = sample_tokens[i:i + self.seq_length]
                target_tokens = sample_tokens[
                    i + self.seq_length:i + self.total_length
                ]

                # Validate sequence continuity
                all_tokens = context_tokens + target_tokens
                self.validation_stats['total_candidates'] += 1

                # Run all validation checks
                valid, reason = self.validate_sequence_comprehensive(all_tokens)

                if valid:
                    self.validation_stats['valid_sequences'] += 1
                    sequences.append({
                        'context_tokens': context_tokens,
                        'target_tokens': target_tokens,
                        'scene_token': scene['token'],
                        'scene_name': scene['name']
                    })
                else:
                    # Track reason for rejection
                    if 'gap' in reason.lower():
                        self.validation_stats['gaps_detected'] += 1
                    elif 'timestamp' in reason.lower():
                        self.validation_stats['timestamp_violations'] += 1
                    elif 'sensor' in reason.lower():
                        self.validation_stats['sensor_missing'] += 1

        return sequences

    def validate_sequence(self, sample_tokens: List[str]) -> bool:
        """
        Check if sequence is temporally continuous (no gaps).

        Args:
            sample_tokens: List of sample tokens to validate

        Returns:
            True if sequence is continuous, False if there are gaps
        """
        if len(sample_tokens) < 2:
            return True

        # Check each consecutive pair
        for i in range(len(sample_tokens) - 1):
            sample = self.nusc.get('sample', sample_tokens[i])
            next_token = sample.get('next', '')

            # Check if next token matches expected
            if next_token != sample_tokens[i + 1]:
                return False  # Gap detected

        return True

    def validate_sequence_comprehensive(self, sample_tokens: List[str]) -> Tuple[bool, str]:
        """
        Perform comprehensive validation on temporal sequence.

        Checks:
        - Temporal continuity (no gaps in sample chain)
        - Timestamp consistency (uniform time intervals)
        - Sensor data availability (if enabled)

        Args:
            sample_tokens: List of sample tokens to validate

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if len(sample_tokens) < 2:
            return True, "OK"

        # Check 1: Temporal continuity
        if not self.validate_sequence(sample_tokens):
            return False, "Gap in sample chain detected"

        # Check 2: Timestamp consistency
        timestamps = []
        for token in sample_tokens:
            sample = self.nusc.get('sample', token)
            timestamps.append(sample['timestamp'] / 1e6)  # Convert to seconds

        # Check time intervals
        time_diffs = np.diff(timestamps)

        # Account for frame skip in expected interval
        expected_interval = 0.5 * self.frame_skip  # nuScenes default is 2Hz (0.5s)
        max_allowed = expected_interval + self.max_time_gap

        if np.any(time_diffs > max_allowed):
            max_gap = np.max(time_diffs)
            return False, f"Timestamp gap too large: {max_gap:.2f}s > {max_allowed:.2f}s"

        # Check for negative time differences (should never happen)
        if np.any(time_diffs <= 0):
            return False, "Non-monotonic timestamps detected"

        # Check 3: Sensor data availability (optional)
        if self.validate_sensors:
            for token in sample_tokens:
                if not self._validate_sensor_availability(token):
                    return False, "Missing required sensor data"

        return True, "OK"

    def _validate_sensor_availability(self, sample_token: str) -> bool:
        """
        Check that all required sensors have data for this sample.

        Args:
            sample_token: Sample token to check

        Returns:
            True if all sensors available, False otherwise
        """
        sample = self.nusc.get('sample', sample_token)

        # Check camera data
        if 'CAM_FRONT' not in sample['data']:
            return False

        # Check LiDAR data
        if 'LIDAR_TOP' not in sample['data']:
            return False

        # Check radar data (at least one radar sensor)
        radar_sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                        'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        has_radar = any(sensor in sample['data'] for sensor in radar_sensors)
        if not has_radar:
            return False

        return True

    def get_validation_statistics(self) -> Dict[str, any]:
        """
        Get statistics about sequence validation.

        Returns:
            Dictionary with validation statistics
        """
        stats = self.validation_stats.copy()

        if stats['total_candidates'] > 0:
            stats['valid_ratio'] = stats['valid_sequences'] / stats['total_candidates']
            stats['rejection_ratio'] = 1.0 - stats['valid_ratio']
        else:
            stats['valid_ratio'] = 0.0
            stats['rejection_ratio'] = 0.0

        return stats

    def print_validation_summary(self):
        """Print human-readable validation summary."""
        stats = self.get_validation_statistics()

        print("\n" + "="*60)
        print("Temporal Sequence Validation Summary")
        print("="*60)
        print(f"Total candidates:        {stats['total_candidates']}")
        print(f"Valid sequences:         {stats['valid_sequences']}")
        print(f"Valid ratio:             {stats['valid_ratio']:.2%}")
        print(f"\nRejection reasons:")
        print(f"  - Sample gaps:         {stats['gaps_detected']}")
        print(f"  - Timestamp issues:    {stats['timestamp_violations']}")
        print(f"  - Missing sensors:     {stats['sensor_missing']}")
        print("="*60 + "\n")
