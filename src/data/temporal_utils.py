"""Utilities for temporal sequence construction from nuScenes."""
import numpy as np
from typing import List, Dict, Optional


class TemporalSequenceBuilder:
    """Build valid temporal sequences from nuScenes samples."""

    def __init__(self, nusc, seq_length=5, pred_horizon=3, frame_skip=1):
        """
        Initialize sequence builder.

        Args:
            nusc: NuScenes instance
            seq_length: Number of context frames (past)
            pred_horizon: Number of future frames (prediction target)
            frame_skip: Sample every Nth frame (1 = all frames, 2 = half)
        """
        self.nusc = nusc
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.frame_skip = frame_skip
        self.total_length = seq_length + pred_horizon

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
                if self.validate_sequence(context_tokens + target_tokens):
                    sequences.append({
                        'context_tokens': context_tokens,
                        'target_tokens': target_tokens,
                        'scene_token': scene['token'],
                        'scene_name': scene['name']
                    })

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
