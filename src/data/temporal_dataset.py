"""Temporal sequence dataset wrapper for JEPA training."""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any, List

from .temporal_utils import TemporalSequenceBuilder


class TemporalNuScenesDataset(Dataset):
    """
    Temporal sequence wrapper for nuScenes dataset.

    Returns sequences of (context, target) frames for JEPA training,
    where context frames are used to predict future target frames.
    """

    def __init__(self, base_dataset, config, split='train'):
        """
        Initialize temporal dataset.

        Args:
            base_dataset: NuScenesMultiModalDataset instance
            config: Configuration dict with temporal parameters
            split: Dataset split ('train', 'val', 'test')
        """
        self.base_dataset = base_dataset
        self.config = config
        self.split = split

        # Extract temporal parameters
        self.seq_length = config.get('seq_length', 5)
        self.pred_horizon = config.get('pred_horizon', 3)
        self.frame_skip = config.get('frame_skip', 1)

        # Build temporal sequences
        print(f"Building temporal sequences for {split} split...")
        print(f"  Context frames: {self.seq_length}")
        print(f"  Future frames: {self.pred_horizon}")
        print(f"  Frame skip: {self.frame_skip}")

        builder = TemporalSequenceBuilder(
            nusc=base_dataset.nusc,
            seq_length=self.seq_length,
            pred_horizon=self.pred_horizon,
            frame_skip=self.frame_skip
        )

        self.sequences = builder.build_sequences(split)
        print(f"Built {len(self.sequences)} temporal sequences")

        # Create token to index mapping for fast lookup
        self._build_token_index()

    def _build_token_index(self):
        """Build mapping from sample token to dataset index."""
        self.token_to_idx = {}
        for idx, sample in enumerate(self.base_dataset.samples):
            self.token_to_idx[sample['token']] = idx

    def __len__(self) -> int:
        """Return number of temporal sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get temporal sequence with context and target frames.

        Args:
            idx: Sequence index

        Returns:
            Dictionary containing:
                - context: Dict with (camera, lidar, radar, actions) sequences
                - target: Dict with future (camera, lidar, radar, actions)
                - metadata: Scene name and token info
        """
        seq_info = self.sequences[idx]

        # Load context frames (past)
        context_data = self._load_frame_sequence(seq_info['context_tokens'])

        # Load target frames (future)
        target_data = self._load_frame_sequence(seq_info['target_tokens'])

        return {
            'context': context_data,
            'target': target_data,
            'scene_name': seq_info['scene_name'],
            'scene_token': seq_info['scene_token']
        }

    def _load_frame_sequence(self, sample_tokens: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load a sequence of frames from base dataset.

        Args:
            sample_tokens: List of sample tokens to load

        Returns:
            Dictionary with stacked temporal tensors
        """
        cameras = []
        lidars = []
        radars = []
        strategic_actions = []
        tactical_actions = []

        for token in sample_tokens:
            # Get sample index in base dataset
            sample_idx = self.token_to_idx.get(token)

            if sample_idx is None:
                raise ValueError(f"Sample token {token} not found in base dataset")

            # Load single frame from base dataset
            sample = self.base_dataset[sample_idx]

            # Collect data
            cameras.append(sample['camera'])
            lidars.append(sample['lidar'])
            radars.append(sample['radar'])
            strategic_actions.append(sample['strategic_action'])
            tactical_actions.append(sample['tactical_action'])

        # Stack into temporal sequences (T, ...)
        result = {
            'camera': torch.stack(cameras, dim=0),  # (T, 3, H, W)
            'lidar': torch.stack(lidars, dim=0),     # (T, N, 3)
            'radar': torch.stack(radars, dim=0),     # (T, 1, H, W)
            'strategic_action': torch.stack(strategic_actions, dim=0),  # (T,)
            'tactical_action': torch.stack(tactical_actions, dim=0)     # (T, 3)
        }

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with statistics about sequences
        """
        scene_counts = {}
        for seq in self.sequences:
            scene_name = seq['scene_name']
            scene_counts[scene_name] = scene_counts.get(scene_name, 0) + 1

        return {
            'num_sequences': len(self.sequences),
            'seq_length': self.seq_length,
            'pred_horizon': self.pred_horizon,
            'frame_skip': self.frame_skip,
            'num_scenes': len(scene_counts),
            'sequences_per_scene': {
                'min': min(scene_counts.values()) if scene_counts else 0,
                'max': max(scene_counts.values()) if scene_counts else 0,
                'mean': np.mean(list(scene_counts.values())) if scene_counts else 0
            }
        }
