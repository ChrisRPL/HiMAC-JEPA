"""Preprocessing utilities for multi-modal sensor data."""
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple


class CameraPreprocessor:
    """Preprocess camera images for ViT encoder."""

    def __init__(self, config):
        """
        Initialize camera preprocessor.

        Args:
            config: Configuration object with image preprocessing parameters
        """
        self.image_size = tuple(config.image_size)
        self.normalize = config.normalize_images

        # Base transforms: resize and convert to tensor
        transforms = [
            T.Resize(self.image_size),
            T.ToTensor()
        ]

        # Add ImageNet normalization if enabled
        if self.normalize:
            transforms.append(T.Normalize(
                mean=config.imagenet_mean,
                std=config.imagenet_std
            ))

        self.transform = T.Compose(transforms)

        # Augmentation transforms (only for training)
        if config.augmentation.enabled:
            aug_transforms = []

            if config.augmentation.color_jitter:
                aug_transforms.append(
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                )

            if config.augmentation.random_crop:
                aug_transforms.append(
                    T.RandomCrop(self.image_size, padding=4)
                )

            self.augment = T.Compose(aug_transforms) if aug_transforms else None
        else:
            self.augment = None

    def __call__(self, image_path: str, augment: bool = False) -> torch.Tensor:
        """
        Load and preprocess camera image.

        Args:
            image_path: Path to image file
            augment: Whether to apply augmentation (for training)

        Returns:
            Preprocessed image tensor of shape (3, H, W)
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Apply augmentation if requested
        if augment and self.augment is not None:
            img = self.augment(img)

        # Apply base transforms
        return self.transform(img)


class LiDARPreprocessor:
    """Preprocess LiDAR point clouds."""

    def __init__(self, config):
        """
        Initialize LiDAR preprocessor.

        Args:
            config: Configuration object with point cloud parameters
        """
        self.num_points = config.num_points

    def __call__(self, pointcloud: np.ndarray) -> torch.Tensor:
        """
        Sample and normalize point cloud to fixed size.

        Args:
            pointcloud: Raw point cloud array of shape (N, C) where C >= 3

        Returns:
            Sampled point cloud tensor of shape (num_points, 3)
        """
        # Remove ego vehicle points (within 2m radius)
        distances = np.linalg.norm(pointcloud[:, :3], axis=1)
        pointcloud = pointcloud[distances > 2.0]

        # Handle empty point cloud
        if len(pointcloud) == 0:
            return torch.zeros(self.num_points, 3, dtype=torch.float32)

        # Random sampling to fixed size
        num_pts = pointcloud.shape[0]

        if num_pts > self.num_points:
            # Downsample: random selection without replacement
            indices = np.random.choice(num_pts, self.num_points, replace=False)
            pointcloud = pointcloud[indices]
        elif num_pts < self.num_points:
            # Upsample: random selection with replacement
            indices = np.random.choice(num_pts, self.num_points, replace=True)
            pointcloud = pointcloud[indices]

        # Keep only xyz coordinates (first 3 channels)
        pointcloud = pointcloud[:, :3]

        return torch.from_numpy(pointcloud).float()


class RadarPreprocessor:
    """Preprocess radar data."""

    def __init__(self, config):
        """
        Initialize radar preprocessor.

        Args:
            config: Configuration object with radar parameters
        """
        self.radar_size = tuple(config.radar_size)

    def __call__(self, radar_pointcloud: np.ndarray) -> torch.Tensor:
        """
        Convert radar point cloud to range-doppler image.

        Args:
            radar_pointcloud: Radar point cloud of shape (N, C)

        Returns:
            Radar image tensor of shape (1, H, W)
        """
        # Create empty radar image
        radar_image = np.zeros(self.radar_size, dtype=np.float32)

        # TODO: Implement proper range-doppler processing
        # For now, create simple accumulation grid
        # This is a placeholder - proper implementation needed for production

        if len(radar_pointcloud) > 0:
            # Extract x, y coordinates (simplified)
            # In production, should process range, doppler, azimuth properly
            x_coords = radar_pointcloud[:, 0]
            y_coords = radar_pointcloud[:, 1]

            # Normalize to grid indices (simple approach)
            x_min, x_max = -50, 50  # meters
            y_min, y_max = -50, 50  # meters

            x_indices = ((x_coords - x_min) / (x_max - x_min) * (self.radar_size[1] - 1)).astype(int)
            y_indices = ((y_coords - y_min) / (y_max - y_min) * (self.radar_size[0] - 1)).astype(int)

            # Clip to valid range
            x_indices = np.clip(x_indices, 0, self.radar_size[1] - 1)
            y_indices = np.clip(y_indices, 0, self.radar_size[0] - 1)

            # Accumulate points into grid
            for x_idx, y_idx in zip(x_indices, y_indices):
                radar_image[y_idx, x_idx] += 1.0

            # Normalize
            if radar_image.max() > 0:
                radar_image = radar_image / radar_image.max()

        # Add channel dimension: (H, W) -> (1, H, W)
        return torch.from_numpy(radar_image).unsqueeze(0).float()
