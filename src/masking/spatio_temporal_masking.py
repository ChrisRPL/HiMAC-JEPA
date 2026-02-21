import torch
import random

class SpatioTemporalMasking:
    """
    Generates spatio-temporal masks for different modalities and time steps.
    The masking strategy is dynamic, covering both spatial (patches) and temporal (future time steps) dimensions.
    """
    def __init__(self, 
                 mask_ratio_spatial: float = 0.5,
                 mask_ratio_temporal: float = 0.5,
                 patch_size_camera: tuple = (16, 16),
                 patch_size_lidar: tuple = (1, 1),
                 patch_size_radar: tuple = (1, 1),
                 num_temporal_steps: int = 5):
        """
        Args:
            mask_ratio_spatial (float): The ratio of spatial patches to mask.
            mask_ratio_temporal (float): The ratio of future temporal steps to mask.
            patch_size_camera (tuple): (height, width) of patches for camera modality.
            patch_size_lidar (tuple): (depth, width, height) or similar for LiDAR modality.
            patch_size_radar (tuple): (depth, width, height) or similar for Radar modality.
            num_temporal_steps (int): The total number of future temporal steps available for masking.
        """
        self.mask_ratio_spatial = mask_ratio_spatial
        self.mask_ratio_temporal = mask_ratio_temporal
        self.patch_size_camera = patch_size_camera
        self.patch_size_lidar = patch_size_lidar
        self.patch_size_radar = patch_size_radar
        self.num_temporal_steps = num_temporal_steps

    def generate_spatial_mask(self, input_shape: tuple, patch_size: tuple, batch_size: int = 1) -> torch.Tensor:
        """
        Generates a 2D spatial mask for a given input shape and patch size.
        Input shape is (H, W) or (C, H, W).
        Output mask is (B, H_patches, W_patches) boolean tensor.

        Args:
            input_shape: Shape of input (H, W) or (C, H, W)
            patch_size: Size of patches (pH, pW)
            batch_size: Number of masks to generate (one per batch element)

        Returns:
            Batch of spatial masks, shape (B, H_patches, W_patches)
        """
        if len(input_shape) == 3:
            _, H, W = input_shape
        else:
            H, W = input_shape

        pH, pW = patch_size

        num_patches_h = H // pH
        num_patches_w = W // pW
        total_patches = num_patches_h * num_patches_w
        num_masked_patches = int(total_patches * self.mask_ratio_spatial)

        # Generate different mask for each batch element
        masks = []
        for _ in range(batch_size):
            mask = torch.zeros(total_patches, dtype=torch.bool)
            mask_indices = random.sample(range(total_patches), num_masked_patches)
            mask[mask_indices] = True
            masks.append(mask.view(num_patches_h, num_patches_w))

        return torch.stack(masks, dim=0)  # (B, H_patches, W_patches)

    def generate_temporal_mask(self) -> torch.Tensor:
        """
        Generates a 1D temporal mask for future time steps.
        Output mask is (num_temporal_steps,) boolean tensor.
        """
        num_masked_steps = int(self.num_temporal_steps * self.mask_ratio_temporal)
        mask = torch.zeros(self.num_temporal_steps, dtype=torch.bool)
        mask_indices = random.sample(range(self.num_temporal_steps), num_masked_steps)
        mask[mask_indices] = True
        return mask

    def apply_mask(self, data: torch.Tensor, mask: torch.Tensor, patch_size: tuple) -> torch.Tensor:
        """
        Applies a spatial mask to the input data.
        Assumes data is (C, H, W) or (H, W) and mask is (H_patches, W_patches).
        """
        if len(data.shape) == 3:
            C, H, W = data.shape
        else:
            H, W = data.shape
            C = 1 # Treat as single channel for consistent patching
            data = data.unsqueeze(0)

        pH, pW = patch_size

        # Unfold into patches
        patches = data.unfold(1, pH, pH).unfold(2, pW, pW)
        patches = patches.reshape(C, -1, pH, pW) # (C, num_patches, pH, pW)

        # Apply mask
        masked_patches = patches.clone()
        masked_patches[:, mask.flatten()] = 0 # Set masked patches to zero

        # Fold back into original shape (this is a simplified re-assembly, might need more complex logic for actual use)
        # For now, this function primarily demonstrates mask application at patch level.
        # Re-assembly is non-trivial and depends on how the model handles masked inputs.
        # This part is illustrative and might need to be adapted based on the actual model architecture.
        reconstructed_data = torch.zeros_like(data)
        patch_idx = 0
        for i in range(H // pH):
            for j in range(W // pW):
                reconstructed_data[:, i*pH:(i+1)*pH, j*pW:(j+1)*pW] = masked_patches[:, patch_idx]
                patch_idx += 1
        
        if C == 1 and len(data.shape) == 2:
            return reconstructed_data.squeeze(0)
        return reconstructed_data

    def get_masked_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the indices of the masked patches/steps.
        """
        return torch.nonzero(mask.flatten(), as_tuple=True)[0]

    def get_unmasked_indices(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the indices of the unmasked patches/steps.
        """
        return torch.nonzero(~mask.flatten(), as_tuple=True)[0]

    def generate_joint_mask(self, camera_shape: tuple, lidar_shape: tuple, radar_shape: tuple, batch_size: int) -> dict:
        """
        Generate synchronized masks across all modalities for a batch.

        Args:
            camera_shape: Shape of camera input (C, H, W)
            lidar_shape: Shape of LiDAR input (N, 3)
            radar_shape: Shape of radar input (C, H, W)
            batch_size: Number of samples in batch

        Returns:
            Dictionary containing masks for each modality:
            {
                'camera': (B, H_patches, W_patches),
                'lidar': (B, num_point_groups),
                'radar': (B, H_patches, W_patches),
                'temporal': (B, num_temporal_steps)
            }
        """
        # Generate spatial masks for each modality
        camera_mask = self.generate_spatial_mask(camera_shape, self.patch_size_camera, batch_size)
        radar_mask = self.generate_spatial_mask(radar_shape, self.patch_size_radar, batch_size)

        # For LiDAR, treat points as patches (simpler approach)
        # In a full implementation, this could use voxelization or clustering
        num_points = lidar_shape[0] if len(lidar_shape) > 1 else lidar_shape[0]
        num_masked_points = int(num_points * self.mask_ratio_spatial)
        lidar_masks = []
        for _ in range(batch_size):
            lidar_mask = torch.zeros(num_points, dtype=torch.bool)
            mask_indices = random.sample(range(num_points), num_masked_points)
            lidar_mask[mask_indices] = True
            lidar_masks.append(lidar_mask)
        lidar_mask = torch.stack(lidar_masks, dim=0)  # (B, N)

        # Generate temporal mask (same for all samples in batch for simplicity)
        # Could be different per sample if needed
        temporal_mask = self.generate_temporal_mask()
        temporal_mask = temporal_mask.unsqueeze(0).expand(batch_size, -1)  # (B, T)

        return {
            'camera': camera_mask,
            'lidar': lidar_mask,
            'radar': radar_mask,
            'temporal': temporal_mask
        }
