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

    def generate_spatial_mask(self, input_shape: tuple, patch_size: tuple) -> torch.Tensor:
        """
        Generates a 2D spatial mask for a given input shape and patch size.
        Input shape is (H, W) or (C, H, W).
        Output mask is (H_patches, W_patches) boolean tensor.
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

        mask = torch.zeros(total_patches, dtype=torch.bool)
        mask_indices = random.sample(range(total_patches), num_masked_patches)
        mask[mask_indices] = True

        return mask.view(num_patches_h, num_patches_w)

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
