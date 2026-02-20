import torch
import pytest
import random

from src.masking.spatio_temporal_masking import SpatioTemporalMasking

# Note: These tests require PyTorch to be installed to run. If PyTorch is not installed,
# these tests will fail with a ModuleNotFoundError. This is expected given the current
# environment constraints.

def test_generate_spatial_mask_2d():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_shape = (64, 64) # H, W
    patch_size = (16, 16)
    mask = masker.generate_spatial_mask(input_shape, patch_size)

    assert mask.shape == (input_shape[0] // patch_size[0], input_shape[1] // patch_size[1])
    total_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])
    expected_masked_patches = int(total_patches * 0.5)
    assert torch.sum(mask).item() == expected_masked_patches

def test_generate_spatial_mask_3d():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_shape = (3, 64, 64) # C, H, W
    patch_size = (16, 16)
    mask = masker.generate_spatial_mask(input_shape, patch_size)

    assert mask.shape == (input_shape[1] // patch_size[0], input_shape[2] // patch_size[1])
    total_patches = (input_shape[1] // patch_size[0]) * (input_shape[2] // patch_size[1])
    expected_masked_patches = int(total_patches * 0.5)
    assert torch.sum(mask).item() == expected_masked_patches

def test_generate_temporal_mask():
    masker = SpatioTemporalMasking(mask_ratio_temporal=0.6, num_temporal_steps=10)
    mask = masker.generate_temporal_mask()

    assert mask.shape == (10,)
    expected_masked_steps = int(10 * 0.6)
    assert torch.sum(mask).item() == expected_masked_steps

def test_apply_mask_2d():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_data = torch.ones(64, 64)
    patch_size = (16, 16)
    spatial_mask = masker.generate_spatial_mask(input_data.shape, patch_size)
    masked_data = masker.apply_mask(input_data, spatial_mask, patch_size)

    assert masked_data.shape == input_data.shape
    # Check if masked patches are zero
    for i in range(spatial_mask.shape[0]):
        for j in range(spatial_mask.shape[1]):
            if spatial_mask[i, j]:
                assert torch.all(masked_data[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] == 0)
            else:
                assert torch.all(masked_data[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] == 1)

def test_apply_mask_3d():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_data = torch.ones(3, 64, 64)
    patch_size = (16, 16)
    spatial_mask = masker.generate_spatial_mask(input_data.shape, patch_size)
    masked_data = masker.apply_mask(input_data, spatial_mask, patch_size)

    assert masked_data.shape == input_data.shape
    # Check if masked patches are zero
    for i in range(spatial_mask.shape[0]):
        for j in range(spatial_mask.shape[1]):
            if spatial_mask[i, j]:
                assert torch.all(masked_data[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] == 0)
            else:
                assert torch.all(masked_data[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]] == 1)

def test_get_masked_indices():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_shape = (32, 32)
    patch_size = (16, 16)
    spatial_mask = masker.generate_spatial_mask(input_shape, patch_size)
    masked_indices = masker.get_masked_indices(spatial_mask)
    assert len(masked_indices) == torch.sum(spatial_mask).item()

def test_get_unmasked_indices():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5)
    input_shape = (32, 32)
    patch_size = (16, 16)
    spatial_mask = masker.generate_spatial_mask(input_shape, patch_size)
    unmasked_indices = masker.get_unmasked_indices(spatial_mask)
    assert len(unmasked_indices) == torch.sum(~spatial_mask).item()
