import torch

from src.masking.spatio_temporal_masking import SpatioTemporalMasking
from src.training.masking import build_batch_masks


def test_build_batch_masks_uses_temporal_sequence_length():
    masker = SpatioTemporalMasking(mask_ratio_spatial=0.5, mask_ratio_temporal=0.5, num_temporal_steps=10)

    camera = torch.randn(2, 4, 3, 224, 224)
    lidar = torch.randn(2, 4, 1024, 3)
    radar = torch.randn(2, 4, 1, 64, 64)

    masks = build_batch_masks(masker, camera, lidar, radar)

    assert masks["camera"].shape == (2, 14, 14)
    assert masks["lidar"].shape == (2, 1024)
    assert masks["radar"].shape == (2, 64, 64)
    assert masks["temporal"].shape == (2, 4)
