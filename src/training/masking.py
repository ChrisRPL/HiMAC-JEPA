"""Utilities for building training masks from batch tensors."""


def build_batch_masks(masker, camera, lidar, radar):
    """Generate JEPA masks for either single-frame or temporal batches."""
    is_temporal = camera.dim() == 5

    return masker.generate_joint_mask(
        camera_shape=camera.shape[2:] if is_temporal else camera.shape[1:],
        lidar_shape=lidar.shape[2:] if is_temporal else lidar.shape[1:],
        radar_shape=radar.shape[2:] if is_temporal else radar.shape[1:],
        batch_size=camera.shape[0],
        num_temporal_steps=camera.shape[1] if is_temporal else None,
    )
