"""Utilities for building action-free EMA targets."""

import torch


def build_target_latent(ema_model, camera, lidar, radar):
    """Encode target observations with the EMA teacher, without action conditioning."""
    was_training = ema_model.training
    ema_model.eval()

    with torch.no_grad():
        target_latent = ema_model.encode_observations(camera, lidar, radar, masks=None).detach()

    if was_training:
        ema_model.train()

    return target_latent
