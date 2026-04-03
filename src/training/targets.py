"""Utilities for building action-free EMA targets."""

import torch

from src.models.himac_teacher import HiMACObservationTeacher


def build_ema_teacher(config, student=None):
    """Create the slim EMA teacher and optionally initialize it from the student."""
    teacher = HiMACObservationTeacher(config)
    if student is not None:
        teacher.load_from_student(student)
    return teacher


def update_ema_teacher(student, teacher, decay):
    """EMA update for the observation-only teacher."""
    teacher.update_from_student(student, decay)


def build_target_latent(teacher, camera, lidar, radar):
    """Encode target observations with the EMA teacher, without action conditioning."""
    was_training = teacher.training
    teacher.eval()

    with torch.no_grad():
        target_latent = teacher.encode_observations(camera, lidar, radar, masks=None).detach()

    if was_training:
        teacher.train()

    return target_latent
