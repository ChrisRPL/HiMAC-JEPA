"""Baseline models for comparison with HiMAC-JEPA."""

from .base import BaselineModel
from .camera_only import CameraOnlyBaseline

__all__ = [
    'BaselineModel',
    'CameraOnlyBaseline',
]
