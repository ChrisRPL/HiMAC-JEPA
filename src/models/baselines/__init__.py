"""Baseline models for comparison with HiMAC-JEPA."""

from .base import BaselineModel
from .camera_only import CameraOnlyBaseline
from .lidar_only import LiDAROnlyBaseline
from .radar_only import RadarOnlyBaseline
from .ijepa import IJEPABaseline

__all__ = [
    'BaselineModel',
    'CameraOnlyBaseline',
    'LiDAROnlyBaseline',
    'RadarOnlyBaseline',
    'IJEPABaseline',
]
