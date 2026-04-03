"""Observation-only EMA teacher for HiMAC-JEPA."""

import torch
import torch.nn as nn

from .himac_jepa import CameraEncoder, LiDAREncoder, MultiModalFusion, RadarEncoder
from .temporal_fusion import TemporalTransformer


class HiMACObservationTeacher(nn.Module):
    """Slim EMA teacher that only encodes observations."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        cam_config = config["model"].get("camera_encoder", {})
        self.camera_encoder = CameraEncoder(
            embed_dim=cam_config.get("embed_dim", 768),
            patch_size=cam_config.get("patch_size", 16),
            depth=cam_config.get("depth", 12),
            num_heads=cam_config.get("num_heads", 12),
            dropout=cam_config.get("dropout", 0.1),
        )

        lidar_config = config["model"].get("lidar_encoder", {})
        self.lidar_encoder = LiDAREncoder(
            out_channels=lidar_config.get("out_channels", 512),
            dropout=lidar_config.get("dropout", 0.1),
        )

        radar_config = config["model"].get("radar_encoder", {})
        self.radar_encoder = RadarEncoder(
            out_channels=radar_config.get("out_channels", 256),
            input_channels=radar_config.get("input_channels", 1),
            dropout=radar_config.get("dropout", 0.1),
        )

        self.fusion = MultiModalFusion(latent_dim=config["model"]["latent_dim"])

        if config["model"].get("use_temporal_fusion", False):
            self.temporal_fusion = TemporalTransformer(
                d_model=config["model"]["latent_dim"],
                nhead=config["model"].get("temporal_heads", 8),
                num_layers=config["model"].get("temporal_layers", 4),
                dropout=config["model"].get("dropout", 0.1),
            )
            self.use_temporal = True
        else:
            self.use_temporal = False

        for param in self.parameters():
            param.requires_grad = False

    def load_from_student(self, student):
        """Initialize teacher weights from the student observation path."""
        self.camera_encoder.load_state_dict(student.camera_encoder.state_dict())
        self.lidar_encoder.load_state_dict(student.lidar_encoder.state_dict())
        self.radar_encoder.load_state_dict(student.radar_encoder.state_dict())
        self.fusion.load_state_dict(student.fusion.state_dict())

        if self.use_temporal:
            self.temporal_fusion.load_state_dict(student.temporal_fusion.state_dict())

    def update_from_student(self, student, decay):
        """EMA update from student observation parameters."""
        teacher_modules = [
            (self.camera_encoder, student.camera_encoder),
            (self.lidar_encoder, student.lidar_encoder),
            (self.radar_encoder, student.radar_encoder),
            (self.fusion, student.fusion),
        ]

        if self.use_temporal:
            teacher_modules.append((self.temporal_fusion, student.temporal_fusion))

        with torch.no_grad():
            for teacher_module, student_module in teacher_modules:
                for teacher_param, student_param in zip(
                    teacher_module.parameters(), student_module.parameters()
                ):
                    teacher_param.mul_(decay).add_(student_param, alpha=1 - decay)
                for teacher_buffer, student_buffer in zip(
                    teacher_module.buffers(), student_module.buffers()
                ):
                    teacher_buffer.copy_(student_buffer)

    def apply_lidar_mask(self, lidar, mask):
        masked_lidar = lidar.clone()
        masked_lidar[mask] = 0.0
        return masked_lidar

    def apply_spatial_mask(self, tensor, mask):
        batch_size, _, height, width = tensor.shape
        mask_h, mask_w = mask.shape[-2:]

        assert height % mask_h == 0 and width % mask_w == 0

        patch_h = height // mask_h
        patch_w = width // mask_w
        expanded_mask = mask.bool().repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)

        return tensor.masked_fill(expanded_mask.unsqueeze(1), 0.0)

    def _get_last_visible_idx(self, temporal_mask, seq_len, device):
        if temporal_mask is None:
            return None

        time_index = torch.arange(seq_len, device=device).unsqueeze(0).expand(temporal_mask.size(0), -1)
        last_visible_idx = time_index.masked_fill(temporal_mask, -1).max(dim=1).values

        return torch.where(
            last_visible_idx >= 0,
            last_visible_idx,
            torch.full_like(last_visible_idx, seq_len - 1),
        )

    def _encode_single_observation(self, camera, lidar, radar, masks=None):
        if masks is not None:
            if "camera" in masks:
                camera = self.apply_spatial_mask(camera, masks["camera"])
            if "lidar" in masks:
                lidar = self.apply_lidar_mask(lidar, masks["lidar"])
            if "radar" in masks:
                radar = self.apply_spatial_mask(radar, masks["radar"])

        cam_feat = self.camera_encoder(camera)
        lidar_feat = self.lidar_encoder(lidar)
        radar_feat = self.radar_encoder(radar)

        return self.fusion(cam_feat, lidar_feat, radar_feat)

    def _encode_temporal_observation(self, camera_seq, lidar_seq, radar_seq, masks=None):
        batch_size, seq_len = camera_seq.shape[:2]
        temporal_mask = None

        if masks is not None and "temporal" in masks:
            temporal_mask = masks["temporal"][:, :seq_len].bool()

        last_visible_idx = self._get_last_visible_idx(temporal_mask, seq_len, camera_seq.device)
        fused_features = []

        for timestep in range(seq_len):
            camera_frame = camera_seq[:, timestep]
            lidar_frame = lidar_seq[:, timestep]
            radar_frame = radar_seq[:, timestep]

            if temporal_mask is not None and timestep < temporal_mask.size(1):
                frame_mask = temporal_mask[:, timestep]
                camera_frame = camera_frame.masked_fill(frame_mask.view(batch_size, 1, 1, 1), 0.0)
                lidar_frame = lidar_frame.masked_fill(frame_mask.view(batch_size, 1, 1), 0.0)
                radar_frame = radar_frame.masked_fill(frame_mask.view(batch_size, 1, 1, 1), 0.0)

            fused_features.append(
                self._encode_single_observation(camera_frame, lidar_frame, radar_frame, masks=masks)
            )

        fused_features = torch.stack(fused_features, dim=1)

        if self.use_temporal:
            return self.temporal_fusion(fused_features, aggregate="last")
        if last_visible_idx is None:
            return fused_features[:, -1, :]

        gather_idx = last_visible_idx.view(batch_size, 1, 1)
        return fused_features.gather(1, gather_idx.expand(-1, 1, fused_features.size(-1))).squeeze(1)

    def encode_observations(self, camera, lidar, radar, masks=None):
        """Encode observations without action conditioning."""
        if camera.dim() == 5:
            return self._encode_temporal_observation(camera, lidar, radar, masks=masks)

        return self._encode_single_observation(camera, lidar, radar, masks=masks)

    def forward(self, camera, lidar, radar, masks=None):
        return self.encode_observations(camera, lidar, radar, masks=masks)
