import torch
import torch.nn as nn
import torch.nn.functional as F
from .trajectory_planning_head import TrajectoryPlanningHead
from .motion_prediction_head import MotionPredictionHead
from .bev_semantic_segmentation_head import BEVSemanticSegmentationHead
from .action_encoder import HierarchicalActionEncoder

class CameraEncoder(nn.Module):
    """Vision Transformer based camera encoder with configurable depth."""
    def __init__(self, embed_dim=768, patch_size=16, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth

        # Patch embedding
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))  # 196 patches + 1 CLS

        # Transformer blocks with GELU activation and configurable depth
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            ) for _ in range(depth)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]

        # Patch embedding
        x = self.proj(x)  # [B, embed_dim, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer expects [seq_len, batch, embed_dim]
        x = x.transpose(0, 1)  # [num_patches+1, B, embed_dim]

        # Apply transformer blocks
        x = self.blocks(x)

        # Apply layer norm
        x = self.norm(x)

        # Return CLS token representation
        x = x[0]  # [B, embed_dim]
        return x

class LiDAREncoder(nn.Module):
    """Enhanced PointNet-style LiDAR encoder with hierarchical feature learning."""
    def __init__(self, out_channels=512, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels

        # Hierarchical MLP layers with batch norm and dropout
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Global feature aggregation
        self.mlp_global = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        # x: [B, N, 3]
        B, N, _ = x.shape

        # Apply point-wise MLPs
        # Reshape for batch norm: [B, N, C] -> [B*N, C]
        x = x.reshape(B * N, 3)
        x = self.mlp1(x)  # [B*N, 64]
        x = self.mlp2(x)  # [B*N, 128]
        x = self.mlp3(x)  # [B*N, 256]

        # Reshape back: [B*N, 256] -> [B, N, 256]
        x = x.reshape(B, N, 256)

        # Global max pooling across points
        x = torch.max(x, dim=1)[0]  # [B, 256]

        # Final global feature extraction
        x = self.mlp_global(x)  # [B, out_channels]

        return x

class RadarEncoder(nn.Module):
    """Velocity-aware CNN-based Radar encoder with dual-branch processing."""
    def __init__(self, out_channels=256, input_channels=1, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.input_channels = input_channels

        # Spatial feature extraction branch
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Feature fusion and output projection
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_channels)
        )

    def forward(self, x):
        # x: [B, C, H, W] where C could include velocity channels
        # Extract spatial features
        x = self.spatial_branch(x)  # [B, 128, 1, 1]
        x = x.flatten(1)  # [B, 128]

        # Project to output dimension
        x = self.fc(x)  # [B, out_channels]

        return x

class MultiModalFusion(nn.Module):
    """Attention-based fusion of multi-modal features."""
    def __init__(self, latent_dim=512):
        super().__init__()
        self.cam_proj = nn.Linear(768, latent_dim)
        self.lidar_proj = nn.Linear(512, latent_dim)
        self.radar_proj = nn.Linear(256, latent_dim)
        self.fusion_layer = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8)

    def forward(self, cam_feat, lidar_feat, radar_feat):
        # Project to common dimension
        f_cam = self.cam_proj(cam_feat).unsqueeze(0)
        f_lidar = self.lidar_proj(lidar_feat).unsqueeze(0)
        f_radar = self.radar_proj(radar_feat).unsqueeze(0)
        
        # Concatenate for attention
        combined = torch.cat([f_cam, f_lidar, f_radar], dim=0)
        attn_output, _ = self.fusion_layer(combined, combined, combined)
        
        # Mean pooling of fused features
        return attn_output.mean(dim=0)

class HiMACJEPA(nn.Module):
    """Full HiMAC-JEPA Model."""
    def __init__(self, config):
        super().__init__()
        # Camera encoder with config
        cam_config = config['model'].get('camera_encoder', {})
        self.camera_encoder = CameraEncoder(
            embed_dim=cam_config.get('embed_dim', 768),
            patch_size=cam_config.get('patch_size', 16),
            depth=cam_config.get('depth', 12),
            num_heads=cam_config.get('num_heads', 12),
            dropout=cam_config.get('dropout', 0.1)
        )
        # LiDAR encoder with config
        lidar_config = config['model'].get('lidar_encoder', {})
        self.lidar_encoder = LiDAREncoder(
            out_channels=lidar_config.get('out_channels', 512),
            dropout=lidar_config.get('dropout', 0.1)
        )
        self.radar_encoder = RadarEncoder()
        self.fusion = MultiModalFusion(latent_dim=config['model']['latent_dim'])

        # Action encoder
        action_config = config['model'].get('action_encoder', {})
        self.action_encoder = HierarchicalActionEncoder(
            strategic_vocab_size=action_config.get('strategic_vocab_size', 10),
            tactical_dim=action_config.get('tactical_dim', 3),
            latent_dim=action_config.get('latent_dim', 128),
            num_heads=action_config.get('num_heads', 8),
            depth=action_config.get('depth', 2),
            dropout=action_config.get('dropout', 0.1)
        )

        # Predictor input dimension: latent_dim + action_latent_dim
        predictor_input_dim = config['model']['latent_dim'] + action_config.get('latent_dim', 128)
        self.predictor = nn.TransformerEncoderLayer(d_model=predictor_input_dim, nhead=8)
        self.dist_head = nn.Linear(predictor_input_dim, config["model"]["latent_dim"] * 2) # mu and log_var
        self.trajectory_head = TrajectoryPlanningHead(latent_dim=config["model"]["latent_dim"], output_dim=config["trajectory_head"]["output_dim"])
        self.motion_prediction_head = MotionPredictionHead(latent_dim=config["model"]["latent_dim"], output_dim=config["motion_prediction_head"]["output_dim"])
        self.bev_segmentation_head = BEVSemanticSegmentationHead(latent_dim=config["model"]["latent_dim"], bev_h=config["bev_segmentation_head"]["bev_h"], bev_w=config["bev_segmentation_head"]["bev_w"], num_classes=config["bev_segmentation_head"]["num_classes"])

    def apply_lidar_mask(self, lidar, mask):
        """Apply mask to LiDAR point cloud by zeroing out masked points."""
        # lidar: (B, N, 3), mask: (B, N)
        masked_lidar = lidar.clone()
        masked_lidar[mask] = 0.0
        return masked_lidar

    def forward(self, camera, lidar, radar, strategic_action, tactical_action, masks=None):
        """Forward pass with action conditioning and optional masking.

        Args:
            camera: Camera input tensor (B, C, H, W)
            lidar: LiDAR input tensor (B, N, 3)
            radar: Radar input tensor (B, C, H, W)
            strategic_action: Strategic action tensor (B,) or (B, 1)
            tactical_action: Tactical action tensor (B, tactical_dim)
            masks: Optional dict of masks for JEPA training
                   {'camera': (B, H_p, W_p), 'lidar': (B, N), 'radar': (B, H_p, W_p)}

        Returns:
            Tuple of (mu, log_var, trajectory, motion_predictions, bev_segmentation_map)
        """
        # Apply masking if provided (for JEPA training)
        if masks is not None:
            # Apply LiDAR mask (simple zero-out approach)
            lidar = self.apply_lidar_mask(lidar, masks['lidar'])
            # Note: Camera and Radar masking would require more complex logic
            # For now, we process full inputs and apply masking at latent level

        # Encode multi-modal sensor inputs
        cam_feat = self.camera_encoder(camera)
        lidar_feat = self.lidar_encoder(lidar)
        radar_feat = self.radar_encoder(radar)

        # Fuse multi-modal features
        z_t = self.fusion(cam_feat, lidar_feat, radar_feat)

        # Encode hierarchical actions
        action_latent = self.action_encoder(strategic_action, tactical_action)

        # Concatenate sensor representation with action representation
        z_t_action = torch.cat([z_t, action_latent], dim=-1)

        # Predict future latent distribution conditioned on actions
        pred_out = self.predictor(z_t_action.unsqueeze(0)).squeeze(0)
        dist_params = self.dist_head(pred_out)
        mu, log_var = torch.chunk(dist_params, 2, dim=-1)

        # Downstream task predictions
        trajectory = self.trajectory_head(mu)
        motion_predictions = self.motion_prediction_head(mu)
        bev_segmentation_map = self.bev_segmentation_head(mu)

        return mu, log_var, trajectory, motion_predictions, bev_segmentation_map
