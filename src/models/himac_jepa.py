import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraEncoder(nn.Module):
    """Vision Transformer based camera encoder."""
    def __init__(self, embed_dim=768, patch_size=16):
        super().__init__()
        # Simplified ViT-like structure for skeleton
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, embed_dim))
        self.blocks = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12) for _ in range(4)])

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        return x[:, 0] # Return CLS token representation

class LiDAREncoder(nn.Module):
    """PointNet++ based LiDAR encoder."""
    def __init__(self, out_channels=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0] # Global max pooling
        return x

class RadarEncoder(nn.Module):
    """CNN based Radar encoder."""
    def __init__(self, out_channels=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        x = self.fc(x)
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
        self.camera_encoder = CameraEncoder()
        self.lidar_encoder = LiDAREncoder()
        self.radar_encoder = RadarEncoder()
        self.fusion = MultiModalFusion(latent_dim=config['model']['latent_dim'])
        
        # Predictor and Distribution Head
        self.predictor = nn.TransformerEncoderLayer(d_model=config['model']['latent_dim'], nhead=8)
        self.dist_head = nn.Linear(config['model']['latent_dim'], config['model']['latent_dim'] * 2) # mu and log_var

    def forward(self, camera, lidar, radar, actions=None):
        cam_feat = self.camera_encoder(camera)
        lidar_feat = self.lidar_encoder(lidar)
        radar_feat = self.radar_encoder(radar)
        
        z_t = self.fusion(cam_feat, lidar_feat, radar_feat)
        
        # Predict future latent distribution
        # In full implementation, this would be conditioned on actions
        pred_out = self.predictor(z_t.unsqueeze(0)).squeeze(0)
        dist_params = self.dist_head(pred_out)
        mu, log_var = torch.chunk(dist_params, 2, dim=-1)
        
        return mu, log_var
