import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVSemanticSegmentationHead(nn.Module):
    """BEV Semantic Segmentation Head that takes latent representations and outputs a BEV segmentation map."""
    def __init__(self, latent_dim: int, bev_h: int = 200, bev_w: int = 200, num_classes: int = 10):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes

        # Project latent_dim to a feature map with initial_channels and small spatial dimensions
        self.initial_channels = 256
        self.initial_h = 8
        self.initial_w = 8
        self.fc = nn.Linear(latent_dim, self.initial_channels * self.initial_h * self.initial_w)
        
        # Deconvolution layers to upsample
        self.deconv1 = nn.ConvTranspose2d(self.initial_channels, 128, kernel_size=4, stride=2, padding=1) # Upsample to 16x16
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # Upsample to 32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, num_classes, kernel_size=4, stride=2, padding=1) # Upsample to 64x64

    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        batch_size = latent_representation.shape[0]
        x = self.fc(latent_representation)
        
        # Reshape to a feature map for deconvolution (batch_size, initial_channels, initial_h, initial_w)
        x = x.view(batch_size, self.initial_channels, self.initial_h, self.initial_w)

        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        segmentation_map = self.deconv3(x)
        
        # Resize to target BEV dimensions
        segmentation_map = F.interpolate(segmentation_map, size=(self.bev_h, self.bev_w), mode='bilinear', align_corners=False)

        return segmentation_map
