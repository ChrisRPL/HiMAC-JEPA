import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.himac_jepa import HiMACJEPA

# Placeholder for a dummy dataset
class DummyBEVSegmentationDataset(Dataset):
    def __init__(self, num_samples=1000, latent_dim=512, bev_h=200, bev_w=200, num_classes=10):
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        
        # Generate random dummy data
        self.dummy_camera = torch.randn(num_samples, 3, 224, 224)
        self.dummy_lidar = torch.randn(num_samples, 1024, 3)
        self.dummy_radar = torch.randn(num_samples, 1, 64, 64)
        self.dummy_bev_maps = torch.randint(0, num_classes, (num_samples, bev_h, bev_w))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.dummy_camera[idx],
            self.dummy_lidar[idx],
            self.dummy_radar[idx],
            self.dummy_bev_maps[idx],
        )

def finetune_bev_segmentation(config_path="configs/config.yaml", epochs=10, batch_size=32):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = HiMACJEPA(config)

    # Freeze encoder and JEPA predictor weights
    for param in model.camera_encoder.parameters():
        param.requires_grad = False
    for param in model.lidar_encoder.parameters():
        param.requires_grad = False
    for param in model.radar_encoder.parameters():
        param.requires_grad = False
    for param in model.fusion.parameters():
        param.requires_grad = False
    for param in model.predictor.parameters():
        param.requires_grad = False
    for param in model.dist_head.parameters():
        param.requires_grad = False
    for param in model.trajectory_head.parameters():
        param.requires_grad = False
    for param in model.motion_prediction_head.parameters():
        param.requires_grad = False

    # Ensure BEV segmentation head is trainable
    for param in model.bev_segmentation_head.parameters():
        param.requires_grad = True

    # Dummy dataset and dataloader
    dataset = DummyBEVSegmentationDataset(
        latent_dim=config["model"]["latent_dim"],
        bev_h=config["bev_segmentation_head"]["bev_h"],
        bev_w=config["bev_segmentation_head"]["bev_w"],
        num_classes=config["bev_segmentation_head"]["num_classes"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # For semantic segmentation
    optimizer = optim.Adam(model.bev_segmentation_head.parameters(), lr=float(config["training"]["learning_rate"]))

    # Training loop
    print(f"\nStarting fine-tuning for BEV Semantic Segmentation Head for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (camera, lidar, radar, target_bev_map) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass - only BEV segmentation head is trained
            _, _, _, _, predicted_bev_map = model(camera, lidar, radar)
            
            loss = criterion(predicted_bev_map, target_bev_map)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    print("Fine-tuning for BEV Semantic Segmentation Head complete.")

if __name__ == "__main__":
    # This script should be run from the HiMAC-JEPA directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    finetune_bev_segmentation()
