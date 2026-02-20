import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.himac_jepa import HiMACJEPA

# Placeholder for a dummy dataset
class DummyTrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, latent_dim=512, output_dim=30):
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Generate random dummy data
        self.dummy_camera = torch.randn(num_samples, 3, 224, 224)
        self.dummy_lidar = torch.randn(num_samples, 1024, 3)
        self.dummy_radar = torch.randn(num_samples, 1, 64, 64)
        self.dummy_trajectories = torch.randn(num_samples, output_dim)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.dummy_camera[idx],
            self.dummy_lidar[idx],
            self.dummy_radar[idx],
            self.dummy_trajectories[idx],
        )

def finetune_trajectory_planning(config_path="configs/config.yaml", epochs=10, batch_size=32):
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

    # Ensure trajectory head is trainable
    for param in model.trajectory_head.parameters():
        param.requires_grad = True

    # Dummy dataset and dataloader
    dataset = DummyTrajectoryDataset(latent_dim=config["model"]["latent_dim"], output_dim=config["trajectory_head"]["output_dim"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.trajectory_head.parameters(), lr=float(config["training"]["learning_rate"]))

    # Training loop
    print(f"\nStarting fine-tuning for Trajectory Planning Head for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (camera, lidar, radar, target_trajectory) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass - only trajectory head is trained
            mu, log_var, predicted_trajectory, _, _ = model(camera, lidar, radar)
            
            loss = criterion(predicted_trajectory, target_trajectory)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    print("Fine-tuning for Trajectory Planning Head complete.")

if __name__ == "__main__":
    # This script should be run from the HiMAC-JEPA directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    finetune_trajectory_planning()
