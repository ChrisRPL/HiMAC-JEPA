import torch
import torch.nn as nn

class TrajectoryPlanningHead(nn.Module):
    """Trajectory Planning Head that takes latent representations and outputs a predicted trajectory."""
    def __init__(self, latent_dim: int, output_dim: int = 30): # output_dim: e.g., 15 future timesteps * 2 (x,y coords)
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, latent_representation: torch.Tensor) -> torch.Tensor:
        x = self.fc1(latent_representation)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        trajectory = self.fc3(x)
        return trajectory
