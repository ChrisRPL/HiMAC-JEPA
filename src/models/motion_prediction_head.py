import torch
import torch.nn as nn

class MotionPredictionHead(nn.Module):
    """Motion Prediction Head that takes latent representations and outputs predicted motion for other agents."""
    def __init__(self, latent_dim: int, output_dim: int = 60): # e.g., 15 future timesteps * 2 (x,y coords) * 2 agents
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
        motion_predictions = self.fc3(x)
        return motion_predictions
