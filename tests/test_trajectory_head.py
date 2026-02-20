import torch
import yaml
from src.models.himac_jepa import HiMACJEPA
from src.models.trajectory_planning_head import TrajectoryPlanningHead

def test_trajectory_planning_head():
    print("\nRunning test_trajectory_planning_head...")
    latent_dim = 512
    output_dim = 30
    head = TrajectoryPlanningHead(latent_dim, output_dim)
    
    # Create a dummy latent representation
    dummy_latent = torch.randn(1, latent_dim)
    
    # Forward pass
    output = head(dummy_latent)
    
    assert output.shape == (1, output_dim), f"Expected output shape (1, {output_dim}), but got {output.shape}"
    print("TrajectoryPlanningHead instantiation and forward pass successful.")

def test_himac_jepa_with_trajectory_head():
    print("\nRunning test_himac_jepa_with_trajectory_head...")
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Instantiate the model
    model = HiMACJEPA(config)

    # Create dummy inputs
    camera_input = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width
    lidar_input = torch.randn(1, 1024, 3)       # Batch, Num_points, Coords
    radar_input = torch.randn(1, 1, 64, 64)     # Batch, Channels, Height, Width

    # Forward pass
    mu, log_var, trajectory = model(camera_input, lidar_input, radar_input)

    expected_latent_dim = config["model"]["latent_dim"]
    assert mu.shape == (1, expected_latent_dim), f"Expected mu shape (1, {expected_latent_dim}), but got {mu.shape}"
    assert log_var.shape == (1, expected_latent_dim), f"Expected log_var shape (1, {expected_latent_dim}), but got {log_var.shape}"
    expected_trajectory_output_dim = config["trajectory_head"]["output_dim"]
    assert trajectory.shape == (1, expected_trajectory_output_dim), f"Expected trajectory shape (1, {expected_trajectory_output_dim}), but got {trajectory.shape}"
    print("HiMACJEPA model with TrajectoryPlanningHead forward pass successful.")

if __name__ == "__main__":
    test_trajectory_planning_head()
    test_himac_jepa_with_trajectory_head()
