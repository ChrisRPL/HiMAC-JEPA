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
    # Load the concrete model config plus downstream head settings.
    with open("configs/config.yaml", "r") as f:
        root_config = yaml.safe_load(f)
    with open("configs/model/default.yaml", "r") as f:
        model_config = yaml.safe_load(f)

    config = {
        "model": model_config,
        "trajectory_head": root_config["trajectory_head"],
        "motion_prediction_head": root_config["motion_prediction_head"],
        "bev_segmentation_head": root_config["bev_segmentation_head"],
    }

    # Instantiate the model
    model = HiMACJEPA(config)
    model.eval()

    # Create dummy inputs
    camera_input = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width
    lidar_input = torch.randn(1, 1024, 3)       # Batch, Num_points, Coords
    radar_input = torch.randn(1, 1, 64, 64)     # Batch, Channels, Height, Width
    strategic_action = torch.randint(0, config["model"]["action_encoder"]["strategic_vocab_size"], (1,))
    tactical_action = torch.randn(1, config["model"]["action_encoder"]["tactical_dim"])

    # Forward pass
    with torch.no_grad():
        mu, log_var, trajectory, _, _ = model(
            camera_input,
            lidar_input,
            radar_input,
            strategic_action,
            tactical_action,
        )

    expected_latent_dim = config["model"]["latent_dim"]
    assert mu.shape == (1, expected_latent_dim), f"Expected mu shape (1, {expected_latent_dim}), but got {mu.shape}"
    assert log_var.shape == (1, expected_latent_dim), f"Expected log_var shape (1, {expected_latent_dim}), but got {log_var.shape}"
    expected_trajectory_output_dim = config["trajectory_head"]["output_dim"]
    assert trajectory.shape == (1, expected_trajectory_output_dim), f"Expected trajectory shape (1, {expected_trajectory_output_dim}), but got {trajectory.shape}"
    print("HiMACJEPA model with TrajectoryPlanningHead forward pass successful.")

if __name__ == "__main__":
    test_trajectory_planning_head()
    test_himac_jepa_with_trajectory_head()
