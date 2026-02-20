import torch
import yaml
from src.models.himac_jepa import HiMACJEPA
from src.models.bev_semantic_segmentation_head import BEVSemanticSegmentationHead

def test_bev_semantic_segmentation_head():
    print("\nRunning test_bev_semantic_segmentation_head...")
    latent_dim = 512
    bev_h = 200
    bev_w = 200
    num_classes = 10
    head = BEVSemanticSegmentationHead(latent_dim, bev_h, bev_w, num_classes)
    
    # Create a dummy latent representation
    dummy_latent = torch.randn(1, latent_dim)
    
    # Forward pass
    output = head(dummy_latent)
    
    assert output.shape == (1, num_classes, bev_h, bev_w), f"Expected output shape (1, {num_classes}, {bev_h}, {bev_w}), but got {output.shape}"
    print("BEVSemanticSegmentationHead instantiation and forward pass successful.")

def test_himac_jepa_with_bev_semantic_segmentation_head():
    print("\nRunning test_himac_jepa_with_bev_semantic_segmentation_head...")
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
    mu, log_var, trajectory, motion_predictions, bev_segmentation_map = model(camera_input, lidar_input, radar_input)

    expected_latent_dim = config["model"]["latent_dim"]
    assert mu.shape == (1, expected_latent_dim), f"Expected mu shape (1, {expected_latent_dim}), but got {mu.shape}"
    assert log_var.shape == (1, expected_latent_dim), f"Expected log_var shape (1, {expected_latent_dim}), but got {log_var.shape}"
    expected_trajectory_output_dim = config["trajectory_head"]["output_dim"]
    assert trajectory.shape == (1, expected_trajectory_output_dim), f"Expected trajectory shape (1, {expected_trajectory_output_dim}), but got {trajectory.shape}"
    expected_motion_prediction_output_dim = config["motion_prediction_head"]["output_dim"]
    assert motion_predictions.shape == (1, expected_motion_prediction_output_dim), f"Expected motion_predictions shape (1, {expected_motion_prediction_output_dim}), but got {motion_predictions.shape}"
    expected_bev_h = config["bev_segmentation_head"]["bev_h"]
    expected_bev_w = config["bev_segmentation_head"]["bev_w"]
    expected_num_classes = config["bev_segmentation_head"]["num_classes"]
    assert bev_segmentation_map.shape == (1, expected_num_classes, expected_bev_h, expected_bev_w), f"Expected BEV segmentation map shape (1, {expected_num_classes}, {expected_bev_h}, {expected_bev_w}), but got {bev_segmentation_map.shape}"
    print("HiMACJEPA model with BEVSemanticSegmentationHead forward pass successful.")

if __name__ == "__main__":
    test_bev_semantic_segmentation_head()
    test_himac_jepa_with_bev_semantic_segmentation_head()
