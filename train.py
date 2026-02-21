import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.himac_jepa import HiMACJEPA
from src.data.dataset import MultiModalDrivingDataset, collate_fn
from src.losses.predictive_loss import KLDivergenceLoss, NLLLoss
from src.losses.vicreg_loss import VICRegLoss
from src.masking.spatio_temporal_masking import SpatioTemporalMasking
def update_ema_params(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(model_p, alpha=1 - decay)

class Config:
    # Model parameters
    latent_dim = 128
    camera_encoder_params = {}
    lidar_encoder_params = {}
    radar_encoder_params = {}
    fusion_module_params = {}
    action_encoder_params = {}
    predictor_params = {}

    # Head parameters (dummy values for now)
    trajectory_head_output_dim = 2  # Example output dimension
    motion_prediction_head_output_dim = 4  # Example output dimension
    bev_segmentation_head_bev_h = 20
    bev_segmentation_head_bev_w = 20
    bev_segmentation_head_num_classes = 5  # Example number of classes

    # Dataset parameters
    data_dir = "./data"
    batch_size = 4
    num_workers = 0

    # Training parameters
    learning_rate = 1e-4
    num_epochs = 10

    # Loss weights
    lambda_param = 25.0  # VICReg invariance
    mu_param = 25.0      # VICReg variance
    nu_param = 1.0       # VICReg covariance
    ema_decay = 0.999    # EMA decay rate

    # Masking parameters
    mask_ratio_spatial = 0.75
    mask_ratio_temporal = 0.5
    use_masking = True  # Enable JEPA masking

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # 1. Data Loading
    # Create a dummy config for the dataset
    dataset_config = {
        'data': {
            'batch_size': cfg.batch_size,
            'num_workers': cfg.num_workers,
            'augmentation': False,
            'num_samples': 10,
            'lidar_points': 1024
        }
    }
    dataset = MultiModalDrivingDataset(dataset_config)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)

    # 2. Model Instantiation
    model_config = {
        "model": {
            "latent_dim": cfg.latent_dim,
            "camera_encoder_params": cfg.camera_encoder_params,
            "lidar_encoder_params": cfg.lidar_encoder_params,
            "radar_encoder_params": cfg.radar_encoder_params,
            "fusion_module_params": cfg.fusion_module_params,
            "action_encoder": {
                "strategic_vocab_size": 10,
                "tactical_dim": 3,
                "latent_dim": 128,
                "num_heads": 8,
                "depth": 2,
                "dropout": 0.1
            },
            "predictor_params": cfg.predictor_params,
        },
        "trajectory_head": {"output_dim": cfg.trajectory_head_output_dim},
        "motion_prediction_head": {"output_dim": cfg.motion_prediction_head_output_dim},
        "bev_segmentation_head": {
            "bev_h": cfg.bev_segmentation_head_bev_h,
            "bev_w": cfg.bev_segmentation_head_bev_w,
            "num_classes": cfg.bev_segmentation_head_num_classes,
        },
    }
    model = HiMACJEPA(model_config)
    ema_model = HiMACJEPA(model_config) # Initialize EMA model
    ema_model.load_state_dict(model.state_dict()) # Copy initial weights
    for param in ema_model.parameters():
        param.requires_grad = False # EMA model is not trained via backprop

    print("Model instantiated successfully:")
    print(model)

    # Instantiate loss functions
    predictive_loss_fn = KLDivergenceLoss(reduction='mean') # Or NLLLoss
    vicreg_loss_fn = VICRegLoss(lambda_param=cfg.lambda_param, mu_param=cfg.mu_param, nu_param=cfg.nu_param)

    # Optimizer (will be used in the next phase)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Initialize masking module for JEPA training
    masker = None
    if cfg.use_masking:
        masker = SpatioTemporalMasking(
            mask_ratio_spatial=cfg.mask_ratio_spatial,
            mask_ratio_temporal=cfg.mask_ratio_temporal,
            patch_size_camera=(16, 16),
            num_temporal_steps=5
        )
        print("Masking module initialized for JEPA training.")

    print("Loss functions instantiated successfully.")
    print("Optimizer instantiated successfully.")

    print("EMA model instantiated and initialized.")

    # Training loop placeholder
    for epoch in range(cfg.num_epochs):
        running_total_loss = 0.0
        running_predictive_loss = 0.0
        running_vicreg_loss = 0.0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            # Extract multi-modal inputs and actions from batch
            camera = batch['camera']
            lidar = batch['lidar']
            radar = batch['radar']
            strategic_action = batch['strategic_action']
            tactical_action = batch['tactical_action']

            # Generate masks if JEPA masking is enabled
            masks = None
            if cfg.use_masking and masker is not None:
                masks = masker.generate_joint_mask(
                    camera_shape=camera.shape[1:],  # (C, H, W)
                    lidar_shape=lidar.shape[1:],     # (N, 3)
                    radar_shape=radar.shape[1:],     # (C, H, W)
                    batch_size=camera.shape[0]
                )

            # 1. Feed multi-modal inputs and actions into the HiMACJEPA model (with masks)
            mu, log_var, _, _, _ = model(camera, lidar, radar, strategic_action, tactical_action, masks)

            # Use EMA model to get target latent representation (without masks - full input)
            with torch.no_grad():
                ema_mu, _, _, _, _ = ema_model(camera, lidar, radar, strategic_action, tactical_action, None)
                target_latent = ema_mu.detach()

            # 2. Compute predictive loss
            # For now, let's assume target_latent is available in the batch for predictive loss
            # In a true JEPA setup, target_latent would be derived from a target encoder or masked input
            # Assuming target_latent is mu_q, and we use a dummy log_var_q (e.g., zeros)
            predictive_loss = predictive_loss_fn(mu, log_var, target_latent, torch.zeros_like(log_var))

            # 3. Compute VICReg regularization loss from latent representations
            vicreg_loss = vicreg_loss_fn(mu, target_latent)

            # 4. Combine losses into total loss for backpropagation
            total_loss = predictive_loss + vicreg_loss

            # 5. Perform backward pass and optimizer step
            total_loss.backward()
            optimizer.step()

            # 6. Add EMA target encoder updates
            update_ema_params(model, ema_model, decay=cfg.ema_decay)

            running_total_loss += total_loss.item()
            running_predictive_loss += predictive_loss.item()
            running_vicreg_loss += vicreg_loss.item()

            if i % 10 == 0: # Log every 10 batches
                print(f"Epoch {epoch}, Batch {i}: Total Loss: {total_loss.item():.4f}, Predictive Loss: {predictive_loss.item():.4f}, VICReg Loss: {vicreg_loss.item():.4f}")
        
        avg_total_loss = running_total_loss / len(dataloader)
        avg_predictive_loss = running_predictive_loss / len(dataloader)
        avg_vicreg_loss = running_vicreg_loss / len(dataloader)
        print(f"Epoch {epoch} Summary: Avg Total Loss: {avg_total_loss:.4f}, Avg Predictive Loss: {avg_predictive_loss:.4f}, Avg VICReg Loss: {avg_vicreg_loss:.4f}")

if __name__ == "__main__":
    main()
