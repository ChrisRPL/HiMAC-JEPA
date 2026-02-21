import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.models.himac_jepa import HiMACJEPA
from src.data.dataset import MultiModalDrivingDataset, collate_fn
from src.losses.predictive_loss import KLDivergenceLoss, NLLLoss
from src.losses.vicreg_loss import VICRegLoss
from src.masking.spatio_temporal_masking import SpatioTemporalMasking
def update_ema_params(model, ema_model, decay):
    """Update EMA model parameters."""
    with torch.no_grad():
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(model_p, alpha=1 - decay)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize Weights & Biases
    if cfg.get('wandb', {}).get('enabled', False):
        wandb.init(
            project=cfg.wandb.get('project', 'himac-jepa'),
            entity=cfg.wandb.get('entity', None),
            name=cfg.get('experiment_name', 'himac-jepa-run'),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.get('tags', [])
        )
        print("Weights & Biases initialized")
    else:
        print("Weights & Biases logging disabled")

    # 1. Data Loading
    # Select dataset based on config
    if cfg.data.dataset == 'nuscenes':
        # Use nuScenes dataset
        from src.data.nuscenes_dataset import NuScenesMultiModalDataset
        train_dataset = NuScenesMultiModalDataset(cfg.data, split='train')
        # nuScenes dataset already returns properly formatted batches
        dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers
        )
        print(f"Using nuScenes dataset: {len(train_dataset)} samples")
    else:
        # Use dummy dataset (default)
        dataset_config = {
            'data': {
                'batch_size': cfg.data.batch_size,
                'num_workers': cfg.data.num_workers,
                'augmentation': False,
                'num_samples': 10,
                'lidar_points': cfg.data.num_points
            }
        }
        dataset = MultiModalDrivingDataset(dataset_config)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn
        )
        print(f"Using dummy dataset: {len(dataset)} samples")

    # 2. Model Instantiation
    # Convert Hydra config to dict format expected by model
    model_config = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "trajectory_head": OmegaConf.to_container(cfg.trajectory_head, resolve=True),
        "motion_prediction_head": OmegaConf.to_container(cfg.motion_prediction_head, resolve=True),
        "bev_segmentation_head": OmegaConf.to_container(cfg.bev_segmentation_head, resolve=True),
    }
    model = HiMACJEPA(model_config)
    ema_model = HiMACJEPA(model_config) # Initialize EMA model
    ema_model.load_state_dict(model.state_dict()) # Copy initial weights
    for param in ema_model.parameters():
        param.requires_grad = False # EMA model is not trained via backprop

    print("Model instantiated successfully:")
    print(model)

    # Instantiate loss functions
    predictive_loss_fn = KLDivergenceLoss(reduction='mean')
    vicreg_loss_fn = VICRegLoss(
        lambda_param=cfg.training.lambda_param,
        mu_param=cfg.training.mu_param,
        nu_param=cfg.training.nu_param
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # Initialize masking module for JEPA training
    masker = None
    if cfg.training.use_masking:
        masker = SpatioTemporalMasking(
            mask_ratio_spatial=cfg.masking.spatial_ratio,
            mask_ratio_temporal=cfg.masking.temporal_ratio,
            patch_size_camera=tuple(cfg.masking.patch_size_camera),
            num_temporal_steps=cfg.masking.num_temporal_steps
        )
        print("Masking module initialized for JEPA training.")

    print("Loss functions instantiated successfully.")
    print("Optimizer instantiated successfully.")
    print("EMA model instantiated and initialized.")

    # Training loop
    for epoch in range(cfg.training.epochs):
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
            if cfg.training.use_masking and masker is not None:
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
            update_ema_params(model, ema_model, decay=cfg.training.ema_decay)

            running_total_loss += total_loss.item()
            running_predictive_loss += predictive_loss.item()
            running_vicreg_loss += vicreg_loss.item()

            if i % 10 == 0: # Log every 10 batches
                print(f"Epoch {epoch}, Batch {i}: Total Loss: {total_loss.item():.4f}, Predictive Loss: {predictive_loss.item():.4f}, VICReg Loss: {vicreg_loss.item():.4f}")

                # Log to W&B
                if cfg.get('wandb', {}).get('enabled', False):
                    wandb.log({
                        "train/total_loss": total_loss.item(),
                        "train/predictive_loss": predictive_loss.item(),
                        "train/vicreg_loss": vicreg_loss.item(),
                        "train/epoch": epoch,
                        "train/batch": i
                    })

        avg_total_loss = running_total_loss / len(dataloader)
        avg_predictive_loss = running_predictive_loss / len(dataloader)
        avg_vicreg_loss = running_vicreg_loss / len(dataloader)
        print(f"Epoch {epoch} Summary: Avg Total Loss: {avg_total_loss:.4f}, Avg Predictive Loss: {avg_predictive_loss:.4f}, Avg VICReg Loss: {avg_vicreg_loss:.4f}")

        # Log epoch summary to W&B
        if cfg.get('wandb', {}).get('enabled', False):
            wandb.log({
                "epoch/avg_total_loss": avg_total_loss,
                "epoch/avg_predictive_loss": avg_predictive_loss,
                "epoch/avg_vicreg_loss": avg_vicreg_loss,
                "epoch": epoch
            })

    # Finish W&B run
    if cfg.get('wandb', {}).get('enabled', False):
        wandb.finish()
        print("Weights & Biases run finished")

if __name__ == "__main__":
    main()
