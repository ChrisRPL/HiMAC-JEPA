"""Training script for baseline models."""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_baseline_model(config: dict):
    """
    Create baseline model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        model: Baseline model instance
    """
    model_name = config['model']['name']

    if model_name == 'camera_only':
        from src.models.baselines import CameraOnlyBaseline
        model = CameraOnlyBaseline(config['model'])

    elif model_name == 'lidar_only':
        from src.models.baselines import LiDAROnlyBaseline
        model = LiDAROnlyBaseline(config['model'])

    elif model_name == 'radar_only':
        from src.models.baselines import RadarOnlyBaseline
        model = RadarOnlyBaseline(config['model'])

    elif model_name == 'ijepa':
        from src.models.baselines import IJEPABaseline
        model = IJEPABaseline(config['model'])

    elif model_name == 'vjepa':
        from src.models.baselines import VJEPABaseline
        model = VJEPABaseline(config['model'])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def create_dataloader(config: dict, split: str):
    """
    Create dataloader for baseline training.

    Args:
        config: Data configuration
        split: 'train' or 'val'

    Returns:
        dataloader: PyTorch DataLoader
    """
    # This is a placeholder - in real implementation, use actual dataset
    # For now, return None to indicate not implemented
    print(f"Warning: Dataloader creation not fully implemented")
    print(f"  Would create {split} dataloader with config: {config['data']}")

    return None


def create_optimizer(model: nn.Module, config: dict):
    """
    Create optimizer from config.

    Args:
        model: Model to optimize
        config: Training configuration

    Returns:
        optimizer: PyTorch optimizer
    """
    lr = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.05)
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config: dict):
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        config: Training configuration

    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 10)

    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-6
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif scheduler_name == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def setup_wandb(config: dict):
    """
    Setup Weights & Biases logging.

    Args:
        config: Logging configuration
    """
    if not config['logging']['wandb']['enabled']:
        return None

    try:
        import wandb

        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb'].get('entity'),
            tags=config['logging']['wandb'].get('tags', []),
            config=config
        )

        return wandb

    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")
        return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    device: torch.device,
    epoch: int,
    config: dict,
    wandb=None
):
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Training config
        wandb: Weights & Biases instance (optional)

    Returns:
        avg_metrics: Dictionary of average metrics
    """
    model.train()

    total_metrics = {}
    num_batches = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Training step
        metrics = model.train_step(batch, optimizer)

        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

        num_batches += 1

        # Update progress bar
        pbar.set_postfix(loss=metrics['loss'])

        # Log to wandb
        if wandb and batch_idx % config['logging']['log_interval'] == 0:
            wandb.log({
                'train/' + k: v for k, v in metrics.items()
            }, step=epoch * len(dataloader) + batch_idx)

    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    return avg_metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    wandb=None
):
    """
    Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device
        epoch: Current epoch
        wandb: Weights & Biases instance (optional)

    Returns:
        avg_metrics: Dictionary of average metrics
    """
    model.eval()

    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Validation step
            metrics = model.val_step(batch)

            # Accumulate metrics
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

            num_batches += 1

    # Average metrics
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    # Log to wandb
    if wandb:
        wandb.log({'val/' + k: v for k, v in avg_metrics.items()}, step=epoch)

    return avg_metrics


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    config: dict,
    best: bool = False
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        config: Configuration
        best: Whether this is the best model so far
    """
    save_dir = Path(config['checkpoint']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"

    model.save_checkpoint(
        str(checkpoint_path),
        epoch=epoch,
        optimizer=optimizer,
        scheduler_state_dict=scheduler.state_dict() if scheduler else None
    )

    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if best:
        best_path = save_dir / "best_model.pth"
        model.save_checkpoint(
            str(best_path),
            epoch=epoch,
            optimizer=optimizer
        )
        print(f"Saved best model: {best_path}")


def train(config: dict, args):
    """
    Main training loop.

    Args:
        config: Training configuration
        args: Command line arguments
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print(f"Creating model: {config['model']['name']}")
    model = create_baseline_model(config)
    model = model.to(device)

    print(f"Model parameters: {model.get_num_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')

    if train_loader is None or val_loader is None:
        print("\nERROR: Dataloaders not implemented yet.")
        print("This script provides the training structure.")
        print("Implement create_dataloader() to use actual data.")
        return

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Setup W&B
    wandb = setup_wandb(config)

    # Training loop
    num_epochs = config['training']['num_epochs']
    save_interval = config['logging']['save_interval']

    best_val_loss = float('inf')

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch, config, wandb
        )

        print(f"Train metrics: {train_metrics}")

        # Validate
        val_metrics = validate(model, val_loader, device, epoch, wandb)

        print(f"Val metrics: {val_metrics}")

        # Update scheduler
        if scheduler is not None and epoch > config['training'].get('warmup_epochs', 10):
            scheduler.step()

        # Save checkpoint
        if epoch % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, config)

        # Save best model
        val_loss = val_metrics.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, config, best=True)

    print("\nTraining complete!")

    if wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['camera_only', 'lidar_only', 'radar_only', 'ijepa', 'vjepa'],
        help='Baseline model to train'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: configs/baseline/{model}.yaml)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate'
    )

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging'
    )

    args = parser.parse_args()

    # Load config
    if args.config is None:
        config_path = project_root / f"configs/baseline/{args.model}.yaml"
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override config with command line args
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.lr is not None:
        config['training']['learning_rate'] = args.lr

    if args.resume is not None:
        config['checkpoint']['resume'] = args.resume

    if args.no_wandb:
        config['logging']['wandb']['enabled'] = False

    # Train
    train(config, args)


if __name__ == '__main__':
    main()
