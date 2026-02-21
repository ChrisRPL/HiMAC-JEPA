"""Evaluation script for HiMAC-JEPA model."""
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.himac_jepa import HiMACJEPA
from src.data.nuscenes_dataset import NuScenesMultiModalDataset
from src.data.dataset import MultiModalDrivingDataset
from src.evaluation.intrinsic_metrics import IntrinsicEvaluator
from src.evaluation.downstream_metrics import DownstreamEvaluator

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed, logging disabled")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device='cuda'):
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from model_state_dict")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded from state_dict")
    else:
        # Assume checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint directly")

    model.to(device)
    model.eval()
    print("Checkpoint loaded successfully")

    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    """
    Run evaluation on trained model.

    Args:
        cfg: Hydra configuration
    """
    print("="*60)
    print("HiMAC-JEPA Evaluation")
    print("="*60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize W&B if available and enabled
    if WANDB_AVAILABLE and cfg.evaluation.wandb.get('enabled', False):
        wandb.init(
            project=cfg.evaluation.wandb.get('project', 'himac-jepa'),
            entity=cfg.evaluation.wandb.get('entity', None),
            name=f"eval_{cfg.get('experiment_name', 'himac-jepa')}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.evaluation.wandb.get('tags', ['evaluation'])
        )
        print("\nWeights & Biases initialized for evaluation")
    else:
        print("\nWeights & Biases logging disabled")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load validation dataset
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)

    if cfg.data.dataset == 'nuscenes':
        print("Using nuScenes dataset...")
        val_dataset = NuScenesMultiModalDataset(cfg.data, split='val')
    else:
        print("Using dummy dataset...")
        dataset_config = {
            'data': {
                'batch_size': cfg.data.batch_size,
                'num_workers': cfg.data.num_workers,
                'augmentation': False,
                'num_samples': 10,
                'lidar_points': cfg.data.num_points
            }
        }
        val_dataset = MultiModalDrivingDataset(dataset_config)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.get('num_workers', 0)
    )
    print(f"Loaded {len(val_dataset)} validation samples")

    # Initialize model
    print("\n" + "="*60)
    print("Initializing Model")
    print("="*60)

    model_config = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "trajectory_head": OmegaConf.to_container(cfg.trajectory_head, resolve=True),
        "motion_prediction_head": OmegaConf.to_container(cfg.motion_prediction_head, resolve=True),
        "bev_segmentation_head": OmegaConf.to_container(cfg.bev_segmentation_head, resolve=True),
    }
    model = HiMACJEPA(model_config)

    # Load checkpoint if available
    checkpoint_path = cfg.evaluation.checkpoint_path
    if Path(checkpoint_path).exists():
        model = load_checkpoint(checkpoint_path, model, device)
    else:
        print(f"\nWarning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating with random initialization (for testing only)")

    # Run evaluations
    results = {}

    # Intrinsic evaluation
    if cfg.evaluation.intrinsic.enabled:
        print("\n" + "="*60)
        print("Running Intrinsic Evaluation")
        print("="*60)

        intrinsic_eval = IntrinsicEvaluator(model, val_loader, device)
        intrinsic_results = intrinsic_eval.run_all(cfg.evaluation)
        results.update(intrinsic_results)

        print("\nIntrinsic Results:")
        for key, value in intrinsic_results.items():
            print(f"  {key}: {value:.4f}")

    # Downstream evaluation
    if cfg.evaluation.downstream.enabled:
        print("\n" + "="*60)
        print("Running Downstream Evaluation")
        print("="*60)

        downstream_eval = DownstreamEvaluator(model, val_loader, device)
        downstream_results = downstream_eval.run_all(cfg.evaluation)
        results.update(downstream_results)

        print("\nDownstream Results:")
        for key, value in downstream_results.items():
            print(f"  {key}: {value:.4f}")

    # Log to W&B
    if WANDB_AVAILABLE and cfg.evaluation.wandb.get('enabled', False):
        wandb.log(results)
        print("\nResults logged to Weights & Biases")

    # Save results to file
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)

    results_dir = Path("evaluation_results")
    results_dir.mkdir(exist_ok=True)

    experiment_name = cfg.get('experiment_name', 'default')
    results_file = results_dir / f"results_{experiment_name}.json"

    with open(results_file, 'w') as f:
        # Convert numpy types to float for JSON serialization
        json_results = {k: float(v) if hasattr(v, 'item') else v
                       for k, v in results.items()}
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {results_file}")

    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for key, value in results.items():
        print(f"{key:40s}: {value:.4f}")

    # Finish W&B
    if WANDB_AVAILABLE and cfg.evaluation.wandb.get('enabled', False):
        wandb.finish()

    print("\n" + "="*60)
    print("Evaluation Complete")
    print("="*60)

    return results


if __name__ == "__main__":
    evaluate()
