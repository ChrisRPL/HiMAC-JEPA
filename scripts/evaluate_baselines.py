"""Evaluation and comparison script for baseline models."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    sns = None
    SEABORN_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.nuscenes_dataset import NuScenesMultiModalDataset
from src.evaluation.batching import collate_evaluation_batch
from src.evaluation.baseline_benchmark import (
    align_motion_targets,
    collect_bev_targets,
    collect_motion_targets,
    collect_probe_targets,
    compute_bev_classification_metrics,
    compute_motion_metrics,
    compute_trajectory_horizon_errors,
    compute_trajectory_horizon_metrics,
    build_motion_probe_targets,
    fit_bev_probe,
    fit_ridge_probe,
    move_batch_to_device,
    paired_sign_flip_test,
    predict_bev_probe,
    predict_ridge_probe,
)


def build_benchmark_loaders(
    data_root: str,
    version: str,
    batch_size: int,
    num_workers: int,
    cache_dir: str,
) -> Tuple[DataLoader, DataLoader]:
    """Build label-backed train/val dataloaders for benchmark evaluation."""
    data_cfg = OmegaConf.load(project_root / "configs" / "data" / "nuscenes.yaml")
    data_cfg.data_root = data_root
    data_cfg.version = version
    data_cfg.batch_size = batch_size
    data_cfg.num_workers = num_workers
    data_cfg.augmentation.enabled = False
    data_cfg.labels.enabled = True
    data_cfg.labels.trajectory.include_agents = False
    data_cfg.labels.cache.cache_dir = cache_dir

    train_dataset = NuScenesMultiModalDataset(data_cfg, split="train")
    val_dataset = NuScenesMultiModalDataset(data_cfg, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_evaluation_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_evaluation_batch,
    )

    return train_loader, val_loader


def load_himac_model(checkpoint_path: str, device: torch.device):
    """Load a HiMAC-JEPA checkpoint without relying on BaselineModel APIs."""
    from src.models.himac_jepa import HiMACJEPA

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("HiMAC-JEPA checkpoint must include the training config")

    model = HiMACJEPA(config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def infer_motion_probe_agents(num_steps: int) -> int:
    """Infer the shared fixed-width motion contract from the HiMAC head config."""
    root_cfg = OmegaConf.load(project_root / "configs" / "config.yaml")
    output_dim = int(root_cfg.motion_prediction_head.output_dim)
    if num_steps <= 0 or output_dim % (num_steps * 2) != 0:
        raise ValueError(
            f"motion_prediction_head.output_dim={output_dim} is incompatible with num_steps={num_steps}"
        )
    return output_dim // (num_steps * 2)


def get_model_num_parameters(model) -> int:
    """Count trainable parameters for any PyTorch model."""
    if hasattr(model, "get_num_parameters"):
        return model.get_num_parameters()
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_model_size_mb(model) -> float:
    """Estimate model size in megabytes."""
    if hasattr(model, "get_model_size_mb"):
        return model.get_model_size_mb()

    param_size = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_time(
    model_name: str,
    model,
    sample_batch: Dict[str, torch.Tensor],
    device: torch.device,
    num_iterations: int = 100,
) -> float:
    """Measure average inference latency."""
    import time

    sample_batch = move_batch_to_device(sample_batch, device)
    model.eval()

    def _forward():
        if model_name == "himac_jepa":
            return model(
                sample_batch["camera"],
                sample_batch["lidar"],
                sample_batch["radar"],
                sample_batch["strategic_action"],
                sample_batch["tactical_action"],
                masks=None,
            )
        return model(sample_batch)

    with torch.no_grad():
        for _ in range(10):
            _forward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _forward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    return ((end - start) / num_iterations) * 1000


def load_baseline_model(model_name: str, checkpoint_path: str, device: torch.device):
    """
    Load trained baseline model from checkpoint.

    Args:
        model_name: Name of the baseline model
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
    """
    if model_name == 'camera_only':
        from src.models.baselines import CameraOnlyBaseline
        # Load config from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = CameraOnlyBaseline(checkpoint['config'])

    elif model_name == 'lidar_only':
        from src.models.baselines import LiDAROnlyBaseline
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = LiDAROnlyBaseline(checkpoint['config'])

    elif model_name == 'radar_only':
        from src.models.baselines import RadarOnlyBaseline
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = RadarOnlyBaseline(checkpoint['config'])

    elif model_name == 'ijepa':
        from src.models.baselines import IJEPABaseline
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = IJEPABaseline(checkpoint['config'])

    elif model_name == 'vjepa':
        from src.models.baselines import VJEPABaseline
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = VJEPABaseline(checkpoint['config'])

    elif model_name == 'himac_jepa':
        model = load_himac_model(checkpoint_path, device)
        print(f"Loaded {model_name} from {checkpoint_path}")
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")

    return model


def extract_probe_data(model, dataloader, device: torch.device):
    """Extract baseline latents plus downstream targets."""
    latents = []
    targets = []
    valid_masks = []
    bev_targets = []
    motion_targets = []
    motion_valid_masks = []
    motion_agent_masks = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            latent = model.get_latent(batch).detach().cpu()
            trajectory, valid_mask = collect_probe_targets(batch)

            latents.append(latent)
            targets.append((trajectory * valid_mask.unsqueeze(-1)).cpu())
            valid_masks.append(valid_mask.cpu())
            if "bev_label" in batch:
                bev_targets.append(collect_bev_targets(batch).cpu())
            if "motion_future_trajectories" in batch:
                motion_target, motion_valid_mask, motion_agent_mask = collect_motion_targets(batch)
                motion_targets.append(motion_target.cpu())
                motion_valid_masks.append(motion_valid_mask.cpu())
                motion_agent_masks.append(motion_agent_mask.cpu())

    probe_data = {
        "latents": torch.cat(latents, dim=0),
        "trajectory_targets": torch.cat(targets, dim=0),
        "trajectory_valid_mask": torch.cat(valid_masks, dim=0),
    }
    if bev_targets:
        probe_data["bev_targets"] = torch.cat(bev_targets, dim=0)
    if motion_targets:
        probe_data["motion_targets"] = torch.cat(motion_targets, dim=0)
        probe_data["motion_valid_mask"] = torch.cat(motion_valid_masks, dim=0)
        probe_data["motion_agent_mask"] = torch.cat(motion_agent_masks, dim=0)

    return probe_data


def evaluate_trajectory_probe(
    train_probe_data,
    val_probe_data,
    probe_alpha: float = 1e-3,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate baselines via a frozen-latent trajectory probe."""
    print("Evaluating trajectory probe...")

    probe = fit_ridge_probe(
        train_probe_data["latents"],
        train_probe_data["trajectory_targets"].view(train_probe_data["trajectory_targets"].size(0), -1),
        alpha=probe_alpha,
    )
    predictions = predict_ridge_probe(probe, val_probe_data["latents"]).view_as(
        val_probe_data["trajectory_targets"]
    )

    metrics = compute_trajectory_horizon_metrics(
        predictions,
        val_probe_data["trajectory_targets"],
        val_probe_data["trajectory_valid_mask"],
    )
    per_sample_metrics = compute_trajectory_horizon_errors(
        predictions,
        val_probe_data["trajectory_targets"],
        val_probe_data["trajectory_valid_mask"],
    )

    return metrics, {
        metric_name: values.cpu().numpy()
        for metric_name, values in per_sample_metrics.items()
    }


def evaluate_bev_probe(
    train_probe_data,
    val_probe_data,
    device: torch.device,
    probe_epochs: int = 5,
    probe_batch_size: int = 16,
    probe_learning_rate: float = 1e-3,
) -> Dict[str, float]:
    """Evaluate baselines on BEV segmentation via a frozen-latent decoder probe."""
    if "bev_targets" not in train_probe_data or "bev_targets" not in val_probe_data:
        return {}

    print("Evaluating BEV probe...")

    train_bev = train_probe_data["bev_targets"]
    val_bev = val_probe_data["bev_targets"]
    bev_h, bev_w = train_bev.shape[-2:]
    num_classes = int(
        max(
            train_bev.max().item(),
            val_bev.max().item(),
        )
    ) + 1

    probe = fit_bev_probe(
        train_latents=train_probe_data["latents"],
        train_labels=train_bev,
        latent_dim=train_probe_data["latents"].size(1),
        num_classes=num_classes,
        bev_h=bev_h,
        bev_w=bev_w,
        device=device,
        epochs=probe_epochs,
        batch_size=probe_batch_size,
        learning_rate=probe_learning_rate,
    )
    predictions = predict_bev_probe(
        probe,
        val_probe_data["latents"],
        device=device,
        batch_size=probe_batch_size,
    )
    return compute_bev_classification_metrics(predictions, val_bev, num_classes=num_classes)


def evaluate_motion_probe(
    train_probe_data,
    val_probe_data,
    max_agents: int,
    probe_alpha: float = 1e-3,
) -> Dict[str, float]:
    """Evaluate baselines on motion forecasting via a frozen-latent ridge probe."""
    required_keys = {"motion_targets", "motion_valid_mask", "motion_agent_mask"}
    if not required_keys.issubset(train_probe_data) or not required_keys.issubset(val_probe_data):
        return {}

    print("Evaluating motion probe...")

    train_targets, train_valid_mask, train_agent_mask = build_motion_probe_targets(
        train_probe_data["motion_targets"],
        train_probe_data["motion_valid_mask"],
        train_probe_data["motion_agent_mask"],
        max_agents=max_agents,
    )
    val_targets, val_valid_mask, val_agent_mask = build_motion_probe_targets(
        val_probe_data["motion_targets"],
        val_probe_data["motion_valid_mask"],
        val_probe_data["motion_agent_mask"],
        max_agents=max_agents,
    )

    probe = fit_ridge_probe(
        train_probe_data["latents"],
        train_targets.view(train_targets.size(0), -1),
        alpha=probe_alpha,
    )
    predictions = predict_ridge_probe(probe, val_probe_data["latents"]).view_as(val_targets)

    return compute_motion_metrics(
        predictions,
        val_targets,
        val_valid_mask,
        val_agent_mask,
    )


def evaluate_himac_downstream(
    model,
    val_loader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate HiMAC-JEPA directly on trajectory, motion, and BEV outputs."""
    print("Evaluating HiMAC-JEPA downstream heads...")

    trajectory_predictions = []
    trajectory_targets = []
    trajectory_masks = []
    motion_predictions = []
    motion_targets = []
    motion_valid_masks = []
    motion_agent_masks = []
    bev_predictions = []
    bev_targets = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)

            _, _, trajectory_pred, motion_pred, bev_pred = model(
                batch["camera"],
                batch["lidar"],
                batch["radar"],
                batch["strategic_action"],
                batch["tactical_action"],
                masks=None,
            )

            trajectory_target, trajectory_mask = collect_probe_targets(batch)
            steps = min(trajectory_pred.size(-1) // 2, trajectory_target.size(1))

            trajectory_predictions.append(trajectory_pred.view(trajectory_pred.size(0), -1, 2)[:, :steps].cpu())
            trajectory_targets.append(trajectory_target[:, :steps].cpu())
            trajectory_masks.append(trajectory_mask[:, :steps].cpu())

            if "motion_future_trajectories" in batch:
                motion_target, motion_valid_mask, motion_agent_mask = collect_motion_targets(batch)
                aligned_motion_pred, aligned_motion_target, aligned_motion_valid_mask, aligned_motion_agent_mask = align_motion_targets(
                    motion_pred.cpu(),
                    motion_target.cpu(),
                    motion_valid_mask.cpu(),
                    motion_agent_mask.cpu(),
                )
                motion_predictions.append(aligned_motion_pred)
                motion_targets.append(aligned_motion_target)
                motion_valid_masks.append(aligned_motion_valid_mask)
                motion_agent_masks.append(aligned_motion_agent_mask)

            if "bev_label" in batch:
                bev_predictions.append(torch.argmax(bev_pred, dim=1).cpu())
                bev_targets.append(batch["bev_label"].cpu())

    trajectory_predictions_tensor = torch.cat(trajectory_predictions, dim=0)
    trajectory_targets_tensor = torch.cat(trajectory_targets, dim=0)
    trajectory_masks_tensor = torch.cat(trajectory_masks, dim=0)

    metrics = compute_trajectory_horizon_metrics(
        trajectory_predictions_tensor,
        trajectory_targets_tensor,
        trajectory_masks_tensor,
    )
    per_sample_metrics = compute_trajectory_horizon_errors(
        trajectory_predictions_tensor,
        trajectory_targets_tensor,
        trajectory_masks_tensor,
    )

    if motion_predictions:
        metrics.update(
            compute_motion_metrics(
                torch.cat(motion_predictions, dim=0),
                torch.cat(motion_targets, dim=0),
                torch.cat(motion_valid_masks, dim=0),
                torch.cat(motion_agent_masks, dim=0),
            )
        )

    if bev_predictions:
        metrics.update(
            compute_bev_classification_metrics(
                torch.cat(bev_predictions, dim=0),
                torch.cat(bev_targets, dim=0),
                num_classes=bev_pred.size(1),
            )
        )

    return metrics, {
        metric_name: values.cpu().numpy()
        for metric_name, values in per_sample_metrics.items()
    }


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    train_loader,
    val_loader,
    sample_batch,
    device: torch.device,
    bev_probe_epochs: int = 5,
    bev_probe_batch_size: int = 16,
    bev_probe_learning_rate: float = 1e-3,
    motion_probe_agents: int | None = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Evaluate a single baseline model on all tasks.

    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint
        train_loader: Train dataloader for probe fitting
        val_loader: Validation dataloader for evaluation
        sample_batch: Representative batch for inference timing
        device: Device

    Returns:
        metrics: Dictionary of aggregate metrics
        per_sample_metrics: Dictionary of aligned per-sample metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # Load model
    model = load_baseline_model(model_name, checkpoint_path, device)

    all_metrics = {}

    if model_name == 'himac_jepa':
        task_metrics, per_sample_metrics = evaluate_himac_downstream(model, val_loader, device)
    else:
        train_probe_data = extract_probe_data(model, train_loader, device)
        val_probe_data = extract_probe_data(model, val_loader, device)
        task_metrics, per_sample_metrics = evaluate_trajectory_probe(
            train_probe_data,
            val_probe_data,
        )
        task_metrics.update(
            evaluate_bev_probe(
                train_probe_data,
                val_probe_data,
                device=device,
                probe_epochs=bev_probe_epochs,
                probe_batch_size=bev_probe_batch_size,
                probe_learning_rate=bev_probe_learning_rate,
            )
        )
        if motion_probe_agents is None and "motion_targets" in train_probe_data:
            motion_probe_agents = infer_motion_probe_agents(train_probe_data["motion_targets"].size(2))
        if motion_probe_agents is not None:
            task_metrics.update(
                evaluate_motion_probe(
                    train_probe_data,
                    val_probe_data,
                    max_agents=motion_probe_agents,
                )
            )

    all_metrics.update(task_metrics)

    # Model stats
    all_metrics['model/num_parameters'] = get_model_num_parameters(model)
    all_metrics['model/size_mb'] = get_model_size_mb(model)

    # Inference time
    inference_time = measure_inference_time(model_name, model, sample_batch, device, num_iterations=100)
    all_metrics['model/inference_time_ms'] = inference_time

    return all_metrics, per_sample_metrics


def save_per_sample_metrics(
    per_sample_results: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
):
    """Persist aligned per-sample metrics for downstream analysis."""
    csv_path = output_path / 'per_sample_metrics.csv'
    rows = []

    for model_name, metric_map in per_sample_results.items():
        for metric_name, values in metric_map.items():
            for sample_index, value in enumerate(np.asarray(values).tolist()):
                if np.isnan(value):
                    continue
                rows.append(
                    {
                        'model': model_name,
                        'metric': metric_name,
                        'sample_index': sample_index,
                        'value': value,
                    }
                )

    pd.DataFrame(rows, columns=['model', 'metric', 'sample_index', 'value']).to_csv(
        csv_path,
        index=False,
    )
    print(f"Saved per-sample metrics: {csv_path}")


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    Create comparison table from results.

    Args:
        results: Dictionary mapping model_name -> metrics
        output_path: Path to save table
    """
    print("\nCreating comparison table...")

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Sort by 3s trajectory ADE when available.
    sort_metric = 'trajectory/ade_3s'
    if sort_metric in df.columns and not df[sort_metric].dropna().empty:
        df = df.sort_values(sort_metric)

    # Save as CSV
    csv_path = output_path / 'metrics.csv'
    df.to_csv(csv_path)
    print(f"Saved metrics: {csv_path}")

    # Create human-readable table
    txt_path = output_path / 'comparison_table.txt'
    with open(txt_path, 'w') as f:
        f.write("Baseline Model Comparison\n")
        f.write("=" * 80 + "\n\n")

        f.write(df.to_string())
        f.write("\n\n")

        # Highlight best models
        f.write("\nBest Performers:\n")
        f.write("-" * 40 + "\n")

        for metric in ['trajectory/ade_3s', 'bev/miou', 'motion/ade', 'model/inference_time_ms']:
            if metric in df.columns and not df[metric].dropna().empty:
                if 'ade' in metric or 'fde' in metric:
                    best_model = df[metric].idxmin()
                    best_value = df[metric].min()
                elif 'time' in metric:
                    best_model = df[metric].idxmin()
                    best_value = df[metric].min()
                else:
                    best_model = df[metric].idxmax()
                    best_value = df[metric].max()

                f.write(f"{metric}: {best_model} ({best_value:.3f})\n")

    print(f"Saved table: {txt_path}")

    # Create LaTeX table
    tex_path = output_path / 'comparison_table.tex'
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Model Comparison}\n")
        f.write("\\label{tab:baselines}\n")

        # Select key metrics
        key_metrics = [
            'trajectory/ade_3s',
            'trajectory/fde_3s',
            'bev/miou',
            'motion/ade',
            'model/inference_time_ms'
        ]

        available_metrics = [
            metric for metric in key_metrics
            if metric in df.columns and not df[metric].dropna().empty
        ]

        if available_metrics:
            df_latex = df[available_metrics]
            f.write(df_latex.to_latex(float_format="%.3f"))
        else:
            f.write("% No comparison metrics available yet.\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table: {tex_path}")


def create_comparison_plots(
    results: Dict[str, Dict[str, float]],
    output_path: Path
):
    """
    Create comparison plots.

    Args:
        results: Dictionary mapping model_name -> metrics
        output_path: Path to save plots
    """
    print("\nCreating comparison plots...")

    plots_dir = output_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Set style when seaborn is available, otherwise use matplotlib defaults.
    if SEABORN_AVAILABLE:
        sns.set_style("whitegrid")

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Plot 1: Trajectory ADE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = [
        metric for metric in ['trajectory/ade_1s', 'trajectory/ade_2s', 'trajectory/ade_3s']
        if metric in df.columns and not df[metric].dropna().empty
    ]
    if metrics:
        df[metrics].plot(kind='bar', ax=ax)
        ax.set_ylabel('ADE (m)')
        ax.set_title('Trajectory Prediction - Average Displacement Error')
        ax.legend([metric.split('_')[-1] for metric in metrics])
        plt.tight_layout()
        plt.savefig(plots_dir / 'trajectory_ade.png', dpi=300)
    plt.close()

    # Plot 2: BEV segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'bev/miou' in df.columns and not df['bev/miou'].dropna().empty:
        df['bev/miou'].plot(kind='bar', ax=ax)
        ax.set_ylabel('mIoU')
        ax.set_title('BEV Segmentation - Mean IoU')
        plt.tight_layout()
        plt.savefig(plots_dir / 'bev_miou.png', dpi=300)
    plt.close()

    # Plot 3: Motion prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'motion/ade' in df.columns and not df['motion/ade'].dropna().empty:
        df['motion/ade'].plot(kind='bar', ax=ax)
        ax.set_ylabel('ADE (m)')
        ax.set_title('Motion Prediction - Average Displacement Error')
        plt.tight_layout()
        plt.savefig(plots_dir / 'motion_ade.png', dpi=300)
    plt.close()

    # Plot 4: Radar plot for overall comparison
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Normalize metrics to 0-1 for radar plot
    metrics_for_radar = [
        metric for metric in ['trajectory/ade_3s', 'bev/miou', 'motion/ade', 'model/inference_time_ms']
        if metric in df.columns and not df[metric].dropna().empty
    ]

    if len(metrics_for_radar) < 2:
        plt.close()
        print("Skipping radar plot (not enough comparable metrics)")
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
    angles += angles[:1]

    for model_name in df.index:
        values = []
        for metric in metrics_for_radar:
            val = df.loc[model_name, metric]

            # Normalize (invert for error metrics)
            if 'ade' in metric or 'fde' in metric or 'time' in metric:
                val = 1.0 / (val + 1.0)  # Invert and normalize

            values.append(val)

        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('Overall Performance Comparison', y=1.08)

    plt.tight_layout()
    plt.savefig(plots_dir / 'radar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to: {plots_dir}")


def run_statistical_tests(
    results: Dict[str, Dict[str, float]],
    per_sample_results: Dict[str, Dict[str, np.ndarray]],
    output_path: Path
):
    """
    Run statistical significance tests.

    Args:
        results: Dictionary mapping model_name -> metrics
        output_path: Path to save results
    """
    print("\nRunning statistical tests...")
    save_per_sample_metrics(per_sample_results, output_path)

    txt_path = output_path / 'statistical_tests.txt'
    with open(txt_path, 'w') as f:
        f.write("Statistical Significance Tests\n")
        f.write("=" * 60 + "\n\n")
        f.write("Method: paired sign-flip permutation test on aligned per-sample errors.\n")
        f.write("Interpretation: lower is better; a negative mean delta means the first model has lower error.\n\n")

        wrote_results = False
        for metric in ['trajectory/ade_3s', 'trajectory/fde_3s', 'trajectory/ade_2s', 'trajectory/fde_2s']:
            available_models = [
                model_name
                for model_name in results
                if metric in results.get(model_name, {})
                and metric in per_sample_results.get(model_name, {})
            ]
            if len(available_models) < 2:
                continue

            ranked_models = sorted(
                available_models,
                key=lambda model_name: results[model_name][metric],
            )
            best_model = ranked_models[0]

            f.write(f"{metric}\n")
            f.write(f"Best aggregate model: {best_model} ({results[best_model][metric]:.4f})\n")

            metric_wrote_results = False
            for challenger in ranked_models[1:]:
                try:
                    mean_delta, p_value, num_pairs = paired_sign_flip_test(
                        per_sample_results[best_model][metric],
                        per_sample_results[challenger][metric],
                    )
                except ValueError:
                    continue

                f.write(
                    f"- {best_model} vs {challenger}: "
                    f"mean_delta={mean_delta:.6f}, p={p_value:.6f}, n={num_pairs}\n"
                )
                wrote_results = True
                metric_wrote_results = True

            if not metric_wrote_results:
                f.write("- skipped: not enough finite aligned per-sample values\n")
            f.write("\n")

        if not wrote_results:
            f.write("Statistical tests skipped.\n\n")
            f.write(
                "Reason: at least two models with aligned per-sample trajectory errors "
                "are required before the paired test is meaningful.\n"
            )

    print(f"Saved statistical tests: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")

    parser.add_argument(
        '--models',
        nargs='+',
        default=['camera_only', 'lidar_only', 'radar_only', 'ijepa', 'vjepa', 'himac_jepa'],
        help='Models to evaluate'
    )

    parser.add_argument(
        '--checkpoints',
        nargs='+',
        required=True,
        help='Paths to model checkpoints (same order as --models)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/baselines',
        help='Output directory for results'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset root'
    )

    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        help='nuScenes version'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Evaluation batch size'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers'
    )

    parser.add_argument(
        '--label-cache-dir',
        type=str,
        default='./cache/labels',
        help='Directory for cached evaluation labels'
    )
    parser.add_argument(
        '--bev-probe-epochs',
        type=int,
        default=5,
        help='Number of epochs for frozen-latent BEV probe fitting'
    )
    parser.add_argument(
        '--bev-probe-batch-size',
        type=int,
        default=16,
        help='Batch size for frozen-latent BEV probe fitting'
    )
    parser.add_argument(
        '--bev-probe-lr',
        type=float,
        default=1e-3,
        help='Learning rate for frozen-latent BEV probe fitting'
    )
    parser.add_argument(
        '--motion-probe-agents',
        type=int,
        default=None,
        help='Number of closest agents used for the shared fixed-width motion probe contract'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check checkpoint count matches model count
    if len(args.checkpoints) != len(args.models):
        raise ValueError(f"Number of checkpoints ({len(args.checkpoints)}) must match number of models ({len(args.models)})")

    # Create real label-backed dataloaders
    print("\nBuilding nuScenes benchmark dataloaders...")
    train_loader, val_loader = build_benchmark_loaders(
        data_root=args.data_dir,
        version=args.version,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.label_cache_dir,
    )
    print("Train and validation dataloaders ready")

    sample_batch = next(iter(val_loader))

    # Evaluate all models
    all_results = {}
    per_sample_results = {}

    for model_name, checkpoint_path in zip(args.models, args.checkpoints):
        try:
            metrics, model_per_sample_metrics = evaluate_model(
                model_name,
                checkpoint_path,
                train_loader,
                val_loader,
                sample_batch,
                device,
                bev_probe_epochs=args.bev_probe_epochs,
                bev_probe_batch_size=args.bev_probe_batch_size,
                bev_probe_learning_rate=args.bev_probe_lr,
                motion_probe_agents=args.motion_probe_agents,
            )
            all_results[model_name] = metrics
            per_sample_results[model_name] = model_per_sample_metrics

            print(f"\nResults for {model_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    if not all_results:
        print("\nNo models evaluated successfully")
        return

    # Generate comparison artifacts
    create_comparison_table(all_results, output_path)
    create_comparison_plots(all_results, output_path)
    run_statistical_tests(all_results, per_sample_results, output_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
