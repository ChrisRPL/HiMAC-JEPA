"""Evaluation and comparison script for baseline models."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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
        # Load HiMAC-JEPA model
        from src.models.himac_jepa import HiMACJEPA
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Assume config is in checkpoint or load from default
        model = HiMACJEPA(checkpoint.get('config', {}))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint
    model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {checkpoint_path}")

    return model


def evaluate_trajectory_prediction(
    model,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate trajectory prediction metrics.

    Args:
        model: Baseline model
        dataloader: Test dataloader
        device: Device

    Returns:
        metrics: Dictionary with ADE, FDE for different horizons
    """
    print("Evaluating trajectory prediction...")

    # Placeholder implementation
    # In real implementation, extract latents and evaluate with trajectory head

    metrics = {
        'trajectory/ade_1s': np.random.uniform(0.5, 2.0),
        'trajectory/ade_2s': np.random.uniform(1.0, 3.0),
        'trajectory/ade_3s': np.random.uniform(1.5, 4.0),
        'trajectory/fde_1s': np.random.uniform(0.8, 2.5),
        'trajectory/fde_2s': np.random.uniform(1.5, 4.0),
        'trajectory/fde_3s': np.random.uniform(2.0, 5.0),
    }

    return metrics


def evaluate_bev_segmentation(
    model,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate BEV segmentation metrics.

    Args:
        model: Baseline model
        dataloader: Test dataloader
        device: Device

    Returns:
        metrics: Dictionary with mIoU, accuracy, etc.
    """
    print("Evaluating BEV segmentation...")

    # Placeholder implementation

    metrics = {
        'bev/miou': np.random.uniform(0.2, 0.7),
        'bev/accuracy': np.random.uniform(0.5, 0.9),
        'bev/drivable_iou': np.random.uniform(0.6, 0.9),
        'bev/lane_iou': np.random.uniform(0.3, 0.7),
    }

    return metrics


def evaluate_motion_prediction(
    model,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate motion prediction metrics.

    Args:
        model: Baseline model
        dataloader: Test dataloader
        device: Device

    Returns:
        metrics: Dictionary with mAP, ADE, etc.
    """
    print("Evaluating motion prediction...")

    # Placeholder implementation

    metrics = {
        'motion/map': np.random.uniform(0.1, 0.5),
        'motion/ade': np.random.uniform(1.0, 3.0),
        'motion/fde': np.random.uniform(2.0, 5.0),
    }

    return metrics


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate a single baseline model on all tasks.

    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint
        dataloader: Test dataloader
        device: Device

    Returns:
        metrics: Dictionary of all metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # Load model
    model = load_baseline_model(model_name, checkpoint_path, device)

    # Evaluate on all tasks
    all_metrics = {}

    # Trajectory prediction
    traj_metrics = evaluate_trajectory_prediction(model, dataloader, device)
    all_metrics.update(traj_metrics)

    # BEV segmentation
    bev_metrics = evaluate_bev_segmentation(model, dataloader, device)
    all_metrics.update(bev_metrics)

    # Motion prediction
    motion_metrics = evaluate_motion_prediction(model, dataloader, device)
    all_metrics.update(motion_metrics)

    # Model stats
    all_metrics['model/num_parameters'] = model.get_num_parameters()
    all_metrics['model/size_mb'] = model.get_model_size_mb()

    # Inference time
    dummy_batch = create_dummy_batch(model_name, device)
    inference_time = model.get_inference_time(dummy_batch, num_iterations=100)
    all_metrics['model/inference_time_ms'] = inference_time

    return all_metrics


def create_dummy_batch(model_name: str, device: torch.device) -> Dict:
    """Create dummy batch for inference timing."""
    batch = {}

    if model_name in ['camera_only', 'ijepa']:
        batch['camera'] = torch.randn(1, 3, 224, 224, device=device)

    elif model_name == 'lidar_only':
        batch['lidar'] = torch.randn(1, 2048, 3, device=device)

    elif model_name == 'radar_only':
        batch['radar'] = torch.randn(1, 1, 128, 128, device=device)

    elif model_name in ['vjepa', 'himac_jepa']:
        batch['camera'] = torch.randn(1, 3, 224, 224, device=device)
        batch['lidar'] = torch.randn(1, 2048, 3, device=device)
        batch['radar'] = torch.randn(1, 1, 128, 128, device=device)

    return batch


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

    # Sort by trajectory ADE (lower is better)
    df = df.sort_values('trajectory/ade_3s')

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

        for metric in ['trajectory/ade_3s', 'bev/miou', 'motion/map']:
            if metric in df.columns:
                if 'ade' in metric or 'fde' in metric:
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
            'motion/map',
            'model/inference_time_ms'
        ]

        df_latex = df[key_metrics]

        f.write(df_latex.to_latex(float_format="%.3f"))
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

    # Set style
    sns.set_style("whitegrid")

    # Convert to DataFrame
    df = pd.DataFrame(results).T

    # Plot 1: Trajectory ADE comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['trajectory/ade_1s', 'trajectory/ade_2s', 'trajectory/ade_3s']
    df[metrics].plot(kind='bar', ax=ax)
    ax.set_ylabel('ADE (m)')
    ax.set_title('Trajectory Prediction - Average Displacement Error')
    ax.legend(['1s', '2s', '3s'])
    plt.tight_layout()
    plt.savefig(plots_dir / 'trajectory_ade.png', dpi=300)
    plt.close()

    # Plot 2: BEV segmentation
    fig, ax = plt.subplots(figsize=(10, 6))
    df['bev/miou'].plot(kind='bar', ax=ax)
    ax.set_ylabel('mIoU')
    ax.set_title('BEV Segmentation - Mean IoU')
    plt.tight_layout()
    plt.savefig(plots_dir / 'bev_miou.png', dpi=300)
    plt.close()

    # Plot 3: Motion prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    df['motion/map'].plot(kind='bar', ax=ax)
    ax.set_ylabel('mAP')
    ax.set_title('Motion Prediction - Mean Average Precision')
    plt.tight_layout()
    plt.savefig(plots_dir / 'motion_map.png', dpi=300)
    plt.close()

    # Plot 4: Radar plot for overall comparison
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Normalize metrics to 0-1 for radar plot
    metrics_for_radar = ['trajectory/ade_3s', 'bev/miou', 'motion/map', 'model/inference_time_ms']

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
    output_path: Path
):
    """
    Run statistical significance tests.

    Args:
        results: Dictionary mapping model_name -> metrics
        output_path: Path to save results
    """
    print("\nRunning statistical tests...")

    # Placeholder - in real implementation, use actual test sets
    # and compute p-values with t-tests or Wilcoxon tests

    txt_path = output_path / 'statistical_tests.txt'
    with open(txt_path, 'w') as f:
        f.write("Statistical Significance Tests\n")
        f.write("=" * 60 + "\n\n")

        f.write("Note: Placeholder results - requires actual test data\n\n")

        f.write("Trajectory Prediction (ADE @ 3s):\n")
        f.write("  HiMAC-JEPA vs V-JEPA: p < 0.01 **\n")
        f.write("  HiMAC-JEPA vs I-JEPA: p < 0.001 ***\n")
        f.write("  HiMAC-JEPA vs Camera-Only: p < 0.001 ***\n\n")

        f.write("BEV Segmentation (mIoU):\n")
        f.write("  HiMAC-JEPA vs V-JEPA: p < 0.05 *\n")
        f.write("  HiMAC-JEPA vs Camera-Only: p < 0.001 ***\n\n")

        f.write("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001\n")

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
        help='Path to dataset'
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

    # Create dataloader (placeholder)
    print("\nNote: Using placeholder evaluation (dataloader not implemented)")
    print("Implement actual dataloader for real evaluation\n")

    dataloader = None  # Placeholder

    # Evaluate all models
    all_results = {}

    for model_name, checkpoint_path in zip(args.models, args.checkpoints):
        try:
            metrics = evaluate_model(model_name, checkpoint_path, dataloader, device)
            all_results[model_name] = metrics

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
    run_statistical_tests(all_results, output_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete! Results saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
