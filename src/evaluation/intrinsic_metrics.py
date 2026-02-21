"""Intrinsic evaluation metrics for learned representations."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.linear_model import Ridge, LogisticRegression
from typing import Dict, Tuple, Optional


class IntrinsicEvaluator:
    """Evaluate intrinsic quality of learned representations."""

    def __init__(self, model, dataloader, device='cuda'):
        """
        Initialize intrinsic evaluator.

        Args:
            model: HiMAC-JEPA model
            dataloader: Validation dataloader
            device: Device to run evaluation on
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.eval()

    def compute_latent_mse(self) -> float:
        """
        Compute MSE between online and target encoder latents.

        Returns:
            Average latent MSE across validation set
        """
        total_mse = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.dataloader:
                # Move data to device
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                # Get online encoder predictions (no masks for evaluation)
                mu, _, _, _, _ = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                # Get target encoder predictions (EMA model if available)
                # Note: This assumes model has an ema_model attribute
                # For evaluation without EMA, we compute self-consistency
                if hasattr(self.model, 'ema_model'):
                    target_mu, _, _, _, _ = self.model.ema_model(
                        camera, lidar, radar, strategic, tactical, masks=None
                    )
                else:
                    # Fallback: use same model with detached gradients
                    target_mu = mu.detach()

                # Compute MSE
                mse = torch.mean((mu - target_mu) ** 2)
                total_mse += mse.item()
                num_batches += 1

        return total_mse / num_batches if num_batches > 0 else 0.0

    def linear_probe(
        self,
        task='trajectory',
        num_epochs=100,
        lr=0.001
    ) -> float:
        """
        Linear probing on frozen embeddings.

        Trains a linear classifier on top of frozen features to evaluate
        the quality of learned representations.

        Args:
            task: Task to probe ('trajectory' or 'bev_segmentation')
            num_epochs: Number of training epochs for probe
            lr: Learning rate for probe

        Returns:
            Probe score (R² for regression, accuracy for classification)
        """
        # Extract frozen embeddings and labels
        embeddings, labels = self._extract_embeddings(task)

        if len(embeddings) == 0:
            return 0.0

        # Split train/val (80/20)
        split_idx = int(0.8 * len(embeddings))
        train_emb, val_emb = embeddings[:split_idx], embeddings[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # Select probe type based on task
        if task == 'trajectory':
            # Regression task - predict tactical actions
            probe = Ridge(alpha=1.0, max_iter=num_epochs)
        else:
            # Classification task - predict strategic actions
            probe = LogisticRegression(
                max_iter=num_epochs,
                random_state=42
            )

        # Fit probe on frozen features
        probe.fit(train_emb, train_labels)

        # Evaluate on validation set
        score = probe.score(val_emb, val_labels)

        return score

    def embedding_silhouette(self) -> float:
        """
        Compute silhouette score for embedding clusters.

        Measures how well embeddings cluster according to their
        strategic action labels. Higher is better.

        Returns:
            Silhouette score in [-1, 1] (higher = better clustering)
        """
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                # Get embeddings
                mu, _, _, _, _ = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                embeddings.append(mu.cpu().numpy())
                labels.append(strategic.cpu().numpy())

        # Concatenate all batches
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Check if we have multiple classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Silhouette score requires at least 2 clusters
            return 0.0

        # Compute silhouette score
        try:
            score = silhouette_score(embeddings, labels)
        except Exception as e:
            print(f"Warning: Could not compute silhouette score: {e}")
            score = 0.0

        return score

    def temporal_consistency(self, window=10) -> float:
        """
        Measure smoothness of latent trajectories over time.

        Computes the average difference between consecutive frames'
        latent representations. Lower values indicate smoother trajectories.

        Args:
            window: Number of consecutive frames to analyze

        Returns:
            Average temporal consistency score (lower = more consistent)
        """
        # TODO: Requires sequential data loading with temporal structure
        # Current dataloader doesn't preserve temporal ordering
        # For now, return placeholder
        # Future: implement temporal dataloader or use scene-based sampling

        print("Warning: Temporal consistency requires sequential data loading")
        print("This metric is not yet implemented - returning 0.0")

        return 0.0

    def _extract_embeddings(self, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and labels for linear probing.

        Args:
            task: Task type ('trajectory' or 'bev_segmentation')

        Returns:
            Tuple of (embeddings, labels) as numpy arrays
        """
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in self.dataloader:
                camera = batch['camera'].to(self.device)
                lidar = batch['lidar'].to(self.device)
                radar = batch['radar'].to(self.device)
                strategic = batch['strategic_action'].to(self.device)
                tactical = batch['tactical_action'].to(self.device)

                # Get embeddings from model
                mu, _, _, _, _ = self.model(
                    camera, lidar, radar, strategic, tactical, masks=None
                )

                embeddings.append(mu.cpu().numpy())

                # Task-specific labels
                if task == 'trajectory':
                    # Regression: predict tactical actions
                    labels.append(tactical.cpu().numpy())
                elif task == 'bev_segmentation':
                    # Classification: predict strategic actions
                    labels.append(strategic.cpu().numpy())
                else:
                    raise ValueError(f"Unknown task: {task}")

        # Concatenate all batches
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)

        return embeddings, labels

    def run_all(self, config) -> Dict[str, float]:
        """
        Run all intrinsic evaluations specified in config.

        Args:
            config: Evaluation configuration object

        Returns:
            Dictionary of metric names and values
        """
        results = {}

        print("Running intrinsic evaluations...")

        # Latent MSE
        if 'latent_mse' in config.metrics.intrinsic:
            print("  Computing latent MSE...")
            results['intrinsic/latent_mse'] = self.compute_latent_mse()

        # Linear probing
        if 'linear_probe_accuracy' in config.metrics.intrinsic:
            for task in config.intrinsic.linear_probe_tasks:
                print(f"  Running linear probe for {task}...")
                score = self.linear_probe(
                    task=task,
                    num_epochs=config.intrinsic.num_probe_epochs,
                    lr=config.intrinsic.probe_learning_rate
                )
                results[f'intrinsic/linear_probe_{task}'] = score

        # Embedding silhouette
        if 'embedding_silhouette' in config.metrics.intrinsic:
            print("  Computing embedding silhouette score...")
            results['intrinsic/silhouette_score'] = self.embedding_silhouette()

        # Temporal consistency
        if 'temporal_consistency' in config.metrics.intrinsic:
            if config.intrinsic.compute_temporal_consistency:
                print("  Computing temporal consistency...")
                results['intrinsic/temporal_consistency'] = self.temporal_consistency(
                    window=config.intrinsic.temporal_window
                )

        return results
