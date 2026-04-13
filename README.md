# HiMAC-JEPA: Hierarchical Multi-Modal Action-Conditioned JEPA

HiMAC-JEPA is a novel world model architecture for autonomous driving, based on the **Joint Embedding Predictive Architecture (JEPA)** paradigm. It integrates deep multi-modal sensor fusion, hierarchical action conditioning, and explicit uncertainty quantification to learn robust, semantically rich, and causally aware representations of the driving environment.

## Research Motivation
Autonomous driving systems operate in highly dynamic and uncertain environments. Current world models often struggle with:
- **Incomplete Sensor Fusion**: Failing to leverage the full complementarity of Camera, LiDAR, and Radar.
- **Action Agnosticism**: Not accounting for how different levels of control (strategic vs. tactical) influence future states.
- **Deterministic Predictions**: Lacking explicit measures of uncertainty, which is critical for safety-critical decision-making.

HiMAC-JEPA addresses these gaps by building a comprehensive, predictive world model that understands complex driving scenarios and anticipates future states under various hierarchical actions.

## Architecture Overview
The architecture consists of several key components:
1.  **Multi-Modal Encoders**: Modality-specific encoders for Camera (ViT), LiDAR (PointNet++), and Radar (CNN).
2.  **Multi-Modal Fusion Module**: An attention-based module that fuses features into a joint latent representation $Z_t$.
3.  **Hierarchical Action Encoder**: Encodes strategic (e.g., "change lane") and tactical (e.g., "steer left") actions into a latent action vector.
4.  **JEPA Predictor**: A core module that predicts future latent distributions conditioned on current state and actions.
5.  **Uncertainty Quantification**: Explicitly outputs distribution parameters ($\mu, \sigma$) to quantify prediction confidence.

## Current Status (April 2026)
- Core training scaffold is stable and regression-tested. Current merge tip passed `pytest -q` with **190 passed, 66 skipped**.
- The active JEPA path now includes:
  - spatio-temporal masking across camera, radar, and temporal context
  - an **observation-only EMA teacher** for target latents
  - action-conditioned online prediction with guards against masked-context leakage
- The repo is ready for controlled research iterations on objective design and temporal prediction.
- The repo is **not benchmark-complete yet**:
  - some downstream evaluation metrics still use placeholder logic
  - temporal-consistency evaluation is still scaffolded
  - action extraction still includes heuristic / placeholder components
  - no benchmark table on real nuScenes runs is checked into the repo yet

## Getting Started
### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Installation
```bash
git clone https://github.com/ChrisRPL/HiMAC-JEPA.git
cd HiMAC-JEPA
pip install -r requirements.txt
```

### Dataset Setup

#### Option 1: Use Dummy Data (Quick Start)
The project includes a dummy dataset for testing. No additional setup required:
```bash
python train.py  # Uses dummy data by default
```

#### Option 2: nuScenes Dataset (Recommended)

**1. Download nuScenes**

For testing (v1.0-mini, ~4GB):
```bash
# Create data directory
mkdir -p /data/nuscenes

# Download mini split
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xzf v1.0-mini.tgz -C /data/nuscenes
```

For full training (v1.0-trainval, ~350GB):
```bash
# Download metadata
wget https://www.nuscenes.org/data/v1.0-trainval_meta.tgz

# Download all 10 parts
for i in {01..10}; do
    wget https://www.nuscenes.org/data/v1.0-trainval_${i}_of_10.tgz
done

# Extract all
tar -xzf v1.0-trainval_meta.tgz -C /data/nuscenes
for i in {01..10}; do
    tar -xzf v1.0-trainval_${i}_of_10.tgz -C /data/nuscenes
done
```

**2. Set Environment Variable**
```bash
export NUSCENES_ROOT=/data/nuscenes
```

Or add to your `.bashrc` / `.zshrc`:
```bash
echo 'export NUSCENES_ROOT=/data/nuscenes' >> ~/.bashrc
source ~/.bashrc
```

**3. Verify Installation**
```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/data/nuscenes')
print(f"Loaded {len(nusc.sample)} samples")
# Expected output: Loaded 404 samples
```

**4. Train with nuScenes**
```bash
# Use mini split for testing
python train.py data=nuscenes

# Use full trainval split
python train.py data=nuscenes data.version=v1.0-trainval
```

### Training

**Basic training with default configuration:**
```bash
python train.py
```

**Override specific parameters:**
```bash
python train.py model.latent_dim=256 training.epochs=50
```

**Use experiment configs:**
```bash
# Quick overfitting test
python train.py +experiment=overfit

# Ablation study without masking
python train.py +experiment=ablation_no_masking
```

**Configuration structure:**
- `configs/model/` - Model architecture settings
- `configs/data/` - Dataset configurations
- `configs/training/` - Training hyperparameters
- `configs/masking/` - JEPA masking strategy
- `configs/experiment/` - Pre-configured experiments
- `configs/evaluation.yaml` - Evaluation metrics configuration

### Evaluation

**Status note:** evaluation support is partial. Intrinsic probes run today, but downstream trajectory / BEV metrics and temporal-consistency still contain placeholder logic and should be treated as scaffolding, not benchmark results.

**Run evaluation on a trained model:**
```bash
# Evaluate with default settings
python scripts/evaluate.py evaluation.checkpoint_path=checkpoints/best_model.pth

# Evaluate with specific experiment
python scripts/evaluate.py +experiment=overfit evaluation.checkpoint_path=checkpoints/overfit.pth

# Enable W&B logging
python scripts/evaluate.py evaluation.checkpoint_path=checkpoints/best_model.pth evaluation.wandb.enabled=true
```

**Evaluation Metrics:**

*Intrinsic Metrics (representation quality):*
- **Latent MSE**: Mean squared error between online and target encoder
- **Linear Probe Accuracy**: Classification/regression accuracy with frozen features
- **Embedding Silhouette**: Cluster quality of learned representations
- **Temporal Consistency**: Smoothness of latent trajectories over time

*Downstream Metrics (task performance):*
- **Trajectory ADE/FDE**: Average and final displacement errors for trajectory prediction
- **BEV Segmentation mIoU**: Mean intersection over union for bird's-eye-view segmentation
- **Motion Prediction mAP**: Mean average precision at different distance thresholds

### Temporal Sequence Training

HiMAC-JEPA supports **temporal JEPA training** on multi-frame sequences, where the model learns to predict future latent representations from past context frames. This enhances the model's ability to capture temporal dynamics and make long-term predictions.

**Enable temporal training:**
```bash
# Default: 5 context frames + 3 future frames
python train.py data=nuscenes data.temporal.enabled=true

# Customize sequence length
python train.py data=nuscenes data.temporal.enabled=true \
    data.temporal.seq_length=10 \
    data.temporal.pred_horizon=5

# Adjust frame skip (use every Nth frame)
python train.py data=nuscenes data.temporal.enabled=true \
    data.temporal.frame_skip=2  # Use every other frame (1Hz instead of 2Hz)
```

**Temporal configuration parameters:**
```yaml
temporal:
  enabled: false              # Enable temporal mode
  seq_length: 5               # Number of context frames (past)
  pred_horizon: 3             # Number of future frames to predict
  frame_skip: 1               # Use every Nth frame (1 = all frames)
  min_scene_length: 10        # Skip scenes with fewer samples
```

**How temporal mode works:**
1. **Sequence Construction**: Dataset builds sliding windows of (context, target) pairs
   - Context: Past frames used to predict future
   - Target: Future frames the model tries to predict

2. **Validation**: Each sequence is validated for:
   - Temporal continuity (no gaps in sample chain)
   - Timestamp consistency (uniform time intervals, max 0.6s gap)
   - Sensor availability (camera, LiDAR, radar present)

3. **Training Objective**:
   - **Online encoder**: Processes context frames → predicts future latent
   - **Observation-only EMA teacher**: Processes target frames → provides target latent without action leakage
   - **Loss**: KL divergence + VICReg between predicted and target latents

**Temporal data shapes:**
- Single-frame mode: `camera: (B, C, H, W)`
- Temporal mode: `camera: (B, T, C, H, W)` where `T` is sequence length

**Example workflow:**
```bash
# 1. Train single-frame baseline
python train.py data=nuscenes

# 2. Train temporal model (5→3 frames)
python train.py data=nuscenes data.temporal.enabled=true \
    experiment_name=temporal-5-3

# 3. Train longer sequences (10→5 frames)
python train.py data=nuscenes data.temporal.enabled=true \
    data.temporal.seq_length=10 \
    data.temporal.pred_horizon=5 \
    experiment_name=temporal-10-5

# 4. Evaluate temporal predictions
python scripts/evaluate.py evaluation.checkpoint_path=checkpoints/temporal-5-3.pth
```

**Validation statistics:**

After building sequences, you'll see validation summary:
```
============================================================
Temporal Sequence Validation Summary
============================================================
Total candidates:        1250
Valid sequences:         1180
Valid ratio:             94.40%

Rejection reasons:
  - Sample gaps:         45
  - Timestamp issues:    15
  - Missing sensors:     10
============================================================
```

### Ground Truth Label Extraction

HiMAC-JEPA supports **ground truth label extraction** from nuScenes for downstream task evaluation. Labels are extracted for trajectory prediction, BEV segmentation, and motion prediction tasks.

**Label Types:**
1. **Trajectory Prediction**: Future ego and agent waypoints at 1s, 2s, 3s horizons
2. **BEV Segmentation**: Semantic map with 6 classes (drivable area, lanes, crossings, vehicles, pedestrians)
3. **Motion Prediction**: Multi-agent future trajectories with current states and valid masks

**Enable label extraction during training:**
```bash
# Extract labels on-the-fly during training
python train.py data=nuscenes data.labels.enabled=true

# Combine with temporal training
python train.py data=nuscenes data.temporal.enabled=true data.labels.enabled=true
```

**Pre-extract labels for entire dataset (recommended):**
```bash
# Extract for train split with 8 workers
python scripts/extract_labels.py --split train --workers 8

# Extract for all splits
python scripts/extract_labels.py --split all --workers 8

# Extract with custom parameters
python scripts/extract_labels.py \
    --split train \
    --workers 16 \
    --bev-size 256 256 \
    --bev-range 75.0 \
    --motion-horizon 5.0

# Force recomputation (bypass cache)
python scripts/extract_labels.py --split train --workers 8 --force
```

**Label Configuration:**
```yaml
labels:
  enabled: true

  trajectory:
    pred_horizons: [1.0, 2.0, 3.0]  # Prediction horizons in seconds
    include_ego: true
    include_agents: true
    max_agents: 20

  bev:
    size: [200, 200]  # BEV image size (pixels)
    range: 50.0       # BEV range (meters)
    classes:
      - background
      - drivable_area
      - lane_divider
      - pedestrian_crossing
      - vehicle
      - pedestrian

  motion:
    pred_horizon: 3.0      # Prediction horizon (seconds)
    max_distance: 50.0     # Max tracking distance (meters)
    min_visibility: 0.5    # Min visibility score (0-4)

  cache:
    enabled: true
    cache_dir: "./cache/labels"
    force_recompute: false
```

**Label Output Format:**
```python
labels = {
    'trajectory_ego': {
        1.0: np.ndarray,  # (T, 2) waypoints at 1s
        2.0: np.ndarray,  # (T, 2) waypoints at 2s
        3.0: np.ndarray   # (T, 2) waypoints at 3s
    },
    'trajectory_agents': {
        'agent_id_1': {
            'class': 'vehicle.car',
            'current_pos': np.ndarray,      # (2,) [x, y]
            'current_vel': np.ndarray,      # (2,) [vx, vy]
            'trajectories': {1.0: ..., 2.0: ..., 3.0: ...}
        },
        ...
    },
    'bev': np.ndarray,  # (H, W) segmentation mask
    'motion': {
        'agent_ids': [...],
        'agent_classes': [...],
        'current_states': np.ndarray,        # (N, 4) [x, y, vx, vy]
        'future_trajectories': np.ndarray,   # (N, T, 2)
        'valid_masks': np.ndarray            # (N, T) boolean
    }
}
```

**Cache Performance:**
- **Initial extraction**: ~10-20 sec per sample (with map rendering)
- **With cache**: <0.1 sec per sample (disk read only)
- **Cache size**: ~500MB for v1.0-mini, ~15GB for v1.0-trainval

**Extraction Statistics Example:**
```
============================================================
Extraction complete for TRAIN split
============================================================
Successful:     404/404
Failed:         0
Time elapsed:   3245.2s
Time per sample: 8.03s

Cache statistics:
  Cached samples: 404
  Total size:     512.34 MB
============================================================
```

### Baseline Models for Comparison

HiMAC-JEPA includes 5 baseline models for comprehensive comparison:

**Status note:** baseline training scripts and report formatting are implemented, and the checked-in comparison path is now label-backed for trajectory evaluation. HiMAC-JEPA is scored directly on trajectory and BEV heads where labels exist. Motion metrics and statistical significance tests are still intentionally skipped until the evaluator persists paired per-sample outputs.

**Single-Modal Baselines:**
1. **Camera-Only**: ResNet18 + LSTM, supervised future frame prediction
2. **LiDAR-Only**: Simplified PointNet++, supervised future cloud prediction
3. **Radar-Only**: 3D CNN, supervised future radar prediction

**JEPA Baselines:**
4. **I-JEPA**: Image JEPA (camera-only) with spatial masking, no action conditioning
5. **V-JEPA**: Multi-modal JEPA with temporal prediction, no action conditioning

**Training Baselines:**
```bash
# Train camera-only baseline
python scripts/train_baselines.py --model camera_only

# Train LiDAR-only baseline
python scripts/train_baselines.py --model lidar_only

# Train radar-only baseline
python scripts/train_baselines.py --model radar_only

# Train I-JEPA baseline (camera + JEPA)
python scripts/train_baselines.py --model ijepa

# Train V-JEPA baseline (multi-modal + JEPA)
python scripts/train_baselines.py --model vjepa

# Override hyperparameters
python scripts/train_baselines.py --model vjepa --epochs 200 --lr 1e-3 --batch-size 64

# Disable W&B logging
python scripts/train_baselines.py --model camera_only --no-wandb
```

**Evaluating Baselines:**
```bash
# Evaluate all baselines
python scripts/evaluate_baselines.py \
  --models camera_only lidar_only radar_only ijepa vjepa himac_jepa \
  --checkpoints \
    checkpoints/baselines/camera_only/best_model.pth \
    checkpoints/baselines/lidar_only/best_model.pth \
    checkpoints/baselines/radar_only/best_model.pth \
    checkpoints/baselines/ijepa/best_model.pth \
    checkpoints/baselines/vjepa/best_model.pth \
    checkpoints/himac_jepa/best_model.pth \
  --data-dir /path/to/nuscenes \
  --version v1.0-mini \
  --label-cache-dir ./cache/labels \
  --output-dir results/baselines

# Results saved to:
# - results/baselines/metrics.csv             (raw metrics)
# - results/baselines/comparison_table.txt    (human-readable)
# - results/baselines/comparison_table.tex    (LaTeX for papers)
# - results/baselines/plots/*.png             (plots for available metrics)
# - results/baselines/statistical_tests.txt   (explicit skip note until paired outputs are logged)
```

**Research Hypothesis: Expected Performance Hierarchy** (not benchmarked yet):
```
Trajectory Prediction (ADE ↓):
  HiMAC-JEPA < V-JEPA < I-JEPA < Camera-Only < LiDAR-Only < Radar-Only

BEV Segmentation (mIoU ↑):
  HiMAC-JEPA > V-JEPA > Camera-Only > I-JEPA > LiDAR-Only > Radar-Only

Motion Prediction (mAP ↑):
  HiMAC-JEPA > V-JEPA > LiDAR-Only > I-JEPA > Camera-Only > Radar-Only

Inference Time (ms ↓):
  Radar-Only < Camera-Only < LiDAR-Only < I-JEPA < V-JEPA < HiMAC-JEPA
```

**Key Comparisons:**
- **Single vs Multi-modal**: Shows benefit of sensor fusion
- **Supervised vs JEPA**: Shows benefit of self-supervised representation learning
- **No Actions vs HiMAC-JEPA**: Shows benefit of hierarchical action conditioning
- **Simple Fusion (V-JEPA) vs Hierarchical (HiMAC-JEPA)**: Shows benefit of hierarchical multi-modal architecture

See `docs/BASELINES.md` for baseline implementation notes and planned comparison workflow.

### Ablation Studies

Test individual component contributions:

```bash
# Camera-only baseline
python train.py +experiment=ablation_camera_only data=nuscenes

# LiDAR-only baseline
python train.py +experiment=ablation_lidar_only data=nuscenes

# Radar-only baseline
python train.py +experiment=ablation_radar_only data=nuscenes

# No action conditioning
python train.py +experiment=ablation_no_actions data=nuscenes

# No masking (supervised learning)
python train.py +experiment=ablation_no_masking data=nuscenes

# Single-frame vs temporal comparison
python train.py +experiment=ablation_camera_only data=nuscenes  # Baseline
python train.py +experiment=ablation_camera_only data=nuscenes data.temporal.enabled=true  # Temporal
```

### Weights & Biases Integration

Track experiments with W&B:

```bash
# Enable W&B logging during training
python train.py wandb.enabled=true wandb.entity=YOUR_USERNAME

# Custom experiment name and tags
python train.py wandb.enabled=true experiment_name=my-experiment wandb.tags='[baseline,nuscenes]'

# Combine with nuScenes dataset
python train.py data=nuscenes wandb.enabled=true
```

## Implementation Status

**Implemented and regression-tested**
- Hydra config structure and training entrypoint
- Core HiMAC-JEPA model: encoders, fusion, predictor, uncertainty head
- Hierarchical action conditioning
- Single-frame and temporal data loading
- Spatio-temporal masking utilities
- Observation-only EMA teacher for target latents
- Downstream heads: trajectory, motion, BEV segmentation
- Baseline model implementations and training/evaluation scripts
- Label extraction and caching pipeline

**Implemented, but still needs honest benchmark validation**
- Downstream evaluation metrics: trajectory and BEV benchmark paths are label-backed; motion/significance evaluation is still partial
- Temporal-consistency metric: scaffolded, not yet backed by sequential evaluation
- nuScenes action extraction: tactical/strategic labels still include heuristic or placeholder logic
- Baseline comparisons: honest evaluation loop exists, but no checked-in real benchmark table yet

**Near-term roadmap**
1. **Future-action temporal JEPA**
   Student sees planned future actions; teacher stays observation-only. Goal: make the action-conditioning claim real for multi-step prediction.
2. **Checked-in nuScenes-mini benchmark table**
   Run the honest benchmark loop end-to-end and check in one reproducible comparison table so architecture changes can be ranked.
3. **Structured JEPA targets**
   Move beyond one pooled latent toward token / slot targets so masking supervises local structure, not just a single global vector.

**Longer-term roadmap**
- Waymo Open Dataset integration
- CARLA closed-loop evaluation
- Multi-GPU distributed training

## License
This project is licensed under the MIT License.
