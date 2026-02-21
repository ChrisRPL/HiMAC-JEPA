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
   - **Target encoder (EMA)**: Processes target frames → provides ground truth latent
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
- [x] Project Structure & Hydra Config System
- [x] Core Model Architecture (Encoders, Fusion, Predictor)
- [x] Hierarchical Action Encoder
- [x] Enhanced Encoder Architectures (ViT-12, Hierarchical PointNet, Velocity-aware Radar)
- [x] Data Loading Pipeline
- [x] nuScenes Dataset Integration
- [x] Multi-Modal Preprocessing (Camera, LiDAR, Radar)
- [x] Action Extraction from Ego Vehicle Data
- [x] JEPA Self-Supervised Training Loop
- [x] Spatio-Temporal Masking Strategy
- [x] EMA Target Encoder
- [x] VICReg Regularization
- [x] Downstream Task Heads (Trajectory, Motion, BEV Segmentation)
- [x] Evaluation Metrics (Intrinsic & Downstream)
- [x] Ablation Study Configurations
- [x] Weights & Biases Integration
- [x] Temporal Sequence Loading & Future Prediction
- [x] Temporal Fusion with Transformer Aggregation
- [x] Comprehensive Temporal Validation (Continuity, Timestamps, Sensors)
- [x] Ground Truth Label Extraction (Trajectory, BEV, Motion)
- [x] Label Caching System for Fast Loading
- [x] Parallel Label Extraction Script
- [ ] Baseline Comparison Implementations
- [ ] Waymo Open Dataset Integration
- [ ] CARLA Closed-Loop Evaluation
- [ ] Multi-GPU Distributed Training

## License
This project is licensed under the MIT License.
