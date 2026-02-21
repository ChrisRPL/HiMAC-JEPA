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
- [ ] Waymo Open Dataset Integration
- [ ] Evaluation Metrics & Baselines
- [ ] CARLA Closed-Loop Evaluation

## License
This project is licensed under the MIT License.
