# Research Specification: HiMAC-JEPA Implementation Plan

This document outlines the implementation strategy for the HiMAC-JEPA model, based on the research design.

## 1. Implementation Steps

### Phase 1: Foundation & Data Pipeline
- **Project Scaffolding**: Setup directory structure and configuration management (Hydra/OmegaConf).
- **Data Loading**: Implement a multi-modal data loader for datasets like nuScenes or Waymo, handling synchronized Camera, LiDAR, and Radar data.
- **Preprocessing**: Implement modality-specific preprocessing (image normalization, point cloud voxelization/sampling, radar feature extraction).

### Phase 2: Core Architecture
- **Encoders**:
    - **Camera**: Spatio-temporal Vision Transformer (ViT).
    - **LiDAR**: PointNet++ or Sparse Convolutional Network.
    - **Radar**: 2D/3D CNN for velocity-aware feature extraction.
- **Fusion**: Multi-head cross-attention module to integrate modality features into a joint latent space.
- **Action Encoder**: Transformer-based encoder for hierarchical (strategic + tactical) action sequences.
- **Predictor**: Masked transformer that predicts future latent states.

### Phase 3: Training Objectives
- **Latent Distribution Head**: Implementation of the Gaussian distribution output ($\mu, \sigma$).
- **Loss Functions**:
    - **Predictive Loss**: KL-Divergence or NLL for the predicted distribution.
    - **Regularization**: VICReg (Variance-Invariance-Covariance Regularization) to prevent representation collapse.
- **Masking Strategy**: Implementation of spatio-temporal masking across modalities.

### Phase 4: Evaluation & Downstream Tasks
- **Intrinsic Evaluation**: Linear probing on frozen embeddings, latent prediction MSE.
- **Extrinsic Evaluation**: Integration with a planning head for closed-loop evaluation in CARLA or open-loop on nuScenes.

## 2. Tools & Frameworks
- **Deep Learning**: PyTorch
- **Experiment Tracking**: Weights & Biases (W&B)
- **Configuration**: Hydra
- **Data Processing**: NumPy, Pandas, PyTorch Geometric (for LiDAR)
- **Simulation**: CARLA (for hierarchical action data generation)

## 3. Testing Strategy
- **Unit Tests**: For each encoder, fusion module, and loss function.
- **Integration Tests**: Verifying the full forward pass from raw sensors to predicted distribution.
- **Overfitting Test**: Ensure the model can overfit on a small subset of data to verify the learning capacity.

## 4. Evaluation Plan
- **Baseline Comparison**: Compare against I-JEPA, V-JEPA, and standard generative world models (e.g., DreamerV3).
- **Ablation Studies**:
    - Impact of multi-modal fusion vs. single modality.
    - Impact of hierarchical action conditioning vs. flat actions.
    - Effectiveness of uncertainty quantification in risky scenarios.
