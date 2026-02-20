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

## Implementation Status
- [x] Project Structure & Config System
- [x] Core Model Architecture Components (Encoders, Fusion, Predictor)
- [x] Data Loading Pipeline (Skeleton)
- [ ] Self-Supervised Pre-training Loop
- [ ] Uncertainty Quantification Module
- [ ] Downstream Task Heads (Planning, Prediction)

## License
This project is licensed under the MIT License.
