# HiMAC-JEPA Notebooks

Interactive Jupyter notebooks for baseline training, evaluation, and analysis.

## Notebooks Overview

### 1. Training: `01_train_baselines.ipynb`
**Purpose**: Train all baseline models interactively

**Features:**
- Progressive training cells for all 5 baselines
- Real-time progress bars with tqdm
- Training curve visualization
- Checkpoint saving
- Training summary table

**Usage:**
```bash
jupyter notebook notebooks/01_train_baselines.ipynb
```

**Configuration:**
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 16)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `USE_WANDB`: Enable W&B logging (default: False)

**Outputs:**
- Checkpoints: `checkpoints/baselines/{model_name}/best_model.pth`
- Training curves: `results/baseline_training_curves.png`
- Summary: `results/baseline_training_summary.csv`

---

### 2. Evaluation: `02_evaluate_baselines.ipynb`
**Purpose**: Evaluate trained models on downstream tasks

**Features:**
- Load models from checkpoints
- Evaluate trajectory prediction (ADE, FDE)
- Evaluate BEV segmentation (mIoU)
- Evaluate motion prediction (mAP)
- Model efficiency metrics
- Comparison tables (CSV, TXT, LaTeX)

**Usage:**
```bash
jupyter notebook notebooks/02_evaluate_baselines.ipynb
```

**Prerequisites:**
- Trained models from `01_train_baselines.ipynb`

**Outputs:**
- `results/baselines/metrics.csv` - Full metrics
- `results/baselines/comparison_table.txt` - Human-readable
- `results/baselines/comparison_table.tex` - LaTeX table

---

### 3. Analysis: `03_results_analysis.ipynb`
**Purpose**: Comprehensive analysis and visualization

**Features:**
- Trajectory prediction plots (ADE/FDE)
- BEV segmentation plots
- Motion prediction plots
- Model efficiency plots
- Radar plot for overall comparison
- Accuracy vs efficiency trade-off
- Statistical significance testing

**Usage:**
```bash
jupyter notebook notebooks/03_results_analysis.ipynb
```

**Prerequisites:**
- Evaluation results from `02_evaluate_baselines.ipynb`

**Outputs:**
- `results/baselines/plots/trajectory_prediction.png`
- `results/baselines/plots/bev_segmentation.png`
- `results/baselines/plots/motion_prediction.png`
- `results/baselines/plots/model_efficiency.png`
- `results/baselines/plots/radar_plot.png`
- `results/baselines/plots/accuracy_vs_efficiency.png`
- `results/baselines/statistical_tests.txt`

---

### 4. Visualization: `04_visualize_predictions.ipynb`
**Purpose**: Qualitative prediction visualization

**Features:**
- Trajectory predictions vs ground truth
- BEV segmentation masks
- Multi-agent motion scenes
- Latent space visualization (t-SNE)

**Usage:**
```bash
jupyter notebook notebooks/04_visualize_predictions.ipynb
```

**Outputs:**
- `results/baselines/visualizations/trajectory_predictions.png`
- `results/baselines/visualizations/bev_segmentation.png`
- `results/baselines/visualizations/multi_agent_motion.png`
- `results/baselines/visualizations/latent_space_tsne.png`

---

## Quick Start

**Run all notebooks in sequence:**

```bash
# 1. Train baselines (will take time)
jupyter notebook notebooks/01_train_baselines.ipynb

# 2. Evaluate models
jupyter notebook notebooks/02_evaluate_baselines.ipynb

# 3. Analyze results
jupyter notebook notebooks/03_results_analysis.ipynb

# 4. Visualize predictions
jupyter notebook notebooks/04_visualize_predictions.ipynb
```

**Or use VS Code:**
1. Open `.ipynb` file in VS Code
2. Select Python kernel
3. Run cells sequentially

---

## Requirements

**Python packages:**
```bash
pip install jupyter numpy pandas matplotlib seaborn torch tqdm pyyaml scipy scikit-learn
```

**Optional:**
```bash
pip install wandb  # For W&B logging
```

---

## Notes

### Dummy Data
All notebooks currently use **dummy data** for demonstration. To use with real data:

1. **Training**: Replace `create_dummy_dataloader()` with actual nuScenes dataset
2. **Evaluation**: Replace dummy metric functions with actual downstream task evaluation
3. **Visualization**: Replace dummy generation with actual model predictions

### Modifying Hyperparameters

**In notebooks:**
- Edit configuration cells at the top
- Modify directly in the notebook

**In config files:**
- Edit `configs/baseline/*.yaml` files
- Notebooks will load updated configs

### Troubleshooting

**Issue: CUDA out of memory**
```python
# In 01_train_baselines.ipynb
BATCH_SIZE = 8  # Reduce from 16
```

**Issue: Notebooks slow**
```python
# In 01_train_baselines.ipynb
NUM_EPOCHS = 20  # Reduce from 50
num_batches = 20  # Reduce from 50 in create_dummy_dataloader
```

**Issue: Plots not showing**
```python
# Add at top of notebook
%matplotlib inline
```

---

## Directory Structure

```
notebooks/
├── README.md                           # This file
├── 01_train_baselines.ipynb           # Training notebook
├── 02_evaluate_baselines.ipynb        # Evaluation notebook
├── 03_results_analysis.ipynb          # Analysis notebook
└── 04_visualize_predictions.ipynb     # Visualization notebook

results/
├── baseline_training_curves.png       # From notebook 1
├── baseline_training_summary.csv      # From notebook 1
└── baselines/
    ├── metrics.csv                    # From notebook 2
    ├── comparison_table.txt           # From notebook 2
    ├── comparison_table.tex           # From notebook 2
    ├── statistical_tests.txt          # From notebook 3
    ├── plots/                         # From notebook 3
    │   ├── trajectory_prediction.png
    │   ├── bev_segmentation.png
    │   ├── motion_prediction.png
    │   ├── model_efficiency.png
    │   ├── radar_plot.png
    │   └── accuracy_vs_efficiency.png
    └── visualizations/                # From notebook 4
        ├── trajectory_predictions.png
        ├── bev_segmentation.png
        ├── multi_agent_motion.png
        └── latent_space_tsne.png

checkpoints/baselines/
├── camera_only/best_model.pth
├── lidar_only/best_model.pth
├── radar_only/best_model.pth
├── ijepa/best_model.pth
└── vjepa/best_model.pth
```

---

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@article{himac-jepa-2024,
  title={HiMAC-JEPA: Hierarchical Multi-Modal Action-Conditioned JEPA for Autonomous Driving},
  author={TODO},
  journal={TODO},
  year={2024}
}
```
