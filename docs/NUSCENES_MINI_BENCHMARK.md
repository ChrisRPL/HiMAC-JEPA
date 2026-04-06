# nuScenes-mini Benchmark Runbook

Use this flow when you want one honest benchmark run with checkpoints and results persisted to Google Drive from Colab.

## Recommended layout

- Dataset: `/content/drive/MyDrive/datasets/nuscenes`
- Checkpoints: `/content/drive/MyDrive/HiMAC-JEPA/checkpoints`
- Label cache: `/content/drive/MyDrive/HiMAC-JEPA/cache/labels`
- Evaluation results: `/content/drive/MyDrive/HiMAC-JEPA/evaluation_results`

Keep the repo itself on Colab local disk (`/content/HiMAC-JEPA`) for speed. Keep large data and artifacts on Drive for persistence.

## 1. Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Clone repo and install deps

```bash
cd /content
git clone https://github.com/ChrisRPL/HiMAC-JEPA.git
cd /content/HiMAC-JEPA
pip install -r requirements.txt
pip install nuscenes-devkit
```

## 3. Put nuScenes-mini on Drive

Expected location:

```text
/content/drive/MyDrive/datasets/nuscenes
```

That directory should contain the extracted `v1.0-mini` files.

## 4. Set environment

```bash
export NUSCENES_ROOT=/content/drive/MyDrive/datasets/nuscenes
```

## 5. Train a checkpointed mini run

Smoke run:

```bash
python train.py \
  data=nuscenes \
  data.labels.enabled=true \
  data.labels.trajectory.include_agents=false \
  data.labels.cache.cache_dir=/content/drive/MyDrive/HiMAC-JEPA/cache/labels \
  training.checkpoint_dir=/content/drive/MyDrive/HiMAC-JEPA/checkpoints \
  training.epochs=1 \
  data.batch_size=2 \
  data.num_workers=2 \
  experiment_name=nuscenes-mini-smoke \
  wandb.enabled=false
```

First honest small benchmark:

```bash
python train.py \
  data=nuscenes \
  data.labels.enabled=true \
  data.labels.trajectory.include_agents=false \
  data.labels.cache.cache_dir=/content/drive/MyDrive/HiMAC-JEPA/cache/labels \
  training.checkpoint_dir=/content/drive/MyDrive/HiMAC-JEPA/checkpoints \
  training.epochs=3 \
  data.batch_size=2 \
  data.num_workers=2 \
  experiment_name=nuscenes-mini-benchmark \
  wandb.enabled=false
```

Outputs written automatically:

- latest: `/content/drive/MyDrive/HiMAC-JEPA/checkpoints/nuscenes-mini-benchmark/last_model.pth`
- best: `/content/drive/MyDrive/HiMAC-JEPA/checkpoints/nuscenes-mini-benchmark/best_model.pth`

## 6. Run evaluation

```bash
python scripts/evaluate.py \
  data=nuscenes \
  data.labels.enabled=true \
  data.labels.trajectory.include_agents=false \
  data.labels.cache.cache_dir=/content/drive/MyDrive/HiMAC-JEPA/cache/labels \
  evaluation.checkpoint_path=/content/drive/MyDrive/HiMAC-JEPA/checkpoints/nuscenes-mini-benchmark/best_model.pth \
  evaluation.results_dir=/content/drive/MyDrive/HiMAC-JEPA/evaluation_results \
  data.batch_size=2 \
  data.num_workers=2 \
  experiment_name=nuscenes-mini-benchmark
```

Results file:

- `/content/drive/MyDrive/HiMAC-JEPA/evaluation_results/results_nuscenes-mini-benchmark.json`

## Notes

- Downstream trajectory and BEV metrics are now label-backed.
- Temporal consistency only runs when you evaluate temporal sequence batches.
- Baseline comparison script still needs the honest dataloader path wired in separately.
