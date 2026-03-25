# Training Runbook

## Prerequisites
- GPU with 8GB+ VRAM (NVIDIA RTX 3070 or better / Google Colab T4)
- Python 3.10+, CUDA 11.8+, cuDNN 8.6+
- Dataset downloaded to `data/raw/`

## Step 1 — Environment Setup
```bash
conda create -n skin_cancer python=3.10
conda activate skin_cancer
pip install -r requirements.txt
```

## Step 2 — Download Dataset
```bash
python tools/scripts/download_dataset.py --source ham10000
# Downloads to data/raw/HAM10000/
```

## Step 3 — Preprocess Data
```bash
python src/data/preprocess.py \
  --input data/raw/HAM10000 \
  --output data/processed \
  --size 224
# Creates train/val/test splits with augmentation config
```

## Step 4 — Train Model
```bash
# Single model
python src/training/train.py \
  --model efficientnetb4 \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-4 \
  --experiment skin_cancer_v1

# Resume from checkpoint
python src/training/train.py --resume checkpoints/last.ckpt
```

## Step 5 — Evaluate
```bash
python src/evaluation/evaluate.py \
  --checkpoint models/best_model.h5 \
  --test_dir data/processed/test \
  --grad_cam True
```

## Step 6 — Export Model
```bash
python tools/scripts/export_model.py --format savedmodel,onnx
```

## Monitoring
- MLflow UI: `mlflow ui --port 5000`
- TensorBoard: `tensorboard --logdir logs/`

## Common Issues
| Issue | Fix |
|-------|-----|
| OOM on GPU | Reduce batch_size to 16 or use gradient accumulation |
| Val loss plateau | Enable ReduceLROnPlateau or unfreeze more layers |
| Class imbalance skew | Verify class_weight dict passed to model.fit() |
