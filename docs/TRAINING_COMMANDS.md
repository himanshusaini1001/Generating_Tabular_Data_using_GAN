# 🚀 Training Commands - Complete Guide

## Dataset: ~50k sequences (training50.fasta)

### ⭐ RECOMMENDED: Early Stopping + Bagging
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --bagging --early_stopping --n_models 3 --batch 64 --seq_len 70
```

**Expected:**
- Training time: ~3-4 hours (all 3 models)
- Epochs: 30-50 per model (auto-stops)
- Best accuracy with ensemble

---

### 🎯 BEST Performance: Early Stopping Only
```bash
python main.py --mode train --data data/training50.fasta --epochs 150 --early_stopping --validation_split 0.2 --batch 64
```

**Expected:**
- Training time: ~1.5-2 hours
- Epochs: 40-70 (auto-stops)
- Good balance of speed and quality

---

### ⚡ Fast Training: Smaller Batch Size
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --early_stopping --batch 32 --seq_len 70
```

**Expected:**
- Training time: ~2-3 hours
- Good for limited GPU memory
- Use if getting OOM errors

---

### 💪 Maximum Quality: Larger Batch
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --bagging --early_stopping --n_models 3 --batch 128 --seq_len 70
```

**Expected:**
- Training time: ~2.5-3.5 hours
- Faster training with large batches
- Requires more GPU memory (8GB+)

---

## Dataset: 20k sequences (training20.fasta)

### Quick Training
```bash
python main.py --mode train --data data/training20.fasta --epochs 80 --early_stopping --batch 64
```

### With Bagging
```bash
python main.py --mode train --data data/training20.fasta --epochs 80 --bagging --early_stopping --n_models 3
```

---

## Dataset: 22k sequences (training.fasta)

```bash
python main.py --mode train --data data/training.fasta --epochs 100 --early_stopping --batch 64
```

---

## 🎛️ Command Options Explained

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | train | Training mode |
| `--data` | required | Path to FASTA file |
| `--epochs` | 50 | Max epochs (early stopping can stop earlier) |
| `--batch` | 64 | Batch size (32/64/128/256) |
| `--seq_len` | 128 | Sequence length to use |
| `--early_stopping` | false | Enable auto-stop |
| `--validation_split` | 0.2 | Validation data ratio |
| `--bagging` | false | Enable ensemble |
| `--n_models` | 3 | Number of models for bagging |
| `--target_gc` | 0.42 | GC content target |

---

## 🖥️ GPU Memory Based Commands

### If you have 4GB GPU:
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --early_stopping --batch 32
```

### If you have 6GB GPU:
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --early_stopping --batch 64
```

### If you have 8GB+ GPU:
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --bagging --early_stopping --n_models 3 --batch 128
```

---

## ⏱️ Time Estimates (GPU)

### For 50k sequences:
- Batch 32: ~3-4 hours (single model)
- Batch 64: ~2-3 hours (single model)
- Batch 128: ~1.5-2 hours (single model)
- With bagging (3 models): ~6-9 hours total

### For 20k sequences:
- Batch 64: ~45-60 minutes
- With bagging: ~2-3 hours total

---

## 🔍 Monitor Training

### Check logs:
```bash
# Real-time log viewing
tail -f logs/train_log.txt

# Or in PowerShell:
Get-Content logs/train_log.txt -Wait
```

### Check GPU usage:
```bash
# Linux/Ubuntu
watch -n 1 nvidia-smi

# Windows (PowerShell)
nvidia-smi -l 1
```

---

## 💾 Checkpoints

After training, checkpoints are saved to:
- Single model: `checkpoints/stacked_epoch{N}.pt`
- Bagging: `checkpoints/bagging/bagging_model_{0,1,2}.pt`

---

## 🎬 Ready to Train?

### For 50k sequences - RECOMMENDED:
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --early_stopping --batch 64 --seq_len 70
```

### For BEST results with 50k:
```bash
python main.py --mode train --data data/training50.fasta --epochs 100 --bagging --early_stopping --n_models 3 --batch 64
```

---

## 📊 What to Expect

### Training Output:
```
📊 Dataset split: 40000 train + 10000 validation
Using device: cuda

Epoch 1/100 | Train D=0.3456 G=1.8901 | Val D=0.3321 G=1.7823 | Time=45.23s
Epoch 2/100 | Train D=0.3123 G=1.6543 | Val D=0.3045 G=1.6234 | Time=44.12s
...
Epoch 45/100 | Train D=0.1123 G=1.1234 | Val D=0.1145 G=1.1456 | Time=42.45s

⏹️  Early stopping triggered at epoch 45!
   No improvement for 10 epochs
   Best validation loss: D=0.1123, G=1.1234

✅ Best epoch: 35 with D_loss=0.1098
Checkpoint saved: checkpoints/stacked_epoch45.pt
```

---

## ❓ Troubleshooting

### Out of Memory (OOM):
- Reduce `--batch` from 64 to 32
- Reduce `--seq_len` from 70 to 50
- Use single model (remove `--bagging`)

### Training too slow:
- Increase `--batch` to 128
- Use CPU if GPU not available
- Reduce `--n_models` for bagging

### Want to pause/resume:
Checkpoints are saved, but currently no resume. You'll need to retrain.

