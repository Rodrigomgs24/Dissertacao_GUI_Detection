# Faster R-CNN — Two-Stage Detection

Faster R-CNN with ResNet-50 + FPN backbone (PyTorch / torchvision).

| Notebook | Description |
|---|---|
| `train.ipynb` | Full training pipeline with AMP, warmup, gradient clipping, early stopping |
| `inference.ipynb` | Load best model and run inference with visual output |

## Training Configuration

- **Backbone:** ResNet-50 + FPN
- **Batch size:** 8 (reduce to 4 if OOM)
- **Epochs:** 100 (early stopping, patience=20)
- **LR:** 0.002 with 5-epoch warmup
- **Augmentations:** Colour jitter, horizontal flip, multi-scale (480–640px)
- **Mixed precision:** Enabled (AMP + GradScaler)
- **Gradient clipping:** max_norm=10
