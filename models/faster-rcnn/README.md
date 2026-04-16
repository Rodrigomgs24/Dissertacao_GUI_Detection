# Faster R-CNN — Two-Stage Detection

Faster R-CNN with ResNet-50 + FPN backbone (PyTorch / torchvision).

| Notebook | Description |
|---|---|
| `train.ipynb` | Full training pipeline with AMP, warmup, gradient clipping, early stopping |
| `inference.ipynb` | Load best model and run inference with visual output |

## Training Configuration

- **Backbone:** ResNet-50 + FPN
- **Batch size:** 2 (gradient accumulation × 4 = effective 8)
- **Epochs:** 8 (early stopping, patience=5)
- **LR:** 0.002 with 2-epoch warmup
- **Augmentations:** Colour jitter, horizontal flip, multi-scale (416–512px)
- **Mixed precision:** AMP + GradScaler
- **Gradient clipping:** max_norm=10
