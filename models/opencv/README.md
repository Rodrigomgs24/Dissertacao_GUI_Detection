# OpenCV — Classical Computer Vision Baseline

GUI element detection using traditional image processing (no deep learning).

| Notebook | Description |
|---|---|
| `detection.ipynb` | Full pipeline: grayscale → blur → Canny → morphology → contours |

## Limitations

- No class distinction — all contours treated equally
- Fails with elements without visible borders
- Sensitive to manual parameter tuning
- Cannot handle overlapping elements

These limitations justify the use of deep learning (YOLO, Faster R-CNN).
