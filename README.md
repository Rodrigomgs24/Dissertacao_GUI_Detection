# GUI Elements Detection

**Detection of Graphical User Interface Elements Using Deep Learning**

> Master's Dissertation in Informatics Engineering — Software Engineering  
> Instituto Superior de Engenharia do Porto (ISEP), 2025  
> **Author:** Rodrigo Santos Magalhães

---

## Overview

This project develops and evaluates deep learning models for detecting graphical elements in user interfaces (GUIs). Accurate identification of GUI components — such as buttons, menus, text fields, and icons — is essential for advancing **test automation**, **accessibility enhancement**, and **intelligent interface generation**.

Three approaches are compared:

| Approach                         | Type                      | Implementation                |
| -------------------------------- | ------------------------- | ----------------------------- |
| [**YOLO**](models/yolo/) (v8, v10, v11) | Single-stage detector     | Ultralytics framework         |
| [**Faster R-CNN**](models/faster-rcnn/) | Two-stage detector        | ResNet-50 + FPN (torchvision) |
| [**OpenCV**](models/opencv/)            | Classical computer vision | Edge detection baseline       |

---

## Repository Structure

```
GUI-Elements-Detection/
│
├── models/                     ← All model implementations
│   ├── yolo/                        YOLO models (v8, v10, v11)
│   │   ├── train.ipynb                   Train all 3 YOLO variants
│   │   └── inference.ipynb               Inference with best model
│   │
│   ├── faster-rcnn/                 Faster R-CNN (ResNet-50 + FPN)
│   │   ├── train.ipynb                   Full training pipeline
│   │   └── inference.ipynb               Inference with trained model
│   │
│   └── opencv/                      Classical CV baseline
│       └── detection.ipynb               Edge detection pipeline
│
├── data/                        ← Dataset preparation
│   ├── convert_data_yaml.ipynb       Generate YOLO config
│   └── convert_yolo_to_coco.ipynb    YOLO → COCO conversion
│
├── results/                     ← Outputs organised by model
│   ├── yolo/
│   ├── faster-rcnn/
│   ├── opencv/
│
└── docs/                        ← Dissertation
```

---

## Dataset

[**Wave-UI**](https://huggingface.co/datasets/agentsea/wave-ui) dataset, annotated with [Label Studio](https://labelstud.io/) for **16 GUI component classes**: Button, Checkbox, Dropdown, Icon, Image, Input, Link, Menu, Modal, Progress bar, Radio button, Slider, Tab, Text, Text field, Toolbar.

**Split:** 70% training · 20% testing · 10% validation

> ⚠️ The dataset is not included in this repository. See [`data/README.md`](data/README.md) for setup instructions.

---

## Setup

```bash
git clone https://github.com/<your-username>/GUI-Elements-Detection.git
```

## Usage

All notebooks are designed to run in **Google Colab**.

1. **Prepare data** — `data/convert_data_yaml.ipynb` → `data/convert_yolo_to_coco.ipynb`
2. **Train** — `models/yolo/train.ipynb` or `models/faster-rcnn/train.ipynb`
3. **Inference** — `models/yolo/inference.ipynb`, `models/faster-rcnn/inference.ipynb`, or `models/opencv/detection.ipynb`

---

## Technologies

[PyTorch](https://pytorch.org/) · [torchvision](https://pytorch.org/vision/) · [Ultralytics YOLO](https://docs.ultralytics.com/) · [OpenCV](https://opencv.org/) · [pycocotools](https://github.com/cocodataset/cocoapi) · [Label Studio](https://labelstud.io/) · [Google Colab](https://colab.research.google.com/) · [Matplotlib](https://matplotlib.org/)

---

## License

MIT — see [LICENSE](LICENSE).
