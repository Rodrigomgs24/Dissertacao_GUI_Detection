# GUI Elements Detection

**Detection of Graphical User Interface Elements Using Deep Learning**

> Master's Dissertation in Informatics Engineering — Software Engineering
> Instituto Superior de Engenharia do Porto (ISEP), 2025
> **Author:** Rodrigo Santos Magalhães

---

## Overview

This project develops and evaluates deep learning models for detecting graphical elements in user interfaces (GUIs). Accurate identification of GUI components — such as buttons, menus, text fields, and icons — is essential for advancing **test automation**, **accessibility enhancement**, and **intelligent interface generation**.

Three detection approaches are compared:

| Approach | Type | Implementation |
|----------|------|----------------|
| [**YOLO**](models/yolo/) (v8, v10, v11) | Single-stage detector | Ultralytics framework |
| [**Faster R-CNN**](models/faster-rcnn/) | Two-stage detector | ResNet-50 + FPN (torchvision) |
| [**OpenCV**](models/opencv/) | Classical computer vision | Edge detection baseline |

---

## Dataset

A cross-platform dataset combining **mobile** and **web** UI screenshots, with **12 unified GUI element classes**:

| Source | Platform | Images | Annotations | Reference |
|--------|----------|--------|-------------|-----------|
| [Rico](https://interactionmining.org/rico) | Android mobile | 63,160 | 1,075,398 | Deka et al., UIST 2017 |
| [WebUI](https://uimodeling.github.io/) | Web (6 viewports) | 41,970 | 2,242,576 | Wu et al., CHI 2023 |
| **Combined** | **Cross-platform** | **105,130** | **3,317,974** | — |

**Split:** 80% train · 10% validation · 10% test

### Unified Classes (12)

```
Button · Text · Image · Icon · Input · Link · Checkbox · Toggle · Toolbar · Navigation · Modal · Tab
```

> The dataset is not included in this repository due to size (~30 GB).
> See [`data/README.md`](data/README.md) for download and preparation instructions.

---

## Repository Structure

```
Dissertacao_GUI_Detection/
│
├── data/                           ← Dataset pipeline
│   ├── class_mapping.py                 Unified taxonomy (12 classes)
│   ├── prepare_rico.py                  Download + convert Rico → YOLO
│   ├── download_webui.py                Download WebUI from Google Drive
│   ├── prepare_webui.py                 Convert WebUI → YOLO
│   ├── merge_datasets.py               Merge Rico + WebUI → combined dataset
│   ├── convert_yolo_to_coco.ipynb       YOLO → COCO format (for Faster R-CNN)
│   └── REPRODUCIBILITY.md              Full pipeline documentation
│
├── models/                         ← Model implementations
│   ├── yolo/                            YOLO v8, v10, v11
│   │   ├── train.ipynb                       Train all 3 variants
│   │   └── inference.ipynb                   Inference with best model
│   │
│   ├── faster-rcnn/                     Faster R-CNN (ResNet-50 + FPN)
│   │   ├── train.ipynb                       Full training pipeline
│   │   └── inference.ipynb                   Inference with trained model
│   │
│   └── opencv/                          Classical CV baseline
│       └── detection.ipynb                   Edge detection pipeline
│
├── results/                        ← Outputs by model
│   ├── yolo/
│   ├── faster-rcnn/
│   └── opencv/
│
├── docs/                           ← Dissertation document
├── requirements.txt                ← Python dependencies
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

```bash
cd data/

# Rico (mobile) — download + convert (~30 min)
python prepare_rico.py --download --convert

# WebUI (web) — download (~1-2h) + convert (~15 min)
python download_webui.py
python prepare_webui.py --raw --webui_dir ./webui_raw

# Merge into combined dataset + generate data.yaml
python merge_datasets.py
```

### 3. Train models

```bash
# YOLO (Ultralytics)
# → Open models/yolo/train.ipynb

# Faster R-CNN (PyTorch)
# → First convert: data/convert_yolo_to_coco.ipynb
# → Then train:    models/faster-rcnn/train.ipynb
```

> See [`data/REPRODUCIBILITY.md`](data/REPRODUCIBILITY.md) for the full step-by-step guide.

---

## Technologies

[PyTorch](https://pytorch.org/) · [torchvision](https://pytorch.org/vision/) · [Ultralytics YOLO](https://docs.ultralytics.com/) · [OpenCV](https://opencv.org/) · [pycocotools](https://github.com/cocodataset/cocoapi) · [Matplotlib](https://matplotlib.org/) · [Google Colab](https://colab.research.google.com/)

---

## License

MIT — see [LICENSE](LICENSE).
