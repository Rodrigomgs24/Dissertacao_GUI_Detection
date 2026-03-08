# Data Pipeline

Download, conversion, and merging of GUI element datasets into YOLO format.

## Pipeline Scripts

Run in order:

| # | Script | Description |
|---|--------|-------------|
| 1 | `prepare_rico.py` | Download Rico (66K Android screenshots) + convert to YOLO |
| 2 | `download_webui.py` | Download WebUI-7k-balanced from Google Drive (gdown + 7z) |
| 3 | `prepare_webui.py` | Convert raw WebUI data (axtree + bounding boxes) to YOLO |
| 4 | `merge_datasets.py` | Merge Rico + WebUI into combined dataset + generate `data.yaml` |

### Quick run

```bash
python prepare_rico.py --download --convert
python download_webui.py
python prepare_webui.py --raw --webui_dir ./webui_raw
python merge_datasets.py
```

## Support Scripts

| Script | Description |
|--------|-------------|
| `class_mapping.py` | Unified taxonomy (12 classes) + Rico/WebUI label mappings |
| `convert_yolo_to_coco.ipynb` | Convert YOLO labels to COCO JSON format (needed for Faster R-CNN) |

## Data Sources

| Dataset | Platform | Size | Format | Source |
|---------|----------|------|--------|--------|
| **Rico** | Android | 66K screenshots (1440x2560) | Semantic JSON trees (`componentLabel` + `bounds`) | [interactionmining.org](https://interactionmining.org/rico) |
| **WebUI** | Web | 7K pages x 6 viewports | Chrome DevTools axtree + bounding boxes (gzip JSON) | [uimodeling.github.io](https://uimodeling.github.io/) |

## Output Structure

```
data/
├── rico/                      ← Raw Rico data (gitignored)
│   ├── combined/                   66K screenshots + view hierarchies
│   └── semantic_annotations/       Semantic labels (componentLabel)
│
├── webui_raw/                 ← Raw WebUI crawl data (gitignored)
│   └── {crawl_id}/
│       ├── {device}-axtree.json.gz
│       ├── {device}-bb.json.gz
│       └── {device}-screenshot.webp
│
└── unified/                   ← YOLO datasets (gitignored)
    ├── rico/
    │   └── {train,val,test}/{images,labels}/
    ├── webui/
    │   └── {train,val,test}/{images,labels}/
    └── combined/
        ├── data.yaml               ← YOLO training config
        └── {train,val,test}/{images,labels}/
```

## Documentation

- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — Full step-by-step reproducibility guide with expected outputs, technical decisions, and troubleshooting.
