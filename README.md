<div align="center">

# 🐦 Fine-Grained Bird Recognition on Edge Devices

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-00FFFF?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Swin](https://img.shields.io/badge/Swin--B-Classifier-7B68EE?style=flat-square)](https://github.com/microsoft/Swin-Transformer)
[![Sophon](https://img.shields.io/badge/Sophon-SE5-FF6B35?style=flat-square)](https://www.sophon.ai)

**Two-stage edge pipeline for 200-class fine-grained bird recognition — YOLOv8s-CBAM detection → TARA-Cascade Swin-B classifier**

> ⚠️ **Showcase Only** — ~15% skeleton. Training code, model weights & dataset processing not included.

</div>

---

## ✨ Overview

Undergraduate thesis project at Sichuan University. The system detects birds in natural images and classifies them across 200 fine-grained species (CUB-200-2011), with hierarchical Order / Family / Genus / Species predictions. Optimized for real-time inference on the Sophon SE5 edge AI box.

**Final performance on edge device:** 22.9 ms / frame · 43 FPS · ~91% Top-1.

---

## 🏗️ Architecture

```
                Input Image
                     │
                     ▼
        ┌────────────────────────────┐
        │  Stage 1 · Detection       │
        │  YOLOv8s + CBAM + SIoU     │
        └──────────────┬─────────────┘
                       │ cropped bird patch
                       ▼
        ┌────────────────────────────┐
        │  Stage 2 · Classification  │
        │  TARA-Cascade Swin-B       │
        │  ├── Order   (13 classes)  │
        │  ├── Family  (37 classes)  │
        │  ├── Genus   (122 classes) │
        │  └── Species (200 classes) │
        └──────────────┬─────────────┘
                       │
                       ▼
              Hierarchical Output
                       │
                       ▼
        ┌────────────────────────────┐
        │  Edge Deployment           │
        │  Sophon SE5 · BMNNSDK      │
        │  22.9ms / 43 FPS           │
        └────────────────────────────┘
```

---

## 📁 Structure

```
fine-grained-bird-recognition/
├── src/
│   ├── detection/
│   │   └── yolov8_cbam.py       # YOLOv8s + CBAM + SIoU
│   ├── classifier/
│   │   └── swin_tara.py         # TARA-Cascade Swin-B
│   └── edge/
│       └── sophon_deploy.py     # Sophon SE5 inference wrapper
├── configs/
│   └── model.yaml
└── requirements.txt
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLOv8s + CBAM attention + SIoU loss |
| Classification | Swin Transformer-Base + TARA Cascade |
| Framework | PyTorch 2.x |
| Edge Runtime | BMNNSDK on Sophon SE5 (BM1684) |
| Dataset | CUB-200-2011 |

---

## 📊 Key Results

| Stage | Metric | Value |
|---|---|---|
| Detection | mAP@50 | 96.2% |
| Classification (Species) | Top-1 | 91.3% |
| Edge Inference | Latency | 22.9 ms |
| Edge Inference | Throughput | 43 FPS |

---

<div align="center">
<sub>Showcase version · Core training code not included · For portfolio reference only</sub>
</div>
