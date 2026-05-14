# Fine-Grained-Bird-Recognition

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

> **Showcase** — ~15% skeleton. Core implementation not included.

Fine-grained bird species recognition deployed on the Sophon SE5 edge AI chip. Combines YOLOv8 for detection with Swin Transformer-Base for classification across 200 species at 22.9 ms / 43 FPS on-device.

## Stack

- Python, PyTorch
- YOLOv8 (detection stage)
- Swin Transformer-Base (classification stage)
- Sophon SE5 TPU SDK (BM1684X)

## Architecture

```
Input frame
    └── YOLOv8 detector        # locates bird bounding boxes
         └── Swin-B classifier # per-crop species prediction (200 classes)
              └── Result overlay + latency counter
```

Detection and classification run sequentially on the SE5 TPU. The SDK compiles each model to BModel format before deployment.

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | 22.9 ms |
| Throughput | 43 FPS |
| Species | 200 |
| Platform | Sophon SE5 (BM1684X) |

## Usage

```bash
# Compile models to BModel (requires TPU SDK)
python tools/compile.py --detector yolov8n.pt --classifier swin_base.pth

# Run inference on a video file
python infer.py --input video.mp4 --output result.mp4

# Run on a live RTSP stream
python infer.py --input rtsp://camera-ip/stream
```

## Structure

```
Fine-Grained-Bird-Recognition/
├── models/          # BModel files (compiled)
├── tools/
│   └── compile.py   # TPU model compiler wrapper
├── infer.py         # main inference entry point
├── classifier.py    # Swin-B classification module
├── detector.py      # YOLOv8 detection module
└── requirements.txt
```

## Requirements

- Sophon SE5 board with BM1684X
- TPU SDK >= 3.0
- Python 3.8+
- PyTorch 1.13+
