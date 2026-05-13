"""
Sophon SE5 edge inference wrapper.
Converts PyTorch models to BModel format and runs via BMNNSDK runtime.
"""

from typing import Dict
import numpy as np


class SophonInferencer:
    def __init__(self, bmodel_path: str, device_id: int = 0):
        self.bmodel_path = bmodel_path
        # [BMRuntime init not shown]
        pass

    def detect(self, image: np.ndarray) -> Dict:
        """Stage 1: run YOLOv8s-CBAM on full image. [Not shown]"""
        pass

    def classify(self, bird_crop: np.ndarray) -> Dict:
        """Stage 2: hierarchical classification. [Not shown]"""
        pass

    def pipeline(self, image: np.ndarray) -> Dict:
        """End-to-end: detect → crop → classify. Avg 22.9 ms / frame. [Not shown]"""
        pass
