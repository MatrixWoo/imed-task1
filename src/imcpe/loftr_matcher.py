# loftr_matcher.py
"""LoFTR dense matcher wrapper (kornia implementation, outdoor weights).

LoFTR matches every position of the coarse feature maps via cross-attention
instead of detecting sparse keypoints first — much more robust in
texture-poor endoscopic scenes, at the cost of heavier compute.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


class LoFTRMatcher:
    def __init__(self, device: str = "cuda", conf_thresh: float = 0.5) -> None:
        from kornia.feature import LoFTR

        self.device = torch.device(device)
        self.model = LoFTR(pretrained="outdoor").to(self.device).eval()
        self.conf_thresh = conf_thresh

    @torch.no_grad()
    def match(self, img0_path: Path, img1_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Match a pair of images; returns (kpts0, kpts1) as (N, 2) numpy arrays
        in original image coordinates (input size must be divisible by 8;
        kornia LoFTR works at the input resolution).
        """
        img0 = cv2.imread(str(img0_path), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
        h, w = img0.shape
        h8, w8 = h // 8 * 8, w // 8 * 8
        if (h8, w8) != (h, w):
            img0 = img0[:h8, :w8]
            img1 = img1[:h8, :w8]

        x0 = torch.from_numpy(img0)[None, None].float().div(255.0).to(self.device)
        x1 = torch.from_numpy(img1)[None, None].float().div(255.0).to(self.device)
        out = self.model({"image0": x0, "image1": x1})

        k0 = out["keypoints0"].cpu().numpy()
        k1 = out["keypoints1"].cpu().numpy()
        conf = out["confidence"].cpu().numpy()
        m = conf >= self.conf_thresh
        return k0[m], k1[m]
