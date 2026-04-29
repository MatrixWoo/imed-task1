from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd


@dataclass
class RelativePoseResult:
    r: np.ndarray
    t: np.ndarray
    num_matches: int
    num_inliers: int


class ALikeLightGlueMatcher:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)

    def match(self, img0_path: Path, img1_path: Path) -> tuple[np.ndarray, np.ndarray]:
        image0 = load_image(img0_path).to(self.device)
        image1 = load_image(img1_path).to(self.device)
        feats0 = self.extractor.extract(image0)
        feats1 = self.extractor.extract(image1)
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]

        kpts0 = feats0["keypoints"][matches[:, 0]].detach().cpu().numpy()
        kpts1 = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy()
        return kpts0, kpts1


def estimate_relative_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K: np.ndarray,
    ransac_thresh_px: float = 1.0,
) -> RelativePoseResult:
    E, mask = cv2.findEssentialMat(
        pts0,
        pts1,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh_px,
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    inliers, R, t, _ = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
    t = t.reshape(3)
    t = t / np.linalg.norm(t)

    return RelativePoseResult(
        r=R.astype(np.float64),
        t=t.astype(np.float64),
        num_matches=int(pts0.shape[0]),
        num_inliers=int(inliers),
    )
