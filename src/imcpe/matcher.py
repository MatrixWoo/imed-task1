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


def _normalize_points(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pts, K, None)
    return normalized.reshape(-1, 2)


def estimate_relative_pose(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray | None = None,
    ransac_thresh: float = 1e-3,
) -> RelativePoseResult:
    """Estimate relative pose from pts0 (camera 0) to pts1 (camera 1).

    Uses a single intrinsics matrix when K1 is None; otherwise normalizes
    with K0 and K1 for cross-camera essential matrix estimation.
    """
    if K1 is None:
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            cameraMatrix=K0,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh if ransac_thresh > 0.01 else 1.0,
        )
        recover_K = K0
        pts0_rec, pts1_rec = pts0, pts1
    else:
        pts0_n = _normalize_points(pts0, K0)
        pts1_n = _normalize_points(pts1, K1)
        E, mask = cv2.findEssentialMat(
            pts0_n,
            pts1_n,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh,
        )
        recover_K = np.eye(3, dtype=np.float64)
        pts0_rec, pts1_rec = pts0_n, pts1_n

    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    inliers, R, t, _ = cv2.recoverPose(E, pts0_rec, pts1_rec, recover_K, mask=mask)
    t = t.reshape(3)
    t = t / np.linalg.norm(t)

    return RelativePoseResult(
        r=R.astype(np.float64),
        t=t.astype(np.float64),
        num_matches=int(pts0.shape[0]),
        num_inliers=int(inliers),
    )
