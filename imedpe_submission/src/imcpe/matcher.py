# matcher.py 
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
    def __init__(
        self,
        device: str = "cuda",
        max_keypoints: int = 2048,
        resize: tuple[int, int] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.resize = resize  # (h, w) optional downscale for speed
        self.extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)
        # default config keeps point pruning enabled (the batched matching
        # path is unused: pruning bookkeeping breaks for batch > 1 and
        # pruning-free matching degrades weak-texture scenes)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)
        # one-slot cache: consecutive calls on the same image reuse features
        self._cache_path: Path | None = None
        self._cache_feats = None

    def _load(self, img_path: Path) -> torch.Tensor:
        """Load an image as RGB (C,H,W), optionally downscaling to self.resize.

        ALIKED is trained on RGB; grayscale loading measurably changes its
        keypoint responses.
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise RuntimeError(f"Could not load image: {img_path}")
        if self.resize is not None:
            h, w = self.resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
        return t.to(self.device)

    def extract(self, img_path: Path):
        """Extract features with a one-slot cache (deterministic extraction)."""
        if img_path == self._cache_path:
            return self._cache_feats
        image = self._load(img_path)[None]  # (1, C, H, W)
        feats = self.extractor.extract(image)
        if self.resize is not None:
            # lightglue keypoints are in the resized frame -> map back
            orig = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).shape[:2]
            rh, rw = self.resize
            scale = torch.tensor(
                [orig[1] / rw, orig[0] / rh], device=self.device, dtype=torch.float32
            )
            feats["keypoints"] = feats["keypoints"] * scale[None]
        self._cache_path = img_path
        self._cache_feats = feats
        return feats

    def match(self, img0_path: Path, img1_path: Path) -> tuple[np.ndarray, np.ndarray]:
        feats0 = self.extract(img0_path)
        feats1 = self.extract(img1_path)
        return self.match_features(feats0, feats1)

    def match_features(self, feats0, feats1) -> tuple[np.ndarray, np.ndarray]:
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]

        kpts0 = feats0["keypoints"][matches[:, 0]].detach().cpu().numpy()
        kpts1 = feats1["keypoints"][matches[:, 1]].detach().cpu().numpy()
        return kpts0, kpts1

    # --- batched variants (chunked inference) ---

    @torch.no_grad()
    def extract_batch(self, paths: list[Path]) -> list[dict]:
        """Extract features for a batch of images in ONE dense-map forward.

        The lightglue ALIKED wrapper asserts batch size 1 and its keypoint
        branch stacks per-image lists (equal sizes assumed), so we call the
        batched dense backbone directly and run the cheap per-image keypoint
        heads afterwards.
        """
        images = torch.stack([self._load(p) for p in paths])
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)  # ALIKED expects RGB
        feature_map, score_map = self.extractor.extract_dense_map(images)
        feats = []
        for b in range(len(paths)):
            kpts, kscores, _ = self.extractor.dkd(score_map[b : b + 1])
            desc, _ = self.extractor.desc_head(feature_map[b : b + 1], kpts)
            k = kpts[0]  # (N, 2), normalized to [-1, 1]
            h, w = images.shape[-2:]
            wh = torch.tensor([w - 1, h - 1], device=images.device, dtype=torch.float32)
            kp_px = wh[None, :] * (k + 1.0) / 2.0
            if self.resize is not None:
                orig = cv2.imread(str(paths[b]), cv2.IMREAD_GRAYSCALE).shape[:2]
                rh, rw = self.resize
                kp_px = kp_px * torch.tensor(
                    [orig[1] / rw, orig[0] / rh], device=images.device, dtype=torch.float32
                )[None]
            feats.append(
                {
                    "keypoints": kp_px,
                    "descriptors": desc[0],
                    "keypoint_scores": kscores[0],
                }
            )
        return feats

    @torch.no_grad()
    def match_batch(
        self, feats0_list: list[dict], feats1_list: list[dict]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Match a batch of feature pairs in ONE LightGlue forward.

        Items are padded to the max keypoint count (zero descriptors and
        scores mark the padding; LightGlue masks them).
        """
        B = len(feats0_list)
        N = max(f["keypoints"].shape[0] for f in feats0_list + feats1_list)
        D = feats0_list[0]["descriptors"].shape[1]

        def pad_stack(feats_list: list[dict]) -> dict:
            kp = torch.zeros(B, N, 2, device=self.device)
            desc = torch.zeros(B, N, D, device=self.device)
            scores = torch.zeros(B, N, device=self.device)
            for b, f in enumerate(feats_list):
                n = f["keypoints"].shape[0]
                kp[b, :n] = f["keypoints"]
                desc[b, :n] = f["descriptors"]
                scores[b, :n] = f["keypoint_scores"]
            return {"keypoints": kp, "descriptors": desc, "keypoint_scores": scores}

        d0 = pad_stack(feats0_list)
        d1 = pad_stack(feats1_list)
        out = self.matcher({"image0": d0, "image1": d1})

        # without point pruning the compact per-item format is used; match
        # indices refer to the PADDED arrays, so index into those
        matches_list = out["matches"]  # list of (N, 2) per batch item
        scores_list = out["scores"]

        results = []
        for b in range(B):
            m0 = matches_list[b]  # (N, 2)
            s0 = scores_list[b]   # (N,)
            valid = (m0[:, 0] >= 0) & (s0 > 0.0)
            k0 = d0["keypoints"][b][m0[valid, 0]].cpu().numpy()
            k1 = d1["keypoints"][b][m0[valid, 1]].cpu().numpy()
            results.append((k0, k1))
        return results


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
