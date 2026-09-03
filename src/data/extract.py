"""Video -> (patches, keypoints) clip extraction. Importable, no import-time side effects.

This is the canonical implementation, factored out of scripts/preprocess.py so that
training extraction and test-time extraction share one code path. They must: if the
two drift, the model is evaluated on clips built differently from the ones it trained
on, and no amount of careful metric work downstream can recover from that.

Faithful to the 6fca412 pipeline, with three corrections:
  * Frame rate is read per video (29.97 for 37 files in this corpus, 30.00 for four).
  * A dead `elif` branch with a condition identical to the `if` above it is removed;
    its intent -- copy-forward when there is no velocity history -- was already the
    behaviour of the velocity branch, which uses vx=vy=0 in that case.
  * Device and model are passed in rather than read from module globals.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config

# torch is imported lazily inside the functions that need it. Probing a video and
# planning its windows must work on a machine without a deep-learning stack -- that
# is what makes `--dry-run` useful for checking a plan before committing hours to it.

# Skeleton tracking constants, preserved from the base pipeline.
CONF_TH = 0.05        # below this a joint counts as unresolved
MAX_MISSING = 5       # frames a joint may be imputed before it is dropped
DECAY_PRED = 0.90     # confidence decay when a joint is velocity-predicted
MAX_JUMP = 80         # px; a larger inter-frame move is rejected as a tracking error
PAD = 64              # border added so a 128x128 crop near an edge stays in bounds
POSE_INPUT_HEIGHT = 256.0
HEATMAP_STRIDE = 8.0


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    n_frames: int
    duration_s: float


def probe_video(path) -> VideoInfo | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0 or n <= 0:
        return None
    return VideoInfo(fps=float(fps), n_frames=n, duration_s=n / float(fps))


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_gaussian_filter(fusion_size=None, sigma=None) -> np.ndarray:
    """[PAPER] Eq. 1 -- a 2D Gaussian centred on the joint, used to fuse the RGB crop."""
    fusion_size = config.FUSION_SIZE if fusion_size is None else fusion_size
    sigma = config.GAUSSIAN_SIGMA if sigma is None else sigma
    xs = np.arange(fusion_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, xs, indexing="xy")
    c = (fusion_size - 1) / 2.0
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma ** 2))
    g /= g.max()
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)


def load_pose_model(device=None, weights=None):
    import torch
    from src.pose.models.with_mobilenet import PoseEstimationWithMobileNet
    from src.pose.modules.load_state import load_state
    device = device or get_device()
    weights = weights or config.POSE_WEIGHTS
    net = PoseEstimationWithMobileNet().to(device).eval()
    load_state(net, torch.load(weights, map_location=device))
    return net


def process_maps_to_coords(heatmap, paf, scale, prev_kpts=None):
    """18 OpenPose joints for one frame, as (18, 3) of (x, y, confidence).

    Three strategies in descending order of reliability:
      A. PAF grouping -- the proper multi-person association. When a previous skeleton
         is known, the candidate pose closest to it is chosen, which keeps identity
         stable when a nurse or visitor enters frame.
      B. ROI search around the previous position of each joint, for when PAF fails.
      C. Global maximum per joint, only when there is no history at all.
    """
    from src.pose.modules.keypoints import extract_keypoints, group_keypoints

    all_kpts, total = [], 0
    for i in range(18):
        total += extract_keypoints(heatmap[i], all_kpts, total)

    coords = np.full((18, 3), -1.0, dtype=np.float32)
    poses, all_res = group_keypoints(all_kpts, paf.transpose(1, 2, 0))

    if poses is not None and len(poses) > 0:
        pose = None
        if prev_kpts is not None:
            best, best_dist = -1, float("inf")
            for p_idx, cand in enumerate(poses):
                dist, n_valid = 0.0, 0
                for i in range(18):
                    if cand[i] != -1 and prev_kpts[i][2] > 0:
                        k = all_res[int(cand[i])]
                        x = k[0] * HEATMAP_STRIDE / scale
                        y = k[1] * HEATMAP_STRIDE / scale
                        dist += float(np.hypot(x - prev_kpts[i][0], y - prev_kpts[i][1]))
                        n_valid += 1
                if n_valid:
                    avg = dist / n_valid
                    if avg < best_dist:
                        best_dist, best = avg, p_idx
            if best != -1:
                pose = poses[best]
        if pose is None:
            pose = poses[max(range(len(poses)),
                             key=lambda p: int(np.sum(poses[p][:18] != -1)))]
        for i in range(18):
            if pose[i] != -1:
                k = all_res[int(pose[i])]
                coords[i] = [k[0] * HEATMAP_STRIDE / scale,
                             k[1] * HEATMAP_STRIDE / scale, k[2]]
        return coords

    hm_h, hm_w = heatmap[0].shape
    radius = max(3, int(50 * scale / HEATMAP_STRIDE))
    for i in range(18):
        if prev_kpts is not None and prev_kpts[i][2] > 0.1:
            px = int(prev_kpts[i][0] * scale / HEATMAP_STRIDE)
            py = int(prev_kpts[i][1] * scale / HEATMAP_STRIDE)
            x0, x1 = max(0, px - radius), min(hm_w, px + radius)
            y0, y1 = max(0, py - radius), min(hm_h, py + radius)
            roi = heatmap[i][y0:y1, x0:x1]
            if roi.size:
                _, conf, _, loc = cv2.minMaxLoc(roi)
                if conf > 0.05:
                    coords[i] = [(x0 + loc[0]) * HEATMAP_STRIDE / scale,
                                 (y0 + loc[1]) * HEATMAP_STRIDE / scale, conf]
                    continue
        _, conf, _, loc = cv2.minMaxLoc(heatmap[i])
        if conf > 0.1:
            coords[i] = [loc[0] * HEATMAP_STRIDE / scale,
                         loc[1] * HEATMAP_STRIDE / scale, conf]
    return coords


def extract_clip(cap, fps: float, t_start_s: float, net, g_filter, device=None):
    """One clip -> (patches (T,15,3,32,32) float32 in [0,1], kpts (T,15,3) pixels).

    Returns (None, None) if the clip cannot be fully read.
    """
    import torch
    device = device or get_device()
    T = config.CLIP_FRAMES

    start_frame = int(round(t_start_s * fps))
    end_frame = int(round((t_start_s + config.CLIP_SECONDS) * fps))
    if end_frame <= start_frame:
        return None, None

    wanted = set(np.linspace(start_frame, end_frame - 1, T).astype(int).tolist())
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    for _ in range(end_frame - start_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) == T:
            break
    if len(frames) < T:
        return None, None

    h = frames[0].shape[0]
    scale = POSE_INPUT_HEIGHT / float(h)
    batch = np.stack([cv2.resize(f, (0, 0), fx=scale, fy=scale) for f in frames])
    batch = (batch.astype(np.float32) - 128.0) / 256.0
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        out = net(tensor)
        heatmaps = out[-2].cpu().numpy()
        pafs = out[-1].cpu().numpy()

    persistent = np.full((18, 3), -1.0, dtype=np.float32)
    prev_positions = np.full((18, 2), -1.0, dtype=np.float32)
    missing = np.zeros(18, dtype=np.int32)
    clip_patches, clip_kpts = [], []

    for i in range(T):
        raw = process_maps_to_coords(heatmaps[i], pafs[i], scale, prev_kpts=persistent)

        for k in range(18):
            x, y, conf = raw[k]
            px, py, pconf = persistent[k]
            if conf > CONF_TH:
                if pconf > CONF_TH and np.hypot(x - px, y - py) > MAX_JUMP:
                    conf = -1.0          # implausible jump: treat as a miss
                if conf > CONF_TH:
                    prev_positions[k] = persistent[k][:2]
                    persistent[k] = raw[k]
                    missing[k] = 0
                    continue
            missing[k] += 1
            if pconf > CONF_TH and missing[k] <= MAX_MISSING:
                # Constant-velocity carry-forward. With no velocity history the
                # deltas are zero, which is exactly a copy-forward -- so the base
                # pipeline's separate copy branch was unreachable and is dropped.
                vx = px - prev_positions[k][0] if prev_positions[k][0] >= 0 else 0.0
                vy = py - prev_positions[k][1] if prev_positions[k][1] >= 0 else 0.0
                raw[k] = np.array([px + vx, py + vy, pconf * DECAY_PRED], np.float32)
                persistent[k] = raw[k]
            else:
                persistent[k] = np.array([-1, -1, -1], np.float32)
                raw[k] = np.array([-1, -1, -1], np.float32)

        coords = raw[config.JOINT_INDICES]
        clip_kpts.append(coords.copy())

        padded = cv2.copyMakeBorder(frames[i], PAD, PAD, PAD, PAD,
                                    cv2.BORDER_CONSTANT, value=0)
        half = config.FUSION_SIZE // 2
        patch_batch = np.zeros((config.N_JOINTS, config.PATCH_SIZE,
                                config.PATCH_SIZE, 3), dtype=np.float32)
        for j, (x, y, _c) in enumerate(coords):
            if x < 0 or y < 0:
                continue
            xp, yp = int(round(x + PAD)), int(round(y + PAD))
            crop = padded[yp - half:yp + half, xp - half:xp + half].astype(np.float32)
            if crop.shape == (config.FUSION_SIZE, config.FUSION_SIZE, 3):
                fused = crop * g_filter
                patch_batch[j] = cv2.resize(
                    fused, (config.PATCH_SIZE, config.PATCH_SIZE),
                    interpolation=cv2.INTER_CUBIC) / 255.0
        clip_patches.append(patch_batch)

    patches = np.array(clip_patches).transpose(0, 1, 4, 2, 3)   # T,15,3,32,32
    return patches, np.array(clip_kpts)


def label_seconds(value):
    """Label.xlsx cell -> seconds. Handles datetime.time and 'HH:MM:SS' / 'MM:SS'."""
    import pandas as pd
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    if hasattr(value, "hour"):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    parts = str(value).strip().split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except ValueError:
        return None
    return None
