"""Clip dataset for every training and evaluation pipeline.

Two corrections to the 6fca412 version, both load-bearing:

1. KEYPOINT CHANNELS. The old code sliced `kpts[:, :, :2]` while STViG instantiates
   `Stem_pe(input_dim=3)`, so the first forward pass raised a channel mismatch --
   6fca412 could not run at all. Channel count is now driven by config.KPT_CHANNELS
   and passed to the model, so the two cannot disagree again.

2. NORMALISATION. The old code divided x by 1920 and y by 1080. The methodology
   (section 3.1) requires centering on the mid-hip and scaling by torso length. See
   src/data/normalize.py.

`eval_mode` returns the clip's START TIME IN SECONDS. The old code called this
`frame_num` and the evaluator divided it by an assumed 25 fps, which corrupted every
latency figure. The name is now `t_start_s` and it is never divided by anything.
"""
import json
import os

import torch
from torch.utils.data import Dataset

import config
from src.data.normalize import normalize_skeleton
from src.utils.naming import parse_clip_name  # re-exported for callers

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class VSViGDataset(Dataset):
    """Label file may be a JSON list [[name, label], ...] or a dict {name: label}."""

    def __init__(self, data_folder, label_file, eval_mode: bool = False):
        self.patches_dir = os.path.join(str(data_folder), "patches")
        self.kpts_dir = os.path.join(str(data_folder), "kpts")
        self.eval_mode = eval_mode

        if not os.path.exists(self.patches_dir) or not os.path.exists(self.kpts_dir):
            raise FileNotFoundError(
                f"'patches' and 'kpts' must exist inside {data_folder}")

        with open(label_file, "r") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            self._labels = raw
        elif isinstance(raw, dict):
            self._labels = [[k, v] for k, v in raw.items()]
        else:
            raise ValueError("Label file must be a JSON list or dict.")

    def __len__(self):
        return len(self._labels)

    def _load(self, name):
        data = torch.load(os.path.join(self.patches_dir, f"{name}.pt"),
                          map_location="cpu").float()
        if data.max() > 2.0:
            data = data / 255.0
        data = (data - MEAN) / STD                      # (30, 15, 3, 32, 32)

        kpts = torch.load(os.path.join(self.kpts_dir, f"{name}.pt"),
                          map_location="cpu")
        kpts = normalize_skeleton(kpts)                 # (30, 15, KPT_CHANNELS)
        return data, kpts

    def __getitem__(self, idx):
        name = str(self._labels[idx][0])
        target = float(self._labels[idx][1])
        data, kpts = self._load(name)

        if not self.eval_mode:
            return {"data": data, "kpts": kpts}, target

        patient, event, t_start_s = parse_clip_name(name)
        return data, kpts, target, f"{patient}_{event}", t_start_s
