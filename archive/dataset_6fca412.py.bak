import json
import os
import torch
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


class VSViGDataset(Dataset):
    """
    Unified dataset for all training and evaluation pipelines.

    Label file can be either:
      - a JSON list:  [[filename, label], ...]
      - a JSON dict:  {filename: label, ...}

    Args:
        data_folder: root folder containing 'patches/' and 'kpts/' subdirs
        label_file:  path to JSON label file
        eval_mode:   if True, __getitem__ also returns (unique_id, frame_num)
                     for use in evaluation scripts
    """

    def __init__(self, data_folder: str, label_file: str, eval_mode: bool = False):
        self.patches_dir = os.path.join(data_folder, "patches")
        self.kpts_dir    = os.path.join(data_folder, "kpts")
        self.eval_mode   = eval_mode

        if not os.path.exists(self.patches_dir) or not os.path.exists(self.kpts_dir):
            raise FileNotFoundError(
                f"'patches' and 'kpts' folders must exist inside {data_folder}"
            )

        with open(label_file, "r") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            self._labels = raw                          # [[name, label], ...]
        elif isinstance(raw, dict):
            self._labels = [[k, v] for k, v in raw.items()]
        else:
            raise ValueError("Label file must be a JSON list or dict.")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        name   = str(self._labels[idx][0])
        target = float(self._labels[idx][1])

        data = torch.load(
            os.path.join(self.patches_dir, f"{name}.pt"), map_location="cpu"
        )
        kpts = torch.load(
            os.path.join(self.kpts_dir, f"{name}.pt"), map_location="cpu"
        )

        # --- patches: ensure float32 in [0,1] then ImageNet-standardize ---
        data = data.float()
        if data.max() > 2.0:
            data = data / 255.0
        data = (data - MEAN) / STD          # shape: (30, 15, 3, 32, 32)

        # --- keypoints: normalize to [0,1] ---
        kpts = kpts.float()
        if kpts.max() > 2.0:
            kpts[:, :, 0] = kpts[:, :, 0] / 1920.0
            kpts[:, :, 1] = kpts[:, :, 1] / 1080.0
        # keep only x,y columns (drop confidence if present)
        kpts = kpts[:, :, :2]               # shape: (30, 15, 2)

        if not self.eval_mode:
            return {"data": data, "kpts": kpts}, target

        # eval_mode: also return identifiers needed for accumulation logic
        parts      = name.split("_")
        unique_id  = f"{parts[0]}_{parts[1]}"
        try:
            frame_num = float(parts[-1])
        except ValueError:
            frame_num = 0.0

        return data, kpts, target, unique_id, frame_num
