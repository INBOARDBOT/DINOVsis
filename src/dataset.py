"""
dataset.py  —  NPC T1 NIfTI → 2-D axial slice dataset for DINOv3 fine-tuning.

Each volume is loaded once, sliced along the axial axis, and each (image, mask)
slice pair is returned as a PyTorch sample.  Only slices that contain at least
one foreground voxel in the mask are kept by default (set keep_empty=True to
disable this filter so that the full volume is used).

CSV format (no header):
    data/NPC_pre/T1/image/Case001.nii.gz,data/NPC_pre/T1/label/Case001.nii.gz
Paths in the CSV are relative to the project root (two levels above this file).
"""

import csv
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

# ---------------------------------------------------------------------------
# Root of the whole project (two levels above this script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Low-level helpers (re-exported so train.py can import them directly)
# ---------------------------------------------------------------------------

def load_nii(path: str) -> Tuple[np.ndarray, tuple]:
    """Load a NIfTI file and return (data, voxel_zooms)."""
    img = nib.load(path)
    data = img.get_fdata()          # (H, W, D) for a typical 3-D volume
    return data, img.header.get_zooms()


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Clip to 1st–99th percentile and normalise to [0, 1] (float32)."""
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-8)
    return volume.astype(np.float32)


def volume_to_slices(volume: np.ndarray, axis: int = 2) -> np.ndarray:
    """Return 2-D slices stacked along the first axis: (D, H, W)."""
    return np.moveaxis(volume, axis, 0)


def make_image_transform(size: int = 896) -> v2.Compose:
    """Standard DINOv3 LVD-1689M image transform."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])


def make_mask_transform(size: int = 896) -> v2.Compose:
    """Nearest-neighbour resize for segmentation masks (no normalisation)."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size),
                  interpolation=v2.InterpolationMode.NEAREST,
                  antialias=False),
        v2.ToDtype(torch.long, scale=False),
    ])


def slice_to_rgb_pil(slice_2d: np.ndarray) -> Image.Image:
    """Convert a (H, W) float32 [0, 1] array to an RGB PIL image."""
    rgb = np.stack([slice_2d] * 3, axis=-1)          # (H, W, 3)
    rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_u8)



class NPCSurvivalDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_transform: Optional[Callable] = None,
        axis: int = 2,
        img_size: int = 896,
        max_slices: Optional[int] = 30,
        keep_empty: bool = True,
    ):
        self.img_transform = img_transform or make_image_transform(img_size)
        self.axis = axis
        self.max_slices = max_slices
        self.keep_empty = keep_empty

        self.df = pd.read_csv(csv_path)
        
        required_cols = ['image_pth', 'survival_label_path']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in {csv_path}")
        
        print(f"Loaded {len(self.df)} patients from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = row['image_pth'].strip()
        label_path = row['survival_label_path'].strip()
        case_number = row.get('case_number', 'unknown')

        volume, _ = load_nii(img_path)  # Returns (H, W, D) or similar
        volume = normalize_volume(volume)  # Normalize to [0, 1] or z-score

        slices = volume_to_slices(volume, self.axis)  # (num_slices, H, W)

        if not self.keep_empty:
            keep_idx = [i for i, s in enumerate(slices) if s.mean() > 1e-3]
            if keep_idx:
                slices = slices[keep_idx]

        if self.max_slices is not None and len(slices) > self.max_slices:
            sel = np.linspace(0, len(slices) - 1, self.max_slices).round().astype(int)
            slices = slices[sel]

        slice_tensors = []
        for s in slices:
            img_pil = slice_to_rgb_pil(s)  # Converts (H, W) to RGB PIL
            img_tensor = self.img_transform(img_pil)  # (3, H, W)
            slice_tensors.append(img_tensor)

        slices_tensor = torch.stack(slice_tensors, dim=0)

        with open(label_path, 'r') as f:
            survival_data = json.load(f)

        time_months = torch.tensor(survival_data['time_months'], dtype=torch.float32)
        event = torch.tensor(survival_data['event'], dtype=torch.float32)

        return slices_tensor, time_months, event

    # ------------------------------------------------------------------
    @staticmethod
    def survival_collate_fn(batch):
        """
        Collate function for NPCSurvivalDataset.

        Each patient can have a different number of slices, so we can't stack
        them into one tensor. Instead, return a list of per-patient slice
        tensors plus stacked time/event tensors.
        
        Args:
            batch: List of tuples (slices_tensor, time_months, event)
            
        Returns:
            slices_list: List of tensors, each of shape (num_slices_i, 3, H, W)
            times: Tensor of shape (batch_size,) with survival times
            events: Tensor of shape (batch_size,) with event indicators
        """
        slices_list, times, events = zip(*batch)
        times = torch.stack(times)
        events = torch.stack(events)
        return list(slices_list), times, events



# class NPCSurvivalDatasetPreprocessed(Dataset):
#     def __init__(self, csv_path: str):
#         """
#         Lightweight Dataset that consumes pre-calculated slice tensors directly.
#         Requires CSV generated by the preprocessing script.
#         """
#         self.df = pd.read_csv(csv_path)
        
#         required_cols = ['preprocessed_tensor_pth', 'survival_label_path']
#         for col in required_cols:
#             if col not in self.df.columns:
#                 raise ValueError(f"Required column '{col}' not found in {csv_path}")
        
#         print(f"Loaded {len(self.df)} preprocessed patient pointers from {csv_path}")

#     def __len__(self) -> int:
#         return len(self.df)

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         row = self.df.iloc[idx]

#         tensor_path = row['preprocessed_tensor_pth'].strip()
#         label_path = row['survival_label_path'].strip()

#         # Instantaneous operation; loads exactly what is needed into memory
#         # Shape: (num_slices, 3, H, W)
#         slices_tensor = torch.load(tensor_path, map_location="cpu")

#         # Load survival JSON info
#         with open(label_path, 'r') as f:
#             survival_data = json.load(f)

#         time_months = torch.tensor(survival_data['time_months'], dtype=torch.float32)
#         event = torch.tensor(survival_data['event'], dtype=torch.float32)

#         return slices_tensor, time_months, event

#     @staticmethod
#     def survival_collate_fn(batch):
#         slices_list, times, events = zip(*batch)
        
#         # Stack individual patient tensors into a single uniform multi-GPU batch matrix
#         # Output shape: [Batch_Size, Num_Slices, 3, H, W]
#         slices_tensor = torch.stack(slices_list, dim=0) 
        
#         times = torch.stack(times)
#         events = torch.stack(events)
        
#         return slices_tensor, times, events