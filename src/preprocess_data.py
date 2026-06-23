import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import numpy as np

# Import helpers from your existing dataset.py
from dataset import load_nii, normalize_volume, volume_to_slices, slice_to_rgb_pil, make_image_transform

def preprocess_csv(csv_path: str, output_dir: Path, img_size: int = 896, max_slices: int = 30, axis: int = 2, keep_empty: bool = True):
    """
    Reads a NIfTI survival CSV, extracts/transforms slices, saves them as a single .pt file 
    per patient, and creates a new metadata CSV.
    """
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_transform = make_image_transform(img_size)
    new_rows = []
    
    print(f"\n[INFO] Preprocessing data from {csv_path}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_pth'].strip()
        label_path = row['survival_label_path'].strip()
        
        # Determine a unique file name using case_number or index
        case_id = row.get('case_number', f"case_{idx}")
        if isinstance(case_id, float) and np.isnan(case_id): # fallback if NaN
            case_id = f"case_{idx}"
            
        # 1. Load and process NIfTI volume (same logic as your old dataset)
        volume, _ = load_nii(img_path)
        volume = normalize_volume(volume)
        slices = volume_to_slices(volume, axis)
        
        if not keep_empty:
            keep_idx = [i for i, s in enumerate(slices) if s.mean() > 1e-3]
            if keep_idx:
                slices = slices[keep_idx]

        if max_slices is not None and len(slices) > max_slices:
            sel = np.linspace(0, len(slices) - 1, max_slices).round().astype(int)
            slices = slices[sel]
            
        # 2. Convert to Tensors and stack
        slice_tensors = []
        for s in slices:
            img_pil = slice_to_rgb_pil(s)
            img_tensor = img_transform(img_pil)  # Shape: (3, H, W)
            slice_tensors.append(img_tensor)
            
        slices_tensor = torch.stack(slice_tensors, dim=0)  # Shape: (Num_Slices, 3, H, W)
        
        # 3. Save tensor directly to disk
        pt_filename = f"{case_id}_slices.pt"
        pt_save_path = output_dir / pt_filename
        torch.save(slices_tensor, pt_save_path)
        
        # 4. Record new rows for our lightweight metadata file
        new_rows.append({
            'preprocessed_tensor_pth': str(pt_save_path),
            'survival_label_path': label_path,
            'case_number': case_id
        })
        
    # Write the new dataset CSV pointing directly to our pre-cooked tensors
    new_df = pd.DataFrame(new_rows)
    out_csv_path = csv_path.replace(".csv", "_preprocessed.csv")
    new_df.to_csv(out_csv_path, index=False)
    print(f"[SUCCESS] Preprocessing completed! New metadata saved to: {out_csv_path}")

if __name__ == "__main__":
    # Define paths based on your configurations
    TRAIN_CSV = "data/splits/train_t1_sanitized.csv"
    VALID_CSV = "data/splits/valid_t1_sanitized.csv"
    
    # Store processed tensors locally on your machine/instance disk
    OUTPUT_TENSOR_DIR = Path("data/preprocessed_tensors")
    
    # Run for both splits
    preprocess_csv(TRAIN_CSV, OUTPUT_TENSOR_DIR / "train", img_size=896, max_slices=30, axis=2, keep_empty=False)
    preprocess_csv(VALID_CSV, OUTPUT_TENSOR_DIR / "valid", img_size=896, max_slices=30, axis=2, keep_empty=False)