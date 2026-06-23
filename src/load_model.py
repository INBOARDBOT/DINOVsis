"""
load_model.py  —  Load a DINOv3 ViT backbone by name.

The backbone variant is resolved through DINOV3_WEIGHTS, a dict that maps
a short human-readable key (matching what you put in config.yaml) to:
  - the .pth filename inside REPO_DIR/dinov3_ViT_model_weights/
  - the torch.hub model string used to instantiate the architecture

Usage
-----
# From another script (backbone key comes from config.yaml):
    from load_model import load_segmentor
    backbone = load_segmentor("vits16")

# Directly (quick sanity-check):
    python load_model.py --backbone vitl16
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

REPO_DIR          = Path(__file__).parent.parent
DINOV3_REPO       = REPO_DIR / "dinov3-main"
MODEL_WEIGHT_DIR  = REPO_DIR / "dinov3_ViT_model_weights"

sys.path.insert(0, str(DINOV3_REPO))

# ---------------------------------------------------------------------------
# Backbone registry
# key          : short name used in config.yaml  → model.backbone
# "hub_model"  : string passed to torch.hub.load as `model=`
# "weight_file": filename inside MODEL_WEIGHT_DIR
# ---------------------------------------------------------------------------
DINOV3_REGISTRY: dict[str, dict[str, str]] = {
    "vitb16": {
        "hub_model":   "dinov3_vitb16",
        "weight_file": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    },
    "vith16plus": {
        "hub_model":   "dinov3_vith16plus",
        "weight_file": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    },
    "vitl16": {
        "hub_model":   "dinov3_vitl16",
        "weight_file": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    },
    "vits16": {
        "hub_model":   "dinov3_vits16",
        "weight_file": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    },
    "vits16plus": {
        "hub_model":   "dinov3_vits16plus",
        "weight_file": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_segmentor(backbone: str = "vits16") -> torch.nn.Module:
    """
    Instantiate and return a DINOv3 backbone loaded onto the best available device.

    Parameters
    ----------
    backbone : str
        One of the keys in DINOV3_REGISTRY, e.g. "vits16", "vitb16", "vitl16".
        This value is read from config.yaml → model.backbone by train.py.

    Raises
    ------
    KeyError  – if `backbone` is not in DINOV3_REGISTRY.
    FileNotFoundError – if the weight file is missing from MODEL_WEIGHT_DIR.
    """
    if backbone not in DINOV3_REGISTRY:
        valid = ", ".join(DINOV3_REGISTRY)
        raise KeyError(
            f"Unknown backbone '{backbone}'. Valid options are: {valid}"
        )

    entry       = DINOV3_REGISTRY[backbone]
    hub_model   = entry["hub_model"]
    weight_file = MODEL_WEIGHT_DIR / entry["weight_file"]

    if not weight_file.exists():
        raise FileNotFoundError(
            f"Weight file not found: {weight_file}\n"
            f"Expected it inside: {MODEL_WEIGHT_DIR}"
        )

    print(f"[INFO] Backbone       : {backbone}  ({hub_model})")
    print(f"[INFO] Weight file    : {weight_file.name}")

    torch.cuda.empty_cache()

    # Instantiate architecture (no pretrained weights from hub)
    segmentor = torch.hub.load(
        str(DINOV3_REPO),
        model=hub_model,
        source="local",
        pretrained=False,
    )

    # Load checkpoint — supports teacher/model/raw state-dict layouts
    print(f"[INFO] Loading weights from: {weight_file}")
    checkpoint = torch.load(weight_file, map_location="cpu")
    state_dict = checkpoint.get(
        "teacher", checkpoint.get("model", checkpoint)
    )
    segmentor.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    segmentor = segmentor.to(device)
    print(f"[INFO] Backbone loaded onto: {device}")

    return segmentor


# ---------------------------------------------------------------------------
# CLI sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and inspect a DINOv3 backbone.")
    parser.add_argument(
        "--backbone",
        default="vits16",
        choices=list(DINOV3_REGISTRY),
        help="Backbone variant to load (default: vits16).",
    )
    args = parser.parse_args()

    segmentor = load_segmentor(args.backbone)

    print("\n--- GPU Memory Usage ---")
    for i in range(torch.cuda.device_count()):
        mb = round(torch.cuda.memory_allocated(i) / 1e6, 2)
        print(f"  GPU {i}: {mb} MB")

    print("\nModel state-dict keys (first 10):")
    keys = list(segmentor.state_dict().keys())
    # for k in keys[:10]:
    for k in keys:
        print(f"  {k}")
    # if len(keys) > 10:
    #     print(f"  … ({len(keys)} total)")