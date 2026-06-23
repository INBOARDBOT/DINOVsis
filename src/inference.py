"""
infer.py — Evaluate a trained CoxRiskNet checkpoint on a survival dataset.

Loads the same DINOv3 backbone + CoxRiskNet head used in train.py, restores
weights from a checkpoint (best_model.pth / final_model.pth), runs inference
over every patient in a CSV (no gradient, no augmentation), and reports:

  - per-patient risk scores (saved to CSV)
  - Harrell's C-index
  - Kaplan-Meier curves for high-risk vs low-risk groups (split at median risk)
  - log-rank test p-value

Usage
-----
    python src/infer.py \
        --checkpoint output/train_vits16plus_20260622_185822/best_model.pth \
        --config     output/train_vits16plus_20260622_185822/config_used.yaml \
        --csv        data/splits/valid_t1_sanitized.csv \
        --out-dir    output/eval_valid

If --csv is omitted, the val_csv path from the config is used.
If --config is omitted, the default config/config.yaml is used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NPCSurvivalDataset
from head_model import CoxRiskNet
from load_model import load_segmentor

REPO_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_DIR / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CoxRiskNet.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a .pth state_dict (best_model.pth or final_model.pth)")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"YAML config used at train time (default: {DEFAULT_CONFIG})")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV of patients to evaluate (default: paths.val_csv from config)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write risk_scores.csv / km_curves.png (default: alongside checkpoint)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="DataLoader batch size for inference (default: 2)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (default: 0, safest for inference)")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_paths(cfg: dict) -> dict:
    paths = cfg.get("paths", {})
    for key, rel in paths.items():
        if not str(rel).startswith("/"):
            paths[key] = str(REPO_DIR / rel)
    return cfg


# ---------------------------------------------------------------------------
# Harrell's C-index (vectorised via lifelines; faster than the O(N^2) loop
# in cox_survival_model.py and gives identical results)
# ---------------------------------------------------------------------------

def compute_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    # lifelines expects higher predicted_scores == lower risk by default for
    # some conventions; concordance_index treats predicted_scores as a risk
    # proxy where higher predicted_scores should mean shorter survival time,
    # which matches our risk score convention (higher risk -> shorter time).
    return concordance_index(time, -risk, event)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_risk, all_time, all_event = [], [], []

    for slices_list, time_months, event in tqdm(loader, desc="Inference", unit="batch"):
        risks = torch.cat([
            model(patient_slices.to(device))
            for patient_slices in slices_list
        ])  # (P,)
        all_risk.append(risks.cpu().numpy())
        all_time.append(time_months.numpy())
        all_event.append(event.numpy())

    return (
        np.concatenate(all_risk),
        np.concatenate(all_time),
        np.concatenate(all_event),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_risk_csv(out_dir: Path, df: pd.DataFrame, risk: np.ndarray,
                  time: np.ndarray, event: np.ndarray) -> Path:
    out = df.copy()
    out["risk_score"] = risk
    out["time_months"] = time
    out["event"] = event
    out_path = out_dir / "risk_scores.csv"
    out.to_csv(out_path, index=False)
    return out_path


def plot_km_curves(out_dir: Path, risk: np.ndarray, time: np.ndarray,
                    event: np.ndarray, ci: float) -> Path:
    median_risk = np.median(risk)
    high_risk = risk >= median_risk
    low_risk = ~high_risk

    t_high, e_high = time[high_risk], event[high_risk]
    t_low, e_low = time[low_risk], event[low_risk]

    lr = logrank_test(t_high, t_low, e_high, e_low)

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()

    kmf.fit(t_high, e_high, label=f"High risk (n={high_risk.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#e74c3c")

    kmf.fit(t_low, e_low, label=f"Low risk (n={low_risk.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2ecc71")

    ax.set_title(f"Kaplan-Meier Curves\nLog-rank p = {lr.p_value:.4f}", fontsize=13)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.05, f"C-index: {ci:.3f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / "km_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[INFO] Log-rank test p-value: {lr.p_value:.4f}")
    return out_path


def print_summary(risk: np.ndarray, time: np.ndarray, event: np.ndarray, ci: float) -> None:
    median_risk = np.median(risk)
    high = risk >= median_risk
    low = ~high
    print("\n" + "=" * 50)
    print("  Inference summary")
    print("=" * 50)
    print(f"  N patients      : {len(risk)}")
    print(f"  Events          : {int(event.sum())} / {len(event)} "
          f"({event.mean()*100:.1f}%)")
    print(f"  C-index         : {ci:.4f}  (0.5 = random, 1.0 = perfect)")
    print(f"  Risk min/median/max : "
          f"{risk.min():.3f} / {median_risk:.3f} / {risk.max():.3f}")
    print(f"  High-risk group : {int(event[high].sum())} / {high.sum()} events "
          f"({event[high].mean()*100:.1f}%)")
    print(f"  Low-risk group  : {int(event[low].sum())} / {low.sum()} events "
          f"({event[low].mean()*100:.1f}%)")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config_path = args.config if args.config is not None else DEFAULT_CONFIG
    cfg = load_config(config_path)
    resolve_paths(cfg)

    paths_cfg = cfg["paths"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    csv_path = args.csv if args.csv is not None else Path(
        paths_cfg.get("test_csv", str(REPO_DIR / "data" / "splits" / "test_t1_sanitized.csv"))
    )
    out_dir = args.out_dir if args.out_dir is not None else args.checkpoint.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device       : {device}")
    print(f"[INFO] Checkpoint   : {args.checkpoint}")
    print(f"[INFO] Eval CSV     : {csv_path}")
    print(f"[INFO] Output dir   : {out_dir}")

    # ---- dataset / loader -------------------------------------------------
    dataset = NPCSurvivalDataset(
        csv_path=str(csv_path),
        img_size=data_cfg.get("img_size", 896),
        axis=data_cfg.get("slice_axis", 2),
        keep_empty=data_cfg.get("keep_empty", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=False,
        collate_fn=NPCSurvivalDataset.survival_collate_fn,
    )
    print(f"[INFO] Patients     : {len(dataset)}")

    # ---- model --------------------------------------------------------
    backbone_key = model_cfg.get("backbone", "vits16")
    embed_dim = model_cfg.get("embed_dim", 384)

    print(f"[INFO] Loading backbone: {backbone_key} (embed_dim={embed_dim})")
    backbone = load_segmentor(backbone_key)

    model = CoxRiskNet(segmentor=backbone, embed_dim=embed_dim).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print("[INFO] Checkpoint loaded.")

    # ---- inference ------------------------------------------------------
    risk, time, event = run_inference(model, loader, device)

    # ---- metrics + report -------------------------------------------------
    ci = compute_c_index(risk, time, event)
    print_summary(risk, time, event, ci)

    csv_out = save_risk_csv(out_dir, dataset.df, risk, time, event)
    print(f"[INFO] Risk scores  → {csv_out}")

    km_out = plot_km_curves(out_dir, risk, time, event, ci)
    print(f"[INFO] KM plot      → {km_out}")


if __name__ == "__main__":
    main()