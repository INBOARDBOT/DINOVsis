import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NPCSurvivalDataset
from head_model import CoxRiskNet
from load_model import load_segmentor

# ---------------------------------------------------------------------------
REPO_DIR       = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_DIR / "config" / "config.yaml"
OUTPUT_ROOT    = REPO_DIR / "output"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    print(f"[INFO] Loaded config: {path}")
    return cfg


def _set_nested(cfg: dict, dotkey: str, _unused: str = "") -> None:
    key_path, _, raw = dotkey.partition("=")
    keys = key_path.strip().split(".")
    for cast in (int, float):
        try:
            value = cast(raw); break
        except ValueError:
            pass
    else:
        value = (raw.lower() == "true") if raw.lower() in ("true", "false") else raw
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    print(f"[INFO] Override: {key_path} = {value!r}")


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides or []:
        _set_nested(cfg, item)
    return cfg


def resolve_paths(cfg: dict) -> dict:
    paths = cfg.get("paths", {})
    for key, rel in paths.items():
        paths[key] = str(REPO_DIR / rel)
    return cfg


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

def make_run_dir(backbone: str) -> Path:
    """Create and return a unique timestamped run directory."""
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"train_{backbone}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(cfg: dict, run_dir: Path, src_config: Path) -> None:
    """Copy the resolved config into the run folder for full reproducibility."""
    dst = run_dir / "config_used.yaml"
    shutil.copy2(src_config, dst)
    # Also dump the post-override state (includes --set changes)
    with open(run_dir / "config_resolved.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


def open_csv_log(run_dir: Path):
    """Open train_log.csv and return (file-handle, csv-writer)."""
    log_path = run_dir / "train_log.csv"
    fh = open(log_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])
    return fh, writer


def write_summary(run_dir: Path, cfg: dict, history: list[dict],
                  best_epoch: int, best_val: float, total_time: float) -> None:
    """Write a human-readable summary.txt to the run folder."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    optim_cfg = cfg.get("optimizer", {})

    lines = [
        "=" * 60,
        "  CoxRiskNet — Training Summary",
        "=" * 60,
        f"  Backbone       : {model_cfg.get('backbone', '?')}",
        f"  Embed dim      : {model_cfg.get('embed_dim', '?')}",
        f"  Epochs         : {train_cfg.get('epochs', '?')}",
        f"  Batch size     : {train_cfg.get('batch_size', '?')}",
        f"  LR             : {optim_cfg.get('lr', '?')}",
        f"  Weight decay   : {optim_cfg.get('weight_decay', '?')}",
        f"  Scheduler      : {optim_cfg.get('scheduler', {}).get('name', 'none')}",
        "-" * 60,
        f"  Best val loss  : {best_val:.6f}  (epoch {best_epoch})",
        f"  Final train loss: {history[-1]['train_loss']:.6f}",
        f"  Final val loss  : {history[-1]['val_loss']:.6f}",
        f"  Total time     : {total_time / 60:.1f} min",
        "=" * 60,
        "",
        "  Epoch log:",
        f"  {'Epoch':>6}  {'Train loss':>12}  {'Val loss':>10}  {'LR':>10}",
        "  " + "-" * 44,
    ]
    for row in history:
        lines.append(
            f"  {row['epoch']:>6}  {row['train_loss']:>12.6f}"
            f"  {row['val_loss']:>10.6f}  {row['lr']:>10.2e}"
        )
    lines += ["=" * 60, ""]

    summary_path = run_dir / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[INFO] Summary written → {summary_path}")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DinoUNetLight.")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"YAML config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE", default=[],
                        help="Dot-notation overrides, e.g. --set training.epochs=20")
    return parser.parse_args()



def cox_loss(risk: torch.Tensor,
             time:  torch.Tensor,
             event: torch.Tensor) -> torch.Tensor:
    """
    Cox partial-likelihood loss (Breslow approximation).
    Mirrors the implementation in cox_model.py.

    Args:
        risk:  (N,) raw risk scores (higher = higher hazard)
        time:  (N,) follow-up / death time in months
        event: (N,) 1 = died, 0 = censored
    Returns:
        Scalar loss (negative partial log-likelihood)
    """
    risk  = risk.reshape(-1)
    time  = time.reshape(-1)
    event = event.reshape(-1).float()

    # Sort descending by time so cumsum builds the risk set
    order = torch.argsort(time, descending=True)
    risk  = risk[order]
    event = event[order]

    log_risk_set = torch.logcumsumexp(risk, dim=0)
    loss = -((risk - log_risk_set) * event).sum() / event.sum().clamp_min(1.0)
    return loss


def main() -> None:
    args = parse_args()

    config_path = args.config if args.config is not None else DEFAULT_CONFIG
    cfg = load_config(config_path)
    if args.set:
        apply_overrides(cfg, args.set)
    resolve_paths(cfg)

    paths_cfg = cfg["paths"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    optim_cfg = cfg["optimizer"]

 
    backbone_key = model_cfg.get("backbone", "vits16")
    run_dir      = make_run_dir(backbone_key)
    print(f"[INFO] Run directory: {run_dir}")

    save_config_snapshot(cfg, run_dir, config_path)
    csv_fh, csv_writer = open_csv_log(run_dir)

  
    device = torch.device(
        train_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")

  
    print("[INFO] Initialising datasets …")
    train_dataset = NPCSurvivalDataset(
        csv_path = paths_cfg.get("train_csv", str(REPO_DIR / "survival_label" / "survival_groundtruth.csv")),
        img_size = data_cfg.get("img_size", 896),
        axis     = data_cfg.get("slice_axis", 2),
        keep_empty=data_cfg.get("keep_empty", False)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg.get("batch_size", 2),
        shuffle     = True,
        num_workers = train_cfg.get("num_workers", 2),
        drop_last   = True,
        pin_memory  = train_cfg.get("pin_memory", True),
        collate_fn  = NPCSurvivalDataset.survival_collate_fn,
    )


    val_dataset = NPCSurvivalDataset(
        csv_path = paths_cfg.get("val_csv", str(REPO_DIR / "survival_label" / "survival_groundtruth.csv")),
        img_size = data_cfg.get("img_size", 896),
        axis     = data_cfg.get("slice_axis", 2),
        keep_empty=data_cfg.get("keep_empty", False)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = train_cfg.get("batch_size", 32),
        shuffle     = False,
        num_workers = train_cfg.get("num_workers", 2),
        drop_last   = False,
        pin_memory  = train_cfg.get("pin_memory", True),
        collate_fn  = NPCSurvivalDataset.survival_collate_fn,
    )

    print(f"[INFO] Train patients: {len(train_dataset)} | Val patients: {len(val_dataset)}")

    embed_dim   = model_cfg.get("embed_dim", 384)

    print(f"[INFO] Loading backbone: {backbone_key} (embed_dim={embed_dim})")
    backbone = load_segmentor(backbone_key)

    print("[INFO] Building CoxRiskNet")
    model = CoxRiskNet(segmentor=backbone, embed_dim=embed_dim)
    model = model.to(device)

    criterion = cox_loss

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = float(optim_cfg.get("lr", 1e-4)),
        weight_decay = float(optim_cfg.get("weight_decay", 1e-4)),
    )

    scheduler  = None
    sched_cfg  = optim_cfg.get("scheduler", {})
    sched_name = sched_cfg.get("name")
    epochs     = train_cfg.get("epochs", 10)

    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(sched_cfg.get("eta_min", 1e-6))
        )
    elif sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = sched_cfg.get("step_size", 5),
            gamma     = sched_cfg.get("gamma", 0.5),
        )
    if scheduler:
        print(f"[INFO] Scheduler: {sched_name}")

    save_every = train_cfg.get("save_every_n_epochs", 0)

  
    print(f"[INFO] Training for {epochs} epochs …\n")
    history       = []
    best_val_loss = float("inf")
    best_epoch    = 0
    t0            = time.time()

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        epoch_start = time.time()

        model.train()
        epoch_loss = 0.0
        batch_bar  = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}",
                          unit="batch", leave=False)
        for slices_list, time_months, event in batch_bar:
            time_months = time_months.to(device)
            event       = event.to(device)
            optimizer.zero_grad()
 
            # Each patient's slice stack has a different length, so run the
            # model once per patient and stack the resulting per-patient
            # risk scores into a single (P,) tensor for this batch.
            risks = torch.cat([
                model(patient_slices.to(device))
                for patient_slices in slices_list
            ])  # (P,) — one risk score per patient in the batch

            loss = criterion(risks, time_months, event)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for slices_list, time_months, event in val_loader:
                time_months = time_months.to(device)
                event       = event.to(device)
                risks = torch.cat([
                    model(patient_slices.to(device))
                    for patient_slices in slices_list
                ])
                val_loss += criterion(risks, time_months, event).item()
        avg_val = val_loss / max(len(val_loader), 1)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - epoch_start

        epoch_bar.set_postfix(
            train=f"{avg_train:.4f}", val=f"{avg_val:.4f}"
        )

        row = dict(epoch=epoch + 1, train_loss=avg_train,
                   val_loss=avg_val, lr=current_lr, elapsed_s=round(elapsed, 1))
        history.append(row)
        csv_writer.writerow([row["epoch"], f"{avg_train:.6f}",
                             f"{avg_val:.6f}", f"{current_lr:.2e}", row["elapsed_s"]])
        csv_fh.flush()

        # ---- checkpoints ----
        if save_every and (epoch + 1) % save_every == 0:
            ckpt = run_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"\n[INFO] Checkpoint → {ckpt.name}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch    = epoch + 1
            torch.save(model.state_dict(), run_dir / "best_model.pth")

        if scheduler:
            scheduler.step()


    csv_fh.close()
    total_time = time.time() - t0

    torch.save(model.state_dict(), run_dir / "final_model.pth")

    write_summary(run_dir, cfg, history, best_epoch, best_val_loss, total_time)

    print(f"\n[INFO] Run folder  : {run_dir}")
    print(f"[INFO] Best model  : epoch {best_epoch}, val loss {best_val_loss:.6f}")
    print(f"[INFO] Total time  : {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()