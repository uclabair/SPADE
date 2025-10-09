
import os
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config as CFG
from models import CLIPModel
from utils import AvgMeter
from dataset import STDataset_PreLoad 


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CLIP-style aligner on precomputed features."
    )
    # Data / splits
    p.add_argument("--folds_path", type=str, default=None)
    p.add_argument("--coords_csv", type=str, default=None)
    p.add_argument(
        "--protos",
        type=str,
        default=None,
        help="Comma-separated prototype IDs to train on (e.g., '0,1,2'). "
             "If not set, defaults to range(16).",
    )
    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--device", type=str, default=None, help="'cuda' or 'cpu' (default from CFG)"
    )
    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument(
        "--exp_name",
        type=str,
        default="run",
        help="String to include in checkpoint filenames.",
    )
    # Scheduler
    p.add_argument(
        "--eta_min",
        type=float,
        default=0.0,
        help="CosineAnnealingLR minimum LR (eta_min).",
    )
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    meter = AvgMeter()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()
                 if k in ["image_features", "reduced_expression"]}
        loss = model(batch)
        meter.update(loss.item(), n=batch["image_features"].size(0))
    return meter.avg


def train_one_proto(proto_id, cfg, device):
    # Load folds/coords
    if cfg.folds_path is not None and os.path.exists(cfg.folds_path):
        _ = np.load(cfg.folds_path, allow_pickle=True)

    df = pd.read_csv(cfg.coords_data_path_v1 if cfg.coords_data_path_v1 else cfg.coords_csv)
    df = df[df["image_prototype"] == proto_id]
    df = df[(df["id"] != "NCBI855") & (df["id"] != "NCBI854")]

    n = len(df)
    n_val = max(1, int(0.1 * n))
    train_df = df.iloc[:-n_val].reset_index(drop=True) if n_val < n else df.copy()
    val_df = df.iloc[-n_val:].reset_index(drop=True) if n_val < n else df.copy()

    print(f"[proto {proto_id}] train={train_df.shape[0]}  val={val_df.shape[0]}")

    train_ds = STDataset_PreLoad(train_df, cfg)
    val_ds = STDataset_PreLoad(val_df, cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Model / Optim / Scheduler
    model = CLIPModel().to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * math.ceil(max(1, len(train_loader)))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=getattr(cfg, "eta_min", 0.0))

    best_val = float("inf")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    step = 0
    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        train_meter = AvgMeter()
        loop = tqdm(train_loader, desc=f"[proto {proto_id}] epoch {epoch}/{cfg.num_epochs}", leave=False)

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()
                     if k in ["image_features", "reduced_expression"]}

            loss = model(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            train_meter.update(loss.item(), n=batch["image_features"].size(0))
            loop.set_postfix(train_loss=f"{train_meter.avg:.4f}")

        # Validation
        val_loss = evaluate(model, val_loader, device)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(
                cfg.checkpoint_dir,
                f"{cfg.exp_name}_best_proto{proto_id}_ep{epoch:03d}_val{best_val:.6f}.pt",
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "proto": proto_id,
                    "val_loss": best_val,
                    "cfg": dict(
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        lr=cfg.lr,
                        weight_decay=cfg.weight_decay,
                        num_epochs=cfg.num_epochs,
                        eta_min=getattr(cfg, "eta_min", 0.0),
                    ),
                },
                best_path,
            )

        print(
            f"[proto {proto_id}] epoch {epoch:03d}  "
            f"train={train_meter.avg:.4f}  val={val_loss:.4f}  best={best_val:.4f}  "
            f"elapsed={time.time()-t0:.1f}s"
        )

    # Also save final checkpoint
    last_path = os.path.join(
        cfg.checkpoint_dir,
        f"{cfg.exp_name}_last_proto{proto_id}_ep{cfg.num_epochs:03d}.pt",
    )
    torch.save(
        {"model": model.state_dict(), "epoch": cfg.num_epochs, "proto": proto_id, "val_loss": best_val},
        last_path,
    )


def main():
    args = parse_args()

    # Create a lightweight view over CFG with possible overrides
    class CfgView:
        pass

    cfg = CfgView()
    # Inherit from config.py, allow overrides
    cfg.folds_path = args.folds_path or getattr(CFG, "folds_path", None)
    cfg.coords_data_path_v1 = args.coords_csv or getattr(CFG, "coords_data_path_v1", None)
    cfg.coords_csv = args.coords_csv or getattr(CFG, "coords_csv", None)

    cfg.batch_size = args.batch_size or getattr(CFG, "batch_size", 128)
    cfg.num_workers = args.num_workers or getattr(CFG, "num_workers", 4)
    cfg.num_epochs = args.epochs or getattr(CFG, "num_epochs", 100)
    cfg.lr = args.lr or getattr(CFG, "lr", 1e-4)
    cfg.weight_decay = args.weight_decay or getattr(CFG, "weight_decay", 1e-2)
    cfg.seed = args.seed or getattr(CFG, "seed", 42)
    cfg.device = args.device or getattr(CFG, "device", "cuda")

    cfg.checkpoint_dir = args.checkpoint_dir or getattr(CFG, "checkpoint_dir", "checkpoints")
    cfg.exp_name = args.exp_name
    cfg.eta_min = args.eta_min

    # Prototypes to run
    if args.protos:
        protos = [int(x) for x in args.protos.split(",") if x.strip() != ""]
    else:
        protos = list(range(16)) 


    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    for proto in protos:
        train_one_proto(proto, cfg, device)


if __name__ == "__main__":
    main()
