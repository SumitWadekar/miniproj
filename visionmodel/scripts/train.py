import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StockCandleImageDataset
from model import VisionStockRegressor, ModelConfig


def rmse_from_mse(mse: float) -> float:
    return float(np.sqrt(mse))


def train_one_epoch(model, loader, device, scaler, optimizer, criterion):
    model.train()
    running = 0.0
    n = 0
    for x, y, _ in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(x)
            loss = criterion(preds, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


def eval_epoch(model, loader, device, criterion):
    model.eval()
    running = 0.0
    n = 0
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            running += loss.item() * x.size(0)
            n += x.size(0)
    mse = running / max(n, 1)
    return mse, rmse_from_mse(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, default="data/samples_metadata.csv")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--backbone", type=str, default="efficientnet_b0")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")

    batch_size = 32 if device.type == 'cuda' else 16

    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    ds_train = StockCandleImageDataset(root / args.metadata, split="train", img_size=args.img_size, augment=True)
    ds_val = StockCandleImageDataset(root / args.metadata, split="val", img_size=args.img_size, augment=False)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    device = torch.device(args.device)
    cfg = ModelConfig(pretrained=True, backbone=args.backbone, out_dim=5)
    model = VisionStockRegressor(cfg).to(device)

    # Freeze backbone for a few epochs
    def set_backbone_requires_grad(req: bool):
        for p in model.backbone.parameters():
            p.requires_grad = req

    set_backbone_requires_grad(False)
    # Ensure classifier requires grad
    for p in model.backbone.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_rmse = float('inf')
    best_path = models_dir / "best_model.pkl"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            # Unfreeze for fine-tuning
            set_backbone_requires_grad(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        print(f"\nEpoch {epoch}/{args.epochs}")
        train_mse = train_one_epoch(model, train_loader, device, scaler, optimizer, criterion)
        train_rmse = rmse_from_mse(train_mse)
        val_mse, val_rmse = eval_epoch(model, val_loader, device, criterion)
        scheduler.step(val_rmse)
        print(f"  train_rmse={train_rmse:.6f}  val_rmse={val_rmse:.6f}")

        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            pickle.dump(model, open(best_path, 'wb'))
            print(f"  ✓ Saved new best: {best_path}  (rmse={best_rmse:.6f})")

    print(f"Best val RMSE: {best_rmse:.6f}")

    # Save final model
    final_path = models_dir / "final_model.pkl"
    pickle.dump(model, open(final_path, 'wb'))
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
