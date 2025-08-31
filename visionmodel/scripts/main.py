import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import mplfinance as mpf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Candlestick generation utils
# ----------------------------
def make_out_dirs(root: Path):
    for split in ["train", "val", "test"]:
        for cls in ["uptrend", "downtrend", "sideways"]:
            (root / split / cls).mkdir(parents=True, exist_ok=True)


def label_trend(close_series: pd.Series, pct: float) -> str:
    start = float(close_series.iloc[0])
    end = float(close_series.iloc[-1])
    up = start * (1 + pct)
    down = start * (1 - pct)
    if end >= up:
        return "uptrend"
    elif end <= down:
        return "downtrend"
    else:
        return "sideways"


def window_indices(n: int, window: int, stride: int) -> List[Tuple[int, int]]:
    """Return start,end (inclusive) indices for sliding windows."""
    idxs = []
    i = 0
    while i + window <= n:
        idxs.append((i, i + window))
        i += stride
    return idxs


def df_to_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the dataframe has columns exactly: Open, High, Low, Close, Volume, indexed by Datetime
    out = df.copy()
    out["Datetime"] = pd.to_datetime(out["Datetime"])
    out = out.set_index("Datetime")
    out = out[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return out


def save_candle_image(ohlc_window: pd.DataFrame, path: Path, add_volume=True, style="yahoo"):
    # mpf saves the file directly
    mpf.plot(
        ohlc_window,
        type="candle",
        volume=add_volume,
        style=style,
        savefig=dict(fname=str(path), dpi=120, bbox_inches="tight", pad_inches=0.05),
    )


def build_image_dataset_from_csv(
    csv_path: Path,
    out_root: Path,
    window: int = 20,
    stride: int = 5,
    label_pct: float = 0.01,
    val_size: float = 0.15,
    test_size: float = 0.15,
    style: str = "yahoo"
):
    """
    Reads the OHLCV CSV, creates sliding windows -> candlestick images, labels them, and
    splits into train/val/test folders on disk in ImageFolder structure.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    make_out_dirs(out_root)

    df = pd.read_csv(csv_path)
    # Optional: if multiple "Unit" symbols exist, you can groupby Unit. For now treat as one series by time.
    # If you want per-Unit windows, uncomment the groupby logic.

    df_ohlc = df_to_ohlc(df).sort_index()
    n = len(df_ohlc)
    idxs = window_indices(n, window, stride)

    # Collect samples
    samples = []
    tmp_images_dir = out_root / "_tmp_all"
    (tmp_images_dir).mkdir(parents=True, exist_ok=True)

    for k, (s, e) in enumerate(tqdm(idxs, desc="Generating charts")):
        win = df_ohlc.iloc[s:e]
        if len(win) < window:
            continue
        lab = label_trend(win["Close"], pct=label_pct)
        img_name = f"win_{k:05d}_{lab}.png"
        img_path = tmp_images_dir / img_name
        save_candle_image(win, img_path, add_volume=True, style=style)
        samples.append((str(img_path), lab))

    if not samples:
        raise RuntimeError("No samples created. Check window/stride settings vs dataset length.")

    # Split train/val/test
    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]

    # stratified split: first train+temp, then temp->val/test
    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=(val_size + test_size),
                                                        random_state=42, stratify=labels)
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=rel_test,
                                                    random_state=42, stratify=y_temp)

    # Move files into split/class folders
    def move_files(xs, ys, split):
        for p, lab in tqdm(list(zip(xs, ys)), desc=f"Organizing {split}", total=len(xs)):
            dest = out_root / split / lab / Path(p).name
            os.replace(p, dest)

    move_files(X_train, y_train, "train")
    move_files(X_val,   y_val,   "val")
    move_files(X_test,  y_test,  "test")

    # Save meta
    meta = {
        "csv_path": str(csv_path),
        "total_windows": len(samples),
        "splits": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "window": window,
        "stride": stride,
        "label_pct": label_pct,
        "style": style
    }
    with open(out_root / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Clean tmp dir if empty
    try:
        tmp_images_dir.rmdir()
    except OSError:
        # If some files remain (e.g. if you abort mid-run), ignore
        pass

    print("Dataset prepared at:", str(out_root))


# ----------------------------
# Training / Evaluation
# ----------------------------
def get_dataloaders(root: Path, img_size: int = 224, batch_size: int = 32, num_workers: int = 2):
    # Augmentations: avoid horizontal flip (time axis), keep small rotations/zoom
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomResizedCrop(size=img_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_ds = datasets.ImageFolder(root=str(root / "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(root=str(root / "val"),   transform=eval_tfms)
    test_ds  = datasets.ImageFolder(root=str(root / "test"),  transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_to_idx = train_ds.class_to_idx
    return train_loader, val_loader, test_loader, class_to_idx


def build_model(num_classes: int = 3):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model


def train(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 15,
    lr: float = 1e-4,
    out_dir: Path = Path("artifacts")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_acc = 0.0
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                v_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total if v_total else 0.0
        val_acc = v_correct / v_total if v_total else 0.0

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved new best to {best_path} (val_acc={best_val_acc:.4f})")

    return best_path


def evaluate(model, test_loader, device, class_names: List[str]):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.cpu().numpy().tolist())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Trader's Eye Vision Model – end to end")
    ap.add_argument("--csv", type=str, required=True, help="Path to OHLCV CSV (Datetime,Open,High,Low,Close,Volume)")
    ap.add_argument("--dataset_out", type=str, default="dataset_images", help="Folder to write ImageFolder dataset")
    ap.add_argument("--window", type=int, default=20, help="Sliding window length (candles)")
    ap.add_argument("--stride", type=int, default=5, help="Stride between windows")
    ap.add_argument("--pct", type=float, default=0.01, help="Label threshold (e.g., 0.01=±1%)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    out_root = Path(args.dataset_out).resolve()

    # 1) build dataset (images + splits)
    build_image_dataset_from_csv(
        csv_path=csv_path,
        out_root=out_root,
        window=args.window,
        stride=args.stride,
        label_pct=args.pct,
        val_size=0.15,
        test_size=0.15,
        style="yahoo"
    )

    # 2) dataloaders
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        out_root, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers
    )
    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k

    # save class map
    artifacts = Path("artifacts"); artifacts.mkdir(exist_ok=True, parents=True)
    with open(artifacts / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    # 3) train
    device = torch.device(args.device)
    model = build_model(num_classes=len(class_names)).to(device)
    best_ckpt = train(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, out_dir=artifacts)

    # 4) load best & evaluate
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    evaluate(model, test_loader, device, class_names)

    print("\nDone. Artifacts saved in ./artifacts and dataset in", str(out_root))


if __name__ == "__main__":
    main()
