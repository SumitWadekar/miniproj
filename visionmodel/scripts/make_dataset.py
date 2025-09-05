import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os, json

def make_mountain_dataset(csv_file, out_path, window=100, stride=10, steps=10):
    """
    Generate mountain chart images + 10-step numeric relative trend labels.
    """
    df = pd.read_csv(csv_file)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")

    n = len(df)
    windows = [(i, i + window) for i in range(0, n - window, stride)]
    samples = []
    tmp_dir = out_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for k, (s, e) in enumerate(tqdm(windows, desc="Generating mountain charts")):
        win = df.iloc[s:e]
        if len(win) < window:
            continue

        # ✅ Compute 10-step relative changes
        seg_len = window // steps
        labels = []
        for j in range(steps):
            seg = win.iloc[j*seg_len:(j+1)*seg_len]
            if len(seg) < 2:
                labels.append(0.0)
                continue
            start, end = seg["Close"].iloc[0], seg["Close"].iloc[-1]
            rel_change = (end - start) / start  # % relative change
            labels.append(rel_change)

        # Save mountain chart (clean version)
        plt.figure(figsize=(4,3))
        plt.plot(win.index, win["Close"], color="black")
        plt.fill_between(win.index, win["Close"], win["Close"].min(), color="black", alpha=0.3)
        plt.axis("off")

        img_name = f"win_{k:05d}.png"
        img_path = tmp_dir / img_name
        plt.savefig(img_path, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

        samples.append((str(img_path), labels))

    # Split into train/val/test
    paths, labels = zip(*samples)
    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    def save_split(xs, ys, split):
        split_dir = out_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        meta = []
        for p, lbls in zip(xs, ys):
            fname = Path(p).name
            dest = split_dir / fname
            os.replace(p, dest)
            meta.append({"image": str(dest), "labels": lbls})
        with open(out_path / f"{split}_labels.json", "w") as f:
            json.dump(meta, f, indent=2)

    save_split(X_train, y_train, "train")
    save_split(X_val, y_val, "val")
    save_split(X_test, y_test, "test")

    print(f"✅ Dataset generated at {out_path}")
    print(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

if __name__ == "__main__":
    csv_file = "data/raw/btc_data.csv"
    make_mountain_dataset(Path(csv_file), Path("dataset_images"), window=100, stride=10, steps=10)
