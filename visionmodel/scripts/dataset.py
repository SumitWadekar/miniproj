from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class StockCandleImageDataset(Dataset):
    def __init__(self, metadata_csv: Path, split: str, img_size: int = 224, augment: bool = True):
        self.meta = pd.read_csv(metadata_csv)
        self.meta = self.meta[self.meta["split"] == split].reset_index(drop=True)
        self.split = split
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(row["image"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor([
            row["y_open"], row["y_high"], row["y_low"], row["y_close"], row["y_volume"]
        ], dtype=torch.float32)
        # also return metadata needed for denormalization
        extras = {
            "window_min": float(row["window_min"]),
            "window_max": float(row["window_max"]),
            "vol_max": float(row["vol_max"]),
            "ticker": row["ticker"],
            "date": row["date"],
            "slot": int(row["slot"]),
            "image": row["image"],
        }
        return x, y, extras
