import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np

# ---------------- Dataset ---------------- #
class MountainDataset(Dataset):
    def __init__(self, label_json, img_size=224):
        with open(label_json, "r") as f:
            self.meta = json.load(f)
        self.transform = transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        item = self.meta[idx]
        img = Image.open(item["image"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(item["labels"], dtype=torch.float32)  # shape [10]
        return x, y

# ---------------- Model ---------------- #
def build_model(steps=10):
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = base.fc.in_features
    base.fc = nn.Linear(in_f, steps)  # regression head
    return base

# ---------------- Train ---------------- #
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, out_dir=Path("artifacts")):
    out_dir.mkdir(parents=True, exist_ok=True)
    criterion = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    best_path = out_dir / "best_model.pth"

    for epoch in range(1, epochs+1):
        model.train(); train_loss=0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train"):
            x,y = x.to(device), y.to(device)
            optim.zero_grad()
            preds = model(x)  # shape [B,10]
            loss = criterion(preds, y)
            loss.backward(); optim.step()
            train_loss += loss.item()*x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval(); val_loss=0
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} val"):
                x,y = x.to(device), y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()*x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val=val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved new best to {best_path}")

    return best_path

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset_images")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_ds = MountainDataset(Path(args.dataset)/"train_labels.json", args.img_size)
    val_ds   = MountainDataset(Path(args.dataset)/"val_labels.json", args.img_size)
    test_ds  = MountainDataset(Path(args.dataset)/"test_labels.json", args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device(args.device)
    model = build_model(steps=10).to(device)

    best_ckpt = train_model(model, train_loader, val_loader, device, epochs=args.epochs)

    # Evaluate on test
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x,y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            p = model(x).cpu().numpy()
            preds.append(p); trues.append(y.numpy())
    preds = np.vstack(preds); trues = np.vstack(trues)
    mse = np.mean((preds-trues)**2)
    print(f"✅ Test MSE: {mse:.6f}")

if __name__ == "__main__":
    main()
