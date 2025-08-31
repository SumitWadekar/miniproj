import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms, models
from PIL import Image
import mplfinance as mpf


def df_to_ohlc(df):
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime")
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def save_last_window_chart(df_ohlc, window, out_png):
    win = df_ohlc.iloc[-window:]
    mpf.plot(win, type="candle", volume=True, style="yahoo",
             savefig=dict(fname=str(out_png), dpi=120, bbox_inches="tight", pad_inches=0.05))


def load_model(ckpt_path: Path, class_map_path: Path, device):
    with open(class_map_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, len(class_to_idx))
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, idx_to_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", default="artifacts/best_model.pth")
    ap.add_argument("--class_map", default="artifacts/class_to_idx.json")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    df = pd.read_csv(args.csv)
    ohlc = df_to_ohlc(df)
    tmp_png = Path("artifacts/last_window.png")
    tmp_png.parent.mkdir(parents=True, exist_ok=True)
    save_last_window_chart(ohlc, args.window, tmp_png)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    img = Image.open(tmp_png).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    model, idx_to_class = load_model(Path(args.ckpt), Path(args.class_map), device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())
        pred_cls = idx_to_class[pred_idx]

    print(f"Prediction: {pred_cls}")
    for i, p in enumerate(probs):
        print(f"  {idx_to_class[i]}: {p:.4f}")


if __name__ == "__main__":
    main()
