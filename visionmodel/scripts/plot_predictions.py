import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms, models

# --- Load model ---
def build_model(steps=10):
    base = models.resnet50(weights=None)
    in_f = base.fc.in_features
    base.fc = torch.nn.Linear(in_f, steps)
    return base

def predict(image_path, ckpt, steps=10, img_size=224, device="cpu"):
    tfm = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    model = build_model(steps)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        preds = model(x).cpu().numpy().flatten()
    return preds

# --- Plot true vs predicted ---
def plot_comparison(image_path, true_labels, preds, window=100):
    # Fake time axis (0..99 candles)
    x = np.arange(window)
    # Approximate reconstruction of "close" prices
    close = np.cumsum(np.random.randn(window)) + 100  # placeholder if raw data not stored

    seg_len = window // len(true_labels)
    true_points, pred_points = [], []
    for i in range(len(true_labels)):
        idx = (i+1)*seg_len - 1
        base = close[i*seg_len]
        true_points.append(base * (1 + true_labels[i]))
        pred_points.append(base * (1 + preds[i]))

    plt.figure(figsize=(8,5))
    plt.plot(x, close, label="Mountain Chart (Close Price)", color="black")
    plt.scatter(np.arange(seg_len-1, window, seg_len), true_points, c="green", label="True Points", marker="o")
    plt.scatter(np.arange(seg_len-1, window, seg_len), pred_points, c="red", label="Predicted Points", marker="x")
    plt.legend()
    plt.title("True vs Predicted 10-Step Relative Trends")
    plt.show()

if __name__ == "__main__":
    # Example usage
    image = "dataset_images/test/win_00276.png"
    labels_json = "dataset_images/test_labels.json"

    with open(labels_json, "r") as f:
        meta = json.load(f)
    true_labels = [m for m in meta if Path(m["image"]).name == Path(image).name][0]["labels"]

    preds = predict(image, "artifacts/best_model.pth", steps=10)
    plot_comparison(image, true_labels, preds)
