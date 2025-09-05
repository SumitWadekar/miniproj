import torch, json
from pathlib import Path
from torchvision import transforms, models
from PIL import Image
import argparse

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

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="artifacts/best_model.pth")
    args = ap.parse_args()

    preds = predict(args.image, args.ckpt)
    print("Predicted 10-step relative changes:")
    for i, p in enumerate(preds,1):
        print(f"Step {i}: {p:+.4f}")
