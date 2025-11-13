import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import StockCandleImageDataset
from model import VisionStockRegressor, ModelConfig


def denormalize(batch_pred: torch.Tensor, extras_list):
    # batch_pred: [B,5] normalized
    outs = []
    for i in range(len(extras_list)):
        ex = extras_list[i]
        pmin = ex["window_min"]
        pmax = ex["window_max"]
        vmax = ex["vol_max"] if ex["vol_max"] > 0 else 1.0
        preds = batch_pred[i].cpu().numpy()
        o, h, l, c, v = preds
        o = o * (pmax - pmin) + pmin
        h = h * (pmax - pmin) + pmin
        l = l * (pmax - pmin) + pmin
        c = c * (pmax - pmin) + pmin
        v = v * vmax
        outs.append([o, h, l, c, v])
    return np.array(outs)


def evaluate(model, loader, device):
    model.eval()
    all_preds_norm = []
    all_trues_norm = []
    all_extras = []
    with torch.no_grad():
        for batch in loader:
            imgs, targets, extras = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            preds = model(imgs)
            all_preds_norm.append(preds)
            all_trues_norm.append(targets)
            # extras is dict of lists, convert to list of dicts
            batch_size = imgs.shape[0]
            for i in range(batch_size):
                ex = {k: v[i] for k, v in extras.items()}
                all_extras.append(ex)
    preds_norm = torch.cat(all_preds_norm, dim=0)
    trues_norm = torch.cat(all_trues_norm, dim=0)

    # Denormalize
    preds_denorm = denormalize(preds_norm, all_extras)
    trues_denorm = denormalize(trues_norm, all_extras)

    # RMSE per column
    mse = np.mean((preds_denorm - trues_denorm) ** 2, axis=0)
    rmse = np.sqrt(mse)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    
    # Special handling for Volume: use log scale
    vol_pred_clipped = np.maximum(0, preds_denorm[:, 4])
    vol_true_clipped = np.maximum(0, trues_denorm[:, 4])
    vol_pred = np.log1p(vol_pred_clipped)  # log(volume + 1)
    vol_true = np.log1p(vol_true_clipped)
    rmse_vol_log = np.sqrt(np.mean((vol_pred - vol_true) ** 2))
    
    for i, (c, r) in enumerate(zip(cols, rmse)):
        if c == "Volume":
            print(f"RMSE {c}: {r:.4f} (linear), {rmse_vol_log:.4f} (log scale)")
        else:
            print(f"RMSE {c}: {r:.4f}")
    
    # Average RMSE excluding Volume
    rmse_avg = np.sqrt(np.mean((preds_denorm[:, :4] - trues_denorm[:, :4]) ** 2))
    print(f"Average RMSE (prices only): {rmse_avg:.4f}")
    # Average with log volume
    price_mse = np.mean((preds_denorm[:, :4] - trues_denorm[:, :4]) ** 2)
    vol_mse_log = np.mean((vol_pred - vol_true) ** 2)
    rmse_combined = np.sqrt((4 * price_mse + vol_mse_log) / 5)
    print(f"Average RMSE (with log Volume): {rmse_combined:.4f}")

    return preds_denorm, trues_denorm, all_extras


def scatter_plots(preds, trues, out_dir: Path):
    cols = ["Open", "High", "Low", "Close", "Volume"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, col in enumerate(cols):
        plt.figure(figsize=(4,4))
        if col == "Volume":
            # Use log scale for volume
            p_clipped = np.maximum(0, preds[:, i])
            t_clipped = np.maximum(0, trues[:, i])
            p = np.log1p(p_clipped)
            t = np.log1p(t_clipped)
            plt.scatter(t, p, alpha=0.6, s=12)
            lims = [min(t.min(), p.min()), max(t.max(), p.max())]
            plt.plot(lims, lims, 'r--', linewidth=1)
            plt.title(f"Pred vs True {col} (log scale)")
            plt.xlabel("True (log)")
            plt.ylabel("Pred (log)")
        else:
            plt.scatter(trues[:, i], preds[:, i], alpha=0.6, s=12)
            lims = [min(trues[:, i].min(), preds[:, i].min()), max(trues[:, i].max(), preds[:, i].max())]
            plt.plot(lims, lims, 'r--', linewidth=1)
            plt.title(f"Pred vs True {col}")
            plt.xlabel("True")
            plt.ylabel("Pred")
        plt.tight_layout()
        fname = out_dir / f"scatter_{col}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"Saved {fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=str, default=None)
    ap.add_argument("--weights", type=str, default="models/best_model.pkl")
    ap.add_argument("--backbone", type=str, default="efficientnet_b0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.metadata is None:
        args.metadata = str(root / "data" / "samples_metadata.csv")
    ds_test = StockCandleImageDataset(Path(args.metadata), split="test", img_size=224, augment=False)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(args.device)
    cfg = ModelConfig(pretrained=False, backbone=args.backbone, out_dim=5)
    model = pickle.load(open(root / args.weights, 'rb'))
    model.to(device)

    preds, trues, extras = evaluate(model, test_loader, device)
    scatter_plots(preds, trues, root / "models" / "eval_plots")


if __name__ == "__main__":
    main()