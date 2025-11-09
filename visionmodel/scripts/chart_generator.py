import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
import mplfinance as mpf
from tqdm import tqdm


def ensure_dirs(root: Path):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)


def load_condensed(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: Date, Slot, Open, High, Low, Close, Volume
    return df


def normalize_context(df_ctx: pd.DataFrame):
    price_min = float(df_ctx["Low"].min())
    price_max = float(df_ctx["High"].max())
    vol_max = float(df_ctx["Volume"].max()) if float(df_ctx["Volume"].max()) > 0 else 1.0
    eps = 1e-8
    def norm_price(x):
        return (x - price_min) / (price_max - price_min + eps)
    def norm_vol(x):
        return x / (vol_max + eps)
    out = df_ctx.copy()
    for col in ["Open", "High", "Low", "Close"]:
        out[col] = norm_price(out[col])
    out["Volume"] = norm_vol(out["Volume"])
    return out, price_min, price_max, vol_max


def plot_sample(df_ctx_norm: pd.DataFrame, out_path: Path, figsize=(2.24, 2.24), dpi=100):
    # Build a synthetic DateTimeIndex for consistent spacing
    n = len(df_ctx_norm)
    dt_index = pd.date_range(start=pd.Timestamp("2000-01-01"), periods=n, freq="H")
    df_plot = df_ctx_norm.copy()
    df_plot.index = dt_index
    mpf.plot(df_plot, type="candle", volume=True, style="charles", axisoff=True,
             figsize=figsize, savefig=dict(fname=str(out_path), dpi=dpi, bbox_inches="tight", pad_inches=0))


def generate_images_and_metadata(root: Path, condensed_paths: List[Path], context: int = 8, img_px: int = 224) -> Path:
    images_dir = root / "images"
    meta_rows: List[Dict] = []

    for csv_path in condensed_paths:
        df = load_condensed(csv_path)
        ticker = csv_path.stem.replace("_condensed", "")
        # Iterate each slot; create context window [i-context+1 .. i]
        for i in tqdm(range(len(df)), desc=f"Images {ticker}"):
            left = max(0, i - context + 1)
            ctx = df.iloc[left:i+1].reset_index(drop=True)
            if ctx.empty:
                continue
            ctx_norm, pmin, pmax, vmax = normalize_context(ctx)
            # output filename
            date = df.iloc[i]["Date"]
            slot = int(df.iloc[i]["Slot"])
            img_name = f"{ticker}_{date}_s{slot:02d}.png"
            out_img = images_dir / img_name
            plot_sample(ctx_norm, out_img, figsize=(img_px/100, img_px/100), dpi=100)

            # Target is the last row (normalized OHLCV)
            y = ctx_norm.iloc[-1]
            meta_rows.append({
                "image": str(out_img),
                "ticker": ticker,
                "date": date,
                "slot": slot,
                "y_open": float(y["Open"]),
                "y_high": float(y["High"]),
                "y_low": float(y["Low"]),
                "y_close": float(y["Close"]),
                "y_volume": float(y["Volume"]),
                "window_min": float(pmin),
                "window_max": float(pmax),
                "vol_max": float(vmax),
            })

    # Build metadata DataFrame
    meta = pd.DataFrame(meta_rows)
    # Split 80/10/10 per ticker
    splits = []
    for t, g in meta.groupby("ticker"):
        g = g.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(g)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        g.loc[:n_train-1, "split"] = "train"
        g.loc[n_train:n_train+n_val-1, "split"] = "val"
        g.loc[n_train+n_val:, "split"] = "test"
        splits.append(g)
    meta = pd.concat(splits, ignore_index=True)

    out_csv = root / "data" / "samples_metadata.csv"
    meta.to_csv(out_csv, index=False)
    print(f"Saved metadata: {out_csv} (rows={len(meta)})")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condensed", nargs="+", help="Paths to condensed CSVs (from data_pipeline)")
    ap.add_argument("--context", type=int, default=8)
    ap.add_argument("--img_px", type=int, default=224)
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    root = scripts_dir.parent  # visionmodel directory
    ensure_dirs(root)

    condensed_paths = [Path(p) for p in args.condensed]
    generate_images_and_metadata(root, condensed_paths, context=args.context, img_px=args.img_px)
