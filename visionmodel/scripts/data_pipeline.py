import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ------------------------------------------------------------ #
#                     CONFIGURATION                            #
# ------------------------------------------------------------ #

CRYPTO_TICKERS = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "SOL-USD",  # Solana
    "XRP-USD",  # Ripple
    "ADA-USD",  # Cardano
]


@dataclass
class PipelineConfig:
    output_dir: Path
    period_days: int = 60
    interval: str = "15m"      # can be 15m / 1h / 1d
    slot_minutes: int = 180    # condense 3 hours
    sleep_seconds: int = 2     # delay between downloads


# ------------------------------------------------------------ #
#                        DATA FETCHING                         #
# ------------------------------------------------------------ #

def get_clean_history(ticker: str, period_days: int, interval: str) -> pd.DataFrame:
    """Fetch clean OHLCV crypto data using modern yfinance (Ticker.history)."""
    print(f"⏳ Fetching {ticker} ({interval}, {period_days}d)...")
    t = yf.Ticker(ticker)
    df = pd.DataFrame()

    # Use start/end for intraday data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    try:
        df = t.history(
            start=start_date,
            end=end_date,
            interval=interval,
            prepost=False,
            auto_adjust=False,
            repair=True,
        )
    except Exception as e:
        print(f"⚠️  {ticker} failed: {e}")
        time.sleep(1)

    if df.empty:
        print(f"❌ No data for {ticker}")
        return df

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    valid_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[valid_cols].dropna()

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"✅ {ticker}: {len(df)} rows fetched.")
    return df


# ------------------------------------------------------------ #
#                   CONDENSE INTO TIME SLOTS                   #
# ------------------------------------------------------------ #

def condense_intraday(df_bars: pd.DataFrame, slot_minutes: int = 180) -> pd.DataFrame:
    """Condense 15m or 1h bars into time slots (e.g., 3-hour)."""
    if df_bars.empty:
        return df_bars

    df = df_bars.copy().sort_index()
    base_interval_minutes = int((df.index[1] - df.index[0]).total_seconds() / 60.0)
    bars_per_slot = max(1, slot_minutes // base_interval_minutes)
    out_rows = []

    for day, g in df.groupby(df.index.date):
        g = g.reset_index(drop=True)
        g["_slot"] = (g.index // bars_per_slot).astype(int)
        for sidx, chunk in g.groupby("_slot"):
            if len(chunk) < bars_per_slot:
                continue
            o = float(chunk["Open"].iloc[0])
            h = float(chunk["High"].max())
            l = float(chunk["Low"].min())
            c = float(chunk["Close"].iloc[-1])
            v = float(chunk["Volume"].sum())
            out_rows.append({
                "Date": str(day),
                "Slot": int(sidx),
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "Volume": v,
            })
    return pd.DataFrame(out_rows)


# ------------------------------------------------------------ #
#                        SAVE TO CSV                           #
# ------------------------------------------------------------ #

def save_csv(ticker: str, df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{ticker.replace('-', '_')}_condensed.csv"
    df.to_csv(path, index=False)
    print(f"💾 Saved: {path} ({len(df)} slots)")
    return path


# ------------------------------------------------------------ #
#                        PIPELINE RUNNER                       #
# ------------------------------------------------------------ #

def run_pipeline(cfg: PipelineConfig, tickers: List[str]):
    print(f"📊 Starting crypto data pipeline for {len(tickers)} assets...\n")
    saved_paths = []

    for ticker in tqdm(tickers, desc="Processing cryptos"):
        df = get_clean_history(ticker, cfg.period_days, cfg.interval)
        if df.empty:
            print(f"⚠️ Skipping {ticker}: empty data.\n")
            continue

        condensed = condense_intraday(df, cfg.slot_minutes)
        if condensed.empty:
            print(f"⚠️ Skipping {ticker}: no complete slots.\n")
            continue

        out_path = save_csv(ticker, condensed, cfg.output_dir)
        saved_paths.append(str(out_path))
        time.sleep(cfg.sleep_seconds)

    print("\n✅ Crypto data pipeline complete.")
    print(f"Saved CSVs: {saved_paths}")
    return saved_paths


# ------------------------------------------------------------ #
#                            MAIN                              #
# ------------------------------------------------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crypto OHLCV Data Pipeline (using yfinance)")
    ap.add_argument("--interval", type=str, default="15m", help="Interval: 15m, 1h, or 1d")
    ap.add_argument("--period_days", type=int, default=60, help="How many days of data to fetch (max 60 for 15m)")
    ap.add_argument("--slot_minutes", type=int, default=180, help="Condensation slot length (minutes)")
    ap.add_argument("--tickers", type=str, nargs="*", default=CRYPTO_TICKERS, help="Crypto tickers to fetch")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg = PipelineConfig(
        output_dir=root / "data",
        period_days=args.period_days,
        interval=args.interval,
        slot_minutes=args.slot_minutes,
    )

    run_pipeline(cfg, args.tickers)
