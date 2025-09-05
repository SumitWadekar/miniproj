import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def fetch_data(ticker="BTC-USD", days=120, interval="1h", out_file="data/raw/train_data.csv"):
    """
    Downloads OHLCV data from Yahoo Finance and saves as CSV.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    df = yf.download(
        tickers=ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        group_by="ticker",
        auto_adjust=False
    )

    # Drop extra column levels if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Save as CSV
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file)

    print(f"✅ Saved data to {out_file}, shape={df.shape}")
    return df

if __name__ == "__main__":
    fetch_data(ticker="BTC-USD", days=120, interval="1h", out_file="data/raw/btc_data.csv")
