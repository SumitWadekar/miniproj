import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ====== Model Definition ======
class StockPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=3, num_layers=2):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ====== Load Model + Scaler ======
def load_model(path="stock_predictor.pth"):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model = StockPredictor()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = checkpoint["scaler"]
    return model, scaler


# ====== Multi-step Forecast ======
def forecast_fixed_range(model, scaler, data, start_date, end_date, lookback=5):
    """
    Forecasts between given start and end dates (6h intervals).
    """
    values = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaled = scaler.transform(values)

    seq = torch.tensor(scaled[-lookback:], dtype=torch.float32).unsqueeze(0)
    future_datetimes = pd.date_range(start=start_date, end=end_date, freq="6H")

    preds = []
    for _ in range(len(future_datetimes)):
        with torch.no_grad():
            pred_scaled = model(seq).numpy()

        dummy = np.zeros((1, 5))
        dummy[0, [0, 1, 3]] = pred_scaled[0]
        pred_original = scaler.inverse_transform(dummy)[0, [0, 1, 3]]
        preds.append(pred_original)

        new_row = np.zeros((1, 5))
        new_row[0, [0, 1, 3]] = pred_scaled[0]
        new_row[0, 2] = seq[0, -1, 2].item()  
        new_row[0, 4] = seq[0, -1, 4].item()  

        new_row = torch.tensor(new_row, dtype=torch.float32)
        seq = torch.cat([seq[:, 1:, :], new_row.unsqueeze(0)], dim=1)

    return np.array(preds), future_datetimes


# ====== Main ======
if __name__ == "__main__":
    # Load model
    model, scaler = load_model("stock_predictor.pth")
    test_df = pd.read_csv("test_dataset.csv")

    # Forecast between fixed range
    preds, future_datetimes = forecast_fixed_range(
        model, scaler,
        test_df,
        start_date="2025-08-05 00:00:00",
        end_date="2025-08-14 23:59:59",
        lookback=5
    )

    # Build output dataframe
    out_df = pd.DataFrame({
        "Date": [dt.strftime("%Y%m%d") for dt in future_datetimes],
        "Time": [dt.strftime("%H") for dt in future_datetimes],
        "Pred_Open": preds[:, 0],
        "Pred_High": preds[:, 1],
        "Pred_Close": preds[:, 2],
    })

    out_df.to_csv("predictions_5to14Aug.csv", index=False)
    print("Saved forecast to predictions_5to14Aug.csv")
