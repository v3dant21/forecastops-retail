"""LSTM forecasting with per-series causal windows and complete persistence."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forecastops.models.base import BaseModel
from forecastops.utils.helpers import set_seed


class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        sequence, _ = self.lstm(inputs)
        return self.output(self.dropout(sequence[:, -1, :])).squeeze(-1)


class LSTMForecaster(BaseModel):
    """Predict each row from its current covariates and earlier rows of its series.

    Supply product/store IDs to keep panel sequences separate. A request without
    history uses zero padding, so results can differ from a request with history.
    Sales themselves must never be supplied as model features.
    """

    def __init__(self, config: dict | None = None):
        super().__init__("lstm", config)
        self.params = {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 128,
            "epochs": 50,
            "seq_length": 28,
        }
        self.params.update((config or {}).get("lstm", {}))
        self.device = torch.device("cpu")

    def _windows(self, frame, values):
        length = self.params["seq_length"]
        windows = np.zeros((len(frame), length, values.shape[1]), dtype=np.float32)
        positions = pd.DataFrame({"position": np.arange(len(frame))})
        keys = [k for k in ("product_id", "store_id") if k in frame]
        for key in keys:
            positions[key] = frame[key].to_numpy()
        if "date" in frame:
            positions["date"] = pd.to_datetime(frame["date"]).to_numpy()
            positions = positions.sort_values("date", kind="stable")
        groups = positions.groupby(keys, sort=False, dropna=False) if keys else [(None, positions)]
        for _, group in groups:
            rows = group["position"].to_numpy()
            for offset, row in enumerate(rows):
                history = values[rows[max(0, offset - length + 1) : offset + 1]]
                windows[row, -len(history) :] = history
        return windows

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if len(X_train) == 0:
            raise ValueError("Training data cannot be empty")
        set_seed(42)
        self.series_columns = [c for c in ("product_id", "store_id") if c in X_train]
        self.feature_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
        values = X_train[self.feature_columns].fillna(0).to_numpy(dtype=np.float32)
        if not self.feature_columns:
            raise ValueError("At least one numeric feature is required")
        self.feature_means = values.mean(axis=0)
        self.feature_stds = values.std(axis=0) + 1e-8
        self.target_mean = float(y_train.mean())
        self.target_std = float(y_train.std(ddof=0)) + 1e-8
        windows = self._windows(X_train, (values - self.feature_means) / self.feature_stds)
        targets = (y_train.to_numpy(dtype=np.float32) - self.target_mean) / self.target_std
        dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(targets))
        loader = DataLoader(dataset, batch_size=self.params["batch_size"], shuffle=True)
        self.model = DemandLSTM(
            len(self.feature_columns),
            self.params["hidden_size"],
            self.params["num_layers"],
            self.params["dropout"],
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])
        self.model.train()
        for _ in range(self.params["epochs"]):
            for inputs, target in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(inputs), target)
                loss.backward()
                optimizer.step()
        self.trained = True
        return self

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model has not been trained")
        missing = set(self.series_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing series identifiers: {sorted(missing)}")
        if X.empty:
            return np.array([])
        values = X[self.feature_columns].fillna(0).to_numpy(dtype=np.float32)
        windows = self._windows(X, (values - self.feature_means) / self.feature_stds)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(windows), self.params["batch_size"]):
                outputs.append(
                    self.model(
                        torch.from_numpy(windows[start : start + self.params["batch_size"]])
                    ).numpy()
                )
        return np.maximum(np.concatenate(outputs) * self.target_std + self.target_mean, 0)

    def get_params(self) -> dict[str, Any]:
        return {"params": self.params, "feature_columns": self.feature_columns}

    def set_params(self, params):
        self.params.update(params.get("params", {}))
        return self
