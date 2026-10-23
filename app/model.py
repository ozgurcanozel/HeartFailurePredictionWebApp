"""Model tanımı ve ön işleme (eğitimdeki ile aynı olmalı)."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "heart_failure_model.pth"
DATA_PATH = BASE_DIR / "heart.csv"

FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
            "Oldpeak", "ST_Slope"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


class HeartClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(11, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class Predictor:
    def __init__(self):
        df = pd.read_csv(DATA_PATH)
        # Label encoding: alfabetik sıra (sklearn LabelEncoder / pd.factorize(sort=True))
        self.encoders = {c: {v: i for i, v in enumerate(sorted(df[c].unique()))}
                         for c in CATEGORICAL}
        enc = self._encode(df)
        # Tüm 11 sütun standardize edilir (CSV üzerinde ~%86 doğrulukla doğrulandı)
        self.mean = enc.mean()
        self.std = enc.std(ddof=0)
        self.df = df

        self.model = HeartClassifier()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[FEATURES].copy()
        for c in CATEGORICAL:
            out[c] = out[c].map(self.encoders[c])
        return out

    def _tensor(self, df: pd.DataFrame) -> torch.Tensor:
        enc = self._encode(df)
        enc = (enc.astype("float64") - self.mean) / self.std
        return torch.tensor(enc.values.astype("float32"))

    def predict(self, record: dict) -> dict:
        x = self._tensor(pd.DataFrame([record]))
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=1)[0]
        return {"prediction": int(probs.argmax()),
                "probability": float(probs[1])}

    def accuracy_on_csv(self) -> float:
        x = self._tensor(self.df)
        y = torch.tensor(self.df["HeartDisease"].values)
        with torch.inference_mode():
            return (self.model(x).argmax(1) == y).float().mean().item()
