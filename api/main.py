
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
import os

# ============================================================
# Initialisation
# ============================================================

app = FastAPI(
    title="Energy Forecasting API",
    description="API de prédiction de consommation énergétique PJM Est",
    version="1.0.0"
)

# Chemin flexible — fonctionne en local ET sur Docker/Render
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

model        = joblib.load(DATA_DIR / "model_xgboost.pkl")
MARGE_IC     = 697
MAPE_ATTENDU = 0.74

# ============================================================
# Schémas de données
# ============================================================

class PredictionInput(BaseModel):
    datetime          : str
    lag_1             : float
    lag_24            : float
    lag_48            : float
    lag_168           : float
    lag_8736          : float
    rolling_mean_24h  : float
    rolling_mean_168h : float
    rolling_mean_720h : float
    rolling_std_24h   : float
    rolling_std_168h  : float
    temp              : float
    humidity          : float
    windspeed         : float
    cloudcover        : float
    is_outlier        : Optional[int] = 0

class PredictionOutput(BaseModel):
    datetime              : str
    prediction_mw         : float
    confidence_interval   : dict
    mape_expected_percent : float

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "status"     : "online",
        "model"      : "XGBoost",
        "mape_test"  : f"{MAPE_ATTENDU}%",
        "description": "API de prédiction de consommation énergétique PJM Est"
    }

@app.get("/health")
def health():
    try:
        test = np.zeros((1, 34))
        model.predict(test)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    try:
        dt = pd.to_datetime(data.datetime)

        features = {
            "is_outlier"       : data.is_outlier,
            "hour"             : dt.hour,
            "dayofweek"        : dt.dayofweek,
            "month"            : dt.month,
            "quarter"          : dt.quarter,
            "year"             : dt.year,
            "dayofyear"        : dt.dayofyear,
            "weekofyear"       : dt.isocalendar()[1],
            "season"           : (dt.month % 12) // 3,
            "is_holiday"       : 0,
            "is_weekend"       : int(dt.dayofweek >= 5),
            "lag_1"            : data.lag_1,
            "lag_24"           : data.lag_24,
            "lag_48"           : data.lag_48,
            "lag_168"          : data.lag_168,
            "lag_8736"         : data.lag_8736,
            "rolling_mean_24h" : data.rolling_mean_24h,
            "rolling_mean_168h": data.rolling_mean_168h,
            "rolling_mean_720h": data.rolling_mean_720h,
            "rolling_std_24h"  : data.rolling_std_24h,
            "rolling_std_168h" : data.rolling_std_168h,
            "temp"             : data.temp,
            "humidity"         : data.humidity,
            "windspeed"        : data.windspeed,
            "cloudcover"       : data.cloudcover,
            "hour_sin"         : np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos"         : np.cos(2 * np.pi * dt.hour / 24),
            "dow_sin"          : np.sin(2 * np.pi * dt.dayofweek / 7),
            "dow_cos"          : np.cos(2 * np.pi * dt.dayofweek / 7),
            "month_sin"        : np.sin(2 * np.pi * dt.month / 12),
            "month_cos"        : np.cos(2 * np.pi * dt.month / 12),
            "temp_x_hour"      : data.temp * dt.hour,
            "temp_x_season"    : data.temp * ((dt.month % 12) // 3),
            "hour_x_weekend"   : dt.hour * int(dt.dayofweek >= 5),
        }

        X    = pd.DataFrame([features])
        pred = float(model.predict(X)[0])

        return PredictionOutput(
            datetime              = data.datetime,
            prediction_mw         = round(pred, 1),
            confidence_interval   = {
                "lower": round(pred - MARGE_IC, 1),
                "upper": round(pred + MARGE_IC, 1)
            },
            mape_expected_percent = MAPE_ATTENDU
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
