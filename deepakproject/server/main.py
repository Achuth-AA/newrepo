"""
IoT Sensor Anomaly Detection API
FastAPI backend for real-time anomaly detection and monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import json
from datetime import datetime
import os

# Initialize FastAPI app
app = FastAPI(
    title="IoT Sensor Anomaly Detection API",
    description="Real-time anomaly detection for IoT sensor data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models = {}
scaler = None
feature_names = []

# Pydantic models
class SensorReading(BaseModel):
    timestamp: str
    moteid: int
    temperature: float
    humidity: float
    light: float
    voltage: float

class SensorReadings(BaseModel):
    readings: List[SensorReading]

class PredictionResponse(BaseModel):
    timestamp: str
    moteid: int
    is_anomaly: bool
    anomaly_score: float
    model: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    total_features: int

class StatisticsResponse(BaseModel):
    total_readings: int
    anomaly_count: int
    anomaly_percentage: float
    sensor_health: Dict[int, str]

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    global models, scaler, feature_names

    models_path = "../models/saved_models"

    try:
        # Load scaler
        scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
        print("✓ Scaler loaded")

        # Load Isolation Forest
        models['isolation_forest'] = joblib.load(os.path.join(models_path, "isolation_forest.pkl"))
        print("✓ Isolation Forest loaded")

        # Load Random Forest
        models['random_forest'] = joblib.load(os.path.join(models_path, "random_forest.pkl"))
        print("✓ Random Forest loaded")

        # Load XGBoost
        import xgboost as xgb
        models['xgboost'] = xgb.XGBClassifier()
        models['xgboost'].load_model(os.path.join(models_path, "xgboost_model.json"))
        print("✓ XGBoost loaded")

        # Load Autoencoder
        models['autoencoder'] = keras.models.load_model(os.path.join(models_path, "autoencoder.h5"))
        models['ae_threshold'] = np.load(os.path.join(models_path, "ae_threshold.npy"))
        print("✓ Autoencoder loaded")

        # Load LSTM
        models['lstm'] = keras.models.load_model(os.path.join(models_path, "lstm_final.h5"))
        print("✓ LSTM loaded")

        # Load feature names
        feature_names = pd.read_csv(os.path.join("../data/processed", "feature_names.csv"), header=None)[0].tolist()
        print(f"✓ {len(feature_names)} feature names loaded")

        print(f"\n🚀 All models loaded successfully!")

    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("⚠️  API will run with limited functionality")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for prediction
    This should match the feature engineering from notebook 02
    """
    # Sort by moteid and timestamp
    df = df.sort_values(['moteid', 'timestamp'])

    # Temporal features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    df['time_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

    # Rolling statistics
    sensor_features = ['temperature', 'humidity', 'light', 'voltage']
    windows = [10, 30, 60]

    for feature in sensor_features:
        for window in windows:
            df[f'{feature}_rolling_mean_{window}'] = df.groupby('moteid')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{feature}_rolling_std_{window}'] = df.groupby('moteid')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'{feature}_rolling_min_{window}'] = df.groupby('moteid')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'{feature}_rolling_max_{window}'] = df.groupby('moteid')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )

    # Rate of change
    for feature in sensor_features:
        df[f'{feature}_diff_1'] = df.groupby('moteid')[feature].diff(1)
        df[f'{feature}_diff_2'] = df.groupby('moteid')[feature].diff(2)
        df[f'{feature}_pct_change'] = df.groupby('moteid')[feature].pct_change()
        df[f'{feature}_deviation_from_mean'] = df[feature] - df[f'{feature}_rolling_mean_30']

    # Lag features
    lag_periods = [1, 2, 5, 10]
    for feature in sensor_features:
        for lag in lag_periods:
            df[f'{feature}_lag_{lag}'] = df.groupby('moteid')[feature].shift(lag)

    # Statistical features
    for feature in sensor_features:
        df[f'{feature}_zscore'] = df.groupby('moteid')[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        df[f'{feature}_ema'] = df.groupby('moteid')[feature].transform(
            lambda x: x.ewm(span=30, adjust=False).mean()
        )

    # Inter-sensor features
    for feature in sensor_features:
        global_stats = df.groupby('timestamp')[feature].agg(['mean', 'std', 'min', 'max'])
        global_stats.columns = [f'{feature}_global_mean', f'{feature}_global_std',
                               f'{feature}_global_min', f'{feature}_global_max']
        df = df.merge(global_stats, on='timestamp', how='left')
        df[f'{feature}_deviation_from_global'] = df[feature] - df[f'{feature}_global_mean']
        df[f'{feature}_global_zscore'] = df[f'{feature}_deviation_from_global'] / (df[f'{feature}_global_std'] + 1e-6)

    # Interaction features
    df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
    df['light_temp_interaction'] = df['light'] * df['temperature']
    df['voltage_drop_rate'] = df.groupby('moteid')['voltage'].transform(
        lambda x: x.diff().rolling(window=50, min_periods=1).mean()
    )

    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    return df

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "total_features": len(feature_names)
    }

@app.post("/predict/isolation_forest", response_model=List[PredictionResponse])
async def predict_isolation_forest(data: SensorReadings):
    """Predict anomalies using Isolation Forest"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([reading.dict() for reading in data.readings])

        # Engineer features
        df = engineer_features(df)

        # Select features
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'epoch']]
        X = df[feature_cols].values

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        predictions = models['isolation_forest'].predict(X_scaled)
        scores = models['isolation_forest'].score_samples(X_scaled)

        # Prepare response
        results = []
        for idx, reading in enumerate(data.readings):
            results.append({
                "timestamp": reading.timestamp,
                "moteid": reading.moteid,
                "is_anomaly": bool(predictions[idx] == -1),
                "anomaly_score": float(scores[idx]),
                "model": "isolation_forest",
                "confidence": float(abs(scores[idx]))
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/random_forest", response_model=List[PredictionResponse])
async def predict_random_forest(data: SensorReadings):
    """Predict anomalies using Random Forest"""
    try:
        df = pd.DataFrame([reading.dict() for reading in data.readings])
        df = engineer_features(df)

        feature_cols = [col for col in df.columns if col not in ['timestamp', 'epoch']]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        predictions = models['random_forest'].predict(X_scaled)
        probabilities = models['random_forest'].predict_proba(X_scaled)[:, 1]

        results = []
        for idx, reading in enumerate(data.readings):
            results.append({
                "timestamp": reading.timestamp,
                "moteid": reading.moteid,
                "is_anomaly": bool(predictions[idx] == 1),
                "anomaly_score": float(probabilities[idx]),
                "model": "random_forest",
                "confidence": float(probabilities[idx] if predictions[idx] == 1 else 1 - probabilities[idx])
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/xgboost", response_model=List[PredictionResponse])
async def predict_xgboost(data: SensorReadings):
    """Predict anomalies using XGBoost"""
    try:
        df = pd.DataFrame([reading.dict() for reading in data.readings])
        df = engineer_features(df)

        feature_cols = [col for col in df.columns if col not in ['timestamp', 'epoch']]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        predictions = models['xgboost'].predict(X_scaled)
        probabilities = models['xgboost'].predict_proba(X_scaled)[:, 1]

        results = []
        for idx, reading in enumerate(data.readings):
            results.append({
                "timestamp": reading.timestamp,
                "moteid": reading.moteid,
                "is_anomaly": bool(predictions[idx] == 1),
                "anomaly_score": float(probabilities[idx]),
                "model": "xgboost",
                "confidence": float(probabilities[idx] if predictions[idx] == 1 else 1 - probabilities[idx])
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/autoencoder", response_model=List[PredictionResponse])
async def predict_autoencoder(data: SensorReadings):
    """Predict anomalies using Autoencoder"""
    try:
        df = pd.DataFrame([reading.dict() for reading in data.readings])
        df = engineer_features(df)

        feature_cols = [col for col in df.columns if col not in ['timestamp', 'epoch']]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        reconstructions = models['autoencoder'].predict(X_scaled, batch_size=len(X_scaled))
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

        threshold = models['ae_threshold']
        predictions = (mse > threshold).astype(int)

        results = []
        for idx, reading in enumerate(data.readings):
            results.append({
                "timestamp": reading.timestamp,
                "moteid": reading.moteid,
                "is_anomaly": bool(predictions[idx] == 1),
                "anomaly_score": float(mse[idx]),
                "model": "autoencoder",
                "confidence": float(abs(mse[idx] - threshold) / threshold)
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/ensemble", response_model=List[PredictionResponse])
async def predict_ensemble(data: SensorReadings):
    """Ensemble prediction using multiple models"""
    try:
        df = pd.DataFrame([reading.dict() for reading in data.readings])
        df = engineer_features(df)

        feature_cols = [col for col in df.columns if col not in ['timestamp', 'epoch']]
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)

        # Get predictions from all models
        iso_pred = (models['isolation_forest'].predict(X_scaled) == -1).astype(int)
        rf_pred = models['random_forest'].predict(X_scaled)
        xgb_pred = models['xgboost'].predict(X_scaled)

        # Autoencoder
        reconstructions = models['autoencoder'].predict(X_scaled, batch_size=len(X_scaled))
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
        ae_pred = (mse > models['ae_threshold']).astype(int)

        # Ensemble voting (majority)
        ensemble_scores = iso_pred + rf_pred + xgb_pred + ae_pred
        ensemble_pred = (ensemble_scores >= 2).astype(int)

        # Calculate ensemble confidence
        confidence = ensemble_scores / 4.0

        results = []
        for idx, reading in enumerate(data.readings):
            results.append({
                "timestamp": reading.timestamp,
                "moteid": reading.moteid,
                "is_anomaly": bool(ensemble_pred[idx] == 1),
                "anomaly_score": float(ensemble_scores[idx]),
                "model": "ensemble",
                "confidence": float(confidence[idx] if ensemble_pred[idx] == 1 else 1 - confidence[idx])
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """Get system statistics"""
    try:
        # Load predictions data if available
        try:
            df = pd.read_csv("../data/processed/unsupervised_predictions.csv")

            total_readings = len(df)
            anomaly_count = df['ensemble_anomaly'].sum() if 'ensemble_anomaly' in df.columns else 0
            anomaly_percentage = (anomaly_count / total_readings * 100) if total_readings > 0 else 0

            # Sensor health (simplified)
            sensor_health = {}
            for sensor_id in df['moteid'].unique():
                sensor_data = df[df['moteid'] == sensor_id]
                if 'ensemble_anomaly' in sensor_data.columns:
                    anomaly_rate = sensor_data['ensemble_anomaly'].mean()
                    health = "good" if anomaly_rate < 0.05 else "warning" if anomaly_rate < 0.15 else "critical"
                    sensor_health[int(sensor_id)] = health

            return {
                "total_readings": int(total_readings),
                "anomaly_count": int(anomaly_count),
                "anomaly_percentage": float(anomaly_percentage),
                "sensor_health": sensor_health
            }
        except:
            return {
                "total_readings": 0,
                "anomaly_count": 0,
                "anomaly_percentage": 0.0,
                "sensor_health": {}
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
