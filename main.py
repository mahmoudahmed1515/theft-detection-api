from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل الملفات عند بدء التشغيل
scaler = joblib.load("stat_scaler.pkl")
model = load_model("base_cnnlstm_final.keras")

class MeterData(BaseModel):
    CONS_NO: str | None = None
    readings: list[float]

@app.post("/predict")
def predict_theft(data: MeterData):
    raw_readings = np.array(data.readings).reshape(1, -1)
    scaled_readings = scaler.transform(raw_readings)
    formatted_input = scaled_readings.reshape((scaled_readings.shape[0], scaled_readings.shape[1], 1))
    
    raw_prediction = float(model.predict(formatted_input, verbose=0)[0][0])
    is_anomaly = raw_prediction >= 0.5
    
    return {
        "CONS_NO": data.CONS_NO,
        "is_anomaly": is_anomaly,
        "anomaly_score": int(raw_prediction * 100),
        "confidence": int(raw_prediction * 100) if is_anomaly else int((1 - raw_prediction) * 100),
        "status": "⚠️ High Risk / Theft Suspected" if is_anomaly else "✅ Normal Meter"
    }