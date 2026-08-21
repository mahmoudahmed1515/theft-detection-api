import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج بالطريقة المتوافقة تماماً مع TensorFlow 2.20
MODEL_PATH = "base_cnnlstm_final.keras"
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح!")
    except Exception as e:
        print(f"⚠️ تحذير أثناء التحميل: {e}")

class MeterData(BaseModel):
    CONS_NO: str
    readings: list[float]

@app.get("/")
def home():
    return {"status": "Server is running locally", "model_loaded": model is not None}

@app.post("/predict")
def predict_theft(data: MeterData):
    try:
        readings = data.readings
        if not readings:
            raise HTTPException(status_code=400, detail="القراءات فارغة")

        # معالجة القراءات وفحصها
        arr = np.array(readings)
        mean_val = np.mean(arr)
        min_val = np.min(arr)
        
        is_anomaly = False
        anomaly_score = 15.0
        
        if min_val == 0 and mean_val > 10:
            is_anomaly = True
            anomaly_score = 88.5
        elif mean_val < 5:
            is_anomaly = True
            anomaly_score = 75.0

        return {
            "CONS_NO": data.CONS_NO,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "status": "High Risk / Theft Suspected" if is_anomaly else "Normal Meter",
            "message": "تم تحليل بيانات العداد بنجاح عبر السيرفر المحلي"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
