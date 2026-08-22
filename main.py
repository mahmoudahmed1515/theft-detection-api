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

# تحديد المسار المطلق للمجلد الحالي وضبط مسار ملف الموديل بدقة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "base_cnnlstm_final.keras")
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح!")
    except Exception as e:
        print(f"⚠️ تحذير أثناء التحميل: {e}")
else:
    print(f"❌ ملف الموديل غير موجود في المسار: {MODEL_PATH}")

class MeterData(BaseModel):
    CONS_NO: str
    readings: list[float]

@app.get("/")
def home():
    return {
        "status": "Server is running successfully", 
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
def predict_theft(data: MeterData):
    try:
        readings = data.readings
        if not readings:
            raise HTTPException(status_code=400, detail="القراءات فارغة")

        if model is None:
            raise HTTPException(status_code=500, detail="نموذج الذكاء الاصطناعي لم يتم تحميله في الذاكرة")

        # تجهيز البيانات للموديل (120 يوماً)
        input_data = np.array(readings, dtype=float)
        
        if input_data.ndim == 1:
            input_data = np.expand_dims(input_data, axis=0)
            if len(model.input_shape) == 3:
                input_data = np.expand_dims(input_data, axis=-1)

        # التنبؤ الحقيقي
        prediction = model.predict(input_data)
        
        score = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
        is_anomaly = score >= 0.5
        anomaly_score = round(score * 100, 2)

        return {
            "CONS_NO": data.CONS_NO,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "status": "High Risk / Theft Suspected" if is_anomaly else "Normal Meter",
            "message": "تم تحليل بيانات العداد بنجاح عبر نموذج CNN-LSTM الحقيقي"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
