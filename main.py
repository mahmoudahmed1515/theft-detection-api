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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "base_cnnlstm_final.h5")
model = None

# دالة مخصصة لتجاوز خصائص الـ Quantization المعترضة أثناء قراءة الطبقات
from tensorflow.keras.layers import Dense
class SafeDense(Dense):
    @classmethod
    from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

if os.path.exists(MODEL_PATH):
    try:
        # محاولة التحميل مع تمرير الـ SafeDense في الـ custom_objects
        model = load_model(MODEL_PATH, compile=False, custom_objects={'Dense': SafeDense})
        print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح تام!")
    except Exception as e:
        print(f"⚠️ فشل التحميل بالطريقة الأولى، جاري المحاولة العادية: {e}")
        try:
            model = load_model(MODEL_PATH, compile=False)
            print("✅ تم التحميل بالطريقة العادية بنجاح!")
        except Exception as ex:
            print(f"❌ خطأ نهائي في تحميل الموديل: {ex}")
else:
    print(f"❌ ملف الموديل غير موجود في المسار: {MODEL_PATH}")
