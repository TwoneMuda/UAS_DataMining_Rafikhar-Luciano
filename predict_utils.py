"""Fungsi bantu untuk aplikasi Streamlit: memuat data & model, prediksi."""
import os
import sys
import joblib
import pandas as pd

# Agar bisa import src/data_utils dari folder app/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import load_raw, clean_data, find_dataset  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")


def models_exist():
    needed = ["model.pkl", "kmeans.pkl", "scaler.pkl", "meta.pkl"]
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed)


def load_models():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    meta = joblib.load(os.path.join(MODEL_DIR, "meta.pkl"))
    return model, kmeans, scaler, meta


def load_dataset():
    """Memuat dataset yang sudah dibersihkan (atau None jika belum ada)."""
    if find_dataset() is None:
        return None
    return clean_data(load_raw())


def predict_pm25(model, input_dict):
    """Prediksi nilai PM2.5 dari satu baris input pengguna."""
    X = pd.DataFrame([input_dict])
    return float(model.predict(X)[0])


def assign_cluster(kmeans, scaler, meta, weather_dict, pm25_value):
    """Tentukan cluster kondisi udara untuk input + prediksi PM2.5."""
    row = {
        "TEMP": weather_dict["TEMP"],
        "DEWP": weather_dict["DEWP"],
        "PRES": weather_dict["PRES"],
        "Iws": weather_dict["Iws"],
        "pm2.5": pm25_value,
    }
    X = pd.DataFrame([row])[meta["cluster_features"]]
    scaled = scaler.transform(X)
    cid = int(kmeans.predict(scaled)[0])
    label = meta["cluster_labels"].get(cid, f"Cluster {cid}")
    return cid, label


def pm25_category(value):
    """Kategori kualitas udara berdasarkan ambang umum PM2.5 (ug/m3)."""
    if value <= 35:
        return "Baik", "#2ecc71"
    if value <= 75:
        return "Sedang", "#f1c40f"
    if value <= 115:
        return "Tidak Sehat (Sensitif)", "#e67e22"
    if value <= 150:
        return "Tidak Sehat", "#e74c3c"
    return "Sangat Tidak Sehat", "#8e44ad"
