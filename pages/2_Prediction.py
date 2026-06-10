"""Halaman Prediction / Analysis."""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_utils import (  # noqa: E402
    models_exist,
    load_models,
    predict_pm25,
    assign_cluster,
    pm25_category,
)

st.set_page_config(page_title="Prediksi PM2.5", page_icon="\U0001F52E", layout="wide")
st.title("\U0001F52E Prediksi PM2.5 & Kondisi Udara")

if not models_exist():
    st.error("Model belum tersedia. Jalankan dulu: python train_model.py")
    st.stop()

model, kmeans, scaler, meta = load_models()
ranges = meta["feature_ranges"]
med = meta["feature_medians"]

st.markdown("Masukkan kondisi cuaca, lalu klik **Prediksi**.")

with st.form("form_prediksi"):
    c1, c2, c3 = st.columns(3)
    with c1:
        month = st.slider("Bulan", 1, 12, 6)
        hour = st.slider("Jam", 0, 23, 12)
        cbwd = st.selectbox("Arah angin (cbwd)", meta["cbwd_categories"])
    with c2:
        temp = st.number_input("Suhu / TEMP (\u00b0C)", value=float(med["TEMP"]))
        dewp = st.number_input("Titik embun / DEWP (\u00b0C)", value=float(med["DEWP"]))
        pres = st.number_input("Tekanan / PRES (hPa)", value=float(med["PRES"]))
    with c3:
        iws = st.number_input("Kecepatan angin / Iws", value=float(med["Iws"]), min_value=0.0)
        is_snow = st.number_input("Jam salju / Is", value=float(med["Is"]), min_value=0.0)
        ir_rain = st.number_input("Jam hujan / Ir", value=float(med["Ir"]), min_value=0.0)
    submitted = st.form_submit_button("\U0001F680 Prediksi", use_container_width=True)

if submitted:
    input_dict = {
        "month": month, "hour": hour, "DEWP": dewp, "TEMP": temp,
        "PRES": pres, "Iws": iws, "Is": is_snow, "Ir": ir_rain, "cbwd": cbwd,
    }
    pm = predict_pm25(model, input_dict)
    pm = max(pm, 0.0)
    cid, cluster_label = assign_cluster(kmeans, scaler, meta, input_dict, pm)
    category, color = pm25_category(pm)

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Prediksi PM2.5", f"{pm:.1f} \u00b5g/m\u00b3")
    b.metric("Kategori Udara", category)
    c.metric("Cluster Kondisi", cluster_label)
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{color};"
        f"color:white;font-weight:bold;text-align:center;'>"
        f"Kualitas udara diperkirakan: {category} ({pm:.1f} \u00b5g/m\u00b3)"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model regresi terbaik: {meta['best_model']}")
