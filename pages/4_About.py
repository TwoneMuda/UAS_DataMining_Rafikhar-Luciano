"""Halaman About."""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_utils import models_exist, load_models  # noqa: E402

st.set_page_config(page_title="About", page_icon="\u2139\uFE0F", layout="wide")
st.title("\u2139\uFE0F Tentang Proyek")

st.markdown(
    """
    ### Penjelasan Metode
    **1. Regresi (Prediksi PM2.5)**
    Memprediksi nilai konsentrasi PM2.5 (variabel kontinu) berdasarkan faktor
    cuaca seperti suhu, titik embun, tekanan udara, kecepatan angin, hujan,
    dan salju. Beberapa algoritma dibandingkan (Linear Regression, Decision
    Tree, Random Forest, Gradient Boosting) dan model terbaik dipakai pada
    halaman Prediction.

    **2. Clustering (K-Means)**
    Mengelompokkan kondisi udara ke dalam beberapa segmen berdasarkan
    kemiripan karakteristik cuaca dan tingkat PM2.5. Jumlah cluster optimal
    ditentukan dengan Elbow Method & Silhouette Score.

    ### Dataset
    **UCI Beijing PM2.5 Data Set** \u2014 data kualitas udara per jam di Beijing
    (2010\u20132014). Sumber: UCI Machine Learning Repository.
    Atribut: PM2.5, DEWP, TEMP, PRES, cbwd, Iws, Is, Ir, dan waktu.

    ### Framework Proses
    **CRISP-DM**: Business Understanding \u2192 Data Understanding \u2192 Data
    Preparation \u2192 Modeling \u2192 Evaluation \u2192 Deployment.

    ### Identitas
    - **Nama:** RAFIKHAR LUCIANO
    - **Tugas:** Proyek Akhir (UAS) Data Mining
    """
)

if models_exist():
    _, _, _, meta = load_models()
    st.markdown("### Performa Model Regresi")
    import pandas as pd

    res = pd.DataFrame(meta["reg_results"]).T
    st.dataframe(res.style.format("{:.3f}"), use_container_width=True)
    st.success(f"Model terbaik: {meta['best_model']}")
