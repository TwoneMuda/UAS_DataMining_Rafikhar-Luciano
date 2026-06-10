"""Halaman Dataset Overview."""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_utils import load_dataset  # noqa: E402

st.set_page_config(page_title="Dataset Overview", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA Dataset Overview")

df = load_dataset()
if df is None:
    st.error(
        "Dataset belum ditemukan. Letakkan file "
        "'PRSA_data_2010.1.1-2014.12.31.csv' di folder dataset/."
    )
    st.stop()

st.markdown("### Informasi Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data", f"{len(df):,}")
c2.metric("Jumlah Atribut", f"{df.shape[1]}")
if "pm2.5" in df.columns:
    c3.metric("Rata-rata PM2.5", f"{df['pm2.5'].mean():.1f} \u00b5g/m\u00b3")

st.markdown("### Sampel Data")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("### Statistik Deskriptif")
st.dataframe(df.describe(), use_container_width=True)

st.markdown("### Distribusi PM2.5")
if "pm2.5" in df.columns:
    import plotly.express as px

    fig = px.histogram(df, x="pm2.5", nbins=60, title="Distribusi nilai PM2.5")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Jumlah Missing Value per Kolom")
st.dataframe(df.isna().sum().rename("missing").to_frame(), use_container_width=True)
