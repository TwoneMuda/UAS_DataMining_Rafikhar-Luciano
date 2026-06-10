"""Halaman Visualization."""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict_utils import load_dataset, models_exist, load_models  # noqa: E402

st.set_page_config(page_title="Visualization", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 Visualisasi & Hasil Analisis")

df = load_dataset()
if df is None:
    st.error("Dataset belum ditemukan di folder dataset/.")
    st.stop()

st.markdown("### Rata-rata PM2.5 per Bulan")
if "month" in df.columns and "pm2.5" in df.columns:
    by_month = df.groupby("month")["pm2.5"].mean().reset_index()
    st.plotly_chart(
        px.line(by_month, x="month", y="pm2.5", markers=True,
                title="Rata-rata PM2.5 per Bulan"),
        use_container_width=True,
    )

st.markdown("### Rata-rata PM2.5 per Jam")
if "hour" in df.columns:
    by_hour = df.groupby("hour")["pm2.5"].mean().reset_index()
    st.plotly_chart(
        px.bar(by_hour, x="hour", y="pm2.5", title="Rata-rata PM2.5 per Jam"),
        use_container_width=True,
    )

st.markdown("### Korelasi Antar Fitur Numerik")
num = df.select_dtypes(include="number")
if not num.empty:
    corr = num.corr(numeric_only=True)
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", aspect="auto",
                  color_continuous_scale="RdBu_r", title="Heatmap Korelasi"),
        use_container_width=True,
    )

st.markdown("### Hasil Clustering K-Means (proyeksi PCA 2D)")
if models_exist():
    from sklearn.decomposition import PCA

    model, kmeans, scaler, meta = load_models()
    cdf = df[meta["cluster_features"]].dropna().sample(
        min(3000, len(df)), random_state=42
    )
    scaled = scaler.transform(cdf)
    labels = kmeans.predict(scaled)
    pca = PCA(n_components=2).fit_transform(scaled)
    plot_df = pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1]})
    plot_df["Cluster"] = [meta["cluster_labels"].get(int(c), str(c)) for c in labels]
    st.plotly_chart(
        px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                   title="Segmentasi Kondisi Udara (K-Means)"),
        use_container_width=True,
    )
    st.info(f"Silhouette Score: {meta['silhouette']:.3f}")
else:
    st.warning("Model belum dilatih. Jalankan python train_model.py.")
