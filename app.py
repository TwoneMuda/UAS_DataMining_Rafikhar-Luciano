import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AirPred — Prediksi Kualitas Udara",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "model")
    reg   = joblib.load(os.path.join(model_dir, "model.pkl"))
    km    = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
    sc    = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    meta  = joblib.load(os.path.join(model_dir, "meta.pkl"))
    return reg, km, sc, meta

@st.cache_data
def load_dataset():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "dataset", "PRSA_data_2010.1.1-2014.12.31.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "No" in df.columns:
        df = df.drop(columns=["No"])
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True)
    NUMERIC = ["month", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[c].median())
    df["cbwd"] = df["cbwd"].fillna(df["cbwd"].mode().iloc[0])
    return df

# colour palette
CLUSTER_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#e67e22", 3: "#27ae60"}
CLUSTER_NAMES  = {0: "Polusi Tinggi", 1: "Polusi Sedang", 2: "Cukup Buruk", 3: "Paling Sehat"}
CLUSTER_ICONS  = {0: "🔴", 1: "🟡", 2: "🟠", 3: "🟢"}

def pm25_category(val):
    if val <= 12:
        return "🟢 Baik", "#27ae60"
    elif val <= 35.4:
        return "🟡 Sedang", "#f1c40f"
    elif val <= 55.4:
        return "🟠 Tidak Sehat (Sensitif)", "#e67e22"
    elif val <= 150.4:
        return "🔴 Tidak Sehat", "#e74c3c"
    elif val <= 250.4:
        return "🟣 Sangat Tidak Sehat", "#8e44ad"
    else:
        return "⚫ Berbahaya", "#2c3e50"

# ──────────────────────────────────────────────
# SIDEBAR NAV
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Beijing_smog_comparison_August_2005.jpg/320px-Beijing_smog_comparison_August_2005.jpg",
             use_container_width=True, caption="Kualitas Udara Beijing")
    st.markdown("---")
    page = st.radio(
        "📌 Navigasi",
        ["🏠 Home", "📊 Dataset Overview", "🔮 Prediction / Analysis",
         "📈 Visualization", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("UAS Data Mining — UNESA 2024/2025")

# ──────────────────────────────────────────────
# LOAD ARTIFACTS
# ──────────────────────────────────────────────
try:
    reg_model, km_model, scaler, meta = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_error = str(e)

df = load_dataset()

# ══════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem'>
        <h1 style='font-size:2.8rem; color:#2c3e50;'>🌫️ AirPred</h1>
        <h3 style='color:#7f8c8d; font-weight:400;'>Prediksi & Segmentasi Kualitas Udara Beijing</h3>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("### 📋 Deskripsi Proyek")
        st.info("""
Proyek ini menerapkan dua metode **Data Mining** pada dataset kualitas udara
Beijing (2010–2014) dari UCI Machine Learning Repository:

- **Regresi** — memprediksi konsentrasi PM2.5 (µg/m³) berdasarkan faktor
  cuaca seperti suhu, titik embun, tekanan, dan kecepatan angin.
- **Clustering (K-Means)** — mengelompokkan kondisi udara menjadi 4 segmen
  berdasarkan karakteristik meteorologi dan kadar PM2.5.

Framework metodologi yang digunakan adalah **CRISP-DM** (Cross-Industry
Standard Process for Data Mining).
        """)

        st.markdown("### 🎯 Tujuan Proyek")
        goals = [
            "Membangun model prediksi PM2.5 yang akurat dari data cuaca",
            "Membandingkan performa Linear Regression, Decision Tree, Random Forest, dan Gradient Boosting",
            "Mengelompokkan kondisi udara untuk memahami pola polusi",
            "Menyediakan antarmuka interaktif untuk prediksi real-time",
        ]
        for g in goals:
            st.markdown(f"✅ {g}")

    with col_r:
        st.markdown("### 👥 Identitas Anggota")
        st.markdown("""
<div style='background:#f8f9fa; padding:1.5rem; border-radius:12px; border-left:4px solid #3498db;'>
    <table style='width:100%; border-collapse:collapse;'>
        <tr>
            <td style='padding:8px 4px; color:#7f8c8d; font-size:0.85rem;'>Nama</td>
            <td style='padding:8px 4px; font-weight:600; color:#2c3e50;'>Rafikhar Luciano</td>
        </tr>
        <tr>
            <td style='padding:8px 4px; color:#7f8c8d; font-size:0.85rem;'>Program Studi</td>
            <td style='padding:8px 4px; font-weight:600; color:#2c3e50;'>Informatika / Teknik Informatika</td>
        </tr>
        <tr>
            <td style='padding:8px 4px; color:#7f8c8d; font-size:0.85rem;'>Universitas</td>
            <td style='padding:8px 4px; font-weight:600; color:#2c3e50;'>Universitas Negeri Surabaya (UNESA)</td>
        </tr>
        <tr>
            <td style='padding:8px 4px; color:#7f8c8d; font-size:0.85rem;'>Mata Kuliah</td>
            <td style='padding:8px 4px; font-weight:600; color:#2c3e50;'>Data Mining — UAS</td>
        </tr>
        <tr>
            <td style='padding:8px 4px; color:#7f8c8d; font-size:0.85rem;'>Tahun</td>
            <td style='padding:8px 4px; font-weight:600; color:#2c3e50;'>2024/2025</td>
        </tr>
    </table>
</div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Teknologi yang Digunakan")
        tech_cols = st.columns(3)
        techs = ["Python", "Scikit-learn", "Streamlit", "Pandas", "Matplotlib", "Seaborn"]
        for i, t in enumerate(techs):
            tech_cols[i % 3].markdown(f"<span style='background:#3498db;color:white;padding:4px 10px;border-radius:20px;font-size:0.8rem;'>{t}</span>", unsafe_allow_html=True)

    # Quick stats bar
    if artifacts_ok:
        st.markdown("---")
        st.markdown("### 📊 Ringkasan Hasil Model")
        m1, m2, m3, m4 = st.columns(4)
        best = meta["best_model"]
        r2   = meta["reg_results"][best]["R2"]
        rmse = meta["reg_results"][best]["RMSE"]
        mae  = meta["reg_results"][best]["MAE"]
        sil  = meta["silhouette"]
        m1.metric("🏆 Model Terbaik", best)
        m2.metric("📐 R² Score", f"{r2:.4f}")
        m3.metric("📉 RMSE", f"{rmse:.2f} µg/m³")
        m4.metric("🔵 Silhouette Score", f"{sil:.3f}")

# ══════════════════════════════════════════════
# PAGE: DATASET OVERVIEW
# ══════════════════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Dataset: **Beijing PM2.5 Data** — UCI Machine Learning Repository (2010–2014)")

    if df is None:
        st.warning("⚠️ File dataset tidak ditemukan. Pastikan file `PRSA_data_2010.1.1-2014.12.31.csv` ada di folder `dataset/`.")
        st.info("Menampilkan informasi statis dari notebook analisis.")

        st.markdown("### 📁 Informasi Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris (Raw)", "43.824")
        c2.metric("Jumlah Fitur", "13")
        c3.metric("Missing PM2.5", "2.067 baris")

        st.markdown("### 📋 Deskripsi Fitur")
        feat_df = pd.DataFrame({
            "Fitur": ["year","month","day","hour","pm2.5","DEWP","TEMP","PRES","cbwd","Iws","Is","Ir"],
            "Tipe": ["int","int","int","int","float","float","float","float","str","float","int","int"],
            "Deskripsi": [
                "Tahun pengukuran","Bulan pengukuran","Hari pengukuran","Jam pengukuran",
                "Konsentrasi PM2.5 (µg/m³) — TARGET",
                "Titik embun / Dew Point (℃)",
                "Suhu udara (℃)",
                "Tekanan udara (hPa)",
                "Arah angin gabungan (NW, NE, SE, cv)",
                "Kecepatan angin kumulatif (m/s)",
                "Jam hujan salju kumulatif","Jam hujan kumulatif",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        return

    # ---- Dataset ada ----
    st.markdown("### 📁 Informasi Umum")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris", f"{len(df):,}")
    c2.metric("Jumlah Fitur", len(df.columns))
    c3.metric("PM2.5 Min", f"{df['pm2.5'].min():.1f} µg/m³")
    c4.metric("PM2.5 Max", f"{df['pm2.5'].max():.1f} µg/m³")

    tab1, tab2, tab3 = st.tabs(["🔍 Preview Data", "📈 Statistik", "📊 Distribusi"])

    with tab1:
        st.dataframe(df.head(100), use_container_width=True, height=350)

    with tab2:
        st.dataframe(df.describe().T.round(3), use_container_width=True)
        st.markdown("#### ❓ Missing Values")
        mv = df.isna().sum()
        mv = mv[mv > 0]
        if mv.empty:
            st.success("Tidak ada missing value setelah preprocessing.")
        else:
            st.dataframe(mv.rename("Missing Count").to_frame(), use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df["pm2.5"], bins=60, kde=True, ax=ax, color="#3498db")
            ax.set_title("Distribusi PM2.5", fontsize=12)
            ax.set_xlabel("PM2.5 (µg/m³)")
            st.pyplot(fig, use_container_width=True)

        with col_b:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            numeric_cols = df.select_dtypes("number").columns
            sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
                        cmap="RdBu_r", ax=ax2, annot_kws={"size": 7})
            ax2.set_title("Korelasi Antar Fitur", fontsize=12)
            st.pyplot(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE: PREDICTION / ANALYSIS
# ══════════════════════════════════════════════
elif page == "🔮 Prediction / Analysis":
    st.title("🔮 Prediction & Analysis")

    if not artifacts_ok:
        st.error(f"Model tidak dapat dimuat. Jalankan notebook terlebih dahulu untuk menghasilkan file model.\n\n`{artifact_error}`")
        st.stop()

    tab_reg, tab_clust = st.tabs(["📐 Prediksi PM2.5 (Regresi)", "🎯 Segmentasi Udara (Clustering)"])

    # ---- REGRESSION TAB ----
    with tab_reg:
        st.markdown("### Masukkan Data Cuaca")
        st.caption("Isi nilai parameter cuaca untuk mendapatkan prediksi konsentrasi PM2.5.")

        ranges = meta["feature_ranges"]
        medians = meta["feature_medians"]
        cbwd_cats = meta["cbwd_categories"]

        c1, c2 = st.columns(2)
        with c1:
            month = st.slider("📅 Bulan", 1, 12, 6)
            hour  = st.slider("🕐 Jam", 0, 23, 12)
            DEWP  = st.number_input("💧 Titik Embun / DEWP (℃)",
                                     float(ranges["DEWP"][0]), float(ranges["DEWP"][1]),
                                     float(medians["DEWP"]), step=0.5)
            TEMP  = st.number_input("🌡️ Suhu / TEMP (℃)",
                                     float(ranges["TEMP"][0]), float(ranges["TEMP"][1]),
                                     float(medians["TEMP"]), step=0.5)
        with c2:
            PRES  = st.number_input("🌬️ Tekanan / PRES (hPa)",
                                     float(ranges["PRES"][0]), float(ranges["PRES"][1]),
                                     float(medians["PRES"]), step=1.0)
            Iws   = st.number_input("💨 Kecepatan Angin / Iws (m/s)",
                                     0.0, float(ranges["Iws"][1]),
                                     float(medians["Iws"]), step=1.0)
            Is    = st.number_input("❄️ Jam Salju / Is", 0.0, float(ranges["Is"][1]), 0.0, step=1.0)
            Ir    = st.number_input("🌧️ Jam Hujan / Ir", 0.0, float(ranges["Ir"][1]), 0.0, step=1.0)

        cbwd = st.selectbox("🧭 Arah Angin / cbwd", cbwd_cats)

        st.markdown("---")
        if st.button("🚀 Prediksi PM2.5", use_container_width=True, type="primary"):
            input_df = pd.DataFrame([{
                "month": month, "hour": hour, "DEWP": DEWP, "TEMP": TEMP,
                "PRES": PRES, "Iws": Iws, "Is": Is, "Ir": Ir, "cbwd": cbwd,
            }])
            pred_val = float(reg_model.predict(input_df)[0])
            pred_val = max(0.0, pred_val)
            label, color = pm25_category(pred_val)

            r1, r2 = st.columns(2)
            r1.markdown(f"""
            <div style='background:{color}22; border:2px solid {color}; border-radius:12px; padding:1.2rem; text-align:center;'>
                <div style='font-size:2.5rem; font-weight:700; color:{color};'>{pred_val:.1f} µg/m³</div>
                <div style='font-size:1.1rem; color:#2c3e50; margin-top:4px;'>Prediksi PM2.5</div>
            </div>
            """, unsafe_allow_html=True)
            r2.markdown(f"""
            <div style='background:#f8f9fa; border:2px solid #dee2e6; border-radius:12px; padding:1.2rem; text-align:center;'>
                <div style='font-size:2rem;'>{label}</div>
                <div style='font-size:0.9rem; color:#7f8c8d; margin-top:8px;'>Kategori Kualitas Udara (AQI)</div>
                <div style='font-size:0.8rem; color:#95a5a6;'>Model: {meta["best_model"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📋 Panduan AQI PM2.5")
            aqi_df = pd.DataFrame({
                "Kategori": ["Baik","Sedang","Tidak Sehat (Sensitif)","Tidak Sehat","Sangat Tidak Sehat","Berbahaya"],
                "Rentang (µg/m³)": ["0–12","12.1–35.4","35.5–55.4","55.5–150.4","150.5–250.4",">250.4"],
                "Keterangan": [
                    "Kualitas udara memuaskan",
                    "Kualitas udara dapat diterima",
                    "Berbahaya untuk kelompok sensitif",
                    "Semua orang mulai terdampak",
                    "Darurat kesehatan",
                    "Peringatan darurat kesehatan",
                ],
            })
            st.dataframe(aqi_df, use_container_width=True, hide_index=True)

    # ---- CLUSTERING TAB ----
    with tab_clust:
        st.markdown("### Masukkan Kondisi Cuaca & PM2.5")
        st.caption("Sistem akan menentukan segmen kualitas udara menggunakan K-Means (k=4).")

        c1, c2 = st.columns(2)
        with c1:
            cl_temp = st.number_input("🌡️ Suhu / TEMP (℃)", -20.0, 42.0, 15.0, step=0.5, key="cl_temp")
            cl_dewp = st.number_input("💧 Titik Embun / DEWP (℃)", -40.0, 28.0, 5.0, step=0.5, key="cl_dewp")
            cl_pres = st.number_input("🌬️ Tekanan / PRES (hPa)", 991.0, 1046.0, 1018.0, step=1.0, key="cl_pres")
        with c2:
            cl_iws  = st.number_input("💨 Kec. Angin / Iws (m/s)", 0.0, 600.0, 10.0, step=1.0, key="cl_iws")
            cl_pm   = st.number_input("🌫️ PM2.5 (µg/m³)", 0.0, 1000.0, 50.0, step=1.0, key="cl_pm")

        if st.button("🎯 Segmentasi Kualitas Udara", use_container_width=True, type="primary"):
            CLUSTER_FEATURES = ["TEMP", "DEWP", "PRES", "Iws", "pm2.5"]
            sample = np.array([[cl_temp, cl_dewp, cl_pres, cl_iws, cl_pm]])
            scaled_sample = scaler.transform(sample)
            cluster_id = int(km_model.predict(scaled_sample)[0])
            cname  = CLUSTER_NAMES[cluster_id]
            ccolor = CLUSTER_COLORS[cluster_id]
            cicon  = CLUSTER_ICONS[cluster_id]

            st.markdown(f"""
            <div style='background:{ccolor}22; border:2px solid {ccolor}; border-radius:12px;
                        padding:1.5rem; text-align:center; margin:1rem 0;'>
                <div style='font-size:3rem;'>{cicon}</div>
                <div style='font-size:1.8rem; font-weight:700; color:{ccolor};'>Cluster {cluster_id}: {cname}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📋 Profil Semua Cluster")
            cl_summary = pd.DataFrame({
                "Cluster": [f"Cluster {i}" for i in range(4)],
                "Nama": list(CLUSTER_NAMES.values()),
                "Deskripsi": [
                    "Kadar PM2.5 tertinggi — polusi sangat tinggi, kondisi cuaca memburuk",
                    "Kadar PM2.5 menengah — polusi sedang, kondisi cukup berawan",
                    "Kadar PM2.5 cukup tinggi — urutan kedua, perlu waspada",
                    "Kadar PM2.5 terendah — udara paling bersih dan sehat",
                ],
            })
            st.dataframe(cl_summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# PAGE: VISUALIZATION
# ══════════════════════════════════════════════
elif page == "📈 Visualization":
    st.title("📈 Visualization")

    if not artifacts_ok:
        st.error("Model tidak dapat dimuat. Jalankan notebook terlebih dahulu.")
        st.stop()

    if df is None:
        st.warning("Dataset tidak ditemukan. Visualisasi membutuhkan file dataset.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribusi & Korelasi",
        "🏆 Perbandingan Model",
        "🎯 Hasil Clustering",
        "📅 Tren Temporal",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df["pm2.5"], bins=60, kde=True, ax=ax, color="#3498db")
            ax.set_title("Distribusi PM2.5", fontsize=13, fontweight="bold")
            ax.set_xlabel("PM2.5 (µg/m³)"); ax.set_ylabel("Frekuensi")
            st.pyplot(fig, use_container_width=True)
        with c2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            numeric_df = df.select_dtypes("number")
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                        cmap="RdBu_r", ax=ax2, annot_kws={"size": 7})
            ax2.set_title("Heatmap Korelasi", fontsize=13, fontweight="bold")
            st.pyplot(fig2, use_container_width=True)

        fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, feat, color in zip(axes, ["TEMP", "DEWP", "PRES"],
                                    ["#e74c3c", "#2980b9", "#27ae60"]):
            ax.scatter(df[feat], df["pm2.5"], alpha=0.1, s=2, color=color)
            ax.set_xlabel(feat); ax.set_ylabel("PM2.5")
            ax.set_title(f"PM2.5 vs {feat}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with tab2:
        results = meta["reg_results"]
        res_df = pd.DataFrame(results).T.round(4)
        best = meta["best_model"]

        st.markdown("#### 📋 Tabel Perbandingan Model")
        st.dataframe(res_df.style.highlight_min(subset=["MAE","RMSE"], color="#d5f5e3")
                                  .highlight_max(subset=["R2"], color="#d5f5e3"),
                     use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        names = list(results.keys())
        colors = ["#3498db" if n != best else "#e74c3c" for n in names]
        for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
            vals = [results[n][metric] for n in names]
            bars = ax.bar(names, vals, color=colors, edgecolor="white")
            ax.set_title(metric, fontsize=12, fontweight="bold")
            ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        plt.suptitle(f"Perbandingan Model (merah = terbaik: {best})", fontsize=11, y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab3:
        CLUSTER_FEATURES = ["TEMP", "DEWP", "PRES", "Iws", "pm2.5"]
        cdf = df[CLUSTER_FEATURES].dropna()
        sc_temp = scaler.transform(cdf)
        labels = km_model.predict(sc_temp)
        cdf2 = cdf.copy()
        cdf2["cluster"] = labels

        from sklearn.decomposition import PCA
        pca_coords = PCA(n_components=2).fit_transform(sc_temp)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                                  c=labels, cmap="viridis", s=4, alpha=0.4)
            plt.colorbar(scatter, ax=ax, label="Cluster")
            ax.set_title("Segmentasi Kondisi Udara (K-Means PCA)", fontsize=12, fontweight="bold")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            st.pyplot(fig, use_container_width=True)

        with c2:
            cluster_summary = cdf2.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
            cluster_summary["jumlah_data"] = cdf2.groupby("cluster").size()
            cluster_summary["nama"] = [CLUSTER_NAMES[i] for i in cluster_summary.index]
            st.markdown("#### 📋 Profil Rata-rata Tiap Cluster")
            st.dataframe(cluster_summary, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        pm_means = cluster_summary["pm2.5"]
        colors_bar = [CLUSTER_COLORS[i] for i in pm_means.index]
        bars = ax2.bar([f"Cluster {i}\n{CLUSTER_NAMES[i]}" for i in pm_means.index],
                        pm_means.values, color=colors_bar, edgecolor="white", linewidth=1.2)
        for b, v in zip(bars, pm_means.values):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax2.set_title("Rata-rata PM2.5 per Cluster", fontsize=13, fontweight="bold")
        ax2.set_ylabel("PM2.5 (µg/m³)")
        st.pyplot(fig2, use_container_width=True)

        st.metric("🔵 Silhouette Score", f"{meta['silhouette']:.3f}",
                  help="Nilai mendekati 1 menunjukkan cluster yang terpisah dengan baik")

    with tab4:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        monthly = df.groupby("month")["pm2.5"].mean()
        axes[0, 0].plot(monthly.index, monthly.values, "o-", color="#e74c3c", linewidth=2)
        axes[0, 0].set_title("Rata-rata PM2.5 per Bulan", fontweight="bold")
        axes[0, 0].set_xlabel("Bulan"); axes[0, 0].set_ylabel("PM2.5 (µg/m³)")
        axes[0, 0].set_xticks(range(1, 13))

        hourly = df.groupby("hour")["pm2.5"].mean()
        axes[0, 1].plot(hourly.index, hourly.values, "o-", color="#3498db", linewidth=2)
        axes[0, 1].set_title("Rata-rata PM2.5 per Jam", fontweight="bold")
        axes[0, 1].set_xlabel("Jam"); axes[0, 1].set_ylabel("PM2.5 (µg/m³)")

        yearly = df.groupby("year")["pm2.5"].mean()
        axes[1, 0].bar(yearly.index, yearly.values, color="#27ae60", edgecolor="white")
        axes[1, 0].set_title("Rata-rata PM2.5 per Tahun", fontweight="bold")
        axes[1, 0].set_xlabel("Tahun"); axes[1, 0].set_ylabel("PM2.5 (µg/m³)")

        wind_pm = df.groupby("cbwd")["pm2.5"].mean().sort_values(ascending=False)
        axes[1, 1].bar(wind_pm.index, wind_pm.values, color="#9b59b6", edgecolor="white")
        axes[1, 1].set_title("Rata-rata PM2.5 per Arah Angin", fontweight="bold")
        axes[1, 1].set_xlabel("Arah Angin"); axes[1, 1].set_ylabel("PM2.5 (µg/m³)")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About")

    tab1, tab2, tab3 = st.tabs(["🔬 Metode", "📂 Dataset", "📋 Informasi Proyek"])

    with tab1:
        st.markdown("## Metode Data Mining yang Digunakan")

        st.markdown("### 1. 📐 Regresi (Prediksi PM2.5)")
        st.info("""
Regresi digunakan untuk memprediksi nilai kontinu — dalam hal ini konsentrasi **PM2.5 (µg/m³)** —
berdasarkan variabel independen (fitur cuaca).

Empat algoritma dibandingkan:

| Algoritma | Kelebihan | Kekurangan |
|---|---|---|
| **Linear Regression** | Sederhana, cepat, interpretable | Tidak menangkap hubungan non-linear |
| **Decision Tree** | Mudah diinterpretasi, non-linear | Rentan overfitting |
| **Random Forest** | Akurasi tinggi, robust | Lebih lambat, kurang interpretable |
| **Gradient Boosting** | Akurasi sangat tinggi | Sensitif terhadap hyperparameter |

Model terbaik dipilih berdasarkan nilai **R² tertinggi** dan **RMSE/MAE terendah** pada data test.
        """)

        st.markdown("### 2. 🎯 Clustering K-Means")
        st.info("""
K-Means Clustering mengelompokkan data ke dalam **k cluster** dengan meminimalkan jumlah
kuadrat jarak dari setiap titik ke centroid cluster-nya.

**Langkah implementasi:**
1. Normalisasi fitur menggunakan `StandardScaler`
2. Penentuan k optimal dengan **Elbow Method** → dipilih k = 4
3. Evaluasi kualitas cluster dengan **Silhouette Score**
4. Visualisasi hasil menggunakan **PCA** (reduksi ke 2 dimensi)

**Fitur clustering:** TEMP, DEWP, PRES, Iws, PM2.5
        """)

        st.markdown("### 3. 🔄 Framework CRISP-DM")
        st.markdown("""
Proyek mengikuti framework **Cross-Industry Standard Process for Data Mining (CRISP-DM)**:

1. **Business Understanding** — Mendefinisikan tujuan bisnis (pemantauan polusi udara)
2. **Data Understanding** — Eksplorasi dan pemahaman dataset Beijing PM2.5
3. **Data Preparation** — Pembersihan data, handling missing values, encoding
4. **Modeling** — Pelatihan model regresi dan clustering
5. **Evaluation** — Perbandingan metrik performa antar model
6. **Deployment** — Ekspor model dan pembuatan aplikasi Streamlit ini
        """)

    with tab2:
        st.markdown("## 📂 Informasi Dataset")
        st.markdown("""
### Beijing PM2.5 Dataset

| Atribut | Keterangan |
|---|---|
| **Nama** | Beijing PM2.5 Data |
| **Sumber** | UCI Machine Learning Repository |
| **Periode** | 1 Januari 2010 – 31 Desember 2014 |
| **Jumlah Data** | 43.824 baris (raw) |
| **Jumlah Fitur** | 13 kolom |
| **Target** | pm2.5 (konsentrasi polutan, µg/m³) |

### Deskripsi Fitur

| Fitur | Satuan | Deskripsi |
|---|---|---|
| year/month/day/hour | — | Waktu pengukuran |
| **pm2.5** | µg/m³ | Konsentrasi partikel halus (TARGET) |
| DEWP | ℃ | Dew Point / Titik Embun |
| TEMP | ℃ | Suhu udara |
| PRES | hPa | Tekanan udara |
| cbwd | — | Arah angin gabungan (NW, NE, SE, cv) |
| Iws | m/s | Kecepatan angin kumulatif |
| Is | jam | Jam hujan salju kumulatif |
| Ir | jam | Jam hujan kumulatif |

### Preprocessing
- Drop kolom `No` (index)
- Hapus baris dengan `pm2.5` = NaN (2.067 baris)
- Imputasi median untuk fitur numerik
- Imputasi modus untuk `cbwd`
- One-Hot Encoding untuk `cbwd`
        """)

    with tab3:
        st.markdown("## 📋 Informasi Proyek")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
### Identitas
| | |
|---|---|
| **Nama** | Rafikhar Luciano |
| **Universitas** | Universitas Negeri Surabaya (UNESA) |
| **Mata Kuliah** | Data Mining |
| **Jenis Tugas** | UAS (Ujian Akhir Semester) |
| **Tahun** | 2024/2025 |

### Tools & Library
- **Python** 3.x
- **Scikit-learn** — Machine Learning
- **Streamlit** — Web App
- **Pandas / NumPy** — Data Processing
- **Matplotlib / Seaborn** — Visualisasi
- **Joblib** — Model Serialization
            """)
        with col2:
            st.markdown("""
### Dua Metode Data Mining
1. **Regresi** (Supervised Learning)
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor ✅ *terbaik*
   - Gradient Boosting Regressor

2. **Clustering** (Unsupervised Learning)
   - K-Means (k=4)
   - Evaluasi: Silhouette Score

### Metrik Evaluasi
- **Regresi:** MAE, RMSE, R²
- **Clustering:** Silhouette Score, Elbow Method
            """)

        st.markdown("---")
        st.caption("© 2024/2025 Rafikhar Luciano — Universitas Negeri Surabaya")
