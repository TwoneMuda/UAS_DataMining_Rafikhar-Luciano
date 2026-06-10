import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, silhouette_score)

# ----------------------------------------------------------------------
# Konfigurasi halaman & konstanta
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi PM2.5 & Kualitas Udara",
    page_icon="\U0001F32B\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC = ['month', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
CATEG = ['cbwd']
TARGET = 'pm2.5'
CLUSTER_FEATURES = ['TEMP', 'DEWP', 'PRES', 'Iws', 'pm2.5']
K_CLUSTERS = 4

LABEL_NUMERIC = {
    'month': 'Bulan', 'hour': 'Jam', 'DEWP': 'Titik Embun / Dew Point (\u00B0C)',
    'TEMP': 'Suhu / Temperature (\u00B0C)', 'PRES': 'Tekanan / Pressure (hPa)',
    'Iws': 'Kecepatan Angin Kumulatif (m/s)', 'Is': 'Jam Salju Kumulatif',
    'Ir': 'Jam Hujan Kumulatif',
}

# Kategori kualitas udara berdasarkan konsentrasi PM2.5 (\u00B5g/m\u00B3) - acuan EPA/WHO
PM25_CATEGORIES = [
    (0.0,   12.0,  'Baik',                 '#009966'),
    (12.1,  35.4,  'Sedang',               '#ffde33'),
    (35.5,  55.4,  'Tidak Sehat (Sensitif)', '#ff9933'),
    (55.5,  150.4, 'Tidak Sehat',          '#cc0033'),
    (150.5, 250.4, 'Sangat Tidak Sehat',   '#660099'),
    (250.5, 1e9,   'Berbahaya',            '#7e0023'),
]

# Kandidat lokasi file dataset
DATA_CANDIDATES = [
    'dataset/PRSA_data_2010.1.1-2014.12.31.csv',
    '../dataset/PRSA_data_2010.1.1-2014.12.31.csv',
    'PRSA_data_2010.1.1-2014.12.31.csv',
    'dataset/PRSA_data.csv',
]


def pm25_category(value):
    """Mengembalikan (label, warna) kategori kualitas udara dari nilai PM2.5."""
    for lo, hi, label, color in PM25_CATEGORIES:
        if lo <= value <= hi:
            return label, color
    return 'Berbahaya', '#7e0023'


# ----------------------------------------------------------------------
# Pemuatan data (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw_data(file_buffer=None):
    """Memuat dataset mentah dari file upload atau dari path lokal."""
    if file_buffer is not None:
        return pd.read_csv(file_buffer)
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def prepare_data(df_raw):
    """Data Preparation sesuai notebook: buang No, drop target NaN, imputasi."""
    df = df_raw.copy()
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())
    df['cbwd'] = df['cbwd'].fillna(df['cbwd'].mode().iloc[0])
    return df


# ----------------------------------------------------------------------
# Pelatihan model (cached resource)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def train_models(df):
    """Melatih model regresi (pilih terbaik) + K-Means. Mengembalikan artefak."""
    # ---- Regresi ----
    X = df[NUMERIC + CATEG]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    prep = ColumnTransformer([
        ('num', 'passthrough', NUMERIC),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEG),
    ])

    candidates = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=42),
        'Random Forest': RandomForestRegressor(
            n_estimators=120, max_depth=18, n_jobs=-1, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    }

    results = {}
    best_name, best_pipe, best_r2 = None, None, -1e9
    for name, est in candidates.items():
        pipe = Pipeline([('prep', prep), ('model', est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
            'R2': r2_score(y_test, pred),
        }
        if results[name]['R2'] > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, results[name]['R2']

    y_pred_best = best_pipe.predict(X_test)

    # ---- Clustering K-Means ----
    cdf = df[CLUSTER_FEATURES].dropna()
    scaler = StandardScaler().fit(cdf)
    scaled = scaler.transform(cdf)

    inertia = []
    krange = list(range(2, 9))
    for kk in krange:
        inertia.append(
            KMeans(n_clusters=kk, random_state=42, n_init=10).fit(scaled).inertia_)

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    sil = silhouette_score(scaled, labels)

    pca = PCA(n_components=2).fit_transform(scaled)

    tmp = cdf.assign(cluster=labels)
    cluster_summary = tmp.groupby('cluster')[CLUSTER_FEATURES].mean().round(2)
    cluster_summary['jumlah_data'] = tmp.groupby('cluster').size()

    # Label cluster dinamis berdasarkan rata-rata PM2.5 (tinggi -> buruk)
    order = cluster_summary['pm2.5'].sort_values(ascending=False).index.tolist()
    quality_names = ['Polusi Tinggi', 'Cukup Buruk', 'Polusi Sedang', 'Paling Sehat']
    cluster_labels = {cid: quality_names[i] for i, cid in enumerate(order)}

    return {
        'best_name': best_name,
        'best_pipe': best_pipe,
        'reg_results': results,
        'y_test': np.asarray(y_test),
        'y_pred': np.asarray(y_pred_best),
        'scaler': scaler,
        'kmeans': kmeans,
        'silhouette': float(sil),
        'inertia': inertia,
        'krange': krange,
        'pca': pca,
        'labels': labels,
        'cluster_summary': cluster_summary,
        'cluster_labels': cluster_labels,
        'feature_medians': {c: float(df[c].median()) for c in NUMERIC},
        'feature_ranges': {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC},
        'cbwd_categories': sorted(df['cbwd'].unique().tolist()),
    }


# ----------------------------------------------------------------------
# Sidebar — navigasi & data
# ----------------------------------------------------------------------
st.sidebar.title("\U0001F32B\uFE0F Navigasi")
page = st.sidebar.radio(
    "Pilih halaman:",
    ["\U0001F3E0 Home", "\U0001F4CA Dataset Overview",
     "\U0001F52E Prediction / Analysis", "\U0001F4C8 Visualization", "\u2139\uFE0F About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Sumber data")
uploaded = st.sidebar.file_uploader(
    "Upload CSV PRSA (opsional)", type=['csv'],
    help="Jika dataset tidak ditemukan otomatis, upload file PRSA di sini.")

df_raw = load_raw_data(uploaded)

if df_raw is None:
    st.title("\U0001F32B\uFE0F Prediksi PM2.5 & Segmentasi Kualitas Udara")
    st.warning(
        "**Dataset belum ditemukan.** Letakkan file "
        "`PRSA_data_2010.1.1-2014.12.31.csv` di folder `dataset/` "
        "di samping `app.py`, atau upload melalui sidebar di kiri.")
    st.info("Dataset: UCI Beijing PM2.5 Data Set.")
    st.stop()

df = prepare_data(df_raw)
art = train_models(df)


# ======================================================================
# 1. HOME
# ======================================================================
if page.endswith("Home"):
    st.title("\U0001F32B\uFE0F Prediksi PM2.5 & Segmentasi Kualitas Udara")
    st.subheader("UAS Data Mining \u2014 Metodologi CRISP-DM")

    st.markdown(
        """
        Aplikasi ini memprediksi konsentrasi **PM2.5** (partikel halus polutan udara)
        berdasarkan faktor cuaca, serta melakukan **segmentasi kondisi udara**
        menggunakan algoritma *clustering* K-Means.
        """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(df):,}")
    c2.metric("Model Terbaik", art['best_name'])
    c3.metric("R\u00B2 (Regresi)", f"{art['reg_results'][art['best_name']]['R2']:.3f}")
    c4.metric("Silhouette (Cluster)", f"{art['silhouette']:.3f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### \U0001F4CC Deskripsi Singkat Proyek")
        st.markdown(
            """
            - **Tujuan 1 \u2014 Regresi:** memprediksi nilai PM2.5 (\u00B5g/m\u00B3) dari
              faktor cuaca (suhu, tekanan, titik embun, kecepatan angin, dll).
            - **Tujuan 2 \u2014 Clustering:** mengelompokkan kondisi udara menjadi
              beberapa segmen kualitas (Polusi Tinggi \u2192 Paling Sehat).
            - **Manfaat:** membantu memahami pola polusi udara dan faktor
              cuaca yang memengaruhinya.
            """)
    with col_b:
        st.markdown("### \U0001F465 Identitas Anggota")
        st.markdown(
            """
            | Peran | Nama |
            |---|---|
            | Ketua / Anggota | **RAFIKHAR LUCIANO** |

            **Mata Kuliah:** Data Mining  
            **Program Studi:** S1 Sistem Informasi  
            **Metodologi:** CRISP-DM
            """)

    st.markdown("---")
    st.markdown("### \U0001F9ED Alur CRISP-DM")
    st.markdown(
        "`Business Understanding` \u2192 `Data Understanding` \u2192 "
        "`Data Preparation` \u2192 `Modeling` \u2192 `Evaluation` \u2192 `Deployment`")


# ======================================================================
# 2. DATASET OVERVIEW
# ======================================================================
elif page.endswith("Dataset Overview"):
    st.title("\U0001F4CA Dataset Overview")

    st.markdown("### \u2139\uFE0F Informasi Dataset")
    st.markdown(
        """
        **UCI Beijing PM2.5 Data Set** \u2014 berisi data per jam konsentrasi PM2.5
        di Kedutaan AS Beijing beserta data cuaca dari Bandara Internasional
        Ibukota Beijing, periode **2010\u20132014**.
        """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Baris (bersih)", f"{len(df):,}")
    c2.metric("Jumlah Kolom", f"{df.shape[1]}")
    c3.metric("Fitur Numerik", f"{len(NUMERIC)}")
    c4.metric("Fitur Kategorik", f"{len(CATEG)}")

    st.markdown("### \U0001F4CB Cuplikan Data")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("\U0001F4D6 Deskripsi Fitur"):
        st.markdown(
            """
            | Fitur | Keterangan |
            |---|---|
            | year, month, day, hour | Waktu pencatatan data |
            | **pm2.5** | Konsentrasi PM2.5 (\u00B5g/m\u00B3) \u2014 *target* |
            | DEWP | Titik embun / Dew Point (\u00B0C) |
            | TEMP | Suhu (\u00B0C) |
            | PRES | Tekanan udara (hPa) |
            | cbwd | Arah angin gabungan (kategorik) |
            | Iws | Kecepatan angin kumulatif (m/s) |
            | Is | Jam salju kumulatif |
            | Ir | Jam hujan kumulatif |
            """)

    st.markdown("### \U0001F4C8 Statistik Sederhana")
    st.dataframe(df.describe().T.round(2), use_container_width=True)

    st.markdown("### \U0001F4CA Visualisasi Data")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Distribusi PM2.5**")
        fig = px.histogram(df, x='pm2.5', nbins=60,
                           color_discrete_sequence=['#cc0033'])
        fig.update_layout(height=350, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        st.markdown("**Rata-rata PM2.5 per Bulan**")
        monthly = df.groupby('month')['pm2.5'].mean().reset_index()
        fig = px.bar(monthly, x='month', y='pm2.5',
                     color_discrete_sequence=['#ff9933'])
        fig.update_layout(height=350, margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Heatmap Korelasi Antar Fitur Numerik**")
    corr = df.select_dtypes('number').corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(height=500, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 3. PREDICTION / ANALYSIS
# ======================================================================
elif page.endswith("Prediction / Analysis"):
    st.title("\U0001F52E Prediction / Analysis")
    st.markdown(
        "Masukkan kondisi cuaca untuk **memprediksi nilai PM2.5** dan "
        "menentukan **segmen kualitas udara** (cluster).")

    medians = art['feature_medians']
    ranges = art['feature_ranges']

    with st.form("prediction_form"):
        st.markdown("#### \U0001F4DD Form Input Pengguna")
        col1, col2 = st.columns(2)
        with col1:
            month = st.slider("Bulan", 1, 12, 6)
            hour = st.slider("Jam", 0, 23, 12)
            temp = st.number_input("Suhu / TEMP (\u00B0C)",
                                   value=float(medians['TEMP']), step=0.5)
            dewp = st.number_input("Titik Embun / DEWP (\u00B0C)",
                                   value=float(medians['DEWP']), step=0.5)
        with col2:
            pres = st.number_input("Tekanan / PRES (hPa)",
                                   value=float(medians['PRES']), step=0.5)
            iws = st.number_input("Kecepatan Angin / Iws (m/s)",
                                  value=float(medians['Iws']), step=0.5, min_value=0.0)
            is_snow = st.number_input("Jam Salju Kumulatif / Is",
                                      value=float(medians['Is']), step=1.0, min_value=0.0)
            ir_rain = st.number_input("Jam Hujan Kumulatif / Ir",
                                      value=float(medians['Ir']), step=1.0, min_value=0.0)
        cbwd = st.selectbox("Arah Angin / cbwd", art['cbwd_categories'])
        submitted = st.form_submit_button("\U0001F680 Proses Prediksi", use_container_width=True)

    if submitted:
        X_input = pd.DataFrame([{
            'month': month, 'hour': hour, 'DEWP': dewp, 'TEMP': temp,
            'PRES': pres, 'Iws': iws, 'Is': is_snow, 'Ir': ir_rain, 'cbwd': cbwd,
        }])
        pm25_pred = float(art['best_pipe'].predict(X_input)[0])
        pm25_pred = max(pm25_pred, 0.0)
        label, color = pm25_category(pm25_pred)

        # Tentukan cluster berdasarkan fitur clustering
        cluster_input = pd.DataFrame([{
            'TEMP': temp, 'DEWP': dewp, 'PRES': pres, 'Iws': iws, 'pm2.5': pm25_pred,
        }])[CLUSTER_FEATURES]
        scaled_in = art['scaler'].transform(cluster_input)
        cluster_id = int(art['kmeans'].predict(scaled_in)[0])
        cluster_name = art['cluster_labels'].get(cluster_id, f"Cluster {cluster_id}")

        st.markdown("### \u2705 Hasil Prediksi")
        r1, r2, r3 = st.columns(3)
        r1.metric("Prediksi PM2.5", f"{pm25_pred:.1f} \u00B5g/m\u00B3")
        r2.metric("Kategori Kualitas Udara", label)
        r3.metric("Segmen (Cluster)", f"#{cluster_id} \u2014 {cluster_name}")

        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{color};"
            f"color:white;font-size:20px;font-weight:600;text-align:center'>"
            f"Kualitas Udara: {label} \u2014 PM2.5 \u2248 {pm25_pred:.1f} \u00B5g/m\u00B3</div>",
            unsafe_allow_html=True)

        st.markdown("#### \U0001F4CD Posisi pada Skala PM2.5")
        fig = px.bar(
            x=[c[2] for c in PM25_CATEGORIES],
            y=[min(c[1], 300) for c in PM25_CATEGORIES],
            color=[c[2] for c in PM25_CATEGORIES],
            color_discrete_sequence=[c[3] for c in PM25_CATEGORIES],
        )
        fig.add_hline(y=pm25_pred, line_dash='dash', line_color='black',
                      annotation_text=f"Prediksi {pm25_pred:.0f}")
        fig.update_layout(showlegend=False, height=350,
                          xaxis_title="Kategori", yaxis_title="Batas atas PM2.5")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"Interpretasi: dengan kondisi cuaca yang dimasukkan, model "
            f"**{art['best_name']}** memprediksi PM2.5 sekitar "
            f"**{pm25_pred:.1f} \u00B5g/m\u00B3** dan K-Means menempatkannya pada "
            f"segmen **{cluster_name}**.")
    else:
        st.info("Isi form di atas lalu tekan **Proses Prediksi**.")


# ======================================================================
# 4. VISUALIZATION
# ======================================================================
elif page.endswith("Visualization"):
    st.title("\U0001F4C8 Visualization")
    st.markdown("Grafik pendukung dan visualisasi hasil analisis model.")

    tab1, tab2, tab3 = st.tabs(
        ["\U0001F3AF Evaluasi Regresi", "\U0001F9E9 Hasil Clustering", "\U0001F4C9 Eksplorasi Data"])

    # ---- Tab 1: Regresi ----
    with tab1:
        st.markdown("#### Perbandingan Model Regresi")
        res_df = pd.DataFrame(art['reg_results']).T.round(3)
        st.dataframe(res_df, use_container_width=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**R\u00B2 per Model** (makin tinggi makin baik)")
            fig = px.bar(res_df.reset_index(), x='index', y='R2',
                         color='R2', color_continuous_scale='Greens')
            fig.update_layout(height=350, xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.markdown("**RMSE per Model** (makin rendah makin baik)")
            fig = px.bar(res_df.reset_index(), x='index', y='RMSE',
                         color='RMSE', color_continuous_scale='Reds')
            fig.update_layout(height=350, xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Prediksi vs Aktual \u2014 Model Terbaik ({art['best_name']})**")
        n = min(3000, len(art['y_test']))
        idx = np.random.RandomState(42).choice(len(art['y_test']), n, replace=False)
        scat = pd.DataFrame({'Aktual': art['y_test'][idx], 'Prediksi': art['y_pred'][idx]})
        fig = px.scatter(scat, x='Aktual', y='Prediksi', opacity=0.4,
                         color_discrete_sequence=['#1f77b4'])
        m = float(max(scat['Aktual'].max(), scat['Prediksi'].max()))
        fig.add_shape(type='line', x0=0, y0=0, x1=m, y1=m,
                      line=dict(color='red', dash='dash'))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Clustering ----
    with tab2:
        st.markdown(f"#### Segmentasi K-Means (k={K_CLUSTERS}) \u2014 Silhouette: {art['silhouette']:.3f}")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Elbow Method**")
            elbow = pd.DataFrame({'k': art['krange'], 'Inertia': art['inertia']})
            fig = px.line(elbow, x='k', y='Inertia', markers=True)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.markdown("**Visualisasi Cluster (PCA 2D)**")
            pca_df = pd.DataFrame(art['pca'], columns=['PC1', 'PC2'])
            pca_df['cluster'] = art['labels'].astype(str)
            samp = pca_df.sample(min(4000, len(pca_df)), random_state=42)
            fig = px.scatter(samp, x='PC1', y='PC2', color='cluster', opacity=0.5)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ringkasan Tiap Cluster**")
        summary = art['cluster_summary'].copy()
        summary['kualitas'] = [art['cluster_labels'].get(i, '-') for i in summary.index]
        st.dataframe(summary, use_container_width=True)

        st.markdown("**Rata-rata PM2.5 per Cluster**")
        bar_df = summary.reset_index()
        fig = px.bar(bar_df, x='cluster', y='pm2.5', color='kualitas',
                     text='pm2.5')
        fig.update_layout(height=380, xaxis_title="Cluster",
                          yaxis_title="PM2.5 (\u00B5g/m\u00B3)")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: Eksplorasi ----
    with tab3:
        st.markdown("#### Eksplorasi Hubungan Fitur")
        cc1, cc2 = st.columns(2)
        with cc1:
            xcol = st.selectbox("Sumbu X", NUMERIC, index=NUMERIC.index('TEMP'))
        with cc2:
            ycol = st.selectbox("Sumbu Y", ['pm2.5'] + NUMERIC, index=0)
        samp = df.sample(min(4000, len(df)), random_state=42)
        fig = px.scatter(samp, x=xcol, y=ycol, color='cbwd', opacity=0.5)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Pola PM2.5 per Jam dalam Sehari**")
        hourly = df.groupby('hour')['pm2.5'].mean().reset_index()
        fig = px.line(hourly, x='hour', y='pm2.5', markers=True,
                      color_discrete_sequence=['#cc0033'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 5. ABOUT
# ======================================================================
elif page.endswith("About"):
    st.title("\u2139\uFE0F About")

    st.markdown("### \U0001F9EA Penjelasan Metode")
    st.markdown(
        """
        Proyek ini mengikuti metodologi **CRISP-DM** dan menggabungkan dua
        teknik *data mining*:

        **1. Regresi (Supervised Learning) \u2014 Prediksi PM2.5**
        - Membandingkan 4 algoritma: *Linear Regression, Decision Tree,
          Random Forest,* dan *Gradient Boosting*.
        - Preprocessing: imputasi median (numerik) & modus (kategorik),
          *One-Hot Encoding* untuk `cbwd`.
        - Evaluasi dengan **MAE, RMSE, dan R\u00B2**; model dengan R\u00B2 tertinggi
          dipilih sebagai model terbaik.

        **2. Clustering (Unsupervised Learning) \u2014 K-Means**
        - Mengelompokkan kondisi udara berdasarkan `TEMP, DEWP, PRES, Iws, pm2.5`.
        - Fitur distandarisasi (*StandardScaler*), jumlah cluster optimal
          ditentukan dengan **Elbow Method** (k=4).
        - Kualitas cluster diukur dengan **Silhouette Score**, divisualisasikan
          dengan **PCA 2D**.
        """)

    st.markdown("### \U0001F5C2\uFE0F Dataset")
    st.markdown(
        """
        - **Nama:** UCI Beijing PM2.5 Data Set
        - **Periode:** 2010\u20132014 (data per jam)
        - **Sumber:** UCI Machine Learning Repository
        - **Target:** `pm2.5` (\u00B5g/m\u00B3)
        - **Fitur cuaca:** suhu, titik embun, tekanan, arah & kecepatan angin,
          jam salju, jam hujan.
        """)

    st.markdown("### \U0001F4CB Informasi Proyek")
    st.markdown(
        f"""
        | Item | Keterangan |
        |---|---|
        | Judul | Prediksi PM2.5 & Segmentasi Kualitas Udara |
        | Mata Kuliah | Data Mining (UAS) |
        | Metodologi | CRISP-DM |
        | Teknik | Regresi + Clustering (K-Means) |
        | Framework | Streamlit |
        | Model terbaik | {art['best_name']} (R\u00B2 = {art['reg_results'][art['best_name']]['R2']:.3f}) |
        | Pembuat | RAFIKHAR LUCIANO |
        """)

    st.markdown("---")
    st.caption("Dibuat untuk UAS Data Mining \u2014 RAFIKHAR LUCIANO.")
