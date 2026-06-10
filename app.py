"""
╔══════════════════════════════════════════════════════════════╗
║   Aplikasi Prediksi Kualitas Udara Global                    ║
║   Dataset: PM2.5 Global Air Pollution 2010-2017             ║
║   Metode: Random Forest + K-Means Clustering                ║
╚══════════════════════════════════════════════════════════════╝

Jalankan dengan:
    streamlit run app.py

Pastikan file berikut ada di folder yang sama:
  - rf_model.pkl
  - cluster_info.json
  - df_history.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Kualitas Udara Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS KUSTOM
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { margin: 0; font-size: 1.8rem; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }

    .cluster-badge {
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .good      { background: #d4edda; color: #155724; }
    .moderate  { background: #cce5ff; color: #004085; }
    .bad       { background: #fff3cd; color: #856404; }
    .very-bad  { background: #f8d7da; color: #721c24; }

    .info-box {
        background: #f0f4f8;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# LOAD MODEL & DATA
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    if not os.path.exists('rf_model.pkl'):
        return None
    return joblib.load('rf_model.pkl')

@st.cache_data(show_spinner="Memuat data kluster...")
def load_cluster_info():
    if not os.path.exists('cluster_info.json'):
        return None
    with open('cluster_info.json') as f:
        info = json.load(f)
    info['label_map'] = {int(k): v for k, v in info['label_map'].items()}
    return info

@st.cache_data(show_spinner="Memuat data historis...")
def load_history():
    if not os.path.exists('df_history.csv'):
        return None
    df = pd.read_csv('df_history.csv')
    return df

rf_model    = load_model()
cluster_info = load_cluster_info()
df_history  = load_history()

MODEL_READY = (rf_model is not None) and (cluster_info is not None) and (df_history is not None)


# FUNGSI UTILITAS
def predict_pm25(country_name, target_year, df_hist, model, ci):
    """Prediksi PM2.5 secara iteratif menggunakan lag features."""
    hist = df_hist[df_hist['Country Name'] == country_name].sort_values('Year')
    if hist.empty:
        return None, None

    pm_vals   = list(hist['PM2.5'].values)
    year_vals = list(hist['Year'].values)
    last_year = year_vals[-1]

    for yr in range(last_year + 1, target_year + 1):
        lag1      = pm_vals[-1]
        lag2      = pm_vals[-2] if len(pm_vals) >= 2 else lag1
        lag3      = pm_vals[-3] if len(pm_vals) >= 3 else lag2
        rollmean2 = np.mean([lag1, lag2])
        diff1     = pm_vals[-1] - (pm_vals[-2] if len(pm_vals) >= 2 else pm_vals[-1])

        feat = pd.DataFrame([{
            'Year': yr,
            'PM2.5_lag1': lag1, 'PM2.5_lag2': lag2, 'PM2.5_lag3': lag3,
            'PM2.5_rollmean2': rollmean2, 'PM2.5_diff1': diff1
        }])
        pred = max(0, round(model.predict(feat)[0], 3))
        pm_vals.append(pred)
        year_vals.append(yr)

    predicted_pm25 = pm_vals[-1]

    # Mapping ke kluster
    thresholds = ci['thresholds']
    order      = ci['order']
    if predicted_pm25 <= thresholds[0]:
        cid = order[0]
    elif predicted_pm25 <= thresholds[1]:
        cid = order[1]
    elif predicted_pm25 <= thresholds[2]:
        cid = order[2]
    else:
        cid = order[3]

    return predicted_pm25, cid


def get_cluster_style(label_str):
    l = label_str.lower()
    if 'sangat baik'  in l: return 'good'
    if 'baik'         in l: return 'moderate'
    if 'sangat buruk' in l: return 'very-bad'
    if 'buruk'        in l: return 'bad'
    return 'moderate'


def get_pm25_category(pm25):
    """Kategori PM2.5 berdasarkan standar WHO/US-EPA."""
    if pm25 <= 12:
        return "🟢 Good (Baik)", "#d4edda", "#155724"
    elif pm25 <= 35.4:
        return "🟡 Moderate (Sedang)", "#fff3cd", "#856404"
    elif pm25 <= 55.4:
        return "🟠 Unhealthy for Sensitive Groups", "#fde8d8", "#7d3c00"
    elif pm25 <= 150.4:
        return "🔴 Unhealthy (Tidak Sehat)", "#f8d7da", "#721c24"
    elif pm25 <= 250.4:
        return "🟣 Very Unhealthy (Sangat Tidak Sehat)", "#e8d0f0", "#4a235a"
    else:
        return "⚫ Hazardous (Berbahaya)", "#d5d5d5", "#1a1a1a"


# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4580/4580438.png", width=80)
    st.markdown("## 🌍 Air Quality Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🔮 Prediksi", "📊 Dashboard Data", "📈 Analisis Kluster", "ℹ️ Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if MODEL_READY:
        st.success("✅ Model siap digunakan")
        countries_list = sorted(df_history['Country Name'].unique())
        min_year = int(df_history['Year'].max()) + 1
        st.markdown(f"**Dataset:** {len(countries_list)} negara")
        st.markdown(f"**Data historis:** 2010–{int(df_history['Year'].max())}")
    else:
        st.error("⚠️ Model belum dimuat")
        st.markdown("""
        **Cara menggunakan:**
        1. Jalankan notebook `air_pollution_datamining.ipynb` terlebih dahulu
        2. Pastikan file berikut ada di folder ini:
           - `rf_model.pkl`
           - `cluster_info.json`
           - `df_history.csv`
        3. Restart aplikasi ini
        """)


# HEADER UTAMA
st.markdown('<p class="main-title">🌍 Prediksi Kualitas Udara Global</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prediksi nilai PM2.5 per tahun dan pengelompokan kluster menggunakan Random Forest + K-Means</p>', unsafe_allow_html=True)


# HALAMAN: PREDIKSI
if "🔮 Prediksi" in page:
    if not MODEL_READY:
        st.warning("⚠️ Model belum dimuat. Silakan jalankan notebook terlebih dahulu.")
        st.stop()

    st.markdown("### 🔮 Prediksi PM2.5 & Kluster Kualitas Udara")

    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("#### Input Prediksi")
        country = st.selectbox(
            "🌐 Pilih Negara",
            options=countries_list,
            index=countries_list.index('Indonesia') if 'Indonesia' in countries_list else 0
        )

        tahun = st.slider(
            "📅 Tahun Prediksi",
            min_value=min_year,
            max_value=min_year + 15,
            value=min_year + 2,
            step=1
        )

        predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True, type="primary")

    with col_result:
        if predict_btn:
            with st.spinner("⏳ Menghitung prediksi..."):
                pm, cid = predict_pm25(country, tahun, df_history, rf_model, cluster_info)

            if pm is None:
                st.error(f"❌ Data untuk negara '{country}' tidak ditemukan.")
            else:
                label_full = cluster_info['label_map'][cid]
                label_short = label_full.split('–')[1].strip() if '–' in label_full else label_full
                css_class = get_cluster_style(label_full)
                pm_cat, pm_bg, pm_fg = get_pm25_category(pm)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                            border-radius: 16px; padding: 1.5rem; border: 1px solid #dee2e6;">
                    <h3 style="margin:0; color:#1a1a2e;">{country} – {tahun}</h3>
                    <hr style="margin: 0.7rem 0; border-color: #ccc;">
                    <div style="font-size: 2.5rem; font-weight: 700; color: #2c3e50;">
                        {pm:.2f} <span style="font-size:1rem; font-weight:400;">µg/m³ PM2.5</span>
                    </div>
                    <div style="background:{pm_bg}; color:{pm_fg}; border-radius:8px;
                                padding: 0.4rem 0.8rem; display:inline-block; margin: 0.5rem 0;">
                        {pm_cat}
                    </div>
                    <br>
                    <span class="cluster-badge {css_class}">{label_full}</span>
                </div>
                """, unsafe_allow_html=True)

                # Rekomendasi
                st.markdown("#### 💡 Rekomendasi")
                recs = {
                    'good':     ["✅ Kualitas udara sangat baik, aktivitas luar ruang aman.",
                                 "🌱 Pertahankan kebijakan lingkungan yang ada."],
                    'moderate': ["😷 Kelompok sensitif sebaiknya mengurangi aktivitas luar ruang.",
                                 "🌿 Tingkatkan monitoring kualitas udara secara berkala."],
                    'bad':      ["⚠️ Gunakan masker saat beraktivitas di luar ruangan.",
                                 "🏭 Evaluasi sumber emisi industri dan transportasi."],
                    'very-bad': ["🚨 Hindari aktivitas luar ruangan yang tidak perlu.",
                                 "🏥 Pemerintah perlu mengambil tindakan darurat segera.",
                                 "😷 Penggunaan masker N95 sangat disarankan."]
                }
                for r in recs.get(css_class, []):
                    st.markdown(f"- {r}")
        else:
            st.markdown("""
            <div class="info-box">
                ℹ️ Pilih negara dan tahun di panel kiri, lalu klik tombol <b>Prediksi Sekarang</b>
                untuk melihat perkiraan nilai PM2.5 dan kluster kualitas udara.
            </div>
            """, unsafe_allow_html=True)

    # ── Grafik tren historis + prediksi ──
    if predict_btn and MODEL_READY:
        st.markdown("---")
        st.markdown("#### 📈 Tren PM2.5 Historis + Prediksi")

        hist_country = df_history[df_history['Country Name'] == country].sort_values('Year')

        # Kumpulkan semua titik prediksi
        all_pred_years, all_pred_vals = [], []
        pm_vals_iter = list(hist_country['PM2.5'].values)
        yr_vals_iter = list(hist_country['Year'].values)

        for yr in range(yr_vals_iter[-1] + 1, tahun + 1):
            lag1 = pm_vals_iter[-1]
            lag2 = pm_vals_iter[-2] if len(pm_vals_iter) >= 2 else lag1
            lag3 = pm_vals_iter[-3] if len(pm_vals_iter) >= 3 else lag2
            feat = pd.DataFrame([{
                'Year': yr,
                'PM2.5_lag1': lag1, 'PM2.5_lag2': lag2, 'PM2.5_lag3': lag3,
                'PM2.5_rollmean2': np.mean([lag1, lag2]),
                'PM2.5_diff1': pm_vals_iter[-1] - (pm_vals_iter[-2] if len(pm_vals_iter) >= 2 else pm_vals_iter[-1])
            }])
            p = max(0, round(rf_model.predict(feat)[0], 3))
            pm_vals_iter.append(p)
            yr_vals_iter.append(yr)
            all_pred_years.append(yr)
            all_pred_vals.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist_country['Year'], hist_country['PM2.5'],
                marker='o', color='steelblue', linewidth=2.2, label='Data Historis', markersize=6)
        if all_pred_years:
            ax.plot([hist_country['Year'].iloc[-1]] + all_pred_years,
                    [hist_country['PM2.5'].iloc[-1]] + all_pred_vals,
                    marker='s', color='coral', linewidth=2.2, linestyle='--',
                    label='Prediksi', markersize=6)
            ax.axvline(x=hist_country['Year'].max() + 0.5, color='gray',
                       linestyle=':', linewidth=1.5, alpha=0.7)
            ax.text(hist_country['Year'].max() + 0.6, ax.get_ylim()[1] * 0.95,
                    '← Historis | Prediksi →', fontsize=8, color='gray')

        ax.set_xlabel('Tahun')
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.set_title(f'Tren PM2.5 – {country}', fontsize=12, fontweight='bold')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Prediksi multi-negara ──
    st.markdown("---")
    with st.expander("🌐 Prediksi Beberapa Negara Sekaligus"):
        sel_countries = st.multiselect(
            "Pilih negara (maks 10)",
            options=countries_list,
            default=['Indonesia', 'India', 'Germany'] if all(
                c in countries_list for c in ['Indonesia', 'India', 'Germany']) else countries_list[:3],
            max_selections=10
        )
        sel_year = st.number_input("Tahun prediksi", min_value=min_year, max_value=min_year + 15,
                                   value=min_year + 2, step=1)

        if st.button("🔍 Bandingkan", use_container_width=True):
            rows_multi = []
            for c in sel_countries:
                pm, cid = predict_pm25(c, sel_year, df_history, rf_model, cluster_info)
                if pm:
                    lbl = cluster_info['label_map'][cid].split('–')[1].strip() if '–' in cluster_info['label_map'][cid] else cluster_info['label_map'][cid]
                    rows_multi.append({'Negara': c, 'PM2.5 Prediksi (µg/m³)': pm, 'Kluster': lbl})
            if rows_multi:
                df_multi = pd.DataFrame(rows_multi).sort_values('PM2.5 Prediksi (µg/m³)', ascending=False)
                st.dataframe(df_multi, use_container_width=True, hide_index=True)

                fig_m, ax_m = plt.subplots(figsize=(10, 4))
                colors_m = ['#e74c3c' if pm > cluster_info['thresholds'][2]
                             else '#e67e22' if pm > cluster_info['thresholds'][1]
                             else '#3498db' if pm > cluster_info['thresholds'][0]
                             else '#2ecc71'
                             for pm in df_multi['PM2.5 Prediksi (µg/m³)']]
                ax_m.barh(df_multi['Negara'], df_multi['PM2.5 Prediksi (µg/m³)'], color=colors_m, edgecolor='white')
                ax_m.set_xlabel('PM2.5 Prediksi (µg/m³)')
                ax_m.set_title(f'Perbandingan PM2.5 Prediksi Tahun {sel_year}', fontweight='bold')
                for i, v in enumerate(df_multi['PM2.5 Prediksi (µg/m³)']):
                    ax_m.text(v + 0.2, i, f'{v:.2f}', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_m)
                plt.close()


# HALAMAN: DASHBOARD DATA
elif "📊 Dashboard Data" in page:
    if not MODEL_READY:
        st.warning("⚠️ Data belum dimuat.")
        st.stop()

    st.markdown("### 📊 Dashboard Data PM2.5 Global (2010–2017)")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌐 Jumlah Negara", df_history['Country Name'].nunique())
    with col2:
        st.metric("📅 Rentang Tahun", f"{int(df_history['Year'].min())}–{int(df_history['Year'].max())}")
    with col3:
        st.metric("📈 Rata-rata PM2.5", f"{df_history['PM2.5'].mean():.2f} µg/m³")
    with col4:
        st.metric("📊 Median PM2.5", f"{df_history['PM2.5'].median():.2f} µg/m³")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        # Distribusi PM2.5
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(df_history['PM2.5'], bins=25, color='steelblue', edgecolor='white', alpha=0.85)
        ax1.set_title('Distribusi Nilai PM2.5 (Seluruh Data)', fontweight='bold')
        ax1.set_xlabel('PM2.5 (µg/m³)')
        ax1.set_ylabel('Frekuensi')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col_r:
        # Rata-rata per tahun
        avg_yr = df_history.groupby('Year')['PM2.5'].mean()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(avg_yr.index, avg_yr.values, marker='o', color='coral', linewidth=2, markersize=7)
        ax2.set_title('Rata-rata PM2.5 Global per Tahun', fontweight='bold')
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Rata-rata PM2.5 (µg/m³)')
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    # Top 20 negara
    top20 = df_history.groupby('Country Name')['PM2.5'].mean().nlargest(20).sort_values()
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors_top = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top20)))
    bars = ax3.barh(top20.index, top20.values, color=colors_top, edgecolor='white')
    ax3.set_title('Top 20 Negara PM2.5 Rata-rata Tertinggi (2010–2017)', fontweight='bold')
    ax3.set_xlabel('Rata-rata PM2.5 (µg/m³)')
    for bar, v in zip(bars, top20.values):
        ax3.text(v + 0.2, bar.get_y() + bar.get_height()/2, f'{v:.1f}', va='center', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")
    st.markdown("#### 🔍 Eksplorasi Data")
    search_country = st.selectbox("Pilih negara untuk dilihat datanya:", options=sorted(df_history['Country Name'].unique()))
    st.dataframe(df_history[df_history['Country Name'] == search_country].set_index('Year').drop(columns=['Country Name'], errors='ignore'),
                 use_container_width=True)


# HALAMAN: ANALISIS KLUSTER
elif "📈 Analisis Kluster" in page:
    if not MODEL_READY:
        st.warning("⚠️ Data kluster belum dimuat.")
        st.stop()

    st.markdown("### 📈 Analisis Kluster K-Means")

    ci = cluster_info
    thresholds = ci['thresholds']
    label_map  = ci['label_map']
    order      = ci['order']
    colors     = ci.get('cluster_colors', ['#2ecc71', '#3498db', '#e67e22', '#e74c3c'])

    # Tabel threshold kluster
    st.markdown("#### 🗂️ Definisi Kluster")
    rows_tbl = [
        {"Kluster": label_map[order[0]], "Batas PM2.5": f"≤ {thresholds[0]:.1f} µg/m³", "Kategori WHO": "Good"},
        {"Kluster": label_map[order[1]], "Batas PM2.5": f"{thresholds[0]:.1f} – {thresholds[1]:.1f} µg/m³", "Kategori WHO": "Moderate"},
        {"Kluster": label_map[order[2]], "Batas PM2.5": f"{thresholds[1]:.1f} – {thresholds[2]:.1f} µg/m³", "Kategori WHO": "Unhealthy"},
        {"Kluster": label_map[order[3]], "Batas PM2.5": f"> {thresholds[2]:.1f} µg/m³", "Kategori WHO": "Hazardous"},
    ]
    st.dataframe(pd.DataFrame(rows_tbl), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Anggota kluster
    if 'Cluster' in df_history.columns:
        st.markdown("#### 🌐 Negara per Kluster")
        tabs = st.tabs([label_map[k].split('–')[1].strip() if '–' in label_map[k] else label_map[k] for k in order])
        for tab, cid in zip(tabs, order):
            with tab:
                members = df_history[df_history['Cluster'] == cid]['Country Name'].unique().tolist()
                cols_tab = st.columns(3)
                for i, m in enumerate(sorted(members)):
                    cols_tab[i % 3].write(f"🌍 {m}")

    st.markdown("---")
    st.markdown("#### ℹ️ Tentang K-Means Clustering")
    st.info("""
    **K-Means Clustering** mengelompokkan negara-negara berdasarkan **pola nilai PM2.5 selama 2010–2017**.

    - **K = 4** dipilih berdasarkan **Elbow Method** (titik siku pada grafik inertia)
    - Setiap kluster memiliki centroid yang mewakili profil polusi rata-rata kelompoknya
    - Batas antar kluster dihitung sebagai **midpoint** antara dua centroid yang berdekatan
    - Hasil prediksi PM2.5 kemudian dipetakan ke kluster menggunakan batas ini
    """)


# HALAMAN: About
elif "ℹ️ Tentang" in page:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini merupakan implementasi proyek **Data Mining** yang mengintegrasikan
    dua metode utama untuk analisis kualitas udara global:

    ---

    #### 🔬 Metode yang Digunakan

    **1. K-Means Clustering (Unsupervised Learning)**
    - Mengelompokkan negara berdasarkan pola PM2.5 historis (2010–2017)
    - Menghasilkan 4 kluster dengan tingkat polusi berbeda
    - K optimal ditentukan menggunakan Elbow Method

    **2. Random Forest Regressor (Supervised Learning)**
    - Memprediksi nilai PM2.5 suatu negara di tahun mendatang
    - Menggunakan fitur: lag features (tahun-1, -2, -3), rolling mean, dan tren
    - Prediksi bersifat iteratif untuk tahun-tahun yang jauh dari data historis

    ---

    #### 📦 Dataset
    | Item | Detail |
    |------|--------|
    | Nama | PM2.5 Global Air Pollution 2010–2017 |
    | Sumber | Kaggle (World Bank) |
    | Cakupan | 240+ negara |
    | Fitur | Nilai PM2.5 tahunan per negara |
    | Link | https://www.kaggle.com/datasets/kweinmeister/pm25-global-air-pollution-20102017 |

    ---

    #### 🔄 Alur Kerja
    ```
    Dataset → Preprocessing → K-Means Clustering → Labeling Kluster
                                                          ↓
                                         Random Forest Regressor
                                                          ↓
                                   Input Tahun → Prediksi PM2.5 → Kluster
    ```

    ---

    #### 📁 File yang Diperlukan
    - `rf_model.pkl` — model Random Forest terlatih
    - `cluster_info.json` — threshold dan label kluster
    - `df_history.csv` — data historis PM2.5

    > Semua file dihasilkan otomatis dari notebook `air_pollution_datamining.ipynb`
    """)
