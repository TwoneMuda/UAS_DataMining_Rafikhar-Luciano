"""Halaman utama (Home) aplikasi Streamlit.
Jalankan dari folder app/:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Prediksi Kualitas Udara PM2.5",
    page_icon="\U0001F32B\uFE0F",
    layout="wide",
)

st.title("\U0001F32B\uFE0F Prediksi & Segmentasi Kualitas Udara (PM2.5)")
st.subheader("Proyek UAS Data Mining")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(
        """
        Aplikasi ini memprediksi konsentrasi **PM2.5** (partikel halus di udara)
        berdasarkan kondisi cuaca, sekaligus mengelompokkan kondisi udara
        menggunakan **clustering**.

        ### Metode Data Mining
        - **Regresi** \u2014 memprediksi nilai PM2.5 (ug/m\u00b3) dari faktor cuaca.
        - **Clustering (K-Means)** \u2014 mengelompokkan kondisi udara menjadi
          beberapa segmen (mis. *Udara Bersih*, *Polusi Tinggi*).

        ### Dataset
        **UCI Beijing PM2.5 Data Set** (data per jam 2010\u20132014).

        ### Cara pakai
        Gunakan menu di sidebar kiri:
        1. **Dataset Overview** \u2014 ringkasan & statistik data.
        2. **Prediction** \u2014 masukkan kondisi cuaca untuk memprediksi PM2.5.
        3. **Visualization** \u2014 grafik pendukung & hasil analisis.
        4. **About** \u2014 penjelasan metode & proyek.
        """
    )
with col2:
    st.info(
        "**Identitas**\n\n"
        "Nama: RAFIKHAR LUCIANO\n\n"
        "Mata Kuliah: Data Mining\n\n"
        "Tugas: Proyek Akhir (UAS)\n\n"
        "Framework: CRISP-DM"
    )
    st.metric("Metode", "Regresi + Clustering")

st.divider()
st.caption(
    "Catatan: jalankan train_model.py terlebih dahulu untuk menghasilkan model, "
    "dan pastikan dataset asli ada di folder dataset/."
)
