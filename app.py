import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FUNGSI UNTUK MEMUAT MODEL DAN PREPROCESSING OBJECTS ---
@st.cache_resource
def load_model_assets():
    """Memuat model, scaler, dan label encoders yang telah dilatih."""
    try:
        model = joblib.load('best_model_gradient_boosting.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("File model atau file pendukung tidak ditemukan.")
        st.info("Pastikan file 'best_model_gradient_boosting.pkl', 'scaler.pkl', dan 'label_encoders.pkl' berada di direktori yang sama dengan 'app.py'")
        return None, None, None

# Memuat aset
model, scaler, label_encoders = load_model_assets()

# --- FUNGSI UNTUK TAMPILAN DAN LOGIKA ---

def show_prediction_page():
    """Menampilkan halaman utama untuk input dan prediksi."""
    st.title("Prediksi Tingkat Obesitas 🩺")
    st.markdown("Isi formulir di bawah ini dengan data yang relevan untuk memprediksi tingkat obesitas.")

    # Membuat layout dengan 2 kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Informasi Pribadi & Keluarga")
        age = st.number_input('**Usia**', min_value=1, max_value=100, value=25)
        gender = st.selectbox('**Jenis Kelamin**', ['Male', 'Female'])
        family_history = st.selectbox('**Riwayat obesitas dalam keluarga?**', ['yes', 'no'])
        height = st.slider('**Tinggi Badan (cm)**', 140.0, 200.0, 170.0, 0.5)
        weight = st.slider('**Berat Badan (kg)**', 40.0, 180.0, 70.0, 0.5)

    with col2:
        st.subheader("Kebiasaan Gaya Hidup")
        favc = st.selectbox('**Sering makan makanan tinggi kalori (FAVC)?**', ['yes', 'no'])
        fcvc = st.slider('**Frekuensi makan sayur (FCVC)**', 1.0, 3.0, 2.0, 0.5, help="1: Tidak pernah, 2: Kadang-kadang, 3: Selalu")
        ncp = st.slider('**Jumlah makan utama per hari (NCP)**', 1.0, 4.0, 3.0, 1.0)
        scc = st.selectbox('**Apakah Anda memonitor kalori (SCC)?**', ['yes', 'no'])
        smoke = st.selectbox('**Apakah Anda merokok (SMOKE)?**', ['yes', 'no'])
        ch2o = st.slider('**Konsumsi air per hari (Liter) (CH2O)**', 1.0, 3.0, 2.0, 0.5)

    st.subheader("Aktivitas Fisik & Transportasi")
    col3, col4 = st.columns(2)

    with col3:
        faf = st.slider('**Frekuensi aktivitas fisik per minggu (FAF)**', 0.0, 3.0, 1.0, 0.5, help="0: Tidak ada, 1: 1-2 hari, 2: 2-4 hari, 3: 4-5 hari")
        tue = st.slider('**Waktu penggunaan gawai per hari (TUE)**', 0.0, 2.0, 1.0, 0.5, help="0: 0-2 jam, 1: 3-5 jam, 2: >5 jam")
    with col4:
        caec = st.selectbox('**Makan di antara waktu makan utama (CAEC)?**', ['no', 'Sometimes', 'Frequently', 'Always'])
        calc = st.selectbox('**Konsumsi alkohol (CALC)?**', ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox('**Transportasi yang digunakan (MTRANS)**', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])

    # Tombol untuk prediksi
    if st.button('**Prediksi Sekarang!**', type="primary", use_container_width=True):
        if model and scaler and label_encoders:
            input_data = pd.DataFrame({
                'Gender': [gender], 'Age': [age], 'Height': [height / 100.0], 'Weight': [weight],
                'family_history_with_overweight': [family_history], 'FAVC': [favc],
                'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke], 'CH2O': [ch2o],
                'SCC': [scc], 'FAF': [faf], 'TUE': [tue], 'CALC': [calc], 'MTRANS': [mtrans]
            })

            # Preprocessing
            numerical_cols = scaler.feature_names_in_
            categorical_cols = [col for col in label_encoders if col != 'NObeyesdad' and col in input_data.columns]

            for col in categorical_cols:
                le = label_encoders[col]
                try:
                    input_data[col] = le.transform(input_data[col])
                except ValueError:
                    input_data[col] = -1

            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

            # === PERBAIKAN UTAMA ADA DI SINI ===
            # Mengatur urutan kolom agar sama persis dengan urutan saat training
            # Kita ambil urutan dari scaler, bukan dari model.
            feature_order = scaler.feature_names_in_
            input_data = input_data[feature_order]

            # Prediksi
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Mengambil nama kelas dari hasil prediksi
            target_encoder = label_encoders['NObeyesdad']
            prediction_class = target_encoder.inverse_transform(prediction)[0]
            
            # Menampilkan hasil
            st.success(f"**Hasil Prediksi: {prediction_class.replace('_', ' ')}**")

            st.write("**Probabilitas Prediksi:**")
            proba_df = pd.DataFrame(
                prediction_proba,
                columns=target_encoder.classes_,
                index=['Probabilitas']
            )
            st.dataframe(proba_df.style.format("{:.2%}"))

def show_about_page():
    """Menampilkan halaman tentang proyek."""
    st.title("Tentang Proyek Ini 📖")
    st.markdown("""
    Aplikasi ini adalah implementasi dari model Machine Learning untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan, kondisi fisik, dan gaya hidup seseorang.

    **Tujuan:**
    Memberikan alat bantu sederhana untuk meningkatkan kesadaran akan risiko obesitas dan mendorong gaya hidup yang lebih sehat.

    **Model:**
    Model yang digunakan adalah **Gradient Boosting**, yang telah dilatih dan dioptimalkan untuk memberikan akurasi terbaik berdasarkan dataset yang tersedia.

    **Dataset:**
    Model ini dilatih menggunakan dataset *'Estimation of Obesity Levels Based on Eating Habits and Physical Condition'* dari UCI Machine Learning Repository.

    **Disclaimer:**
    Hasil prediksi dari aplikasi ini tidak boleh dianggap sebagai diagnosis medis. Selalu konsultasikan dengan tenaga medis profesional untuk evaluasi kesehatan yang akurat.
    """)

# --- Sidebar dan Navigasi Halaman ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Prediksi Obesitas", "Tentang Proyek"])

if page == "Prediksi Obesitas":
    show_prediction_page()
elif page == "Tentang Proyek":
    show_about_page()

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi dikembangkan untuk memenuhi tugas deployment model Machine Learning.")