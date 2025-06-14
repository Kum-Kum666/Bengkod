import streamlit as st
import pandas as pd
import numpy as np
import joblib
from urllib.request import urlopen

# Set up a title and a description for the app
st.set_page_config(page_title="Prediksi Tingkat Obesitas", layout="wide")
st.title("Prediksi Tingkat Obesitas 🧑‍⚕️")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning* untuk memprediksi tingkat obesitas berdasarkan kebiasaan gaya hidup dan atribut fisik.
Silakan masukkan data Anda pada *sidebar* di sebelah kiri.
""")

# --- Function to load model and encoders from GitHub ---
@st.cache_resource
def load_model():
    """Load the model, scaler, and encoders from GitHub."""
    try:
        # Define URLs for the files in your GitHub repository (make sure it's a raw link)
        base_url = "https://github.com/Kum-Kum666/Bengkod.git"
        
        return model, scaler, label_encoders
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None, None

model, scaler, label_encoders = load_model()

# --- Sidebar for User Input ---
st.sidebar.header("Masukkan Data Anda di Sini")

def user_input_features():
    """Create sidebar inputs for user data."""
    # Define categorical options based on the notebook
    gender_options = ['Male', 'Female']
    yes_no_options = ['no', 'yes']
    caec_options = ['no', 'Sometimes', 'Frequently', 'Always']
    calc_options = ['no', 'Sometimes', 'Frequently', 'Always']
    mtrans_options = ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike']

    # Create inputs
    Gender = st.sidebar.selectbox("Jenis Kelamin", options=gender_options)
    Age = st.sidebar.slider("Usia", 14, 65, 25)
    Height = st.sidebar.slider("Tinggi Badan (meter)", 1.40, 2.00, 1.70, 0.01)
    Weight = st.sidebar.slider("Berat Badan (kg)", 35, 180, 70)
    family_history_with_overweight = st.sidebar.selectbox("Riwayat keluarga dengan overweight?", options=yes_no_options)
    FAVC = st.sidebar.selectbox("Sering mengonsumsi makanan tinggi kalori (FAVC)?", options=yes_no_options)
    FCVC = st.sidebar.slider("Frekuensi makan sayur (FCVC)", 1, 3, 2)
    NCP = st.sidebar.slider("Jumlah makanan utama per hari (NCP)", 1, 4, 3)
    CAEC = st.sidebar.selectbox("Konsumsi makanan di antara waktu makan (CAEC)?", options=caec_options)
    SMOKE = st.sidebar.selectbox("Apakah Anda merokok (SMOKE)?", options=yes_no_options)
    CH2O = st.sidebar.slider("Konsumsi air per hari (liter) (CH2O)", 1, 3, 2)
    SCC = st.sidebar.selectbox("Memantau konsumsi kalori (SCC)?", options=yes_no_options)
    FAF = st.sidebar.slider("Frekuensi aktivitas fisik (hari per minggu) (FAF)", 0, 3, 1)
    TUE = st.sidebar.slider("Waktu penggunaan gawai (jam per hari) (TUE)", 0, 2, 1)
    CALC = st.sidebar.selectbox("Konsumsi alkohol (CALC)?", options=calc_options)
    MTRANS = st.sidebar.selectbox("Moda transportasi utama (MTRANS)", options=mtrans_options)
    
    # Create a dictionary of the inputs
    data = {
        'Gender': Gender, 'Age': Age, 'Height': Height, 'Weight': Weight,
        'family_history_with_overweight': family_history_with_overweight,
        'FAVC': FAVC, 'FCVC': FCVC, 'NCP': NCP, 'CAEC': CAEC, 'SMOKE': SMOKE,
        'CH2O': CH2O, 'SCC': SCC, 'FAF': FAF, 'TUE': TUE, 'CALC': CALC, 'MTRANS': MTRANS
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

if model and scaler and label_encoders:
    input_df = user_input_features()

    # --- Preprocessing and Prediction ---
    # Display user input
    st.subheader("Data yang Anda Masukkan:")
    st.write(input_df)

    # Separate numerical and categorical columns from the input
    numerical_cols = input_df.select_dtypes(include=np.number).columns
    categorical_cols = input_df.select_dtypes(include='object').columns

    # Apply label encoding to categorical features
    input_df_encoded = input_df.copy()
    for col in categorical_cols:
        le = label_encoders[col]
        input_df_encoded[col] = le.transform(input_df_encoded[col])
    
    # Apply standard scaling to numerical features
    input_df_encoded[numerical_cols] = scaler.transform(input_df_encoded[numerical_cols])

    # Reorder columns to match the training order if necessary
    # (Assuming the order from the notebook is consistent)
    
    # --- Make Prediction ---
    if st.button("Prediksi Sekarang!"):
        prediction = model.predict(input_df_encoded)
        
        # Decode the prediction back to the original label
        target_encoder = label_encoders['NObeyesdad']
        prediction_label = target_encoder.inverse_transform(prediction)[0]
        
        st.subheader("Hasil Prediksi")
        st.success(f"Berdasarkan data Anda, tingkat obesitas yang diprediksi adalah: **{prediction_label}**")
else:
    st.warning("Model belum berhasil dimuat. Pastikan file model ada di repository GitHub.")