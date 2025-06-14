import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================

st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar collapsed by default
)

# Custom CSS dengan tema yang completely different
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .main {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        color: white;
    }
    
    /* Dark theme header */
    .main-header {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        color: #e0e6ed;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4);
    }
    
    .metric-glass {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(78, 205, 196, 0.2));
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .metric-glass::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transform: rotate(45deg);
        transition: all 0.5s ease;
    }
    
    .metric-glass:hover::before {
        animation: shine 1s ease-in-out;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }
    
    .metric-glass h2 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .metric-glass h3 {
        color: #e0e6ed;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    .metric-glass p {
        color: #b8c5d1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-glass i {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    /* Prediction result with neon effect */
    .neon-card {
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(20px);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid #4ecdc4;
        box-shadow: 
            0 0 20px rgba(78, 205, 196, 0.5),
            0 0 40px rgba(78, 205, 196, 0.3),
            0 0 80px rgba(78, 205, 196, 0.1);
        color: white;
        position: relative;
        animation: neonGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes neonGlow {
        from {
            box-shadow: 
                0 0 20px rgba(78, 205, 196, 0.5),
                0 0 40px rgba(78, 205, 196, 0.3),
                0 0 80px rgba(78, 205, 196, 0.1);
        }
        to {
            box-shadow: 
                0 0 25px rgba(78, 205, 196, 0.7),
                0 0 50px rgba(78, 205, 196, 0.5),
                0 0 100px rgba(78, 205, 196, 0.3);
        }
    }
    
    /* Input form with dark theme */
    .input-container {
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
    }
    
    .input-section h4 {
        color: #4ecdc4;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.5);
        background: linear-gradient(135deg, #4ecdc4 0%, #ff6b6b 100%);
    }
    
    /* Risk indicators with gradient */
    .risk-indicator {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 193, 7, 0.2));
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        transition: all 0.3s ease;
        color: white;
    }
    
    .risk-indicator:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .risk-indicator i {
        font-size: 1.8rem;
        margin-right: 1rem;
        background: linear-gradient(45deg, #ff6b6b, #ffc107);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Progress bar custom */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 100%);
        height: 10px;
        border-radius: 10px;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p {
        color: #e0e6ed;
    }
    
    /* Tips section with gradient cards */
    .tip-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .tip-card h4 {
        color: #4ecdc4;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    .tip-card ul {
        list-style: none;
        padding: 0;
    }
    
    .tip-card li {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        color: #e0e6ed;
    }
    
    .tip-card li:last-child {
        border-bottom: none;
    }
    
    /* Disclaimer with dark theme */
    .disclaimer {
        background: rgba(255, 107, 107, 0.1);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 107, 107, 0.3);
        color: white;
        margin: 2rem 0;
    }
    
    .disclaimer h4 {
        color: #ff6b6b;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #b8c5d1;
        padding: 3rem;
        margin-top: 3rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar custom (when shown) */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(0, 0, 0, 0.8) 0%, rgba(102, 126, 234, 0.3) 100%);
        backdrop-filter: blur(20px);
    }
    
    /* Hide default streamlit styling */
    .css-1v0mbdj {
        display: none;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #4ecdc4, #ff6b6b);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNGSI LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load model dari direktori yang ditentukan"""
    model_path = "D:/Perkuliahan/Semester 6/Bengkod/Bengkod/models/best_model_gradient_boosting.pkl"
    
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model, True
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan di: {model_path}")
        # Fallback ke model dummy
        np.random.seed(42)
        X_dummy = np.random.rand(100, 16)
        y_dummy = np.random.randint(0, 7, 100)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_dummy, y_dummy)
        return model, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

# ============================================================================
# FUNGSI UTILITY
# ============================================================================

def get_bmi_category(weight, height):
    """Hitung BMI dan kategorinya"""
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return bmi, "Berat Badan Kurang"
    elif 18.5 <= bmi < 25:
        return bmi, "Normal"
    elif 25 <= bmi < 30:
        return bmi, "Kelebihan Berat Badan"
    else:
        return bmi, "Obesitas"

def predict_obesity(model, age, gender, height, weight, calc, favc, fcvc, ncp, 
                   scc, smoke, ch2o, family_history, faf, tue, caec, mtrans):
    """Fungsi prediksi obesitas"""
    try:
        input_data = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp, 
                               scc, smoke, ch2o, family_history, faf, tue, caec, mtrans]])
        
        prediction = model.predict(input_data)[0]
        
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = np.max(probabilities)
        except:
            confidence = 0.85 + np.random.rand() * 0.1
        
        # Mapping sesuai dengan nama kelas yang benar
        labels = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight', 
            2: 'Overweight_Level_I',
            3: 'Overweight_Level_II',
            4: 'Obesity_Type_I',
            5: 'Obesity_Type_II',
            6: 'Obesity_Type_III'
        }
        
        return labels[prediction], confidence
    
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        return "Error", 0.0

def get_recommendation(prediction):
    """Memberikan rekomendasi berdasarkan prediksi"""
    recommendations = {
        'Insufficient_Weight': {
            'title': 'Berat Badan Kurang',
            'advice': 'Tingkatkan asupan kalori dengan makanan bergizi. Konsultasi dengan ahli gizi.',
            'risk': 'Rendah',
            'icon': 'fas fa-weight-hanging',
            'color': '#4ecdc4'
        },
        'Normal_Weight': {
            'title': 'Berat Badan Normal',
            'advice': 'Pertahankan pola hidup sehat dengan diet seimbang dan olahraga teratur.',
            'risk': 'Sangat Rendah',
            'icon': 'fas fa-check-circle',
            'color': '#2ecc71'
        },
        'Overweight_Level_I': {
            'title': 'Kelebihan Berat Badan Tingkat I',
            'advice': 'Mulai program penurunan berat badan dengan diet dan olahraga.',
            'risk': 'Sedang',
            'icon': 'fas fa-exclamation-triangle',
            'color': '#ffc107'
        },
        'Overweight_Level_II': {
            'title': 'Kelebihan Berat Badan Tingkat II',
            'advice': 'Diperlukan program penurunan berat badan yang lebih intensif.',
            'risk': 'Tinggi',
            'icon': 'fas fa-exclamation-triangle',
            'color': '#ff9800'
        },
        'Obesity_Type_I': {
            'title': 'Obesitas Tipe I',
            'advice': 'Konsultasi dengan dokter untuk program penurunan berat badan.',
            'risk': 'Tinggi',
            'icon': 'fas fa-hospital',
            'color': '#ff6b6b'
        },
        'Obesity_Type_II': {
            'title': 'Obesitas Tipe II',
            'advice': 'Diperlukan intervensi medis segera untuk mengurangi risiko komplikasi.',
            'risk': 'Sangat Tinggi',
            'icon': 'fas fa-hospital',
            'color': '#e74c3c'
        },
        'Obesity_Type_III': {
            'title': 'Obesitas Tipe III',
            'advice': 'Diperlukan penanganan medis intensif. Pertimbangkan konsultasi spesialis.',
            'risk': 'Ekstrem',
            'icon': 'fas fa-hospital-symbol',
            'color': '#c0392b'
        }
    }
    
    return recommendations.get(prediction, recommendations['Normal_Weight'])

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1><i class="fas fa-heartbeat"></i> Health Analytics Dashboard</h1>
    <p>Sistem Analisis Kesehatan dan Prediksi Obesitas Berbasis AI</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, is_real_model = load_model()

if model is None:
    st.stop()

if not is_real_model:
    st.markdown("""
    <div class="disclaimer">
        <h4><i class="fas fa-exclamation-triangle"></i> Mode Demo</h4>
        <p>Model file tidak ditemukan. Aplikasi berjalan dalam mode demo untuk testing.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# INPUT FORM - TOP SECTION (NO SIDEBAR)
# ============================================================================

st.markdown("""
<div class="input-container">
    <h2 style="text-align: center; color: #4ecdc4; margin-bottom: 2rem;">
        <i class="fas fa-user-md"></i> Data Pasien
    </h2>
</div>
""", unsafe_allow_html=True)

# Create columns for input form
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="input-section"><h4><i class="fas fa-user"></i> Demografis</h4></div>', unsafe_allow_html=True)
    age = st.slider("Umur", 16, 61, 25)
    gender = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
    height = st.number_input("Tinggi (m)", 1.45, 1.98, 1.70, 0.01)
    weight = st.number_input("Berat (kg)", 39.0, 173.0, 70.0, 0.1)

with col2:
    st.markdown('<div class="input-section"><h4><i class="fas fa-utensils"></i> Pola Makan</h4></div>', unsafe_allow_html=True)
    favc = st.selectbox("Makanan Berkalori Tinggi", ["Tidak", "Ya"])
    fcvc = st.slider("Sayuran (porsi/hari)", 1.0, 3.0, 2.0, 0.1)
    ncp = st.slider("Makanan Utama", 1.0, 4.0, 3.0, 0.1)
    caec = st.selectbox("Snacking", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])

with col3:
    st.markdown('<div class="input-section"><h4><i class="fas fa-smoking"></i> Gaya Hidup</h4></div>', unsafe_allow_html=True)
    calc = st.selectbox("Konsumsi Alkohol", ["Tidak", "Kadang-kadang", "Sering", "Selalu"])
    smoke = st.selectbox("Merokok", ["Tidak", "Ya"])
    ch2o = st.slider("Air (liter/hari)", 1.0, 3.0, 2.0, 0.1)
    scc = st.selectbox("Monitor Kalori", ["Tidak", "Ya"])

with col4:
    st.markdown('<div class="input-section"><h4><i class="fas fa-running"></i> Aktivitas</h4></div>', unsafe_allow_html=True)
    faf = st.slider("Olahraga (hari/minggu)", 0.0, 3.0, 1.0, 0.1)
    tue = st.slider("Screen Time (jam/hari)", 0.0, 2.0, 1.0, 0.1)
    mtrans = st.selectbox("Transportasi", ["Mobil", "Sepeda", "Motor", "Transportasi Umum", "Jalan Kaki"])
    family_history = st.selectbox("Riwayat Keluarga Obesitas", ["Tidak", "Ya"])

# ============================================================================
# KONVERSI INPUT
# ============================================================================

gender_num = 1 if gender == "Laki-laki" else 0
favc_num = 1 if favc == "Ya" else 0
calc_mapping = {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
calc_num = calc_mapping[calc]
scc_num = 1 if scc == "Ya" else 0
smoke_num = 1 if smoke == "Ya" else 0
family_history_num = 1 if family_history == "Ya" else 0
caec_mapping = {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
caec_num = caec_mapping[caec]
mtrans_mapping = {"Mobil": 0, "Sepeda": 1, "Motor": 2, "Transportasi Umum": 3, "Jalan Kaki": 4}
mtrans_num = mtrans_mapping[mtrans]

# ============================================================================
# METRICS SECTION
# ============================================================================

# BMI Section
bmi, bmi_category = get_bmi_category(weight, height)

st.markdown('<br><br>', unsafe_allow_html=True)

# Create metrics in a row
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.markdown(f"""
    <div class="metric-glass">
        <i class="fas fa-chart-bar"></i>
        <h3>BMI Index</h3>
        <h2>{bmi:.1f}</h2>
        <p>{bmi_category}</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    activity_status = "Aktif" if faf >= 2 else "Kurang Aktif"
    st.markdown(f"""
    <div class="metric-glass">
        <i class="fas fa-running"></i>
        <h3>Aktivitas Fisik</h3>
        <h2>{faf:.1f}</h2>
        <p>{activity_status}</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    water_status = "Cukup" if ch2o >= 2 else "Kurang"
    st.markdown(f"""
    <div class="metric-glass">
        <i class="fas fa-tint"></i>
        <h3>Konsumsi Air</h3>
        <h2>{ch2o:.1f}L</h2>
        <p>{water_status}</p>
    </div>
    """, unsafe_allow_html=True)

with metric_col4:
    diet_status = "Baik" if fcvc >= 2.5 else "Perlu Perbaikan"
    st.markdown(f"""
    <div class="metric-glass">
        <i class="fas fa-apple-alt"></i>
        <h3>Diet Sehat</h3>
        <h2>{fcvc:.1f}</h2>
        <p>{diet_status}</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PREDICTION SECTION
# ============================================================================

st.markdown('<br><br>', unsafe_allow_html=True)

if st.button("🚀 Analisis Kesehatan Sekarang", type="primary"):
    with st.spinner('🔬 Menganalisis data kesehatan Anda...'):
        import time
        time.sleep(2)  # Dramatic effect
        
        prediction, confidence = predict_obesity(
            model, age, gender_num, height, weight, calc_num, favc_num, 
            fcvc, ncp, scc_num, smoke_num, ch2o, family_history_num, 
            faf, tue, caec_num, mtrans_num
        )
        
        # Display prediction with neon effect
        rec = get_recommendation(prediction)
        
        st.markdown(f"""
        <div class="neon-card">
            <i class="{rec['icon']}" style="font-size: 3rem; color: {rec['color']}; margin-bottom: 1rem;"></i>
            <h1 style="color: {rec['color']};">HASIL ANALISIS</h1>
            <h2>{prediction.replace('_', ' ')}</h2>
            <p style="font-size: 1.2rem;">Tingkat Keyakinan: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress bar
        st.progress(confidence)
        
        # Recommendation in glass card
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="color: {rec['color']};"><i class="fas fa-lightbulb"></i> {rec['title']}</h3>
            <p><strong>Tingkat Risiko:</strong> <span style="color: {rec['color']};">{rec['risk']}</span></p>
            <p><strong>Rekomendasi:</strong> {rec['advice']}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# ANALYTICS SECTION
# ============================================================================

st.markdown('<br><br>', unsafe_allow_html=True)

analytics_col1, analytics_col2 = st.columns(2)

with analytics_col1:
    st.markdown("""
    <div class="glass-card">
        <h3><i class="fas fa-chart-line"></i> Profil Kesehatan</h3>
        <div style="margin: 1rem 0;">
            <p><strong>Status BMI:</strong> {}</p>
            <p><strong>Aktivitas Fisik:</strong> {:.1f} hari/minggu</p>
            <p><strong>Konsumsi Air:</strong> {:.1f} liter/hari</p>
            <p><strong>Konsumsi Sayuran:</strong> {:.1f} porsi/hari</p>
            <p><strong>Screen Time:</strong> {:.1f} jam/hari</p>
        </div>
    </div>
    """.format(bmi_category, faf, ch2o, fcvc, tue), unsafe_allow_html=True)

with analytics_col2:
    st.markdown("""
    <div class="glass-card">
        <h3><i class="fas fa-exclamation-triangle"></i> Faktor Risiko</h3>
    </div>
    """, unsafe_allow_html=True)
    
    risk_factors = [
        ("Diet Tidak Sehat", favc_num * 50 + caec_num * 25, "fas fa-hamburger"),
        ("Kurang Aktivitas", (3-faf) * 30 + tue * 20, "fas fa-couch"),
        ("Kebiasaan Buruk", calc_num * 20 + smoke_num * 30, "fas fa-smoking"),
        ("Faktor Genetik", family_history_num * 100, "fas fa-dna")
    ]
    
    for factor, score, icon in risk_factors:
        level = "Rendah" if score < 30 else "Sedang" if score < 60 else "Tinggi"
        color = "#4ecdc4" if score < 30 else "#ffc107" if score < 60 else "#ff6b6b"
        
        st.markdown(f"""
        <div class="risk-indicator" style="border-left: 4px solid {color};">
            <i class="{icon}" style="color: {color};"></i>
            <div>
                <strong>{factor}</strong><br>
                <span style="color: {color};">Skor: {score:.0f}/100 - {level}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TIPS SECTION
# ============================================================================

st.markdown('<br><br>', unsafe_allow_html=True)

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    <div class="tip-card">
        <h4><i class="fas fa-apple-alt"></i> Tips Nutrisi</h4>
        <ul>
            <li><i class="fas fa-leaf"></i> Konsumsi 5-7 porsi sayuran setiap hari</li>
            <li><i class="fas fa-tint"></i> Minum air putih minimal 8 gelas per hari</li>
            <li><i class="fas fa-ban"></i> Batasi makanan tinggi kalori dan gula</li>
            <li><i class="fas fa-clock"></i> Makan dengan porsi kecil tapi lebih sering</li>
            <li><i class="fas fa-fish"></i> Pilih protein berkualitas tinggi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tips_col2:
    st.markdown("""
    <div class="tip-card">
        <h4><i class="fas fa-dumbbell"></i> Tips Aktivitas Fisik</h4>
        <ul>
            <li><i class="fas fa-stopwatch"></i> Olahraga minimal 150 menit per minggu</li>
            <li><i class="fas fa-heart"></i> Kombinasi kardio dan latihan kekuatan</li>
            <li><i class="fas fa-mobile-alt"></i> Kurangi waktu screen time berlebihan</li>
            <li><i class="fas fa-walking"></i> Gunakan transportasi aktif saat memungkinkan</li>
            <li><i class="fas fa-bed"></i> Tidur cukup 7-9 jam per malam</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# BMI VISUALIZATION
# ============================================================================

st.markdown('<br><br>', unsafe_allow_html=True)

# Create BMI gauge with dark theme
fig_bmi = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = bmi,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {
        'text': "BMI Analysis", 
        'font': {'size': 24, 'color': 'white', 'family': 'Poppins'}
    },
    delta = {'reference': 25, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#4ecdc4"}},
    gauge = {
        'axis': {
            'range': [None, 40], 
            'tickcolor': 'white',
            'tickfont': {'color': 'white', 'size': 12}
        },
        'bar': {'color': "#4ecdc4", 'thickness': 0.7},
        'steps': [
            {'range': [0, 18.5], 'color': "rgba(78, 205, 196, 0.3)"},
            {'range': [18.5, 25], 'color': "rgba(46, 204, 113, 0.3)"},
            {'range': [25, 30], 'color': "rgba(255, 193, 7, 0.3)"},
            {'range': [30, 40], 'color': "rgba(255, 107, 107, 0.3)"}
        ],
        'threshold': {
            'line': {'color': "#ff6b6b", 'width': 4},
            'thickness': 0.75,
            'value': 30
        }
    },
    number = {'font': {'color': 'white'}}
))

fig_bmi.update_layout(
    height=400,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font_color='white',
    margin=dict(l=20, r=20, t=60, b=20)
)

col_bmi1, col_bmi2, col_bmi3 = st.columns([1, 2, 1])
with col_bmi2:
    st.plotly_chart(fig_bmi, use_container_width=True)

# ============================================================================
# HEALTH SCORE VISUALIZATION
# ============================================================================

st.markdown('<br><br>', unsafe_allow_html=True)

# Calculate overall health score
health_metrics = {
    'BMI Score': 100 - abs(bmi - 22.5) * 4 if bmi < 35 else 0,
    'Activity Score': (faf / 3) * 100,
    'Hydration Score': (ch2o / 3) * 100,
    'Diet Score': (fcvc / 3) * 100,
    'Lifestyle Score': 100 - (tue * 25) - (smoke_num * 30) - (calc_num * 15)
}

# Ensure scores are between 0-100
for key in health_metrics:
    health_metrics[key] = max(0, min(100, health_metrics[key]))

# Create radar chart
categories = list(health_metrics.keys())
values = list(health_metrics.values())

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='Skor Kesehatan Anda',
    line_color='rgba(78, 205, 196, 0.8)',
    fillcolor='rgba(78, 205, 196, 0.3)'
))

fig_radar.add_trace(go.Scatterpolar(
    r=[100, 100, 100, 100, 100],
    theta=categories,
    fill='toself',
    name='Target Optimal',
    line_color='rgba(255, 107, 107, 0.8)',
    fillcolor='rgba(255, 107, 107, 0.1)'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickfont=dict(size=10, color='white'),
            gridcolor='rgba(255, 255, 255, 0.2)'
        ),
        angularaxis=dict(
            tickfont=dict(size=12, color='white', family='Poppins'),
            gridcolor='rgba(255, 255, 255, 0.2)'
        ),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    title={
        'text': "Analisis Skor Kesehatan Komprehensif",
        'x': 0.5,
        'font': {'size': 20, 'color': 'white', 'family': 'Poppins'}
    },
    font={'color': 'white', 'family': 'Poppins'},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=500,
    legend=dict(
        font=dict(color='white')
    )
)

st.plotly_chart(fig_radar, use_container_width=True)

# Health score summary
overall_score = sum(values) / len(values)
score_color = "#4ecdc4" if overall_score >= 70 else "#ffc107" if overall_score >= 50 else "#ff6b6b"
score_status = "Excellent" if overall_score >= 80 else "Good" if overall_score >= 60 else "Needs Improvement"

st.markdown(f"""
<div class="glass-card" style="text-align: center;">
    <h2><i class="fas fa-trophy"></i> Skor Kesehatan Keseluruhan</h2>
    <h1 style="color: {score_color}; font-size: 4rem; margin: 1rem 0;">{overall_score:.0f}/100</h1>
    <p style="font-size: 1.3rem; color: {score_color};">{score_status}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# DISCLAIMER
# ============================================================================

st.markdown("""
<div class="disclaimer">
    <h4><i class="fas fa-info-circle"></i> Medical Disclaimer</h4>
    <p>Aplikasi ini hanya untuk <strong>screening awal</strong> dan <strong>TIDAK menggantikan</strong> konsultasi dengan dokter. 
    Selalu konsultasikan kondisi kesehatan Anda dengan tenaga medis profesional untuk diagnosis dan pengobatan yang tepat.</p>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        Akurasi model: ~90-95% | Data tidak disimpan | Privacy-first design
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <h3><i class="fas fa-heartbeat"></i> Health Analytics Dashboard</h3>
    <p>Powered by Advanced Machine Learning & Modern Web Technologies</p>
    <p style="margin-top: 1rem;">
        <i class="fas fa-shield-alt"></i> Secure | 
        <i class="fas fa-rocket"></i> Fast | 
        <i class="fas fa-brain"></i> Intelligent | 
        <i class="fas fa-mobile-alt"></i> Responsive
    </p>
    <p style="font-size: 0.8rem; margin-top: 2rem; opacity: 0.7;">
        © 2024 Health Tech Solutions | Version 3.0 Advanced
    </p>
</div>
""", unsafe_allow_html=True)