import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(page_title="Skin Disease Detector", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 3rem; }
    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #e94560;
    }
    .title-box h1 {
        color: #ffffff;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: 2px;
    }
    .title-box p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .upload-box {
        background: #1a1a2e;
        border: 2px dashed #e94560;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-card {
        background: #1a1a2e;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #2d2d44;
        height: 100%;
    }
    .confidence-bar-bg {
        background: #2d2d44;
        border-radius: 10px;
        height: 10px;
        margin: 6px 0 12px 0;
    }
    .stat-box {
        background: #16213e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        border: 1px solid #2d2d44;
    }
    .stat-box h2 { color: #e94560; margin: 0; font-size: 1.8rem; }
    .stat-box p  { color: #a0aec0; margin: 0; font-size: 0.85rem; }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2d2d44;
    }
    div[data-testid="stFileUploader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('skin_disease_model.keras')
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

DISEASE_INFO = {
    'akiec': ('Actinic Keratosis',    '#f39c12', 'Medium Risk',  '#f39c12', 'Pre-cancerous skin patch caused by sun damage.'),
    'bcc':   ('Basal Cell Carcinoma', '#e74c3c', 'High Risk',    '#e74c3c', 'Most common type of skin cancer. Rarely spreads.'),
    'bkl':   ('Benign Keratosis',     '#3498db', 'Low Risk',     '#3498db', 'Non-cancerous skin growth. Usually harmless.'),
    'df':    ('Dermatofibroma',       '#2ecc71', 'Very Low Risk','#2ecc71', 'Benign skin nodule, often from minor injury.'),
    'mel':   ('Melanoma',             '#e74c3c', 'Very High Risk','#c0392b','Dangerous skin cancer. Early detection is critical.'),
    'nv':    ('Melanocytic Nevi',     '#2ecc71', 'Low Risk',     '#27ae60', 'Common mole. Usually benign and harmless.'),
    'vasc':  ('Vascular Lesion',      '#9b59b6', 'Low Risk',     '#8e44ad', 'Blood vessel abnormality on the skin surface.'),
}

# ── Header ─────────────────────────────────────────────
st.markdown("""
<div class='title-box'>
    <h1>🔬 Skin Disease Detector</h1>
    <p>AI-powered skin lesion analysis using Deep Learning (MobileNetV2 + HAM10000)</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='stat-box'><h2>7</h2><p>Disease Classes</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='stat-box'><h2>10K</h2><p>Training Images</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='stat-box'><h2>89%</h2><p>Model Accuracy</p></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='stat-box'><h2>CNN</h2><p>MobileNetV2</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Upload + Result ────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True, caption="Uploaded skin lesion image")

        img_array = np.array(image.resize((96, 96)), dtype='float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array, verbose=0)[0]
    else:
        st.markdown("""
        <div class='upload-box'>
            <h3 style='color:#e94560'>⬆ Drop your image here</h3>
            <p style='color:#a0aec0'>Supports JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### 🧠 Analysis Result")
    if uploaded_file:
        top_idx        = int(np.argmax(predictions))
        top_class      = class_names[top_idx]
        top_confidence = predictions[top_idx] * 100
        name, color, risk, rcolor, desc = DISEASE_INFO.get(
            top_class, (top_class, '#3498db', 'Unknown', '#3498db', '')
        )

        st.markdown(f"""
        <div class='result-card'>
            <span class='badge' style='background:{rcolor}33; color:{rcolor}'>{risk}</span>
            <h2 style='color:{color}; margin:0.2rem 0'>{name}</h2>
            <p style='color:#a0aec0; font-size:0.9rem; margin-bottom:1rem'>{desc}</p>
            <hr style='border-color:#2d2d44; margin:1rem 0'>
            <p style='color:#a0aec0; margin:0'>Confidence</p>
            <h1 style='color:white; margin:0.2rem 0'>{top_confidence:.1f}%</h1>
            <div class='confidence-bar-bg'>
                <div style='width:{top_confidence:.0f}%; background:{color};
                            height:10px; border-radius:10px;'></div>
            </div>
            <p style='color:#4a5568; font-size:0.8rem'>Disease code: {top_class.upper()}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 All Probabilities")
        for idx in np.argsort(predictions)[::-1]:
            cls   = class_names[idx]
            prob  = predictions[idx] * 100
            dname = DISEASE_INFO.get(cls, (cls,))[0]
            dcol  = DISEASE_INFO.get(cls, ('','#3498db'))[1]
            st.markdown(f"""
            <div style='margin-bottom:8px'>
                <div style='display:flex; justify-content:space-between;
                            color:#a0aec0; font-size:0.85rem'>
                    <span>{dname}</span><span>{prob:.1f}%</span>
                </div>
                <div class='confidence-bar-bg'>
                    <div style='width:{prob:.0f}%; background:{dcol};
                                height:10px; border-radius:10px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='result-card' style='text-align:center; padding:4rem 2rem'>
            <h2 style='color:#2d2d44'>⬅ Upload an image</h2>
            <p style='color:#4a5568'>Results will appear here after analysis</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    ⚠️ For educational purposes only. This is not a medical diagnosis tool.
    Always consult a qualified dermatologist for medical advice.<br>
    Built with TensorFlow · MobileNetV2 · HAM10000 · Streamlit
</div>
""", unsafe_allow_html=True)