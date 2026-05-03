import streamlit as st
import numpy as np
import joblib
import re
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS (COLORFUL UI 🎨)
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .title {
        text-align: center;
        font-size: 40px;
        color: #38bdf8;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        background-color: #1e293b;
        color: white;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #38bdf8;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    ft_model = KeyedVectors.load("fasttext_vectors.kv")
    xg_model = joblib.load("xg_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return ft_model, xg_model, le

ft_model, xg_model, le = load_models()

# -----------------------------
# TEXT CLEANING
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

# -----------------------------
# SENTENCE VECTOR
# -----------------------------
def sent_vector(tokens):
    vectors = []
    for word in tokens:
        if word in ft_model:
            vectors.append(ft_model[word])
    
    if len(vectors) == 0:
        return np.zeros(ft_model.vector_size)
    
    return np.mean(vectors, axis=0)

# -----------------------------
# UI HEADER
# -----------------------------
st.markdown('<div class="title">🧠 Mental Health Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions from text using AI</div>', unsafe_allow_html=True)

# -----------------------------
# INPUT
# -----------------------------
user_input = st.text_area("💬 Enter your thoughts:", height=150)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🚀 Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        tokens = clean_text(user_input)
        vec = sent_vector(tokens).reshape(1, -1)

        pred = xg_model.predict(vec)
        label = le.inverse_transform(pred)[0]

        probs = xg_model.predict_proba(vec)
        confidence = np.max(probs) * 100

        # -----------------------------
        # RESULT DISPLAY (COLORFUL)
        # -----------------------------
        if label == "Anxiety":
            st.error(f"😟 Anxiety Detected")
        elif label == "Depression":
            st.warning(f"😞 Depression Detected")
        elif label == "Suicidal":
            st.error(f"⚠️ Suicidal Thoughts Detected")
        else:
            st.success(f"😊 Normal State")

        st.markdown(f"### 🔍 Confidence: `{confidence:.2f}%`")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("✨ Built with FastText + XGBoost + Streamlit")

