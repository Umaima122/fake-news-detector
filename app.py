import streamlit as st
import joblib

# ------------------ Load model safely ------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except Exception as e:
        st.error(f"Could not load model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model()

# ------------------ Prediction function ------------------
def predict_news(news_text: str) -> bool:
    try:
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)[0]
        return bool(prediction)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CSS Styling ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2c2c2c 100%);
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
        overflow-x: hidden;
    }

    .main-container {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 40px;
        margin: 40px auto;
        max-width: 1100px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
    }

    .main-title {
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        font-weight: 400;
        color: #b0b0b0;
        margin-bottom: 30px;
    }

    textarea {
        border-radius: 12px !important;
        border: 1px solid #404040 !important;
        font-size: 16px !important;
        padding: 15px !important;
        background-color: #252525;
        color: #ffffff;
        resize: vertical;
        min-height: 150px;
    }
    textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 4px rgba(74, 144, 226, 0.2) !important;
    }

    div.stButton > button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        padding: 14px 30px;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #3a7bc8;
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(74, 144, 226, 0.3);
    }

    .result-box {
        text-align: center;
        font-size: 1.5em;
        font-weight: 600;
        padding: 20px;
        border-radius: 15px;
        margin-top: 25px;
        color: #ffffff;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .real {
        background-color: rgba(74, 144, 226, 0.8);
        border: 1px solid #4a90e2;
    }
    .fake {
        background-color: rgba(255, 82, 82, 0.8);
        border: 1px solid #ff5252;
    }

    .footer {
        text-align: center;
        font-size: 0.95em;
        color: #888888;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #404040;
    }

    .example-box {
        background-color: #252525;
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        font-size: 14px;
        margin-bottom: 15px;
    }

    h4 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .stColumn > div > div > div > div > button {
        width: 100%;
        margin-top: 10px;
    }

    .sidebar .stMarkdown {
        color: #d0d0d0;
    }

    .sidebar h3 {
        color: #ffffff;
    }

    a {
        color: #4a90e2;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<h3>About the App</h3>", unsafe_allow_html=True)
    st.info("This app uses a machine learning model to classify news articles as Real or Fake.")
    st.markdown("<p>Developer: <strong>Umaima Qureshi</strong></p>", unsafe_allow_html=True)
    st.markdown("<p>Dataset: <a href='https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset' target='_blank'>Kaggle Fake & Real News</a></p>", unsafe_allow_html=True)
    st.markdown("<p>Tech: Python, Scikit-learn, Streamlit</p>", unsafe_allow_html=True)

# ------------------ Main Container ------------------
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Verify the authenticity of news articles quickly and reliably.</div>", unsafe_allow_html=True)

# Input
user_input = st.text_area("Enter news headline or article here:")

# Prediction
if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please provide text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model not loaded. Check deployment.")
    else:
        with st.spinner("Analyzing..."):
            result = predict_news(user_input)
            if result is True:
                st.markdown("<div class='result-box real'>This appears to be Real News – Credible and trustworthy.</div>", unsafe_allow_html=True)
            elif result is False:
                st.markdown("<div class='result-box fake'>This seems like Fake News – Be cautious.</div>", unsafe_allow_html=True)
            else:
                st.error("Prediction failed. Please try again.")

# Examples
st.markdown("<h4>Try Examples</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("Real News Example", key="real"):
        example_text = "Economic growth rises by 2.3% in the third quarter, according to official government statistics."
        st.markdown(f"<div class='example-box'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("Detected as Real News")
        elif res is False:
            st.error("Detected as Fake News")
        else:
            st.error("Prediction failed")

with col2:
    if st.button("Fake News Example", key="fake"):
        example_text = "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        st.markdown(f"<div class='example-box'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("Detected as Real News (unexpected)")
        elif res is False:
            st.error("Detected as Fake News")
        else:
            st.error("Prediction failed")

st.markdown("</div>", unsafe_allow_html=True)  # close main container

# Footer
st.markdown("<div class='footer'>Open-source project for educational purposes.<br>Developed by Umaima Qureshi | Streamlit & ML Enthusiast</div>", unsafe_allow_html=True)








