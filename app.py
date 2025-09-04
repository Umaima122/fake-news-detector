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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        overflow-x: hidden;
    }

    .main-container {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 25px;
        padding: 50px;
        margin: 50px auto;
        max-width: 1200px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255,255,255,0.1);
    }

    .main-title {
        text-align: center;
        font-size: 3em;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 15px;
        letter-spacing: -1px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        font-size: 1.4em;
        font-weight: 300;
        color: #d0d0d0;
        margin-bottom: 40px;
    }

    textarea {
        border-radius: 15px !important;
        border: 1px solid #505050 !important;
        font-size: 16px !important;
        padding: 20px !important;
        background-color: #1e1e1e;
        color: #ffffff;
        resize: vertical;
        min-height: 200px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
    }
    textarea:focus {
        border-color: #6a5acd !important;
        box-shadow: 0 0 0 4px rgba(106,90,205,0.3) !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #6a5acd 0%, #483d8b 100%);
        color: white;
        border-radius: 12px;
        padding: 15px 35px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #5a4ab8 0%, #3a2f7a 100%);
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(106,90,205,0.4);
    }

    .result-box {
        text-align: center;
        font-size: 1.6em;
        font-weight: 600;
        padding: 25px;
        border-radius: 18px;
        margin-top: 30px;
        color: #ffffff;
        animation: fadeIn 0.6s ease-in-out;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .real {
        background: linear-gradient(135deg, #32cd32 0%, #228b22 100%);
        border: 1px solid #228b22;
    }
    .fake {
        background: linear-gradient(135deg, #ff4500 0%, #cc3300 100%);
        border: 1px solid #cc3300;
    }

    .footer {
        text-align: center;
        font-size: 1em;
        color: #a0a0a0;
        margin-top: 50px;
        padding-top: 25px;
        border-top: 1px solid #404040;
    }

    .example-box {
        background-color: #252525;
        padding: 20px;
        border-radius: 12px;
        color: #ffffff;
        font-size: 15px;
        margin-bottom: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }

    h4 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        font-size: 1.8em;
    }

    .stColumn > div > div > div > div > button {
        width: 100%;
        margin-top: 15px;
        font-size: 16px;
        padding: 12px 25px;
    }

    .sidebar .stMarkdown {
        color: #d0d0d0;
    }

    .sidebar h3 {
        color: #ffffff;
        font-weight: 600;
    }

    a {
        color: #6a5acd;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        text-decoration: underline;
        color: #5a4ab8;
    }

    /* Ensure consistent button sizes in columns */
    .stButton {
        display: flex;
        justify-content: center;
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





