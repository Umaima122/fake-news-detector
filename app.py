import streamlit as st
import joblib

# ------------------ Load model safely with caching ------------------
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
    layout="wide"
)

# ------------------ Custom CSS for Full-page Modern UI ------------------
st.markdown(
    """
    <style>
    /* Import clean modern font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: url("https://img.freepik.com/free-vector/abstract-dark-blue-wave-background_53876-111548.jpg") no-repeat center center fixed;
        background-size: cover;
        color: #ffffff;
    }

    /* Titles */
    .main-title {
        text-align: center;
        font-size: 2.6em;
        font-weight: 600;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        font-weight: 300;
        color: #dcdcdc;
        margin-bottom: 35px;
    }

    /* Input box */
    textarea {
        border-radius: 10px !important;
        border: none !important;
        font-size: 15px !important;
        padding: 15px !important;
        background: rgba(255,255,255,0.1) !important;
        color: #ffffff !important;
    }
    textarea::placeholder {
        color: #cccccc !important;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #1a73e8, #0059b3);
        color: #ffffff;
        border-radius: 8px;
        padding: 12px 25px;
        font-size: 15px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #0059b3, #003d80);
        transform: scale(1.03);
    }

    /* Result Box */
    .result-box {
        text-align: center;
        font-size: 1.4em;
        font-weight: 500;
        padding: 20px;
        border-radius: 10px;
        margin-top: 25px;
        backdrop-filter: blur(6px);
    }
    .real {
        background: rgba(0, 128, 0, 0.3);
        border: 1px solid #00e676;
        color: #00ff99;
    }
    .fake {
        background: rgba(139, 0, 0, 0.3);
        border: 1px solid #ff4d4d;
        color: #ff8080;
    }

    /* Examples */
    .example-info {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.9em;
        color: #f1f1f1;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        color: #cccccc;
        margin-top: 60px;
        padding: 15px 0;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Main Title ------------------
st.markdown("<div class='main-title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Verify the authenticity of news articles with advanced machine learning.</div>", unsafe_allow_html=True)

# ------------------ Input ------------------
user_input = st.text_area(
    "Paste or type the news article here:",
    placeholder="Enter the headline or full text for analysis...",
    height=150
)

# ------------------ Prediction ------------------
if st.button("Analyze News"):
    if not user_input.strip():
        st.warning("Please provide some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model loading failed. Please contact support.")
    else:
        with st.spinner("Analyzing the content..."):
            result = predict_news(user_input)
            if result is True:
                st.markdown("<div class='result-box real'>This appears to be <strong>Real News</strong></div>", unsafe_allow_html=True)
            elif result is False:
                st.markdown("<div class='result-box fake'>This seems like <strong>Fake News</strong></div>", unsafe_allow_html=True)
            else:
                st.error("Analysis failed. Please try again.")

# ------------------ Examples ------------------
st.subheader("Try with Examples")
col1, col2 = st.columns(2)

with col1:
    if st.button("Real News Example"):
        example_text = "Economic growth rises by 2.3% in the third quarter, according to official government statistics."
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        st.success("Real News" if res else "Fake News")

with col2:
    if st.button("Fake News Example"):
        example_text = "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        st.success("Real News" if res else "Fake News")

# ------------------ Footer ------------------
st.markdown(
    """
    <div class='footer'>
    © 2025 Fake News Detector | Developed by Umaima Qureshi | For educational use
    </div>
    """,
    unsafe_allow_html=True
)






