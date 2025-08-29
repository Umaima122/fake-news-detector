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
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    .stApp {
        background: url("mimi.png") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Roboto', sans-serif;
        color: #f0f0f0;
        overflow-x: hidden;
    }

    .main-container {
        background-color: rgba(0, 0, 0, 0.65);
        border-radius: 15px;
        padding: 30px;
        margin: 30px auto;
        max-width: 1000px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }

    .main-title {
        text-align: center;
        font-size: 2.6em;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        font-weight: 400;
        color: #d0d0d0;
        margin-bottom: 25px;
    }

    textarea {
        border-radius: 10px !important;
        border: 1px solid #cccccc !important;
        font-size: 15px !important;
        padding: 12px !important;
        background-color: #1c1c1c;
        color: #ffffff;
    }
    textarea:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.2) !important;
    }

    div.stButton > button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        padding: 12px 25px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1669c1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .result-box {
        text-align: center;
        font-size: 1.4em;
        font-weight: 500;
        padding: 18px;
        border-radius: 12px;
        margin-top: 20px;
        color: #ffffff;
    }
    .real {
        background-color: rgba(26, 115, 232, 0.7);
        border: 1px solid #1a73e8;
    }
    .fake {
        background-color: rgba(220, 53, 69, 0.7);
        border: 1px solid #dc3545;
    }

    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #bbbbbb;
        margin-top: 35px;
        padding-top: 15px;
        border-top: 1px solid #555555;
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
        st.markdown(f"<div style='background-color:#1c1c1c;padding:10px;border-radius:6px;color:#fff;'>{example_text}</div>", unsafe_allow_html=True)
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
        st.markdown(f"<div style='background-color:#1c1c1c;padding:10px;border-radius:6px;color:#fff;'>{example_text}</div>", unsafe_allow_html=True)
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








