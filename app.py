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
    page_icon=":newspaper:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS for Modern, Professional, Engaging UI ------------------
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
        background-color: #f4f6f9;
        color: #333333;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        max-width: 1100px;
        margin: 30px auto;
        padding: 30px;
    }

    /* Titles */
    .main-title {
        text-align: center;
        font-size: 2.4em;
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 12px;
        letter-spacing: -0.4px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #5f6368;
        margin-bottom: 35px;
        font-weight: 300;
        line-height: 1.6;
    }

    /* Result Box */
    .result-box {
        text-align: center;
        font-size: 1.5em;
        font-weight: 500;
        padding: 18px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.06);
        transition: box-shadow 0.3s ease, transform 0.3s ease;
    }
    .result-box:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
    }
    .real {
        background-color: #e8f4fd;
        color: #1967d2;
        border: 1px solid #d2e3fc;
    }
    .fake {
        background-color: #fce8e6;
        color: #c5221f;
        border: 1px solid #f6d0ce;
    }

    /* Text Area */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #dadce0 !important;
        font-size: 15px !important;
        padding: 12px !important;
        background-color: #ffffff;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    textarea:focus {
        border-color: #1a73e8 !important;
        box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.1) !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #1a73e8;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: 15px;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s, box-shadow 0.3s, transform 0.3s;
        box-shadow: 0 1px 4px rgba(26, 115, 232, 0.15);
    }
    div.stButton > button:hover {
        background-color: #1669c1;
        box-shadow: 0 2px 6px rgba(26, 115, 232, 0.25);
        transform: translateY(-1px);
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f8f9fc;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    }
    .sidebar-title {
        font-size: 1.3em;
        font-weight: 500;
        color: #202124;
        margin-bottom: 12px;
    }
    .sidebar-info {
        font-size: 0.95em;
        color: #5f6368;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    /* Examples Section */
    .examples-header {
        font-size: 1.4em;
        font-weight: 500;
        color: #202124;
        margin-top: 45px;
        margin-bottom: 18px;
    }
    .example-button {
        width: 100%;
        margin-bottom: 12px;
    }
    .example-info {
        background-color: #f8f9fc;
        border-left: 3px solid #1a73e8;
        padding: 12px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.9em;
        color: #3c4043;
        line-height: 1.4;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85em;
        color: #5f6368;
        margin-top: 45px;
        padding-top: 18px;
        border-top: 1px solid #e0e0e0;
    }

    /* Subtle Animations */
    @keyframes subtleFadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown, .stTextArea, .stButton {
        animation: subtleFadeIn 0.4s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>About the App</div>", unsafe_allow_html=True)
    st.info("This application employs a machine learning model to classify news articles as either genuine or misleading. Built with Streamlit for an intuitive user interface.")
    st.markdown("<div class='sidebar-info'>Developed by: <strong>Umaima Qureshi</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>Dataset Source: <a href='https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset' target='_blank'>Kaggle Fake & Real News</a></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>Technologies: Python, Scikit-learn, Streamlit</div>", unsafe_allow_html=True)

# ------------------ Main Title ------------------
st.markdown("<div class='main-title'>Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Verify the authenticity of news articles with advanced machine learning analysis.</div>", unsafe_allow_html=True)

# ------------------ Input ------------------
user_input = st.text_area(
    "Paste or type the news article here:",
    placeholder="Enter the headline or full text for analysis...",
    height=150
)

# ------------------ Prediction ------------------
if st.button("Analyze News", key="analyze_button"):
    if not user_input.strip():
        st.warning("Please provide some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("Model loading failed. Please contact support.")
    else:
        with st.spinner("Analyzing the content..."):
            result = predict_news(user_input)
            if result is True:
                st.markdown("<div class='result-box real'>This appears to be <strong>Real News</strong> – Credible and trustworthy.</div>", unsafe_allow_html=True)
            elif result is False:
                st.markdown("<div class='result-box fake'>This seems like <strong>Fake News</strong> – Proceed with caution.</div>", unsafe_allow_html=True)
            else:
                st.error("Analysis failed. Please try again.")

# ------------------ Examples ------------------
st.markdown("<div class='examples-header'>Test with Examples</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("Real News Example", key="real_example", help="Try a sample genuine news snippet"):
        example_text = "Economic growth rises by 2.3% in the third quarter, according to official government statistics."
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("Detected as Real News")
        elif res is False:
            st.error("Detected as Fake News")
        else:
            st.error("Prediction failed")

with col2:
    if st.button("Fake News Example", key="fake_example", help="Try a sample misleading news snippet"):
        example_text = "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("Detected as Real News (unexpected)")
        elif res is False:
            st.error("Detected as Fake News")
        else:
            st.error("Prediction failed")

# ------------------ Footer ------------------
st.markdown(
    """
    <div class='footer'>
    This project is open-source and intended for educational purposes.<br>
    Developed by Umaima Qureshi | Python & ML Enthusiast | Streamlit App Creator
    </div>
    """,
    unsafe_allow_html=True
)





