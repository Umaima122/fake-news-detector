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
        st.error(f"⚠️ Could not load model/vectorizer: {e}")
        return None, None

model, vectorizer = load_model()

# ------------------ Prediction function ------------------
def predict_news(news_text: str) -> bool:
    try:
        input_data = vectorizer.transform([news_text])
        prediction = model.predict(input_data)[0]
        return bool(prediction)
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None

# ------------------ Page Setup ------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Custom CSS for Modern, Professional, Engaging UI ------------------
st.markdown(
    """
    <style>
    /* Global Styles */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #212529;
    }
    .stApp {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        max-width: 1200px;
        margin: 20px auto;
        padding: 20px;
    }

    /* Titles */
    .main-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 700;
        color: #007bff;
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #6c757d;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* Result Box */
    .result-box {
        text-align: center;
        font-size: 1.6em;
        font-weight: 600;
        padding: 20px;
        border-radius: 12px;
        margin-top: 25px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: translateY(-5px);
    }
    .real {
        background-color: #e6fffa;
        color: #0c8599;
        border: 2px solid #22b8cf;
    }
    .fake {
        background-color: #fff5f5;
        color: #c92a2a;
        border: 2px solid #ff6b6b;
    }

    /* Text Area */
    textarea {
        border-radius: 10px !important;
        border: 1px solid #ced4da !important;
        font-size: 16px !important;
        padding: 15px !important;
        background-color: #ffffff;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: border-color 0.3s ease;
    }
    textarea:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1) !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        transition: background-color 0.3s, transform 0.3s;
        box-shadow: 0 2px 5px rgba(0, 123, 255, 0.2);
    }
    div.stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #f1f3f5;
        border-radius: 10px;
        padding: 20px;
    }
    .sidebar-title {
        font-size: 1.4em;
        font-weight: 600;
        color: #343a40;
        margin-bottom: 15px;
    }
    .sidebar-info {
        font-size: 1em;
        color: #495057;
        line-height: 1.6;
    }

    /* Examples Section */
    .examples-header {
        font-size: 1.5em;
        font-weight: 600;
        color: #212529;
        margin-top: 50px;
        margin-bottom: 20px;
    }
    .example-button {
        width: 100%;
        margin-bottom: 15px;
    }
    .example-info {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 0.95em;
        color: #495057;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #6c757d;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #dee2e6;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stMarkdown, .stTextArea, .stButton {
        animation: fadeIn 0.5s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚙️ About the App</div>", unsafe_allow_html=True)
    st.info("This interactive app uses a trained Machine Learning model to classify news articles as Fake or Real. Powered by Streamlit for seamless user experience.")
    st.markdown("<div class='sidebar-info'>👩‍💻 Developed by: <strong>Umaima Qureshi</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>📚 Dataset Source: <a href='https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset' target='_blank'>Kaggle Fake & Real News</a></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-info'>🛠️ Tech Stack: Python, Scikit-learn, Streamlit</div>", unsafe_allow_html=True)

# ------------------ Main Title ------------------
st.markdown("<div class='main-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Instantly verify if a news article is <strong>genuine</strong> or <strong>misleading</strong> using advanced AI analysis.</div>", unsafe_allow_html=True)

# ------------------ Input ------------------
user_input = st.text_area(
    "📝 Paste or type the news article here:",
    placeholder="Enter the headline or full text for analysis...",
    height=150
)

# ------------------ Prediction ------------------
if st.button("🔍 Analyze News", key="analyze_button"):
    if not user_input.strip():
        st.warning("⚠️ Please provide some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model loading failed. Please contact support.")
    else:
        with st.spinner("Analyzing the content... ⏳"):
            result = predict_news(user_input)
            if result is True:
                st.markdown("<div class='result-box real'>✅ This appears to be <strong>Real News</strong> – Credible and trustworthy.</div>", unsafe_allow_html=True)
            elif result is False:
                st.markdown("<div class='result-box fake'>🚨 This seems like <strong>Fake News</strong> – Proceed with caution.</div>", unsafe_allow_html=True)
            else:
                st.error("❌ Analysis failed. Please try again.")

# ------------------ Examples ------------------
st.markdown("<div class='examples-header'>🎯 Test with Examples</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    if st.button("Real News Example", key="real_example", help="Try a sample genuine news snippet"):
        example_text = "Economic growth rises by 2.3% in the third quarter, according to official government statistics."
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("✅ Detected as Real News")
        elif res is False:
            st.error("🚨 Detected as Fake News")
        else:
            st.error("❌ Prediction failed")

with col2:
    if st.button("Fake News Example", key="fake_example", help="Try a sample misleading news snippet"):
        example_text = "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        st.markdown(f"<div class='example-info'>{example_text}</div>", unsafe_allow_html=True)
        res = predict_news(example_text)
        if res is True:
            st.success("✅ Detected as Real News (unexpected)")
        elif res is False:
            st.error("🚨 Detected as Fake News")
        else:
            st.error("❌ Prediction failed")

# ------------------ Footer ------------------
st.markdown(
    """
    <div class='footer'>
    This project is open-source and intended for educational purposes.<br>
    Developed with ❤️ by Umaima Qureshi | Python & ML Enthusiast | Streamlit App Creator
    </div>
    """,
    unsafe_allow_html=True
)




