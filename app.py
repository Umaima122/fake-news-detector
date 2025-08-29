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

# Prediction function
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
    layout="wide",   # ✅ better for phones
    initial_sidebar_state="collapsed"
)

# ------------------ Custom CSS for Professional Look ------------------
st.markdown(
    """
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #5D6D7E;
        margin-bottom: 25px;
    }
    .result-box {
        text-align: center;
        font-size: 1.5em;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .real {
        background-color: #D5F5E3;
        color: #1E8449;
        border: 2px solid #27AE60;
    }
    .fake {
        background-color: #FADBD8;
        color: #C0392B;
        border: 2px solid #E74C3C;
    }
    textarea {
        border-radius: 8px !important;
        font-size: 15px !important;
    }
    div.stButton > button {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1B4F72;
        transform: scale(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar ------------------
st.sidebar.title("⚙️ About")
st.sidebar.info("This demo app detects Fake vs Real News using a Machine Learning model.")
st.sidebar.write("👩‍💻 Developer: **Omaima**")
st.sidebar.write("📚 Dataset: [Kaggle Fake & Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")

# ------------------ Main Title ------------------
st.markdown("<div class='main-title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Check whether a news article is <b>Fake</b> or <b>Real</b> instantly.</div>", unsafe_allow_html=True)

# ------------------ Input ------------------
user_input = st.text_area(
    "📝 Enter a news headline or article:",
    placeholder="Paste or type the news text here...",
    height=120
)

# ------------------ Prediction ------------------
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model not loaded. Please check deployment.")
    else:
        with st.spinner("Analyzing... ⏳"):
            result = predict_news(user_input)
            if result is True:
                st.markdown("<div class='result-box real'>✅ Real News – This article seems genuine.</div>", unsafe_allow_html=True)
            elif result is False:
                st.markdown("<div class='result-box fake'>🚨 Fake News – This article may be misleading.</div>", unsafe_allow_html=True)

# ------------------ Examples ------------------
st.subheader("🎯 Try Examples")
col1, col2 = st.columns(2)

with col1:
    if st.button("Example: Real News"):
        example_text = "Economic growth rises by 2.3% in the third quarter, according to official government statistics."
        st.info(example_text)
        res = predict_news(example_text)
        if res: 
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")

with col2:
    if st.button("Example: Fake News"):
        example_text = "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        st.info(example_text)
        res = predict_news(example_text)
        if res:
            st.success("✅ Real News (unexpected)")
        else:
            st.error("🚨 Fake News")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Made with  using Streamlit & Scikit-learn | Optimized for Mobile 📱")


