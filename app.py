import streamlit as st
import joblib

# Load trained model + vectorizer (ensure the model files are present in your app directory)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction function
def predict_news(news_text: str) -> bool:
    input_data = vectorizer.transform([news_text])
    prediction = model.predict(input_data)[0]
    # Return True if real (1), False if fake (0)
    return bool(prediction)

# ------------------ UI Setup ------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar with settings and info
st.sidebar.title("Settings")
st.sidebar.info(
    "Demo Fake News Detection app using a machine learning model trained on Kaggle's Fake & Real News dataset."
)
st.sidebar.write("Developer: Umaima")
st.sidebar.write(
    "Dataset: [Fake & Real News on Kaggle]"
    "(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)"
)

# Main Title
st.title("Fake News Detector")
st.write(
    "Check whether a news article is **Fake** or **Real** "
    "using a machine learning model."
)

# Input Section
st.subheader("Enter a news headline or article:")
user_input = st.text_area(
    "News Text:",
    placeholder="Paste or type the news text here...",
    height=200
)

# Prediction button and output
if st.button("Check News"):
    if user_input.strip():
        result = predict_news(user_input)
        if result:
            st.success("✅ Real News – This article seems genuine.")
        else:
            st.error("🚨 Fake News – This article may be misleading.")
    else:
        st.error("Please enter some text to analyze.")

# Examples Section
st.subheader("Try Some Examples")
col1, col2 = st.columns(2)

with col1:
    if st.button("Real News Example"):
        example_text = (
            "Economic growth rises by 2.3% in the third quarter, "
            "according to official government statistics."
        )
        st.write("Example Input:", example_text)
        example_result = predict_news(example_text)
        if example_result:
            st.success("Real News – The model detected it as genuine.")
        else:
            st.error("Fake News – The model detected it as misleading.")

with col2:
    if st.button("Fake News Example"):
        example_text = (
            "Breaking: New study confirms consuming chocolate cures diabetes overnight!"
        )
        st.write("Example Input:", example_text)
        example_result = predict_news(example_text)
        if example_result:
            st.success("Real News – The model detected it as genuine (unexpected).")
        else:
            st.error("Fake News – The model detected it as misleading.")

# Footer
st.markdown("---")
st.markdown("Made with Streamlit & Scikit-learn | By Umaima")
