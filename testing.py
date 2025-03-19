import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# File paths for model and vectorizer
MODEL_FILE = "phishing_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

st.title("🔍 Phishing URL Detection using ML")

# File uploaders for datasets
phishing_file = st.file_uploader("Upload Phishing URLs Dataset (CSV)", type=["csv"])
legitimate_file = st.file_uploader("Upload Legitimate URLs Dataset (CSV)", type=["csv"])

def load_datasets(phishing_file, legitimate_file):
    phishing_df = pd.read_csv(phishing_file) if phishing_file else pd.DataFrame()
    legitimate_df = pd.read_csv(legitimate_file) if legitimate_file else pd.DataFrame()

    if phishing_df.empty or legitimate_df.empty:
        return None, None

    phishing_urls = phishing_df.iloc[:, 0].tolist()
    legitimate_urls = legitimate_df.iloc[:, 0].tolist()

    all_urls = phishing_urls + legitimate_urls
    labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)

    return all_urls, labels

def train_model(all_urls, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_urls)

    model = RandomForestClassifier()
    model.fit(X, labels)

    # Save model and vectorizer
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        model, vectorizer = None, None
    return model, vectorizer

if phishing_file and legitimate_file:
    st.write("✅ Datasets uploaded successfully!")

    all_urls, labels = load_datasets(phishing_file, legitimate_file)
    if all_urls and labels:
        st.write("🔄 Training the model...")
        model, vectorizer = train_model(all_urls, labels)
        st.write("✅ Model trained successfully!")

        # Save trained status
        st.session_state["model"] = model
        st.session_state["vectorizer"] = vectorizer

model, vectorizer = load_model()

# URL Input for Prediction
st.write("### 🔗 Check a URL")
url_input = st.text_input("Enter a URL to check:")

if st.button("Check"):
    if model and vectorizer:
        url_features = vectorizer.transform([url_input])
        prediction = model.predict(url_features)[0]
        result = "🚨 Fake URL" if prediction == 1 else "✅ Safe URL"
        st.write(f"**Result:** {result}")
    else:
        st.error("❌ Model is not trained yet. Upload datasets and train first.")

st.write("📥 **Download Trained Model**")
with open(MODEL_FILE, "rb") as f:
    st.download_button("Download Model", f, file_name="phishing_model.pkl")

with open(VECTORIZER_FILE, "rb") as f:
    st.download_button("Download Vectorizer", f, file_name="vectorizer.pkl")
