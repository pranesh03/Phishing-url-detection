import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# File paths for model and vectorizer
MODEL_FILE = "phishing_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
PHISHING_DATASET = "D:\Phising websit\dataset\phishing-urls.csv"
LEGITIMATE_DATASET = "D:\Phising websit\dataset\legitimate-urls.csv"

def load_datasets():
    phishing_df = pd.read_csv(PHISHING_DATASET)
    legitimate_df = pd.read_csv(LEGITIMATE_DATASET)
    
    phishing_urls = phishing_df.iloc[:, 0].tolist()
    legitimate_urls = legitimate_df.iloc[:, 0].tolist()
    
    all_urls = phishing_urls + legitimate_urls
    labels = [1] * len(phishing_urls) + [0] * len(legitimate_urls)
    
    return all_urls, labels

def train_model():
    all_urls, labels = load_datasets()
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_urls)
    
    model = RandomForestClassifier()
    model.fit(X, labels)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        train_model()
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get("url", "")
    
    url_features = vectorizer.transform([url])
    prediction = model.predict(url_features)[0]
    
    return jsonify({"status": "fake" if prediction == 1 else "safe"})

if __name__ == '__main__':
    app.run(debug=True)
