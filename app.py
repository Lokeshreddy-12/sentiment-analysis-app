from flask import Flask, render_template, request
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import emoji
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = Flask(__name__)

label_mapping = {1: "Positive 🙂", 0: "Negative 😞"}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load the trained model
if os.path.exists("bert_sentiment_model"):
    model = AutoModelForSequenceClassification.from_pretrained("bert_sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("bert_sentiment_model")
    model_type = "BERT"
elif os.path.exists("lstm_sentiment_model.h5"):
    model = tf.keras.models.load_model("lstm_sentiment_model.h5")
    tokenizer = joblib.load("lstm_tokenizer.pkl")
    model_type = "LSTM"
else:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    model_type = "Traditional"

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    confidence_score = 0.5
    prediction = 0

    if model_type == "BERT":
        inputs = tokenizer(cleaned_text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
        prediction = np.argmax(probabilities)
        confidence_score = probabilities[prediction]
    elif model_type == "LSTM":
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=100, padding='post')
        probabilities = model.predict(padded, verbose=0)[0][0]
        prediction = 1 if probabilities > 0.5 else 0
        confidence_score = probabilities if prediction == 1 else 1 - probabilities
    else:
        vec_text = vectorizer.transform([cleaned_text])
        probabilities = model.predict_proba(vec_text)[0]
        prediction = model.predict(vec_text)[0]
        confidence_score = probabilities[prediction]

    neutral_threshold = 0.7
    if confidence_score < neutral_threshold:
        sentiment = "Neutral 😐"
        score = 50
    else:
        sentiment = label_mapping[prediction]
        if prediction == 0:
            score = (1 - confidence_score) * 50
        else:
            score = 50 + (confidence_score * 50)

    return sentiment, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['review']
        sentiment, score = predict_sentiment(text)
        return render_template('predict.html', review=text, sentiment=sentiment, score=score, model_type=model_type)
    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    df = pd.read_csv(file)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    df['sentiment'] = [predict_sentiment(text)[0] for text in df['text']]
    return render_template('results.html', tables=[df[['text', 'sentiment']].to_html(classes='data')], titles=df.columns.values)

@app.route('/metrics')
def metrics():
    if os.path.exists("model_metrics.pkl"):
        results = joblib.load("model_metrics.pkl")
        metrics_html = ""
        roc_data = []
        accuracy_data = []
        colors = ['#007bff', '#28a745', '#dc3545', '#ff6b6b', '#4ecdc4', '#6c757d']  # Colors for charts
        for idx, (model_name, metrics) in enumerate(results.items()):
            report = metrics["report"]
            metrics_html += f"<h3>{model_name}</h3>"
            metrics_html += "<table class='data'><tr><th>Label</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>"
            for label in report.keys():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    metrics_html += f"<tr><td>{label}</td><td>{report[label]['precision']:.2f}</td><td>{report[label]['recall']:.2f}</td><td>{report[label]['f1-score']:.2f}</td><td>{report[label]['support']}</td></tr>"
            metrics_html += f"<tr><td colspan='4'><strong>Accuracy</strong></td><td>{metrics['accuracy']:.2f}</td></tr>"
            metrics_html += f"<tr><td colspan='4'><strong>AUC</strong></td><td>{metrics['roc_auc']:.2f}</td></tr></table>"
            # Prepare ROC data for Plotly
            roc_data.append({
                "name": model_name,
                "fpr": metrics["fpr"],
                "tpr": metrics["tpr"],
                "color": colors[idx % len(colors)]
            })
            # Prepare accuracy data for Plotly
            accuracy_data.append({
                "name": model_name,
                "accuracy": metrics["accuracy"],
                "color": colors[idx % len(colors)]
            })
        return render_template('metrics.html', metrics_html=metrics_html, roc_data=roc_data, accuracy_data=accuracy_data)
    return "Metrics not available", 404

@app.route('/realtime', methods=['GET', 'POST'])
def realtime():
    if request.method == 'POST':
        query = request.form['query']
        # Simulated social media posts
        simulated_posts = [
            {"text": f"I love {query}! It's amazing 😊", "username": "user1", "likes": 10},
            {"text": f"{query} is okay, nothing special 😐", "username": "user2", "likes": 5},
            {"text": f"I really dislike {query}, it's terrible 😞", "username": "user3", "likes": 2},
            {"text": f"{query} has its pros and cons", "username": "user4", "likes": 7},
            {"text": f"Wow, {query} exceeded my expectations! 🎉", "username": "user5", "likes": 15}
        ]
        sentiments = []
        for post in simulated_posts:
            sentiment, score = predict_sentiment(post["text"])
            sentiments.append({
                "text": post["text"],
                "username": post["username"],
                "sentiment": sentiment,
                "score": score,
                "likes": post["likes"]
            })
        return render_template('realtime.html', query=query, sentiments=sentiments)
    return render_template('realtime.html')

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """JSON API endpoint. POST JSON like: {"text": "I love this"} """
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return {"error": "JSON body must include 'text' field"}, 400
    sentiment, score = predict_sentiment(data['text'])
    return {"sentiment": sentiment, "score": score, "model_type": model_type}, 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)