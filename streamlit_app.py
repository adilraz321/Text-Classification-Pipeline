# app/streamlit_app.py
import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

st.set_page_config(page_title="Sentiment Analysis Demo", page_icon="🤖", layout="centered")
st.title("🤖 Sentiment Analysis Demo")

st.write("This app lets you test two models: **TF-IDF + Logistic Regression** and **BERT Transformer**.")

# 🔸 Select model
method = st.radio("Select Method:", ("TF-IDF + LR", "Transformer (BERT)"))

# ======================= TF-IDF + Logistic Regression =======================
if method == "TF-IDF + LR":
    try:
        # Load trained model
        tfidf, model = joblib.load('models/tfidf_lr.joblib')
    except Exception as e:
        st.error(f"❌ Could not load TF-IDF model. Error: {e}")
    else:
        st.subheader("🧮 TF-IDF + Logistic Regression")
        text = st.text_area("Enter text to analyze")
        if st.button("Predict (LR)"):
            if text.strip():
                vec = tfidf.transform([text])
                pred = model.predict(vec)[0]
                label = "✅ Positive" if pred == 1 else "❌ Negative"
                st.success(f"Prediction: {label}")
            else:
                st.warning("⚠️ Please enter some text.")

# ======================= Transformer (BERT) =======================
else:
    try:
        # Define id2label and label2id for clean output
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}

        model = AutoModelForSequenceClassification.from_pretrained(
            'models/bert_sentiment',
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    except Exception as e:
        st.error(f"❌ Could not load Transformer model. Error: {e}")
    else:
        st.subheader("🧠 Transformer (BERT)")
        text = st.text_area("Enter text to analyze")
        if st.button("Predict (BERT)"):
            if text.strip():
                res = pipe(text)[0]
                label = res['label']
                score = float(res['score'])
                if label == "POSITIVE":
                    st.success(f"✅ Positive ({score:.2f})")
                else:
                    st.error(f"❌ Negative ({score:.2f})")
            else:
                st.warning("⚠️ Please enter some text.")

tab1, tab2 = st.tabs(["🔸 Prediction", "📈 Insights"])

with tab2:
    st.subheader("Sentiment Distribution")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('data/train.csv')
    counts = df['label'].value_counts()
    labels = ['Positive', 'Negative']

    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

uploaded_file = st.file_uploader("📂 Upload CSV for Batch Prediction", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    preds = model.predict(tfidf.transform(df['text']))
    df['prediction'] = preds
    st.write(df.head())
    st.download_button(
        "Download Predictions",
        df.to_csv(index=False).encode('utf-8'),
        "predictions.csv",
        "text/csv"
    )

st.sidebar.title("🧠 Model Info")
st.sidebar.write(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
st.sidebar.write("LR Accuracy: ~85%")
st.sidebar.write("BERT Accuracy: ~88%")
st.sidebar.write("Dataset: IMDB (50k reviews)")

if text.strip():
    res = pipe(text)[0]
    st.write(f"Prediction: {res['label']}")
    st.progress(int(res['score']*100))
