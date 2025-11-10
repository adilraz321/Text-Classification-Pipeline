# src/train_baselines.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from preprocess import load_and_preprocess

# ✅ Ensure model directory exists before saving anything
os.makedirs('models', exist_ok=True)

def train_and_eval():
    # Load preprocessed data
    train_df, val_df, test_df = load_and_preprocess()

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_df['text_clean'])
    X_val = tfidf.transform(val_df['text_clean'])
    X_test = tfidf.transform(test_df['text_clean'])

    # =============================
    # Logistic Regression Baseline
    # =============================
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1)
    lr.fit(X_train, train_df['label'])
    preds_lr = lr.predict(X_test)

    print("\n=== Logistic Regression Results ===")
    print("LR Accuracy:", accuracy_score(test_df['label'], preds_lr))
    print("LR F1:", f1_score(test_df['label'], preds_lr))
    print(classification_report(test_df['label'], preds_lr))

    # ✅ Save Logistic Regression model
    joblib.dump((tfidf, lr), 'models/tfidf_lr.joblib')
    print("✅ Saved Logistic Regression model to 'models/tfidf_lr.joblib'")

    # =============================
    # Support Vector Machine Model
    # =============================
    svm = LinearSVC(C=1.0)
    svm.fit(X_train, train_df['label'])
    preds_svm = svm.predict(X_test)

    print("\n=== SVM Results ===")
    print("SVM Accuracy:", accuracy_score(test_df['label'], preds_svm))
    print("SVM F1:", f1_score(test_df['label'], preds_svm))
    print(classification_report(test_df['label'], preds_svm))

    # ✅ Save SVM model
    joblib.dump((tfidf, svm), 'models/tfidf_svm.joblib')
    print("✅ Saved SVM model to 'models/tfidf_svm.joblib'")

if __name__ == '__main__':
    train_and_eval()
