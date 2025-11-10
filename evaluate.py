# src/evaluate.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion(y_true, y_pred, labels=[0,1], title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Example usage for TF-IDF LR:
def eval_tfidf_lr(test_csv='data/test.csv'):
    tfidf, lr = joblib.load('models/tfidf_lr.joblib')
    df = pd.read_csv(test_csv)
    X_test = tfidf.transform(df['text_clean'])
    preds = lr.predict(X_test)
    print(classification_report(df['label'], preds))
    plot_confusion(df['label'], preds)

if __name__ == '__main__':
    eval_tfidf_lr()

