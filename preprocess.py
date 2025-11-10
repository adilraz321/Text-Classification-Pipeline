# src/preprocess.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os

# create 'data' folder if it doesn't exist
os.makedirs('data', exist_ok=True)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab') 

STOP = set(stopwords.words('english'))
LEMMA = WordNetLemmatizer()

def clean_text(text):
    # lower, remove HTML, non-alpha, extra spaces
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [LEMMA.lemmatize(t) for t in tokens if t not in STOP and len(t) > 1]
    return ' '.join(tokens)

def load_and_preprocess(csv_path='IMDB.csv'):
    df = pd.read_csv(csv_path)  # columns: review, sentiment
    df = df.dropna().reset_index(drop=True)
    df['text_clean'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'negative':0, 'positive':1})
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42, stratify=train_df['label'])
    # Train: 70%, Val: 15%, Test: 15%
    return train_df, val_df, test_df

if __name__ == '__main__':
    train_df, val_df, test_df = load_and_preprocess()
    print(train_df.shape, val_df.shape, test_df.shape)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
