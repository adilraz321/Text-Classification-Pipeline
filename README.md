# Text-Classification-Pipeline

This repository contains the complete code for an end-to-end text classification pipeline designed to perform sentiment analysis on a benchmark dataset of 50,000+ text samples. The project compares the performance of traditional machine learning models against modern, Transformer-based models, achieving upto 90+% accuracy with a fine-tuned Transformer.

This project was built to explore the full lifecycle of an NLP project, from raw data ingestion and preprocessing to model training, evaluation, and fine-tuning.

🚀 Features

Full Preprocessing Pipeline: Implements standard text cleaning techniques, including tokenization, stopword removal, stemming/lemmatization, and TF-IDF vectorization.

Model Benchmarking: Trains and evaluates several classification models, including:

Logistic Regression

Support Vector Machine (SVM)

Deep Learning Integration: Utilizes Transformer-based models from Hugging Face for fine-tuning, achieving state-of-the-art results (88% accuracy).

Data Analysis: Includes scripts for exploratory data analysis (EDA) to extract insights and visualize text patterns using Pandas and Matplotlib.

🛠️ Tech Stack
Python

Scikit-learn: For traditional ML models (Logistic Regression, SVM) and preprocessing (TF-IDF).

Pandas: For data manipulation and analysis.

Matplotlib: For data visualization.

Hugging Face transformers: For loading and fine-tuning Transformer models.

NLTK / spaCy (Assumed for tokenization/stemming)

📦 Installation & Usage
To get started with this project, clone the repository and install the required dependencies.

# Clone the repository
git clone https://github.com/adilraz321/Text-Classification-Pipeline.git
cd Text-Classification-Pipeline

# Install the required libraries
pip install -r requirements.txt

Running the Pipeline
Run the preprocessing script to clean the data (assuming preprocess.py):

Bash

python preprocess.py

Run the model training notebook or script (e.g., train.ipynb or train.py) to train the models and see the evaluation results.

Finally, Run the Streamlit app.
