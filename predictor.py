import re
import string
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load model and label encoder once
pipeline = joblib.load('model_pipeline.joblib')
le = joblib.load('label_encoder.joblib')


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw job descriptions DataFrame for model prediction.
    """
    jobs_df = raw_df.copy()

    # Combine text columns into a single column
    text_cols = ['job_description', 'skills', 'company_profile', 'benefits', 'responsibilities']
    jobs_df['combined_text'] = jobs_df[text_cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

    # Clean the combined text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    jobs_df['combined_text'] = jobs_df['combined_text'].apply(clean_text)
    jobs_df.drop(columns=text_cols, inplace=True)

    return jobs_df


def predict_job_titles(raw_data: pd.DataFrame) -> list:
    """
    Given a raw DataFrame, preprocesses and returns the predicted job titles.
    """
    preprocessed_data = preprocess_data(raw_data)

    # Get class probabilities
    probas = pipeline.predict_proba(preprocessed_data)

    # Get top 5 predictions for each row
    top_results = []
    for row in probas:
        top_indices = np.argsort(row)[-10:][::-1]  # Indices of top 5 predictions
        top_titles = le.inverse_transform(top_indices)  # Decode to job titles
        top_scores = row[top_indices]  # Corresponding probabilities
        top_results.append(list(zip(top_titles, top_scores)))

    return top_results
