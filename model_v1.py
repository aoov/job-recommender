# Step 1: Import Libraries and Load Data
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import joblib

# Load the dataset
raw_df = pd.read_csv('job_descriptions.csv')
jobs_df = raw_df.copy()

# Ensure downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === Step 1: Preprocessing ===
# Drop unnecessary columns
columns_to_drop = ['latitude', 'longitude', 'Job Posting Date', 'Contact Person', 'Contact', 'Job Id']
jobs_df.drop(columns=columns_to_drop, inplace=True)
# Rename columns to snake_case
jobs_df.columns = jobs_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Fill missing values in 'company_profile' with 'Unavailable'
jobs_df['company_profile'].fillna('Unavailable', inplace=True)

# Extract numerical features from 'experience' and 'salary_range'
jobs_df[['min_experience', 'max_experience']] = jobs_df['experience'].str.extract(r'(\d+)\s*to\s*(\d+)').astype(float)
jobs_df[['min_salary', 'max_salary']] = jobs_df['salary_range'].str.replace('K', '').str.replace('$', '') \
    .str.extract(r'(\d+)\s*-\s*(\d+)').astype(float)

# Drop original 'experience' and 'salary_range' columns
jobs_df.drop(columns=['experience', 'salary_range'], inplace=True)

# Combine text fields into 'combined_text'
text_cols = ['job_description', 'skills', 'company_profile', 'benefits', 'responsibilities']
jobs_df['combined_text'] = jobs_df[text_cols].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Clean text fields
def clean_text_columns(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

jobs_df['combined_text'] = jobs_df['combined_text'].apply(clean_text_columns)

# Drop individual text columns
jobs_df.drop(columns=text_cols, inplace=True)

# === Step 2: Encode the Target Variable (job_title) ===
le = LabelEncoder()
jobs_df['job_title'] = le.fit_transform(jobs_df['job_title']).astype(str)

# Save the LabelEncoder
joblib.dump(le, 'label_encoder.joblib')

# Split the dataset into train and test sets
train_set, test_set = train_test_split(jobs_df, test_size=0.2, random_state=42, stratify=jobs_df['job_title'])

# Define preprocessing steps
text_transformer = TfidfVectorizer(max_features=1000)
numerical_transformer = StandardScaler()

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'combined_text'),
        ('num', numerical_transformer, ['min_experience', 'max_experience', 'min_salary', 'max_salary'])
    ]
)

# Create the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
X_train = train_set.drop(columns=['job_title', 'Country', 'Salary Range', 'Job Posting Date',
                                  'Contact Person', "Contact", "Job Portal", "Benefits"])
y_train = train_set['job_title']
pipeline.fit(X_train, y_train)

# Save the pipeline
joblib.dump(pipeline, 'model_pipeline.joblib')

le.inverse_transform([122])
