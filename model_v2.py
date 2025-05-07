# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read Data
df = pd.read_pickle("./data/processed_data.pkl")
df.head(5)

# Create Copy of Imported Data for safety
copy_df = df.copy()
copy_df

# Drop Columns
copy_df.drop(
    columns=[
        "skills",
        "responsibilities",
        "work_type",
        "min_experience",
        "max_experience",
        "qualifications",
    ],
    inplace=True,
)
# Combine Role with the feature vector
copy_df["feature_combined"] = copy_df["role"] + " " + copy_df["combined_text"]
copy_df.drop(columns=["role", "combined_text"], inplace=True)

# Clean the feature vector
import re
import string

def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"\n", " ", text)  # Remove newlines
    text = re.sub(r"\[.*?\]", "", text)  # Remove text in brackets
    text = re.sub(
        r"[%s]" % re.escape(string.punctuation), "", text
    )  # Remove punctuation
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words with numbers
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()


copy_df["feature_combined"] = copy_df["feature_combined"].swifter.apply(clean_text)

"""
Model Construction
"""
model_df = copy_df.copy()

X = model_df["feature_combined"]
y = model_df["job_title"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

"""
Random Forest Baseline
"""
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
y_pred = rf.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
KNN Algorithm
"""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_tfidf, y_train)
y_pred_knn = knn.predict(X_test_tfidf)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy: {accuracy:.4f}")

# Pickle file both models and tfidf vectorizer

import pickle

pickle.dump(rf, open("randomforest_predictor.pkl", "wb"))
pickle.dump(knn, open("knn_predictor.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
