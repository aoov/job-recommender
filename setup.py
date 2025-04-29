import nltk
from spacy.cli import download

nltk.download('stopwords', download_dir="venv/nltk_data")
download("en_core_web_sm")
