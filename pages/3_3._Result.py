import numpy as np
import streamlit
import pickle
import re
import string

# Get the data
if 'input' not in streamlit.session_state:
    streamlit.switch_page("1_1._Resume_Upload.py")
else:
    input_data = streamlit.session_state['input']


def load_pickle_file(file_path):
    """
    Loads data from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The deserialized Python object, or None if an error occurs.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


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


rf_model = load_pickle_file("rf.pkl")
knn_model = load_pickle_file("knn_predictor.pkl")
vectorizer = load_pickle_file("tfidf_vectorizer.pkl")

cleaned_text = clean_text(input_data)
vectorized_input = vectorizer.transform([cleaned_text])

def get_top_3(mod, vec_input):
    probas = mod.predict_proba(vec_input)[0]
    top_n = 3
    top_indices = np.argsort(probas)[::-1][:top_n]
    top_classes = mod.classes_[top_indices]
    top_scores = probas[top_indices]
    return list(zip(top_classes, top_scores))

predictions = get_top_3(rf_model, vectorized_input) + get_top_3(knn_model, vectorized_input)
sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
print(sorted_predictions)

streamlit.markdown("# **Top 5 Predicted Roles**")

# Displaying the sorted job titles
x = 1
for job, __ in sorted_predictions:
    streamlit.markdown("## " + str(x) + ". " + job)
    x += 1
