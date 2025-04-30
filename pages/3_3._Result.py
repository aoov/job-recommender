import streamlit
from predictor import predict_job_titles

if 'input_df' not in streamlit.session_state:
    streamlit.switch_page("1_1._Resume_Upload.py")
else:
    input_df = streamlit.session_state['input_df']
titles = predict_job_titles(input_df)
# Extracting the inner list and sorting it by the second value (probability) in descending order
sorted_jobs = [job for job, _ in sorted(titles[0], key=lambda x: x[1], reverse=True)]

streamlit.markdown("# **Recommended Jobs For You**")

# Displaying the sorted job titles
x = 1
for job in sorted_jobs:
    streamlit.markdown("## " + str(x) + ". " + job)
    x += 1
