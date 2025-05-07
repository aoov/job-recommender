import streamlit
from PyPDF2 import PdfReader
from pyresparser import ResumeParser
from streamlit_pdf_viewer import pdf_viewer

# TODO Better title?

streamlit.markdown(
    """
    # Welcome to Job Role Finder!

To begin, upload your resume below and we will find roles and job titles that best match your education, previous positions and skills.
"""
)

resume_upload = streamlit.file_uploader("Upload your resume", type=["csv", "pdf"])

if 'uploaded' in streamlit.session_state and streamlit.session_state['uploaded'] is not None:
    resume_upload = streamlit.session_state['uploaded']

if resume_upload is not None:
    data = ResumeParser(resume_upload).get_extracted_data()
    reader = PdfReader(resume_upload)
    page = reader.pages[0]
    pdfText = page.extract_text()
    streamlit.session_state['raw'] = pdfText
    streamlit.session_state['data'] = data
    streamlit.session_state['uploaded'] = resume_upload
    try:
        pdf_viewer(resume_upload.getvalue())
    except:
        print("Error in pdf viewer")
    if streamlit.button("Continue"):
        streamlit.switch_page("pages/2_2._Data_Review.py")
