import streamlit
import pandas as pd

streamlit.title("Review the parsed data")

if 'data' in streamlit.session_state:

    dfSkills = pd.DataFrame(columns=["Skills"])
    streamlit.markdown("# Skills")
    edited_dfSkills = streamlit.data_editor(dfSkills, num_rows="dynamic")

    dfEducation = pd.DataFrame(columns=["School", "Field of Study", "Degree Level"])
    streamlit.markdown("# Education")
    edited_dfEducation = streamlit.data_editor(dfEducation, num_rows="dynamic", column_config=
    {"Degree Level": streamlit.column_config.SelectboxColumn("Degree Level",
                                                             options=["PhD", "Master's", "Bachelor's", "Associate's"])})


    dfExperience = pd.DataFrame(columns=["Title", "Duration", "Company"])
    streamlit.markdown("# Experience")
    edited_dfExperience = streamlit.data_editor(dfExperience, num_rows="dynamic")

    streamlit.write(streamlit.session_state['data'])

else:
    streamlit.write("We were unable to parse your resume :(")
    if streamlit.button("Back"):
        streamlit.switch_page("1_1._Resume_Upload.py")
