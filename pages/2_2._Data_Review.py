import streamlit
import pandas as pd
from utils import parse_experience_list

streamlit.title("Review the parsed data")
if 'data' not in streamlit.session_state:
    streamlit.write("We were unable to parse your resume :(")
    streamlit.write("You can try again or manually input your data below")
    if streamlit.button("Back"):
        streamlit.switch_page("1_1._Resume_Upload.py")
if 'data' in streamlit.session_state:
    data = streamlit.session_state['data']
else:
    data = {'skills': [], 'college_name': "", 'degree': "", "experience": [], 'designation': ''}
dfSkills = pd.DataFrame({'Skills': data['skills']})
streamlit.markdown("# Skills")
edited_dfSkills = streamlit.data_editor(dfSkills, num_rows="dynamic")

dfEducation = pd.DataFrame(columns=["School", "Field of Study", "Degree Level"])
new_row = pd.DataFrame([{
    "School": data.get('college_name', ''),
    "Field of Study": data.get('degree', ''),
    "Degree Level": None
}])
dfEducation = pd.concat([dfEducation, new_row], ignore_index=True)
streamlit.markdown("# Education")
edited_dfEducation = streamlit.data_editor(dfEducation, num_rows="dynamic", column_config=
{"Degree Level": streamlit.column_config.SelectboxColumn("Degree Level",
                                                         options=["M.Tech", "BCA",
                                                                  "PhD", "MBA", "MCA", "MA",
                                                                  "M.Com", "BBA", "B.Tech", "B.Com", "BA"])})
dfExperience = parse_experience_list(data['experience'])
streamlit.markdown("# Experience")
edited_dfExperience = streamlit.data_editor(dfExperience, num_rows="dynamic")

skills_text = ' '.join(edited_dfSkills.applymap(str).values.flatten())
education_text = ' '.join(edited_dfEducation.applymap(str).values.flatten())
experience_text = ' '.join(edited_dfExperience.applymap(str).values.flatten())

if streamlit.button("Continue"):
    print(experience_text)
    input_df = pd.DataFrame({
        'qualifications': [education_text],
        'location': [''],
        'country': [''],
        'work_type': [''],
        'company_size': [26801],
        'preference': [''],
        'role': [data['designation']],
        'min_experience': [0],
        'max_experience': [15],
        'min_salary': [59],
        'max_salary': [99],
        'job_description': [experience_text],
        'benefits': [""],
        'skills': [edited_dfSkills['Skills'].tolist()],
        'responsibilities': [""],
        'company': [''],
        'company_profile': [""]
    })
    streamlit.session_state['input_df'] = input_df
    streamlit.switch_page("pages/3_3._Result.py")


