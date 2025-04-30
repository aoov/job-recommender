import re

import pandas as pd


def parse_experience_list(raw_list):
    # Helper to clean tabs, newlines, and excess spaces
    def clean_text(text):
        return re.sub(r'[\t\r\n]+', ' ', text).strip()

    if raw_list is None or len(raw_list) == 0:
        return pd.DataFrame(columns=["Title", "Duration", "Company", "Job Description", 'Job Responsibilities'])

    entries = []
    i = 0

    while i < len(raw_list):
        line = raw_list[i]

        # Handle the case where title and duration are on the same line
        title_duration_line = raw_list[i]
        title_duration_parts = title_duration_line.split("–")
        if len(title_duration_parts) > 1:  # This indicates a duration is present in the line
            title_line = title_duration_parts[0].strip()
            duration = title_duration_parts[-1].strip()
        else:
            title_line = title_duration_line.strip()
            duration = ""

        # Identify the company on the next line
        company_line = raw_list[i + 1].strip() if i + 1 < len(raw_list) else ""
        job_desc = ""
        responsibilities = []

        # Next lines might be bullets or description
        j = i + 2
        while j < len(raw_list):
            current_line = raw_list[j].strip()

            if current_line.startswith("•") or current_line.startswith("●"):  # Bullet points
                text = clean_text(current_line.lstrip("•●").strip())
                if not job_desc:
                    job_desc = text
                else:
                    responsibilities.append(text)
            elif any(keyword in current_line for keyword in ["–", "-", "to", "present", "Present"]):
                # A line with duration format, break when found
                if not duration:
                    duration = clean_text(current_line)
                break
            else:
                # Continuation of a line
                if responsibilities:
                    responsibilities[-1] += " " + current_line.strip()
                elif job_desc:
                    job_desc += " " + current_line.strip()
            j += 1

        # Clean up title and company
        title_parts = [clean_text(part) for part in title_line.split(",")]
        title = title_parts[0] if title_parts else ""
        company = ", ".join(title_parts[1:]).strip() if len(title_parts) > 1 else company_line.strip()

        # Add entry to the list
        entries.append({
            "Title": title,
            "Duration": duration,
            "Company": company,
            "Job Description": job_desc,
            "Job Responsibilities": "\n".join(responsibilities)
        })

        # Move index pointer
        i = j

    return pd.DataFrame(entries)