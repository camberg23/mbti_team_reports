import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import io
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as ReportLabImage,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup

# -------------------------------
# Initial Context and Prompts
# -------------------------------

# Define the initial context shared across all LLM calls
initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments based on the TypeFinder framework.

**Team Size:** {TEAM_SIZE}

**Team Members and their TypeFinder Types:**

{TEAM_MEMBERS_LIST}

**TypeFinder Preference Breakdowns:**

{PREFERENCE_BREAKDOWNS}

**TypeFinder Type Breakdowns:**

{TYPE_BREAKDOWNS}

Your task is to contribute to a comprehensive team personality report. The report consists of five sections, and you will be responsible for generating one of them. Please ensure that your section aligns seamlessly with the previous sections and maintains a consistent tone and style throughout the report.

The sections are:

1. **Team Profile**

2. **Type Distribution**

3. **Team Insights**

4. **Type Preference Breakdown**

5. **Actions and Next Steps**

**Formatting Requirements:**

- Use clear headings and subheadings for your section.
- Write in Markdown format for easy readability.
- Use bullet points and tables where appropriate.
- Ensure the content is specific to the provided team TypeFinder types and offers unique insights.
- Avoid redundancy with previous sections!
- Round all percentages to nearest whole number (eg, 60%, not 60.0%)
- CRITICAL: NEVER OUTPUT THE PHRASE 'MBTI,' USE 'TypeFinder' IN PLACE OF IT!
"""

# Define prompts for each section
prompts = {
    "Team Profile": """
{INITIAL_CONTEXT}

**Your Role:**

You are responsible for writing the **Team Profile** section of the report.

**Section 1: Team Profile**

- Begin by explaining each TypeFinder preference (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving).
- For each preference, use the provided counts and percentages to describe the team composition (e.g., "There are X who are Introverted (I) and Y who are Extraverted (E), representing A% and B% of the team, respectively"), and subsequently, the 'team preference' for that specific dichotomy.
- After covering all four preferences, present the overall 'team type' based on the most common traits.
- Provide an analysis of the overall team type, including key characteristics and how it influences team dynamics.
- **Use the provided data; do not compute any new statistics.**
- Required length: Approximately 500 words.

**Begin your section below:**
""",
    "Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Type Distribution** section of the report.

**Section 2: Type Distribution**

- Begin with a **TypeFinder Type Breakdown**: Present the percentage breakdown of each TypeFinder type within the team using the provided data.
- Include a section on **Team Similarity**: Discuss how similarities among team members might influence team cohesion and collaboration.
- Include a section on **Team Diversity**: Discuss how differences among team members contribute to a variety of perspectives and skills within the team.
- **Use the provided data; do not compute any new statistics.**
- Required length: Approximately 500 words.

**Continue the report by adding your section below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Team Insights** section of the report.

**Section 3: Team Insights**

- Under the **Team Insights** header, create two subheadings: **Strengths** and **Potential Blind Spots**.
- For **Strengths**, identify at least four strengths of the team. Each strength should be presented as a bolded sentence, followed by a paragraph expanding on it.
- For **Potential Blind Spots**, identify at least four potential blind spots or challenges. Each should be presented as a bolded sentence, followed by a paragraph expanding on it.
- Ensure that the strengths and blind spots are based on the prevalent and less represented personality traits present in the team.
- Required length: Approximately 700 words total (350 words for strengths, 350 words for blind spots).

**Continue the report by adding your section below:**
""",
    "Type Preference Breakdown": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Type Preference Breakdown** section of the report.

**Section 4: Type Preference Breakdown**

- For each TypeFinder dimension (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving):
  - Use the provided counts and percentages to describe the team's percentage distribution.
  - Create a separate table for each dimension using the data provided.
  - Under each table, immediately provide a 200-word discussion of the implications of this specific distribution for workplace dynamics (no new header necessary).
- Explain what these percentages mean for team communication, decision-making, and problem-solving.
- **Use the provided data; do not compute any new statistics.**
- Required length: Approximately 800 words.

**Continue the report by adding your section below:**
""",
    "Actions and Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

You are responsible for writing the **Actions and Next Steps** section of the report.

**Section 5: Actions and Next Steps**

- Provide actionable recommendations for team leaders to enhance collaboration and performance, based on the analysis in the previous sections and the specific TypeFinder types present in the team.
- Structure the recommendations with subheadings for each area of action. For each area, briefly justify why the personality composition leads you to make that recommendation.
- Under each subheading, provide some bullet points or numbered lists of specific actions.
- Don't make these suggestions too laborious or impractical to actually implement.
- Immediately end your outputs after the last bullet, do not add anything after the final bullet! i.e., NO concluding filler text after this (NO filler concluding paragraph like "By following these recommendations...", "By integrating these actions...").
- Required length: Approximately 400 words.

**Conclude the report by adding your section below:**
"""
}

# -------------------------------
# TypeFinder Types
# -------------------------------

typefinder_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

# -------------------------------
# Callback Function
# -------------------------------

def randomize_types_callback():
    randomized_types = [random.choice(typefinder_types) for _ in range(int(st.session_state['team_size']))]
    for i in range(int(st.session_state['team_size'])):
        key = f'mbti_{i}'
        st.session_state[key] = randomized_types[i]

# -------------------------------
# Streamlit App Layout
# -------------------------------

# Initialize the 'team_size' in session_state if not present
if 'team_size' not in st.session_state:
    st.session_state['team_size'] = 5

# Input for team size
team_size = st.number_input(
    'Enter the number of team members (up to 30)', 
    min_value=1, max_value=30, value=5, key='team_size'
)

# Add a button to randomize TypeFinder types
st.button('Randomize Types', on_click=randomize_types_callback)

# Initialize list to store TypeFinder types
team_typefinder_types = []

# Input for TypeFinder types of each team member
st.header('Enter TypeFinder types for each team member')

# Ensure that session_state has entries for all team members
for i in range(int(team_size)):
    if f'mbti_{i}' not in st.session_state:
        st.session_state[f'mbti_{i}'] = 'Select TypeFinder Type'

# Display selection boxes
for i in range(int(team_size)):
    mbti_type = st.selectbox(
        f'Team Member {i+1}',
        options=['Select TypeFinder Type'] + typefinder_types,
        key=f'mbti_{i}'
    )
    if mbti_type != 'Select TypeFinder Type':
        team_typefinder_types.append(mbti_type)
    else:
        team_typefinder_types.append(None)  # Ensure the list has the same length as team_size

# Submit button
if st.button('Generate Report'):
    if None in team_typefinder_types:
        st.error('Please select TypeFinder types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Prepare the team types as a string
            team_types_str = ', '.join(team_typefinder_types)
            
            # Prepare the team members list
            team_members_list = "\n".join([
                f"{i+1}. Team Member {i+1}: {mbti_type}" 
                for i, mbti_type in enumerate(team_typefinder_types)
            ])
            
            # Compute counts and percentages for preferences
            preference_counts = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
            for t in team_typefinder_types:
                if len(t) == 4:
                    preference_counts[t[0]] += 1  # E or I
                    preference_counts[t[1]] += 1  # S or N
                    preference_counts[t[2]] += 1  # T or F
                    preference_counts[t[3]] += 1  # J or P
            total_members = len(team_typefinder_types)
            preference_percentages = {k: round((v / total_members) * 100) for k, v in preference_counts.items()}

            # Compute counts and percentages for types
            type_counts = Counter(team_typefinder_types)
            type_percentages = {k: int((v / total_members) * 100) for k, v in type_counts.items()}

            # Prepare the preference breakdowns string
            preference_breakdowns = ""
            for dichotomy in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
                count1 = preference_counts[dichotomy[0]]
                count2 = preference_counts[dichotomy[1]]
                perc1 = preference_percentages[dichotomy[0]]
                perc2 = preference_percentages[dichotomy[1]]
                preference_breakdowns += f"**{dichotomy[0]} vs {dichotomy[1]}**\n"
                preference_breakdowns += f"- {dichotomy[0]}: {count1} members ({perc1}%)\n"
                preference_breakdowns += f"- {dichotomy[1]}: {count2} members ({perc2}%)\n\n"

            # Prepare the type breakdowns string
            type_breakdowns = "**TypeFinder Type Breakdown**\n"
            for t, count in type_counts.items():
                perc = type_percentages[t]
                type_breakdowns += f"- {t}: {count} members ({perc}%)\n"

            # Generate plots
            sns.set_style('whitegrid')  # Improve plot aesthetics
            plt.rcParams.update({'font.family': 'serif'})  # Set plot fonts to serif

            plots = {}
            # Plot for TypeFinder Type Distribution
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=list(type_counts.keys()), 
                y=list(type_counts.values()), 
                palette='viridis'
            )
            plt.title('TypeFinder Type Distribution', fontsize=16)
            plt.xlabel('TypeFinder Types', fontsize=14)
            plt.ylabel('Number of Team Members', fontsize=14)
            plt.xticks(rotation=45)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            type_distribution_plot = buf.getvalue()
            plots['type_distribution'] = type_distribution_plot
            plt.close()

            # Plot for each preference dichotomy
            preference_plots = {}
            for dichotomy in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
                labels = [dichotomy[0], dichotomy[1]]
                sizes = [preference_counts[dichotomy[0]], preference_counts[dichotomy[1]]]
                plt.figure(figsize=(6, 6))
                plt.pie(
                    sizes, 
                    labels=labels, 
                    autopct='%1.1f%%', 
                    startangle=90, 
                    colors=sns.color_palette('pastel'),
                    textprops={'fontsize': 12, 'fontfamily': 'serif'}
                )
                plt.title(f'{dichotomy[0]} vs {dichotomy[1]} Preference Distribution', fontsize=14)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_data = buf.getvalue()
                preference_plots[''.join(dichotomy)] = plot_data
                plt.close()

            # Initialize the LLM
            chat_model = ChatOpenAI(
                openai_api_key=st.secrets['API_KEY'], 
                model_name='gpt-4o-2024-08-06', 
                temperature=0.2
            )

            # Prepare the initial context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list,
                TEAM_TYPES=team_types_str,
                PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
                TYPE_BREAKDOWNS=type_breakdowns.strip()
            )

            # Initialize variables to store the report
            report_sections = {}
            report_so_far = ""
            # Iterate over each section
            for section_name in [
                "Team Profile", 
                "Type Distribution", 
                "Team Insights", 
                "Type Preference Breakdown", 
                "Actions and Next Steps"
            ]:
                # Prepare the prompt
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                # Prepare the variables for the prompt
                prompt_variables = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                # Create the chain
                chat_chain = LLMChain(prompt=prompt_template, llm=chat_model)
                # Generate the section
                section_text = chat_chain.run(**prompt_variables)
                # Store the section
                report_sections[section_name] = section_text.strip()
                # Update the report so far
                report_so_far += f"\n\n{section_text.strip()}"

            # Combine all sections into the final report
            final_report = "\n\n".join([
                report_sections["Team Profile"],
                report_sections["Type Distribution"],
                report_sections["Team Insights"],
                report_sections["Type Preference Breakdown"],
                report_sections["Actions and Next Steps"]
            ])

            # Display the report sections and intersperse plots
            for section_name in [
                "Team Profile", 
                "Type Distribution", 
                "Team Insights", 
                "Type Preference Breakdown", 
                "Actions and Next Steps"
            ]:
                st.markdown(report_sections[section_name])
                if section_name == "Type Distribution":
                    # Display Type Distribution plot
                    st.header("Type Distribution Plot")
                    st.image(plots['type_distribution'], use_column_width=True)
                if section_name == "Type Preference Breakdown":
                    # Display preference plots
                    for dichotomy in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
                        key = ''.join(dichotomy)
                        st.header(f"{dichotomy[0]} vs {dichotomy[1]} Preference Distribution")
                        st.image(preference_plots[key], use_column_width=True)

            # -------------------------------
            # PDF Generation Function
            # -------------------------------

            def convert_markdown_to_pdf(report_sections_dict, plots_dict, preference_plots_dict):
                # Initialize PDF document
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()
                
                # Define a serif font style
                styleN = ParagraphStyle(
                    'Normal',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                )
                styleH = ParagraphStyle(
                    'Heading',
                    parent=styles['Heading1'],
                    fontName='Times-Bold',
                    fontSize=18,
                    leading=22,
                    spaceAfter=10,
                )
                styleB = ParagraphStyle(
                    'Body',
                    parent=styles['BodyText'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                )
                styleList = ParagraphStyle(
                    'List',
                    parent=styles['Normal'],
                    fontName='Times-Roman',
                    fontSize=12,
                    leading=14,
                    leftIndent=20,
                )
                
                from markdown2 import markdown
                from bs4 import BeautifulSoup

                def process_markdown(text):
                    html = markdown(text)
                    soup = BeautifulSoup(html, 'html.parser')
                    for elem in soup:
                        if elem.name in ['h1', 'h2', 'h3']:
                            elements.append(Paragraph(elem.text, styleH))
                        elif elem.name == 'p':
                            elements.append(Paragraph(elem.decode_contents(), styleN))
                        elif elem.name == 'ul':
                            for li in elem.find_all('li'):
                                elements.append(Paragraph('• ' + li.text, styleList))
                        elif elem.name == 'strong':
                            elements.append(Paragraph('<b>%s</b>' % elem.text, styleN))
                        elif elem.name == 'table':
                            # Parse table
                            table_data = []
                            rows = elem.find_all('tr')
                            for row in rows:
                                cols = row.find_all(['td', 'th'])
                                table_row = [col.text.strip() for col in cols]
                                # Skip separator rows
                                if all(cell.strip().replace('-', '').replace(':', '') == '' for cell in table_row):
                                    continue
                                table_data.append(table_row)
                            if table_data:
                                t = Table(table_data, hAlign='LEFT')
                                t.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                                ]))
                                elements.append(t)
                                elements.append(Spacer(1, 12))
                        else:
                            elements.append(Paragraph(elem.text, styleN))
                        elements.append(Spacer(1, 12))

                # Add each section and corresponding plots
                for section_name in [
                    "Team Profile", 
                    "Type Distribution", 
                    "Team Insights", 
                    "Type Preference Breakdown", 
                    "Actions and Next Steps"
                ]:
                    # Add the section text
                    section_text = report_sections_dict[section_name]
                    process_markdown(section_text)
                    
                    # After specific sections, add plots
                    if section_name == "Type Distribution":
                        # Add Type Distribution plot
                        elements.append(Spacer(1, 12))
                        img_buffer = io.BytesIO(plots_dict['type_distribution'])
                        img = ReportLabImage(img_buffer, width=400, height=240)  # Adjust size as needed
                        elements.append(img)
                        elements.append(Spacer(1, 12))
                    if section_name == "Type Preference Breakdown":
                        # Add preference plots
                        for dichotomy in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
                            key = ''.join(dichotomy)
                            elements.append(Spacer(1, 12))
                            img_buffer = io.BytesIO(preference_plots_dict[key])
                            elements.append(Paragraph(f"{dichotomy[0]} vs {dichotomy[1]} Preference Distribution", styleH))
                            img = ReportLabImage(img_buffer, width=300, height=300)  # Adjust size as needed
                            elements.append(img)
                            elements.append(Spacer(1, 12))

                # Build PDF
                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            # -------------------------------
            # Generate PDF
            # -------------------------------

            # Convert the report to PDF
            pdf_data = convert_markdown_to_pdf(report_sections, plots, preference_plots)

            # Download button for the report
            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="team_personality_report.pdf",
                mime="application/pdf"
            )


# import streamlit as st
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import random
# from collections import Counter

# # Define the initial context shared across all LLM calls
# initial_context = """
# You are an expert organizational psychologist specializing in team dynamics and personality assessments based on the TypeFinder framework.

# **Team Size:** {TEAM_SIZE}

# **Team Members and their TypeFinder Types:**

# {TEAM_MEMBERS_LIST}

# **TypeFinder Preference Breakdowns:**

# {PREFERENCE_BREAKDOWNS}

# **TypeFinder Type Breakdowns:**

# {TYPE_BREAKDOWNS}

# Your task is to contribute to a comprehensive team personality report. The report consists of five sections, and you will be responsible for generating one of them. Please ensure that your section aligns seamlessly with the previous sections and maintains a consistent tone and style throughout the report.

# The sections are:

# 1. **Team Profile**

# 2. **Type Distribution**

# 3. **Team Insights**

# 4. **Type Preference Breakdown**

# 5. **Actions and Next Steps**

# **Formatting Requirements:**

# - Use clear headings and subheadings for your section.
# - Write in Markdown format for easy readability.
# - Use bullet points and tables where appropriate.
# - Ensure the content is specific to the provided team TypeFinder types and offers unique insights.
# - Avoid redundancy with previous sections!
# - Round all percentages to nearest whole number (eg, 60%, not 60.0%)
# - CRITICAL: NEVER OUTPUT THE PHRASE 'MBTI,' USE 'TypeFinder' IN PLACE OF IT!
# """

# # Define prompts for each section, incorporating your feedback
# prompts = {
#     "Team Profile": """
# {INITIAL_CONTEXT}

# **Your Role:**

# You are responsible for writing the **Team Profile** section of the report.

# **Section 1: Team Profile**

# - Begin by explaining each TypeFinder preference (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving).
# - For each preference, use the provided counts and percentages to describe the team composition (e.g., "There are X who are Introverted (I) and Y who are Extraverted (E), representing A% and B% of the team, respectively"), and subsequently, the 'team preference' for that specific dichotomy.
# - After covering all four preferences, present the overall 'team type' based on the most common traits.
# - Provide an analysis of the overall team type, including key characteristics and how it influences team dynamics.
# - **Use the provided data; do not compute any new statistics.**
# - Required length: Approximately 500 words.

# **Begin your section below:**
# """,
#     "Type Distribution": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# You are responsible for writing the **Type Distribution** section of the report.

# **Section 2: Type Distribution**

# - Begin with a **TypeFinder Type Breakdown**: Present the percentage breakdown of each TypeFinder type within the team using the provided data.
# - Include a section on **Team Similarity**: Discuss how similarities among team members might influence team cohesion and collaboration.
# - Include a section on **Team Diversity**: Discuss how differences among team members contribute to a variety of perspectives and skills within the team.
# - **Use the provided data; do not compute any new statistics.**
# - Required length: Approximately 500 words.

# **Continue the report by adding your section below:**
# """,
#     "Team Insights": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# You are responsible for writing the **Team Insights** section of the report.

# **Section 3: Team Insights**

# - Under the **Team Insights** header, create two subheadings: **Strengths** and **Potential Blind Spots**.
# - For **Strengths**, identify at least four strengths of the team. Each strength should be presented as a bolded sentence, followed by a paragraph expanding on it.
# - For **Potential Blind Spots**, identify at least four potential blind spots or challenges. Each should be presented as a bolded sentence, followed by a paragraph expanding on it.
# - Ensure that the strengths and blind spots are based on the prevalent and less represented personality traits present in the team.
# - Required length: Approximately 700 words total (350 words for strengths, 350 words for blind spots).

# **Continue the report by adding your section below:**
# """,
#     "Type Preference Breakdown": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# You are responsible for writing the **Type Preference Breakdown** section of the report.

# **Section 4: Type Preference Breakdown**

# - For each TypeFinder dimension (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving):
#   - Use the provided counts and percentages to describe the team's percentage distribution.
#   - Create a separate table for each dimension using the data provided.
#   - Under each table, immediately provide a 200-word discussion of the implications of this specific distribution for workplace dynamics (no new header necessary).
# - Explain what these percentages mean for team communication, decision-making, and problem-solving.
# - **Use the provided data; do not compute any new statistics.**
# - Required length: Approximately 800 words.

# **Continue the report by adding your section below:**
# """,
#     "Actions and Next Steps": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# You are responsible for writing the **Actions and Next Steps** section of the report.

# **Section 5: Actions and Next Steps**

# - Provide actionable recommendations for team leaders to enhance collaboration and performance, based on the analysis in the previous sections and the specific TypeFinder types present in the team.
# - Structure the recommendations with subheadings for each area of action. For each area, briefly justify why the personality composition leads you to make that recommendation.
# - Under each subheading, provide some bullet points or numbered lists of specific actions.
# - Don't make these suggestions too laborious or impractical to actually implement.
# - Immediately end your outputs after the last bullet, do not add anything after the final bullet! i.e., NO concluding filler text after this (NO filler concluding paragraph like "By following these recommendations...", "By integrating these actions...").
# - Required length: Approximately 400 words.

# **Conclude the report by adding your section below:**
# """
# }

# # Define the list of TypeFinder types
# typefinder_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP',
#                     'INFJ', 'INFP', 'ENFJ', 'ENFP',
#                     'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
#                     'ISTP', 'ISFP', 'ESTP', 'ESFP']

# # Function to randomize TypeFinder types and trigger a rerun
# def randomize_types_callback():
#     randomized_types = [random.choice(typefinder_types) for _ in range(int(st.session_state['team_size']))]
#     for i in range(int(st.session_state['team_size'])):
#         key = f'mbti_{i}'
#         st.session_state[key] = randomized_types[i]
#     # st.rerun()  # Removed to avoid re-running the script

# # Initialize the 'team_size' in session_state if not present
# if 'team_size' not in st.session_state:
#     st.session_state['team_size'] = 5

# # Input for team size
# team_size = st.number_input('Enter the number of team members (up to 30)', 
#                             min_value=1, max_value=30, value=5, key='team_size')

# # Add a button to randomize TypeFinder types
# st.button('Randomize Types', on_click=randomize_types_callback)

# # Initialize list to store TypeFinder types
# team_typefinder_types = []

# # Input for TypeFinder types of each team member
# st.header('Enter TypeFinder types for each team member')

# # Ensure that session_state has entries for all team members
# for i in range(int(team_size)):
#     if f'mbti_{i}' not in st.session_state:
#         st.session_state[f'mbti_{i}'] = 'Select TypeFinder Type'

# # Display selection boxes
# for i in range(int(team_size)):
#     mbti_type = st.selectbox(
#         f'Team Member {i+1}',
#         options=['Select TypeFinder Type'] + typefinder_types,
#         key=f'mbti_{i}'
#     )
#     if mbti_type != 'Select TypeFinder Type':
#         team_typefinder_types.append(mbti_type)
#     else:
#         team_typefinder_types.append(None)  # Ensure the list has the same length as team_size

# # Submit button
# if st.button('Generate Report'):
#     if None in team_typefinder_types:
#         st.error('Please select TypeFinder types for all team members.')
#     else:
#         with st.spinner('Generating report, please wait...'):
#             # Prepare the team types as a string
#             team_types_str = ', '.join(team_typefinder_types)
#             # Prepare the team members list
#             team_members_list = "\n".join([f"{i+1}. Team Member {i+1}: {mbti_type}" 
#                                            for i, mbti_type in enumerate(team_typefinder_types)])
#             # Compute counts and percentages for preferences
#             preference_counts = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
#             for t in team_typefinder_types:
#                 if len(t) == 4:
#                     preference_counts[t[0]] += 1  # E or I
#                     preference_counts[t[1]] += 1  # S or N
#                     preference_counts[t[2]] += 1  # T or F
#                     preference_counts[t[3]] += 1  # J or P
#             total_members = len(team_typefinder_types)
#             preference_percentages = {k: (v / total_members) * 100 for k, v in preference_counts.items()}

#             # Compute counts and percentages for types
#             type_counts = Counter(team_typefinder_types)
#             type_percentages = {k: int((v / total_members) * 100) for k, v in type_counts.items()}

#             # Prepare the preference breakdowns string
#             preference_breakdowns = ""
#             for dichotomy in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
#                 count1 = preference_counts[dichotomy[0]]
#                 count2 = preference_counts[dichotomy[1]]
#                 perc1 = preference_percentages[dichotomy[0]]
#                 perc2 = preference_percentages[dichotomy[1]]
#                 preference_breakdowns += f"**{dichotomy[0]} vs {dichotomy[1]}**\n"
#                 preference_breakdowns += f"- {dichotomy[0]}: {count1} members ({perc1:.1f}%)\n"
#                 preference_breakdowns += f"- {dichotomy[1]}: {count2} members ({perc2:.1f}%)\n\n"

#             # Prepare the type breakdowns string
#             type_breakdowns = "**TypeFinder Type Breakdown**\n"
#             for t, count in type_counts.items():
#                 perc = type_percentages[t]
#                 type_breakdowns += f"- {t}: {count} members ({perc:.1f}%)\n"

#             # Initialize the LLM
#             chat_model = ChatOpenAI(
#                 openai_api_key=st.secrets['API_KEY'], 
#                 model_name='gpt-4o-2024-08-06', 
#                 temperature=0.2
#             )

#             # Prepare the initial context
#             initial_context_template = PromptTemplate.from_template(initial_context)
#             formatted_initial_context = initial_context_template.format(
#                 TEAM_SIZE=str(team_size),
#                 TEAM_MEMBERS_LIST=team_members_list,
#                 TEAM_TYPES=team_types_str,
#                 PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
#                 TYPE_BREAKDOWNS=type_breakdowns.strip()
#             )

#             # Initialize variables to store the report
#             report_sections = []
#             report_so_far = ""
#             # Iterate over each section
#             for section_name in ["Team Profile", "Type Distribution", "Team Insights", "Type Preference Breakdown", "Actions and Next Steps"]:
#                 # Prepare the prompt
#                 prompt_template = PromptTemplate.from_template(prompts[section_name])
#                 # Prepare the variables for the prompt
#                 prompt_variables = {
#                     "INITIAL_CONTEXT": formatted_initial_context.strip(),
#                     "REPORT_SO_FAR": report_so_far.strip()
#                 }
#                 # Create the chain
#                 chat_chain = LLMChain(prompt=prompt_template, llm=chat_model)
#                 # Generate the section
#                 section_text = chat_chain.run(**prompt_variables)
#                 # Append the section to the report
#                 report_sections.append(section_text.strip())
#                 # Update the report so far
#                 report_so_far += f"\n\n{section_text.strip()}"

#             # Combine all sections into the final report
#             final_report = "\n\n".join(report_sections)

#             # Display the report using markdown
#             st.markdown(final_report)

#             # Download button for the report
#             st.download_button(
#                 label="Download Report",
#                 data=final_report,
#                 file_name="team_personality_report.md",
#                 mime="text/markdown"
#             )
