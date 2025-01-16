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

# -----------------------------------------------------------------------------------
# TypeFinder Types (16 common variations)
# -----------------------------------------------------------------------------------
typefinder_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

def randomize_types_callback():
    randomized_types = [random.choice(typefinder_types) for _ in range(int(st.session_state['team_size']))]
    for i in range(int(st.session_state['team_size'])):
        key = f'mbti_{i}'
        st.session_state[key] = randomized_types[i]

# -----------------------------------------------------------------------------------
# Updated Prompts Reflecting Best Practices
# -----------------------------------------------------------------------------------

initial_context = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments using the TypeFinder framework.

TypeFinder has four primary dimensions:
1) Extraversion (E) vs. Introversion (I)
2) Sensing (S) vs. Intuition (N)
3) Thinking (T) vs. Feeling (F)
4) Judging (J) vs. Perceiving (P)

We avoid the term "MBTI" and call it "TypeFinder."
We use “dimension” to describe the four pairs, and “preference” only when referencing one side (e.g., “preference for Thinking”).

**Team Size:** {TEAM_SIZE}

**Team Members and their TypeFinder Types:**
{TEAM_MEMBERS_LIST}

**Preference Counts/Percentages:**
{PREFERENCE_BREAKDOWNS}

**Type Counts/Percentages:**
{TYPE_BREAKDOWNS}

Your goal:
Create a comprehensive team report with the following structure:

1. Introduction
2. Analysis of Type Distribution
3. Analysis of Dimension Preferences
4. Team Insights
5. Next Steps

Follow these guidelines:
- Always write in Markdown with clear headings (`##`, `###`, etc.).
- Round percentages to the nearest whole number.
- Never use the word “MBTI.”
- Maintain a professional, neutral tone.
"""

prompts = {
    "Introduction": """
{INITIAL_CONTEXT}

**Your Role:**

Write **Section 1: Introduction** to the TypeFinder Team Report.

## Section 1: Introduction

- Briefly explain the TypeFinder framework (4 dimensions, each with 2 preferences).
- Provide a short rationale for how these preferences/types can help teams collaborate effectively.
- Optionally introduce the concept of an overall "team type" if you'd like, or hold that for a later section.
- ~400 words.

**Begin your section below:**
""",
    "Analysis of Type Distribution": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 2: Analysis of Type Distribution**.

## Section 2: Analysis of Type Distribution

- Present the percentages for each TypeFinder type (e.g., ISTJ, ENFP) from the provided data.
- List the types **on the team** (with short bullet points describing each, plus count & %).
- List any types **not** on the team (0%, absent).
- Include a brief discussion of how certain distributions might affect communication and decision-making.
- ~500 words.

**Continue your report below:**
""",
    "Analysis of Dimension Preferences": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 3: Analysis of Dimension Preferences**.

## Section 3: Analysis of Dimension Preferences

- For each dimension (E vs I, S vs N, T vs F, J vs P):
  - Provide the counts/percentages for each preference (already in context, no new math).
  - 1–2 paragraphs discussing how that preference split affects the team.
- Insert references to relevant visual aids (pie charts) if desired.
- If you wish, you can also mention a "Team Type" if it helps summarize the majority preferences. 
- ~600 words total.

**Continue your report below:**
""",
    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 4: Team Insights**.

## Section 4: Team Insights

Create subheadings:

1. **Strengths**  
   - At least four strengths, each in **bold** on one line, then a paragraph.

2. **Potential Blind Spots**  
   - At least four challenges. Same formatting (bold line, then paragraph).

3. **Communication**  
   - 1–2 paragraphs on how dimension splits shape communication.

4. **Teamwork**  
   - 1–2 paragraphs on collaboration, workflow, synergy.

5. **Conflict**  
   - 1–2 paragraphs on possible friction points, plus suggestions.

Total ~700 words.

**Continue the report below:**
""",
    "Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 5: Next Steps**.

## Section 5: Next Steps

- Provide **actionable** recommendations for leveraging the TypeFinder composition.
- Use subheadings (###) for each category of recommendations.
- Under each, bullet points or numbered lists with blank lines between items.
- End immediately after the final bullet (no concluding paragraph).
- ~400 words.

**Conclude the report below:**
"""
}

# -----------------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------------

st.title('TypeFinder Team Report Generator')

if 'team_size' not in st.session_state:
    st.session_state['team_size'] = 5

team_size = st.number_input(
    'Enter the number of team members (up to 30)', 
    min_value=1, max_value=30, value=5, key='team_size'
)

st.button('Randomize Types', on_click=randomize_types_callback)

st.subheader('Enter TypeFinder types for each team member')
for i in range(int(team_size)):
    if f'mbti_{i}' not in st.session_state:
        st.session_state[f'mbti_{i}'] = 'Select TypeFinder Type'

team_typefinder_types = []
for i in range(int(team_size)):
    selected_type = st.selectbox(
        f'Team Member {i+1}',
        options=['Select TypeFinder Type'] + typefinder_types,
        key=f'mbti_{i}'
    )
    if selected_type != 'Select TypeFinder Type':
        team_typefinder_types.append(selected_type)
    else:
        team_typefinder_types.append(None)

if st.button('Generate Report'):
    if None in team_typefinder_types:
        st.error('Please select TypeFinder types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Build a list of members
            team_members_list = "\n".join([
                f"{i+1}. Team Member {i+1}: {t}"
                for i, t in enumerate(team_typefinder_types)
            ])

            # Count preferences
            preference_counts = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
            total_members = len(team_typefinder_types)
            for t in team_typefinder_types:
                if len(t) == 4:
                    preference_counts[t[0]] += 1  # E or I
                    preference_counts[t[1]] += 1  # S or N
                    preference_counts[t[2]] += 1  # T or F
                    preference_counts[t[3]] += 1  # J or P

            preference_percentages = {
                k: round((v / total_members) * 100)
                for k, v in preference_counts.items()
            }

            # Build a preference breakdown string
            preference_breakdowns = ""
            for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                a, b = pair[0], pair[1]
                preference_breakdowns += f"**{a} vs {b}**\n"
                preference_breakdowns += f"- {a}: {preference_counts[a]} members ({preference_percentages[a]}%)\n"
                preference_breakdowns += f"- {b}: {preference_counts[b]} members ({preference_percentages[b]}%)\n\n"

            # Count TypeFinder type distribution
            type_counts = Counter(team_typefinder_types)
            type_percentages = {
                k: round((v / total_members) * 100)
                for k, v in type_counts.items()
            }
            
            # Build a type breakdown string
            type_breakdowns = ""
            for t, c in type_counts.items():
                p = type_percentages[t]
                type_breakdowns += f"- {t}: {c} members ({p}%)\n"

            # Generate bar plot for type distribution
            sns.set_style('whitegrid')
            plt.rcParams.update({'font.family':'serif'})
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
            plt.close()

            # Generate preference pie charts
            preference_plots = {}
            for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                labels = [pair[0], pair[1]]
                sizes = [preference_counts[pair[0]], preference_counts[pair[1]]]
                plt.figure(figsize=(6,6))
                plt.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=sns.color_palette('pastel'),
                    textprops={'fontsize':12, 'fontfamily':'serif'}
                )
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                preference_plots[''.join(pair)] = buf.getvalue()
                plt.close()

            # Prepare LLM
            chat_model = ChatOpenAI(
                openai_api_key=st.secrets['API_KEY'],
                model_name='gpt-4o-2024-08-06',
                temperature=0.2
            )

            # Format the initial context
            initial_context_template = PromptTemplate.from_template(initial_context)
            formatted_initial_context = initial_context_template.format(
                TEAM_SIZE=str(team_size),
                TEAM_MEMBERS_LIST=team_members_list,
                PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
                TYPE_BREAKDOWNS=type_breakdowns.strip()
            )

            # Generate sections
            report_sections = {}
            report_so_far = ""

            # The five sections in our new structure:
            # 1) Introduction
            # 2) Analysis of Type Distribution
            # 3) Analysis of Dimension Preferences
            # 4) Team Insights
            # 5) Next Steps
            section_names = [
                "Introduction",
                "Analysis of Type Distribution",
                "Analysis of Dimension Preferences",
                "Team Insights",
                "Next Steps"
            ]

            for section_name in section_names:
                prompt_template = PromptTemplate.from_template(prompts[section_name])
                prompt_vars = {
                    "INITIAL_CONTEXT": formatted_initial_context.strip(),
                    "REPORT_SO_FAR": report_so_far.strip()
                }
                chain = LLMChain(prompt=prompt_template, llm=chat_model)
                section_text = chain.run(**prompt_vars)
                report_sections[section_name] = section_text.strip()
                report_so_far += f"\n\n{section_text.strip()}"

            # Display the final text in the Streamlit app
            for s_name in section_names:
                st.markdown(report_sections[s_name])
                # Show distribution plot after "Analysis of Type Distribution"
                if s_name == "Analysis of Type Distribution":
                    st.header("Type Distribution Plot")
                    st.image(type_distribution_plot, use_column_width=True)
                # Show preference pie charts after "Analysis of Dimension Preferences"
                if s_name == "Analysis of Dimension Preferences":
                    for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                        key = ''.join(pair)
                        st.header(f"{pair[0]} vs {pair[1]} Preference Distribution")
                        st.image(preference_plots[key], use_column_width=True)

            # ----------------------------------------------------
            # PDF Generation
            # ----------------------------------------------------
            def convert_markdown_to_pdf(report_dict, dist_plot, pref_plots):
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                elements = []
                styles = getSampleStyleSheet()

                styleH1 = ParagraphStyle(
                    'Heading1Custom',
                    parent=styles['Heading1'],
                    fontName='Times-Bold',
                    fontSize=18,
                    leading=22,
                    spaceAfter=10,
                )
                styleH2 = ParagraphStyle(
                    'Heading2Custom',
                    parent=styles['Heading2'],
                    fontName='Times-Bold',
                    fontSize=16,
                    leading=20,
                    spaceAfter=8,
                )
                styleH3 = ParagraphStyle(
                    'Heading3Custom',
                    parent=styles['Heading3'],
                    fontName='Times-Bold',
                    fontSize=14,
                    leading=18,
                    spaceAfter=6,
                )
                styleH4 = ParagraphStyle(
                    'Heading4Custom',
                    parent=styles['Heading4'],
                    fontName='Times-Bold',
                    fontSize=12,
                    leading=16,
                    spaceAfter=4,
                )

                styleN = ParagraphStyle(
                    'Normal',
                    parent=styles['Normal'],
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

                def process_markdown(md_text):
                    html = markdown(md_text, extras=['tables'])
                    soup = BeautifulSoup(html, 'html.parser')
                    for elem in soup.contents:
                        if isinstance(elem, str):
                            continue

                        # Table
                        if elem.name == 'table':
                            table_data = []
                            thead = elem.find('thead')
                            if thead:
                                header_row = []
                                for th in thead.find_all('th'):
                                    header_row.append(th.get_text(strip=True))
                                if header_row:
                                    table_data.append(header_row)
                            tbody = elem.find('tbody')
                            if tbody:
                                rows = tbody.find_all('tr')
                            else:
                                rows = elem.find_all('tr')
                            for row in rows:
                                cols = row.find_all(['td','th'])
                                table_row = [col.get_text(strip=True) for col in cols]
                                table_data.append(table_row)
                            if table_data:
                                t = Table(table_data, hAlign='LEFT')
                                t.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                                    ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                                    ('BOTTOMPADDING', (0,0),(-1,0),12),
                                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                                ]))
                                elements.append(t)
                                elements.append(Spacer(1,12))

                        elif elem.name == 'h1':
                            elements.append(Paragraph(elem.text, styleH1))
                            elements.append(Spacer(1,12))
                        elif elem.name == 'h2':
                            elements.append(Paragraph(elem.text, styleH2))
                            elements.append(Spacer(1,12))
                        elif elem.name == 'h3':
                            elements.append(Paragraph(elem.text, styleH3))
                            elements.append(Spacer(1,12))
                        elif elem.name == 'h4':
                            elements.append(Paragraph(elem.text, styleH4))
                            elements.append(Spacer(1,12))

                        elif elem.name == 'p':
                            elements.append(Paragraph(elem.decode_contents(), styleN))
                            elements.append(Spacer(1,12))

                        elif elem.name == 'ul':
                            for li in elem.find_all('li', recursive=False):
                                elements.append(Paragraph('• ' + li.text, styleList))
                                elements.append(Spacer(1,6))

                        else:
                            elements.append(Paragraph(elem.get_text(strip=True), styleN))
                            elements.append(Spacer(1,12))

                # Build PDF from each section
                for s_name in [
                    "Introduction",
                    "Analysis of Type Distribution",
                    "Analysis of Dimension Preferences",
                    "Team Insights",
                    "Next Steps"
                ]:
                    process_markdown(report_dict[s_name])
                    # Insert distribution plot after Type Distribution
                    if s_name == "Analysis of Type Distribution":
                        elements.append(Spacer(1,12))
                        img_buf = io.BytesIO(dist_plot)
                        img = ReportLabImage(img_buf, width=400, height=240)
                        elements.append(img)
                        elements.append(Spacer(1,12))
                    # Insert preference plots after Dimension Preferences
                    if s_name == "Analysis of Dimension Preferences":
                        for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                            key = ''.join(pair)
                            elements.append(Spacer(1,12))
                            elements.append(Paragraph(f"{pair[0]} vs {pair[1]} Preference Distribution", styleH2))
                            pbuf = io.BytesIO(pref_plots[key])
                            pimg = ReportLabImage(pbuf, width=300, height=300)
                            elements.append(pimg)
                            elements.append(Spacer(1,12))

                doc.build(elements)
                pdf_buffer.seek(0)
                return pdf_buffer

            pdf_data = convert_markdown_to_pdf(report_sections, type_distribution_plot, preference_plots)

            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="typefinder_team_report.pdf",
                mime="application/pdf"
            )
