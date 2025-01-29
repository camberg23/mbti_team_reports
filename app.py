import streamlit as st
import pandas as pd
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
    HRFlowable,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from markdown2 import markdown
from bs4 import BeautifulSoup
import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -----------------------------------------------------------------------------------
# Valid TypeFinder Types (16 variations)
# -----------------------------------------------------------------------------------
typefinder_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

def parse_tf_type(tf_str: str) -> str:
    """
    Return a valid 4-letter TypeFinder code if recognized, else empty.
    (As requested by user)
    """
    if not tf_str:
        return ""
    tf_str = str(tf_str).strip().upper()
    if tf_str in typefinder_types:
        return tf_str
    return ""

# -----------------------------------------------------------------------------------
# Updated Prompts (Minimal Adjustments)
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
Create a comprehensive team report with the following sections:

1. Intro & Type Distribution (combined)
2. Analysis of Dimension Preferences
3. Team Insights
4. Next Steps

Follow these guidelines:
- Keep language short, clear, and professional.
- Round all percentages to the nearest whole number.
- Never use "MBTI" anywhere.
- “Dimension” for the four pairs, “preference” only for one side of a dimension.
"""

prompts = {
    # Added note: "mention each user who has that type"
    "Intro_and_Type_Distribution": """
{INITIAL_CONTEXT}

**Types Not on the Team:**
{NOT_ON_TEAM_LIST}

**Your Role:**

Write **Section 1: Intro & Type Distribution** as a single combined section.

## Section 1: Intro & Type Distribution

1. **Introduction (short)**
   - Briefly (1–2 paragraphs) explain the TypeFinder framework (the four dimensions).
   - State how understanding these types helps teams collaborate effectively.

2. **Type Distribution**
   - Present the percentages for each TypeFinder type (ISTJ, ENFP, etc.) based on the provided data.
   - Under a subheading `### Types on the Team`, list each type present (with short bullet points describing it, plus count & %), **and explicitly include the user names** for each type (e.g., "ISTJ (2 members: Alice, Bob)...").
   - Under a subheading `### Types Not on the Team`, incorporate the data above (count=0, 0%).
   - Briefly discuss how certain distributions might affect communication/decision-making.

- Approximately 600 words total.

**Begin your combined section below:**
""",

    # Added note: "reference fractional/partial dimension data" explicitly
    "Analysis of Dimension Preferences": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 2: Analysis of Dimension Preferences**.

## Section 2: Analysis of Dimension Preferences

- Use the **fractional preference counts** (e.g. E=5.4, I=4.6) provided in the context to explain how each dimension (E vs I, S vs N, T vs F, J vs P) is distributed across the team.
- Emphasize that these counts are derived from partial percentages where available.
- 1–2 paragraphs per dimension describing how the team’s unique blend shapes collaboration.
- ~600 words total.

**Continue your report below:**
""",

    "Team Insights": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 3: Team Insights**.

## Section 3: Team Insights

Use the following subheadings:

1. **Strengths**  
   - At least four strengths, each in **bold** on a single line, followed by a paragraph.

2. **Potential Blind Spots**  
   - At least four potential challenges, same bolded format + paragraph.

3. **Communication**  
   - 1–2 paragraphs about how dimension splits influence communication patterns.

4. **Teamwork**  
   - 1–2 paragraphs about collaboration, synergy, workflow.

5. **Conflict**  
   - 1–2 paragraphs about friction points and suggestions for healthy resolution.

~700 words total.

**Continue the report below:**
""",

    "Next Steps": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 4: Next Steps**.

## Section 4: Next Steps

- Provide actionable recommendations for team leaders to leverage the TypeFinder composition.
- Use subheadings (###) for each category.
- Bullet points or numbered lists, with blank lines between items.
- End immediately after the final bullet (no concluding paragraph).
- ~400 words.

**Conclude the report below:**
"""
}

# -----------------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------------

st.title('TypeFinder Team Report Generator (CSV-based, with fractional dimension data)')

# Cover Page Inputs
st.subheader("Cover Page Details")
logo_path = "truity_logo.png"
company_name = st.text_input("Company Name (for cover page)", "Channing Realty")
team_name = st.text_input("Team Name (for cover page)", "Marketing Team")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

# CSV Upload
st.subheader("Upload CSV with columns: User Name, TF Type, TF E/I, TF N/S, TF F/T, TF J/P")
uploaded_csv = st.file_uploader("CSV File", type=["csv"])

if st.button("Generate Report from CSV"):
    if not uploaded_csv:
        st.error("Please upload a CSV first.")
    else:
        with st.spinner("Generating..."):
            df = pd.read_csv(uploaded_csv)

            # Gather valid rows
            valid_rows = []
            for i, row in df.iterrows():
                nm_val = row.get("User Name", "")
                tf_val = row.get("TF Type", "")
                e_val = row.get("TF E/I", "")
                n_val = row.get("TF N/S", "")
                f_val = row.get("TF F/T", "")
                j_val = row.get("TF J/P", "")

                name_str = str(nm_val).strip()
                tf_parsed = parse_tf_type(tf_val)

                if name_str and tf_parsed:
                    # dimension floats
                    try:
                        eF = float(e_val) if e_val != "" else None
                        nF = float(n_val) if n_val != "" else None
                        fF = float(f_val) if f_val != "" else None
                        jF = float(j_val) if j_val != "" else None
                    except:
                        eF = nF = fF = jF = None
                    valid_rows.append((name_str, tf_parsed, eF, nF, fF, jF))

            if not valid_rows:
                st.error("No valid TypeFinder rows found in CSV.")
            else:
                # Summation of fractional data
                dimension_sums = {'E':0.0,'I':0.0,'S':0.0,'N':0.0,'T':0.0,'F':0.0,'J':0.0,'P':0.0}
                dimension_rows_count = 0

                # We'll also store names by type for "Types on the Team"
                name_map_by_type = {}

                # Build "Team Members and their TF Types"
                team_list_str = ""
                tf_types_list = []

                for idx, (username, code, eF, nF, fF, jF) in enumerate(valid_rows, start=1):
                    team_list_str += f"{idx}. {username}: {code}\n"
                    tf_types_list.append(code)

                    # For naming each user in "### Types on the Team"
                    if code not in name_map_by_type:
                        name_map_by_type[code] = []
                    name_map_by_type[code].append(username)

                    # If row has dimension data, accumulate fractional sums
                    if (eF is not None and nF is not None and fF is not None and jF is not None):
                        e_frac = eF/100.0
                        dimension_sums['E'] += e_frac
                        dimension_sums['I'] += (1.0 - e_frac)

                        n_frac = nF/100.0
                        dimension_sums['N'] += n_frac
                        dimension_sums['S'] += (1.0 - n_frac)

                        f_frac = fF/100.0
                        dimension_sums['F'] += f_frac
                        dimension_sums['T'] += (1.0 - f_frac)

                        j_frac = jF/100.0
                        dimension_sums['J'] += j_frac
                        dimension_sums['P'] += (1.0 - j_frac)

                        dimension_rows_count += 1

                total_members = len(valid_rows)

                # If partial dimension data, scale
                if dimension_rows_count > 0:
                    scale_factor = float(total_members)/dimension_rows_count
                else:
                    scale_factor = 1.0

                for k in dimension_sums:
                    dimension_sums[k] *= scale_factor

                # Round to get integer "counts"
                preference_counts = {k: round(v) for k, v in dimension_sums.items()}

                # Pair-based % function
                def pair_percent(a,b):
                    s = preference_counts[a] + preference_counts[b]
                    if s <= 0:
                        return (0,0)
                    pa = round((preference_counts[a]/s)*100)
                    pb = 100 - pa
                    return (pa,pb)

                final_pref_counts = {}
                final_pref_pcts = {}
                for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                    a, b = pair
                    final_pref_counts[a] = preference_counts[a]
                    final_pref_counts[b] = preference_counts[b]
                    pa, pb = pair_percent(a,b)
                    final_pref_pcts[a] = pa
                    final_pref_pcts[b] = pb

                # Build preference breakdown text
                preference_breakdowns = ""
                for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                    a, b = pair
                    preference_breakdowns += f"**{a} vs {b}**\n"
                    preference_breakdowns += f"- {a}: {final_pref_counts[a]} equivalent members ({final_pref_pcts[a]}%)\n"
                    preference_breakdowns += f"- {b}: {final_pref_counts[b]} equivalent members ({final_pref_pcts[b]}%)\n\n"

                # Type distribution
                type_counts = Counter(tf_types_list)
                type_percentages = {
                    t: round((c/total_members)*100)
                    for t,c in type_counts.items()
                }
                # For "Types Not on the Team"
                absent_types = [t for t in typefinder_types if t not in type_counts]
                not_on_team_list_str = "None (All Types Represented)" if not absent_types else ""
                for missing_t in absent_types:
                    not_on_team_list_str += f"- {missing_t} (0%)\n"

                type_breakdowns = ""
                for t, c in type_counts.items():
                    pc = type_percentages[t]
                    type_breakdowns += f"- {t}: {c} members ({pc}%)\n"

                # Build bar chart for type distribution
                sns.set_style('whitegrid')
                plt.rcParams.update({'font.family':'serif'})
                plt.figure(figsize=(10,6))
                sorted_t = sorted(type_counts.keys())
                sorted_ct = [type_counts[x] for x in sorted_t]
                sns.barplot(x=sorted_t, y=sorted_ct, palette='viridis')
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

                # Build preference pie charts
                preference_plots = {}
                for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                    a, b = pair
                    valA = final_pref_counts[a]
                    valB = final_pref_counts[b]
                    plt.figure(figsize=(6,6))
                    plt.pie(
                        [valA, valB],
                        labels=[a,b],
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=sns.color_palette('pastel'),
                        textprops={'fontsize':12,'fontfamily':'serif'}
                    )
                    plt.tight_layout()
                    buf2 = io.BytesIO()
                    plt.savefig(buf2, format='png')
                    buf2.seek(0)
                    preference_plots[a+b] = buf2.getvalue()
                    plt.close()

                # Build the "TEAM_MEMBERS_LIST" but LLM also needs to see exactly which names are in each type
                # We'll embed that in the "TEAM_MEMBERS_LIST" so the LLM can mention them
                # Already we have a line for each user, but let's also provide a short summary
                # This is minimal: we won't break anything else
                # We'll place an additional "Type -> user names" note near the bottom
                user_names_by_type_text = "\n\n**User Names by Type**:\n"
                for t in sorted(name_map_by_type.keys()):
                    user_list_str = ", ".join(name_map_by_type[t])
                    user_names_by_type_text += f"- {t}: {user_list_str}\n"

                # Now we incorporate this into team_members_list so LLM sees it
                team_members_list = team_list_str + user_names_by_type_text

                # Prepare the LLM context
                from langchain.prompts import PromptTemplate
                initial_context_template = PromptTemplate.from_template(initial_context)
                formatted_initial_context = initial_context_template.format(
                    TEAM_SIZE=str(total_members),
                    TEAM_MEMBERS_LIST=team_members_list,
                    PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
                    TYPE_BREAKDOWNS=type_breakdowns.strip()
                )

                # LLM
                chat_model = ChatOpenAI(
                    openai_api_key=st.secrets['API_KEY'],
                    model_name='gpt-4o-2024-08-06',
                    temperature=0.2
                )

                # Generate sections
                report_sections = {}
                report_so_far = ""
                section_order = [
                    "Intro_and_Type_Distribution",
                    "Analysis of Dimension Preferences",
                    "Team Insights",
                    "Next Steps"
                ]
                for sec in section_order:
                    if sec == "Intro_and_Type_Distribution":
                        # pass the list of missing types
                        prompt_template = PromptTemplate.from_template(prompts[sec])
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_initial_context.strip(),
                            "REPORT_SO_FAR": report_so_far.strip(),
                            "NOT_ON_TEAM_LIST": not_on_team_list_str
                        }
                    else:
                        prompt_template = PromptTemplate.from_template(prompts[sec])
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_initial_context.strip(),
                            "REPORT_SO_FAR": report_so_far.strip()
                        }

                    chain = LLMChain(prompt=prompt_template, llm=chat_model)
                    section_txt = chain.run(**prompt_vars)
                    report_sections[sec] = section_txt.strip()
                    report_so_far += "\n\n" + section_txt.strip()

                # Display in Streamlit
                for s in section_order:
                    st.markdown(report_sections[s])
                    if s == "Intro_and_Type_Distribution":
                        st.header("Type Distribution Plot")
                        st.image(type_distribution_plot, use_column_width=True)
                    if s == "Analysis of Dimension Preferences":
                        for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                            st.header(f"{pair[0]} vs {pair[1]} Preference Distribution")
                            st.image(preference_plots[pair[0]+pair[1]], use_column_width=True)

                # PDF with cover
                def build_cover_page(logo_path, type_system_name, company_name, team_name, date_str):
                    from reportlab.platypus import Spacer, Paragraph, Image as RImage, HRFlowable, PageBreak
                    from reportlab.lib.enums import TA_CENTER
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib import colors

                    c_elems = []
                    styles = getSampleStyleSheet()

                    cover_title_style = ParagraphStyle(
                        'CoverTitle',
                        parent=styles['Title'],
                        fontName='Times-Bold',
                        fontSize=24,
                        leading=28,
                        alignment=TA_CENTER,
                        spaceAfter=20
                    )
                    cover_text_style = ParagraphStyle(
                        'CoverText',
                        parent=styles['Normal'],
                        fontName='Times-Roman',
                        fontSize=14,
                        alignment=TA_CENTER,
                        spaceAfter=8
                    )
                    c_elems.append(Spacer(1,80))
                    try:
                        logo_img = RImage(logo_path, width=140, height=52)
                        c_elems.append(logo_img)
                    except:
                        pass

                    c_elems.append(Spacer(1,50))
                    title_p = Paragraph(f"{type_system_name} For The Workplace<br/>Team Report", cover_title_style)
                    c_elems.append(title_p)
                    c_elems.append(Spacer(1,50))
                    sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                    c_elems.append(sep)
                    c_elems.append(Spacer(1,20))
                    comp_p = Paragraph(company_name, cover_text_style)
                    c_elems.append(comp_p)
                    tm_p = Paragraph(team_name, cover_text_style)
                    c_elems.append(tm_p)
                    dt_p = Paragraph(date_str, cover_text_style)
                    c_elems.append(dt_p)
                    c_elems.append(Spacer(1,60))
                    c_elems.append(PageBreak())
                    return c_elems

                def convert_markdown_to_pdf(
                    report_dict,
                    distribution_plot,
                    preference_plots_dict,
                    logo_path,
                    company_name,
                    team_name,
                    date_str
                ):
                    pdf_buf = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buf, pagesize=letter)
                    elements = []

                    # Cover
                    cover_elems = build_cover_page(logo_path,"TypeFinder",company_name,team_name,date_str)
                    elements.extend(cover_elems)

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

                    def process_md(md_text):
                        html = markdown(md_text, extras=['tables'])
                        soup = BeautifulSoup(html, 'html.parser')
                        for elem in soup.contents:
                            if isinstance(elem, str):
                                continue
                            if elem.name == 'h1':
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
                            elif elem.name == 'table':
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
                                    table_row = [c.get_text(strip=True) for c in cols]
                                    table_data.append(table_row)
                                if table_data:
                                    t = Table(table_data, hAlign='LEFT')
                                    t.setStyle(TableStyle([
                                        ('BACKGROUND',(0,0),(-1,0),colors.grey),
                                        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                        ('ALIGN',(0,0),(-1,-1),'CENTER'),
                                        ('FONTNAME',(0,0),(-1,0),'Times-Bold'),
                                        ('FONTNAME',(0,1),(-1,-1),'Times-Roman'),
                                        ('BOTTOMPADDING',(0,0),(-1,0),12),
                                        ('GRID',(0,0),(-1,-1),1,colors.black),
                                    ]))
                                    elements.append(t)
                                    elements.append(Spacer(1,12))
                            else:
                                elements.append(Paragraph(elem.get_text(strip=True), styleN))
                                elements.append(Spacer(1,12))

                    # Process each of the 4 sections
                    for s_name in ["Intro_and_Type_Distribution","Analysis of Dimension Preferences","Team Insights","Next Steps"]:
                        process_md(report_dict[s_name])
                        if s_name == "Intro_and_Type_Distribution":
                            elements.append(Spacer(1,12))
                            dist_buf = io.BytesIO(distribution_plot)
                            dist_img = ReportLabImage(dist_buf, width=400, height=240)
                            elements.append(dist_img)
                            elements.append(Spacer(1,12))
                        if s_name == "Analysis of Dimension Preferences":
                            for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                                elements.append(Spacer(1,12))
                                elements.append(Paragraph(f"{pair[0]} vs {pair[1]} Preference Distribution", styleH2))
                                pbuf = io.BytesIO(preference_plots_dict[pair[0]+pair[1]])
                                pimg = ReportLabImage(pbuf, width=300, height=300)
                                elements.append(pimg)
                                elements.append(Spacer(1,12))

                    doc.build(elements)
                    pdf_buffer.seek(0)
                    return pdf_buffer

                pdf_data = convert_markdown_to_pdf(
                    report_dict=report_sections,
                    distribution_plot=type_distribution_plot,
                    pref_plots=preference_plots,
                    logo_path=logo_path,
                    company_name=company_name,
                    team_name=team_name,
                    date_str=custom_date
                )

                st.download_button(
                    "Download Report as PDF",
                    data=pdf_data,
                    file_name="typefinder_team_report.pdf",
                    mime="application/pdf"
                )
