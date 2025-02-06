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
import numpy as np  # NEW: Required for enhanced plotting

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# -----------------------------------------------------------------------------------
# Enhanced Plotting Functions (for TypeFinder / MBTI)
# -----------------------------------------------------------------------------------
def _generate_pie_chart(data, slices):
    # Filter out slices with 0 counts
    filtered_slices = [s for s in slices if data.get(s['label'], 0) > 0]

    # Ensure all types are included, set count to 0 if missing
    total = sum(data.get(s['label'], 0) for s in filtered_slices)
    if total == 0:
        print("No data to plot.")
        return None

    # Calculate proportions and angles
    current_angle = 0
    for s in filtered_slices:
        count = data.get(s['label'], 0)
        proportion = count / total
        angle = proportion * 360
        s['theta1'] = current_angle
        s['theta2'] = current_angle + angle
        s['radius'] = 1.0 + proportion * 1.6  # Exaggerate radius based on proportion
        current_angle += angle

    # Plot the chart
    fig, ax = plt.subplots(figsize=(8, 8))
    for s in filtered_slices:
        wedge = plt.matplotlib.patches.Wedge(
            center=(0, 0),
            r=s['radius'],
            theta1=s['theta1'],
            theta2=s['theta2'],
            color=s['color'],
            edgecolor='white'
        )
        ax.add_patch(wedge)

    # Add labels
    for s in filtered_slices:
        theta_mid = (s['theta1'] + s['theta2']) / 2
        x = 1.2 * s['radius'] * np.cos(np.radians(theta_mid))  # Adjust label positions closer to slices
        y = 1.2 * s['radius'] * np.sin(np.radians(theta_mid))
        ax.text(
            x, y, s['label'],
            ha='center', va='center', fontsize=12, color=s['color'], fontweight='bold'
        )

    # Final touches
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def plot_mbti_chart(data):
    slices = [
        {'label': 'INTJ', 'color': '#355760'}, {'label': 'ENTP', 'color': '#3E7279'},
        {'label': 'INFJ', 'color': '#B6823E'}, {'label': 'ISFP', 'color': '#D2A26C'},
        {'label': 'ESTJ', 'color': '#DD7C65'}, {'label': 'ISTP', 'color': '#D35858'},
        {'label': 'ENFP', 'color': '#C5744B'}, {'label': 'INTP', 'color': '#87AC73'},
        {'label': 'ESTP', 'color': '#6E9973'}, {'label': 'ENTJ', 'color': '#4D8570'},
        {'label': 'INFP', 'color': '#7B8A2A'}, {'label': 'ESFJ', 'color': '#CC9D3B'},
        {'label': 'ESFP', 'color': '#FFCC00'}, {'label': 'ISFJ', 'color': '#8FA14F'},
        {'label': 'ISTJ', 'color': '#B8682E'}, {'label': 'ENFJ', 'color': '#F98274'}
    ]
    return _generate_pie_chart(data, slices)

def plot_preference_chart(pair, counts):
    # Define color schemes for each preference pair
    if pair == ('E', 'I'):
        slices = [{'label': 'E', 'color': '#A6CEE3'}, {'label': 'I', 'color': '#1F78B4'}]
    elif pair == ('S', 'N'):
        slices = [{'label': 'S', 'color': '#B2DF8A'}, {'label': 'N', 'color': '#33A02C'}]
    elif pair == ('T', 'F'):
        slices = [{'label': 'T', 'color': '#FB9A99'}, {'label': 'F', 'color': '#E31A1C'}]
    elif pair == ('J', 'P'):
        slices = [{'label': 'J', 'color': '#FDBF6F'}, {'label': 'P', 'color': '#FF7F00'}]
    else:
        slices = []
    data = {pair[0]: counts[pair[0]], pair[1]: counts[pair[1]]}
    return _generate_pie_chart(data, slices)

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
    """Return a valid 4-letter TypeFinder code if recognized, else empty."""
    if not tf_str:
        return ""
    tf_str = str(tf_str).strip().upper()
    if tf_str in typefinder_types:
        return tf_str
    return ""

# -----------------------------------------------------------------------------------
# Minimal Prompt Edits to Mention User Names + Fractional Sums
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
    # NOTE: We add instructions about naming each user in "Types on the Team"
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
   - Under a subheading `### Types on the Team`, list each type present (with short bullet points describing it, plus count & %),
     **and explicitly name each user who holds that type** (e.g., "ISTJ (2 members: Alice, Bob)...").
   - Under a subheading `### Types Not on the Team`, incorporate the data above (count=0, 0%).
   - Briefly discuss how certain distributions might affect communication/decision-making.

- Approximately 600 words total.

**Begin your combined section below:**
""",

    # NOTE: We add instructions to reference fractional sums explicitly
    "Analysis of Dimension Preferences": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 2: Analysis of Dimension Preferences**.

## Section 2: Analysis of Dimension Preferences

- Use the provided preference counts/percentages (which may be based on fractional sums) for each dimension (E vs I, S vs N, T vs F, J vs P).
- Clearly mention how the fractional or partial data was used to arrive at the final counts, and interpret these results for collaboration and workflow.
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
   - At least four strengths, each in **bold** (one line), followed by a paragraph.

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

st.title('TypeFinder Team Report Generator')

st.subheader("Cover Page Details")
logo_path = "truity_logo.png"
company_name = st.text_input("Company Name (for cover page)", "")
team_name = st.text_input("Team Name (for cover page)", "")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

st.subheader("Upload CSV")
csv_file = st.file_uploader("Choose a CSV file", type=["csv"])

if st.button("Generate Report from CSV"):
    if not csv_file:
        st.error("Please upload a valid CSV file first.")
    else:
        with st.spinner("Generating report..."):
            df = pd.read_csv(csv_file)

            valid_rows = []
            for i, row in df.iterrows():
                nm_val = row.get("User Name", "")
                tf_val = row.get("TF Type", "")
                ei_val = row.get("TF E/I", "")
                ns_val = row.get("TF N/S", "")
                ft_val = row.get("TF F/T", "")
                jp_val = row.get("TF J/P", "")

                name_str = str(nm_val).strip()
                tf_parsed = parse_tf_type(tf_val)

                if name_str and tf_parsed:
                    # Try dimension floats
                    try:
                        eFloat = float(ei_val) if ei_val != "" else None
                        nFloat = float(ns_val) if ns_val != "" else None
                        fFloat = float(ft_val) if ft_val != "" else None
                        jFloat = float(jp_val) if jp_val != "" else None
                    except:
                        eFloat = nFloat = fFloat = jFloat = None

                    valid_rows.append((name_str, tf_parsed, eFloat, nFloat, fFloat, jFloat))

            if not valid_rows:
                st.error("No valid TypeFinder rows found.")
            else:
                # dimension sums
                dimension_sums = {'E':0.0,'I':0.0,'S':0.0,'N':0.0,'T':0.0,'F':0.0,'J':0.0,'P':0.0}
                dimension_rows_count = 0

                team_list_str = ""
                type_list = []

                for idx, (nm, code, eF, nF, fF, jF) in enumerate(valid_rows, start=1):
                    team_list_str += f"{idx}. {nm}: {code}\n"
                    type_list.append(code)

                    if (eF is not None and nF is not None and fF is not None and jF is not None):
                        e_frac = eF/100.0
                        i_frac = 1 - e_frac
                        dimension_sums['E'] += e_frac
                        dimension_sums['I'] += i_frac

                        n_frac = nF/100.0
                        s_frac = 1 - n_frac
                        dimension_sums['N'] += n_frac
                        dimension_sums['S'] += s_frac

                        f_frac = fF/100.0
                        t_frac = 1 - f_frac
                        dimension_sums['F'] += f_frac
                        dimension_sums['T'] += t_frac

                        j_frac = jF/100.0
                        p_frac = 1 - j_frac
                        dimension_sums['J'] += j_frac
                        dimension_sums['P'] += p_frac

                        dimension_rows_count += 1

                total_members = len(valid_rows)

                # scale if partial dimension data
                if dimension_rows_count > 0:
                    scale_factor = float(total_members)/dimension_rows_count
                else:
                    scale_factor = 1.0

                for k in dimension_sums:
                    dimension_sums[k] *= scale_factor

                # Round to get "counts"
                preference_counts = {k: round(v) for k,v in dimension_sums.items()}

                # Convert to pair-based percentages
                def pair_percent(a,b):
                    s = preference_counts[a] + preference_counts[b]
                    if s <= 0: return (0,0)
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

                preference_breakdowns = ""
                for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                    a, b = pair
                    preference_breakdowns += f"**{a} vs {b}**\n"
                    preference_breakdowns += f"- {a}: {final_pref_counts[a]} members ({final_pref_pcts[a]}%)\n"
                    preference_breakdowns += f"- {b}: {final_pref_counts[b]} members ({final_pref_pcts[b]}%)\n\n"

                # Type distribution breakdown
                type_counts = Counter(type_list)
                type_percentages = {t: round((c/total_members)*100) for t,c in type_counts.items()}
                absent_types = [t for t in typefinder_types if t not in type_counts]
                not_on_team_list_str = ""
                if absent_types:
                    for t in absent_types:
                        not_on_team_list_str += f"- {t} (0%)\n"
                else:
                    not_on_team_list_str = "None (All Types Represented)"

                type_breakdowns = ""
                for t, c in type_counts.items():
                    pct = type_percentages[t]
                    type_breakdowns += f"- {t}: {c} members ({pct}%)\n"

                # Enhanced Type Distribution Plot using MBTI-style pie chart
                type_distribution_plot = plot_mbti_chart(type_counts)

                # Enhanced preference pie charts using the new plotting function
                preference_plots = {}
                for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                    plot_img = plot_preference_chart(pair, final_pref_counts)
                    preference_plots[''.join(pair)] = plot_img

                # Prepare LLM
                chat_model = ChatOpenAI(
                    openai_api_key=st.secrets['API_KEY'],
                    model_name='gpt-4o-2024-08-06',
                    temperature=0.2
                )

                # Format context
                initial_context_template = PromptTemplate.from_template(initial_context)
                formatted_initial_context = initial_context_template.format(
                    TEAM_SIZE=str(total_members),
                    TEAM_MEMBERS_LIST=team_list_str,
                    PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
                    TYPE_BREAKDOWNS=type_breakdowns.strip()
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
                    report_so_far += f"\n\n{section_txt.strip()}"

                for sec in section_order:
                    st.markdown(report_sections[sec])
                    if sec == "Intro_and_Type_Distribution":
                        st.header("Type Distribution Plot")
                        st.image(type_distribution_plot, use_column_width=True)
                    if sec == "Analysis of Dimension Preferences":
                        for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                            st.header(f"{pair[0]} vs {pair[1]} Preference Distribution")
                            st.image(preference_plots[''.join(pair)], use_column_width=True)

                # PDF with cover
                def build_cover_page(logo_path, type_system_name, company_name, team_name, date_str):
                    cov_elems = []
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
                    cov_elems.append(Spacer(1,80))
                    try:
                        lg = RepImage(logo_path, width=140, height=52)
                    except:
                        lg = None
                    if lg:
                        cov_elems.append(lg)
                    cov_elems.append(Spacer(1,50))
                    title_para = Paragraph(f"{type_system_name} For The Workplace<br/>Team Report", cover_title_style)
                    cov_elems.append(title_para)
                    cov_elems.append(Spacer(1,50))
                    sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                    cov_elems.append(sep)
                    cov_elems.append(Spacer(1,20))
                    comp_p = Paragraph(company_name, cover_text_style)
                    cov_elems.append(comp_p)
                    tm_p = Paragraph(team_name, cover_text_style)
                    cov_elems.append(tm_p)
                    dt_p = Paragraph(date_str, cover_text_style)
                    cov_elems.append(dt_p)
                    cov_elems.append(Spacer(1,60))
                    cov_elems.append(PageBreak())
                    return cov_elems

                from reportlab.platypus import Image as RepImage

                def convert_markdown_to_pdf(
                    report_dict,
                    distribution_plot,
                    pref_plots,
                    logo_path,
                    company_name,
                    team_name,
                    date_str
                ):
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                    elements = []

                    # Cover
                    cover_page = build_cover_page(
                        logo_path, "TypeFinder", company_name, team_name, date_str
                    )
                    elements.extend(cover_page)

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
                                    tb = Table(table_data, hAlign='LEFT')
                                    tb.setStyle(TableStyle([
                                        ('BACKGROUND',(0,0),(-1,0),colors.grey),
                                        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                        ('ALIGN',(0,0),(-1,-1),'CENTER'),
                                        ('FONTNAME',(0,0),(-1,0),'Times-Bold'),
                                        ('FONTNAME',(0,1),(-1,-1),'Times-Roman'),
                                        ('BOTTOMPADDING',(0,0),(-1,0),12),
                                        ('GRID',(0,0),(-1,-1),1,colors.black),
                                    ]))
                                    elements.append(tb)
                                    elements.append(Spacer(1,12))
                            else:
                                elements.append(Paragraph(elem.get_text(strip=True), styleN))
                                elements.append(Spacer(1,12))

                    for s in ["Intro_and_Type_Distribution","Analysis of Dimension Preferences",
                              "Team Insights","Next Steps"]:
                        process_markdown(report_dict[s])
                        if s == "Intro_and_Type_Distribution":
                            elements.append(Spacer(1,12))
                            dist_buf = io.BytesIO(distribution_plot)
                            dist_img = ReportLabImage(dist_buf, width=400, height=240)
                            elements.append(dist_img)
                            elements.append(Spacer(1,12))
                        if s == "Analysis of Dimension Preferences":
                            for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                                elements.append(Spacer(1,12))
                                elements.append(Paragraph(f"{pair[0]} vs {pair[1]} Preference Distribution", styleH2))
                                pfBuf = io.BytesIO(pref_plots[''.join(pair)])
                                pfImg = ReportLabImage(pfBuf, width=300, height=300)
                                elements.append(pfImg)
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
                    label="Download Report as PDF",
                    data=pdf_data,
                    file_name="typefinder_team_report.pdf",
                    mime="application/pdf"
                )


# import streamlit as st
# import pandas as pd
# import random
# from collections import Counter
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# from reportlab.platypus import (
#     SimpleDocTemplate,
#     Paragraph,
#     Spacer,
#     Image as ReportLabImage,
#     Table,
#     TableStyle,
#     HRFlowable,
#     PageBreak
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.enums import TA_CENTER
# from reportlab.lib import colors
# from markdown2 import markdown
# from bs4 import BeautifulSoup
# import datetime

# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# # -----------------------------------------------------------------------------------
# # Valid TypeFinder Types (16 variations)
# # -----------------------------------------------------------------------------------
# typefinder_types = [
#     'INTJ', 'INTP', 'ENTJ', 'ENTP',
#     'INFJ', 'INFP', 'ENFJ', 'ENFP',
#     'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
#     'ISTP', 'ISFP', 'ESTP', 'ESFP'
# ]

# def parse_tf_type(tf_str: str) -> str:
#     """Return a valid 4-letter TypeFinder code if recognized, else empty."""
#     if not tf_str:
#         return ""
#     tf_str = str(tf_str).strip().upper()
#     if tf_str in typefinder_types:
#         return tf_str
#     return ""

# # -----------------------------------------------------------------------------------
# # Minimal Prompt Edits to Mention User Names + Fractional Sums
# # -----------------------------------------------------------------------------------

# initial_context = """
# You are an expert organizational psychologist specializing in team dynamics and personality assessments using the TypeFinder framework.

# TypeFinder has four primary dimensions:
# 1) Extraversion (E) vs. Introversion (I)
# 2) Sensing (S) vs. Intuition (N)
# 3) Thinking (T) vs. Feeling (F)
# 4) Judging (J) vs. Perceiving (P)

# We avoid the term "MBTI" and call it "TypeFinder."
# We use “dimension” to describe the four pairs, and “preference” only when referencing one side (e.g., “preference for Thinking”).

# **Team Size:** {TEAM_SIZE}

# **Team Members and their TypeFinder Types:**
# {TEAM_MEMBERS_LIST}

# **Preference Counts/Percentages:**
# {PREFERENCE_BREAKDOWNS}

# **Type Counts/Percentages:**
# {TYPE_BREAKDOWNS}

# Your goal:
# Create a comprehensive team report with the following sections:

# 1. Intro & Type Distribution (combined)
# 2. Analysis of Dimension Preferences
# 3. Team Insights
# 4. Next Steps

# Follow these guidelines:
# - Keep language short, clear, and professional.
# - Round all percentages to the nearest whole number.
# - Never use "MBTI" anywhere.
# - “Dimension” for the four pairs, “preference” only for one side of a dimension.
# """

# prompts = {
#     # NOTE: We add instructions about naming each user in "Types on the Team"
#     "Intro_and_Type_Distribution": """
# {INITIAL_CONTEXT}

# **Types Not on the Team:**
# {NOT_ON_TEAM_LIST}

# **Your Role:**

# Write **Section 1: Intro & Type Distribution** as a single combined section.

# ## Section 1: Intro & Type Distribution

# 1. **Introduction (short)**
#    - Briefly (1–2 paragraphs) explain the TypeFinder framework (the four dimensions).
#    - State how understanding these types helps teams collaborate effectively.

# 2. **Type Distribution**
#    - Present the percentages for each TypeFinder type (ISTJ, ENFP, etc.) based on the provided data.
#    - Under a subheading `### Types on the Team`, list each type present (with short bullet points describing it, plus count & %),
#      **and explicitly name each user who holds that type** (e.g., "ISTJ (2 members: Alice, Bob)...").
#    - Under a subheading `### Types Not on the Team`, incorporate the data above (count=0, 0%).
#    - Briefly discuss how certain distributions might affect communication/decision-making.

# - Approximately 600 words total.

# **Begin your combined section below:**
# """,

#     # NOTE: We add instructions to reference fractional sums explicitly
#     "Analysis of Dimension Preferences": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# Write **Section 2: Analysis of Dimension Preferences**.

# ## Section 2: Analysis of Dimension Preferences

# - Use the provided preference counts/percentages (which may be based on fractional sums) for each dimension (E vs I, S vs N, T vs F, J vs P).
# - Clearly mention how the fractional or partial data was used to arrive at the final counts, and interpret these results for collaboration and workflow.
# - ~600 words total.

# **Continue your report below:**
# """,

#     "Team Insights": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# Write **Section 3: Team Insights**.

# ## Section 3: Team Insights

# Use the following subheadings:

# 1. **Strengths**
#    - At least four strengths, each in **bold** (one line), followed by a paragraph.

# 2. **Potential Blind Spots**
#    - At least four potential challenges, same bolded format + paragraph.

# 3. **Communication**
#    - 1–2 paragraphs about how dimension splits influence communication patterns.

# 4. **Teamwork**
#    - 1–2 paragraphs about collaboration, synergy, workflow.

# 5. **Conflict**
#    - 1–2 paragraphs about friction points and suggestions for healthy resolution.

# ~700 words total.

# **Continue the report below:**
# """,

#     "Next Steps": """
# {INITIAL_CONTEXT}

# **Report So Far:**

# {REPORT_SO_FAR}

# **Your Role:**

# Write **Section 4: Next Steps**.

# ## Section 4: Next Steps

# - Provide actionable recommendations for team leaders to leverage the TypeFinder composition.
# - Use subheadings (###) for each category.
# - Bullet points or numbered lists, with blank lines between items.
# - End immediately after the final bullet (no concluding paragraph).
# - ~400 words.

# **Conclude the report below:**
# """
# }

# # -----------------------------------------------------------------------------------
# # Streamlit App
# # -----------------------------------------------------------------------------------

# st.title('TypeFinder Team Report Generator')

# st.subheader("Cover Page Details")
# logo_path = "truity_logo.png"
# company_name = st.text_input("Company Name (for cover page)", "")
# team_name = st.text_input("Team Name (for cover page)", "")
# today_str = datetime.date.today().strftime("%B %d, %Y")
# custom_date = st.text_input("Date (for cover page)", today_str)

# st.subheader("Upload CSV")
# csv_file = st.file_uploader("Choose a CSV file", type=["csv"])

# if st.button("Generate Report from CSV"):
#     if not csv_file:
#         st.error("Please upload a valid CSV file first.")
#     else:
#         with st.spinner("Generating report..."):
#             df = pd.read_csv(csv_file)

#             valid_rows = []
#             for i, row in df.iterrows():
#                 nm_val = row.get("User Name", "")
#                 tf_val = row.get("TF Type", "")
#                 ei_val = row.get("TF E/I", "")
#                 ns_val = row.get("TF N/S", "")
#                 ft_val = row.get("TF F/T", "")
#                 jp_val = row.get("TF J/P", "")

#                 name_str = str(nm_val).strip()
#                 tf_parsed = parse_tf_type(tf_val)

#                 if name_str and tf_parsed:
#                     # Try dimension floats
#                     try:
#                         eFloat = float(ei_val) if ei_val != "" else None
#                         nFloat = float(ns_val) if ns_val != "" else None
#                         fFloat = float(ft_val) if ft_val != "" else None
#                         jFloat = float(jp_val) if jp_val != "" else None
#                     except:
#                         eFloat = nFloat = fFloat = jFloat = None

#                     valid_rows.append((name_str, tf_parsed, eFloat, nFloat, fFloat, jFloat))

#             if not valid_rows:
#                 st.error("No valid TypeFinder rows found.")
#             else:
#                 # dimension sums
#                 dimension_sums = {'E':0.0,'I':0.0,'S':0.0,'N':0.0,'T':0.0,'F':0.0,'J':0.0,'P':0.0}
#                 dimension_rows_count = 0

#                 team_list_str = ""
#                 type_list = []

#                 for idx, (nm, code, eF, nF, fF, jF) in enumerate(valid_rows, start=1):
#                     team_list_str += f"{idx}. {nm}: {code}\n"
#                     type_list.append(code)

#                     if (eF is not None and nF is not None and fF is not None and jF is not None):
#                         e_frac = eF/100.0
#                         i_frac = 1 - e_frac
#                         dimension_sums['E'] += e_frac
#                         dimension_sums['I'] += i_frac

#                         n_frac = nF/100.0
#                         s_frac = 1 - n_frac
#                         dimension_sums['N'] += n_frac
#                         dimension_sums['S'] += s_frac

#                         f_frac = fF/100.0
#                         t_frac = 1 - f_frac
#                         dimension_sums['F'] += f_frac
#                         dimension_sums['T'] += t_frac

#                         j_frac = jF/100.0
#                         p_frac = 1 - j_frac
#                         dimension_sums['J'] += j_frac
#                         dimension_sums['P'] += p_frac

#                         dimension_rows_count += 1

#                 total_members = len(valid_rows)

#                 # scale if partial dimension data
#                 if dimension_rows_count > 0:
#                     scale_factor = float(total_members)/dimension_rows_count
#                 else:
#                     scale_factor = 1.0

#                 for k in dimension_sums:
#                     dimension_sums[k] *= scale_factor

#                 # Round to get "counts"
#                 preference_counts = {k: round(v) for k,v in dimension_sums.items()}

#                 # Convert to pair-based percentages
#                 def pair_percent(a,b):
#                     s = preference_counts[a] + preference_counts[b]
#                     if s <= 0: return (0,0)
#                     pa = round((preference_counts[a]/s)*100)
#                     pb = 100 - pa
#                     return (pa,pb)

#                 final_pref_counts = {}
#                 final_pref_pcts = {}
#                 for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
#                     a, b = pair
#                     final_pref_counts[a] = preference_counts[a]
#                     final_pref_counts[b] = preference_counts[b]
#                     pa, pb = pair_percent(a,b)
#                     final_pref_pcts[a] = pa
#                     final_pref_pcts[b] = pb

#                 preference_breakdowns = ""
#                 for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
#                     a, b = pair
#                     preference_breakdowns += f"**{a} vs {b}**\n"
#                     preference_breakdowns += f"- {a}: {final_pref_counts[a]} members ({final_pref_pcts[a]}%)\n"
#                     preference_breakdowns += f"- {b}: {final_pref_counts[b]} members ({final_pref_pcts[b]}%)\n\n"

#                 # Type distribution
#                 type_counts = Counter(type_list)
#                 type_percentages = {t: round((c/total_members)*100) for t,c in type_counts.items()}
#                 absent_types = [t for t in typefinder_types if t not in type_counts]
#                 not_on_team_list_str = ""
#                 if absent_types:
#                     for t in absent_types:
#                         not_on_team_list_str += f"- {t} (0%)\n"
#                 else:
#                     not_on_team_list_str = "None (All Types Represented)"

#                 type_breakdowns = ""
#                 for t, c in type_counts.items():
#                     pct = type_percentages[t]
#                     type_breakdowns += f"- {t}: {c} members ({pct}%)\n"

#                 # bar chart
#                 sns.set_style('whitegrid')
#                 plt.rcParams.update({'font.family':'serif'})
#                 plt.figure(figsize=(10,6))
#                 sorted_t = sorted(type_counts.keys())
#                 sorted_ct = [type_counts[x] for x in sorted_t]
#                 sns.barplot(x=sorted_t, y=sorted_ct, palette='viridis')
#                 plt.title('TypeFinder Type Distribution', fontsize=16)
#                 plt.xlabel('TypeFinder Types', fontsize=14)
#                 plt.ylabel('Number of Team Members', fontsize=14)
#                 plt.xticks(rotation=45)
#                 plt.tight_layout()
#                 buf = io.BytesIO()
#                 plt.savefig(buf, format='png')
#                 buf.seek(0)
#                 type_distribution_plot = buf.getvalue()
#                 plt.close()

#                 # preference pie charts
#                 preference_plots = {}
#                 for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
#                     lbl = [pair[0], pair[1]]
#                     c1 = final_pref_counts[pair[0]]
#                     c2 = final_pref_counts[pair[1]]
#                     plt.figure(figsize=(6,6))
#                     plt.pie(
#                         [c1,c2],
#                         labels=lbl,
#                         autopct='%1.1f%%',
#                         startangle=90,
#                         colors=sns.color_palette('pastel'),
#                         textprops={'fontsize':12, 'fontfamily':'serif'}
#                     )
#                     plt.tight_layout()
#                     pbuf = io.BytesIO()
#                     plt.savefig(pbuf, format='png')
#                     pbuf.seek(0)
#                     preference_plots[''.join(pair)] = pbuf.getvalue()
#                     plt.close()

#                 # Prepare LLM
#                 chat_model = ChatOpenAI(
#                     openai_api_key=st.secrets['API_KEY'],
#                     model_name='gpt-4o-2024-08-06',
#                     temperature=0.2
#                 )

#                 # Format context
#                 initial_context_template = PromptTemplate.from_template(initial_context)
#                 formatted_initial_context = initial_context_template.format(
#                     TEAM_SIZE=str(total_members),
#                     TEAM_MEMBERS_LIST=team_list_str,
#                     PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
#                     TYPE_BREAKDOWNS=type_breakdowns.strip()
#                 )

#                 # Generate sections
#                 report_sections = {}
#                 report_so_far = ""
#                 section_order = [
#                     "Intro_and_Type_Distribution",
#                     "Analysis of Dimension Preferences",
#                     "Team Insights",
#                     "Next Steps"
#                 ]
#                 for sec in section_order:
#                     if sec == "Intro_and_Type_Distribution":
#                         prompt_template = PromptTemplate.from_template(prompts[sec])
#                         prompt_vars = {
#                             "INITIAL_CONTEXT": formatted_initial_context.strip(),
#                             "REPORT_SO_FAR": report_so_far.strip(),
#                             "NOT_ON_TEAM_LIST": not_on_team_list_str
#                         }
#                     else:
#                         prompt_template = PromptTemplate.from_template(prompts[sec])
#                         prompt_vars = {
#                             "INITIAL_CONTEXT": formatted_initial_context.strip(),
#                             "REPORT_SO_FAR": report_so_far.strip()
#                         }

#                     chain = LLMChain(prompt=prompt_template, llm=chat_model)
#                     section_txt = chain.run(**prompt_vars)
#                     report_sections[sec] = section_txt.strip()
#                     report_so_far += f"\n\n{section_txt.strip()}"

#                 for sec in section_order:
#                     st.markdown(report_sections[sec])
#                     if sec == "Intro_and_Type_Distribution":
#                         st.header("Type Distribution Plot")
#                         st.image(type_distribution_plot, use_column_width=True)
#                     if sec == "Analysis of Dimension Preferences":
#                         for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
#                             st.header(f"{pair[0]} vs {pair[1]} Preference Distribution")
#                             st.image(preference_plots[''.join(pair)], use_column_width=True)

#                 # PDF with cover
#                 def build_cover_page(logo_path, type_system_name, company_name, team_name, date_str):
#                     cov_elems = []
#                     styles = getSampleStyleSheet()
#                     cover_title_style = ParagraphStyle(
#                         'CoverTitle',
#                         parent=styles['Title'],
#                         fontName='Times-Bold',
#                         fontSize=24,
#                         leading=28,
#                         alignment=TA_CENTER,
#                         spaceAfter=20
#                     )
#                     cover_text_style = ParagraphStyle(
#                         'CoverText',
#                         parent=styles['Normal'],
#                         fontName='Times-Roman',
#                         fontSize=14,
#                         alignment=TA_CENTER,
#                         spaceAfter=8
#                     )
#                     cov_elems.append(Spacer(1,80))
#                     try:
#                         lg = Image(logo_path, width=140, height=52)
#                     except:
#                         lg = None
#                     if lg:
#                         cov_elems.append(lg)
#                     cov_elems.append(Spacer(1,50))
#                     title_para = Paragraph(f"{type_system_name} For The Workplace<br/>Team Report", cover_title_style)
#                     cov_elems.append(title_para)
#                     cov_elems.append(Spacer(1,50))
#                     sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
#                     cov_elems.append(sep)
#                     cov_elems.append(Spacer(1,20))
#                     comp_p = Paragraph(company_name, cover_text_style)
#                     cov_elems.append(comp_p)
#                     tm_p = Paragraph(team_name, cover_text_style)
#                     cov_elems.append(tm_p)
#                     dt_p = Paragraph(date_str, cover_text_style)
#                     cov_elems.append(dt_p)
#                     cov_elems.append(Spacer(1,60))
#                     cov_elems.append(PageBreak())
#                     return cov_elems

#                 from reportlab.platypus import Image as RepImage

#                 def convert_markdown_to_pdf(
#                     report_dict,
#                     distribution_plot,
#                     pref_plots,
#                     logo_path,
#                     company_name,
#                     team_name,
#                     date_str
#                 ):
#                     pdf_buffer = io.BytesIO()
#                     doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
#                     elements = []

#                     # Cover
#                     cover_page = build_cover_page(
#                         logo_path, "TypeFinder", company_name, team_name, date_str
#                     )
#                     elements.extend(cover_page)

#                     styles = getSampleStyleSheet()
#                     styleH1 = ParagraphStyle(
#                         'Heading1Custom',
#                         parent=styles['Heading1'],
#                         fontName='Times-Bold',
#                         fontSize=18,
#                         leading=22,
#                         spaceAfter=10,
#                     )
#                     styleH2 = ParagraphStyle(
#                         'Heading2Custom',
#                         parent=styles['Heading2'],
#                         fontName='Times-Bold',
#                         fontSize=16,
#                         leading=20,
#                         spaceAfter=8,
#                     )
#                     styleH3 = ParagraphStyle(
#                         'Heading3Custom',
#                         parent=styles['Heading3'],
#                         fontName='Times-Bold',
#                         fontSize=14,
#                         leading=18,
#                         spaceAfter=6,
#                     )
#                     styleH4 = ParagraphStyle(
#                         'Heading4Custom',
#                         parent=styles['Heading4'],
#                         fontName='Times-Bold',
#                         fontSize=12,
#                         leading=16,
#                         spaceAfter=4,
#                     )
#                     styleN = ParagraphStyle(
#                         'Normal',
#                         parent=styles['Normal'],
#                         fontName='Times-Roman',
#                         fontSize=12,
#                         leading=14,
#                     )
#                     styleList = ParagraphStyle(
#                         'List',
#                         parent=styles['Normal'],
#                         fontName='Times-Roman',
#                         fontSize=12,
#                         leading=14,
#                         leftIndent=20,
#                     )

#                     def process_markdown(md_text):
#                         html = markdown(md_text, extras=['tables'])
#                         soup = BeautifulSoup(html, 'html.parser')
#                         for elem in soup.contents:
#                             if isinstance(elem, str):
#                                 continue
#                             if elem.name == 'h1':
#                                 elements.append(Paragraph(elem.text, styleH1))
#                                 elements.append(Spacer(1,12))
#                             elif elem.name == 'h2':
#                                 elements.append(Paragraph(elem.text, styleH2))
#                                 elements.append(Spacer(1,12))
#                             elif elem.name == 'h3':
#                                 elements.append(Paragraph(elem.text, styleH3))
#                                 elements.append(Spacer(1,12))
#                             elif elem.name == 'h4':
#                                 elements.append(Paragraph(elem.text, styleH4))
#                                 elements.append(Spacer(1,12))
#                             elif elem.name == 'p':
#                                 elements.append(Paragraph(elem.decode_contents(), styleN))
#                                 elements.append(Spacer(1,12))
#                             elif elem.name == 'ul':
#                                 for li in elem.find_all('li', recursive=False):
#                                     elements.append(Paragraph('• ' + li.text, styleList))
#                                     elements.append(Spacer(1,6))
#                             elif elem.name == 'table':
#                                 table_data = []
#                                 thead = elem.find('thead')
#                                 if thead:
#                                     header_row = []
#                                     for th in thead.find_all('th'):
#                                         header_row.append(th.get_text(strip=True))
#                                     if header_row:
#                                         table_data.append(header_row)
#                                 tbody = elem.find('tbody')
#                                 if tbody:
#                                     rows = tbody.find_all('tr')
#                                 else:
#                                     rows = elem.find_all('tr')
#                                 for row in rows:
#                                     cols = row.find_all(['td','th'])
#                                     table_row = [c.get_text(strip=True) for c in cols]
#                                     table_data.append(table_row)
#                                 if table_data:
#                                     tb = Table(table_data, hAlign='LEFT')
#                                     tb.setStyle(TableStyle([
#                                         ('BACKGROUND',(0,0),(-1,0),colors.grey),
#                                         ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
#                                         ('ALIGN',(0,0),(-1,-1),'CENTER'),
#                                         ('FONTNAME',(0,0),(-1,0),'Times-Bold'),
#                                         ('FONTNAME',(0,1),(-1,-1),'Times-Roman'),
#                                         ('BOTTOMPADDING',(0,0),(-1,0),12),
#                                         ('GRID',(0,0),(-1,-1),1,colors.black),
#                                     ]))
#                                     elements.append(tb)
#                                     elements.append(Spacer(1,12))
#                             else:
#                                 elements.append(Paragraph(elem.get_text(strip=True), styleN))
#                                 elements.append(Spacer(1,12))

#                     for s in ["Intro_and_Type_Distribution","Analysis of Dimension Preferences",
#                               "Team Insights","Next Steps"]:
#                         process_markdown(report_dict[s])
#                         if s == "Intro_and_Type_Distribution":
#                             elements.append(Spacer(1,12))
#                             dist_buf = io.BytesIO(distribution_plot)
#                             dist_img = ReportLabImage(dist_buf, width=400, height=240)
#                             elements.append(dist_img)
#                             elements.append(Spacer(1,12))
#                         if s == "Analysis of Dimension Preferences":
#                             for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
#                                 elements.append(Spacer(1,12))
#                                 elements.append(Paragraph(f"{pair[0]} vs {pair[1]} Preference Distribution", styleH2))
#                                 pfBuf = io.BytesIO(pref_plots[''.join(pair)])
#                                 pfImg = ReportLabImage(pfBuf, width=300, height=300)
#                                 elements.append(pfImg)
#                                 elements.append(Spacer(1,12))

#                     doc.build(elements)
#                     pdf_buffer.seek(0)
#                     return pdf_buffer

#                 pdf_data = convert_markdown_to_pdf(
#                     report_dict=report_sections,
#                     distribution_plot=type_distribution_plot,
#                     pref_plots=preference_plots,
#                     logo_path=logo_path,
#                     company_name=company_name,
#                     team_name=team_name,
#                     date_str=custom_date
#                 )

#                 st.download_button(
#                     label="Download Report as PDF",
#                     data=pdf_data,
#                     file_name="typefinder_team_report.pdf",
#                     mime="application/pdf"
#                 )
