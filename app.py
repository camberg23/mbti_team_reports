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
    """Return a valid 4-letter TypeFinder code if recognized, else empty."""
    if not tf_str:
        return ""
    tf_str = tf_str.strip().upper()
    if tf_str in typefinder_types:
        return tf_str
    return ""

# -----------------------------------------------------------------------------------
# Updated Prompts
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
   - Under a subheading `### Types on the Team`, list each type present (with short bullet points describing it, plus count & %).
   - Under a subheading `### Types Not on the Team`, incorporate the data above (count=0, 0%).
   - Briefly discuss how certain distributions might affect communication/decision-making.

- Approximately 600 words total.

**Begin your combined section below:**
""",
    "Analysis of Dimension Preferences": """
{INITIAL_CONTEXT}

**Report So Far:**

{REPORT_SO_FAR}

**Your Role:**

Write **Section 2: Analysis of Dimension Preferences**.

## Section 2: Analysis of Dimension Preferences

- For each dimension (E vs I, S vs N, T vs F, J vs P):
  - Provide the counts/percentages for each preference (already in context).
  - 1–2 paragraphs discussing how that preference split affects team collaboration and workflow.
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

st.title('TypeFinder Team Report Generator (CSV-based with dimension data)')

st.subheader("Cover Page Details")
logo_path = "truity_logo.png"  # your logo path
company_name = st.text_input("Company Name (for cover page)", "Channing Realty")
team_name = st.text_input("Team Name (for cover page)", "Marketing Team")
today_str = datetime.date.today().strftime("%B %d, %Y")
custom_date = st.text_input("Date (for cover page)", today_str)

st.subheader("Upload CSV with columns:")
st.markdown("""
- **User Name** (string)
- **TF Type** (one of the 16, e.g. INFP, ESFJ, etc.)
- **TF E/I** (float, % favoring E, e.g. 55 -> 55% E, 45% I)
- **TF N/S** (float, % favoring N)
- **TF F/T** (float, % favoring F)
- **TF J/P** (float, % favoring J)
""")

csv_file = st.file_uploader("Choose a CSV file", type=["csv"])

if st.button("Generate Report from CSV"):
    if not csv_file:
        st.error("Please upload a valid CSV file first.")
    else:
        with st.spinner("Generating report..."):
            df = pd.read_csv(csv_file)

            # We'll parse valid rows (have a recognized TF Type) and gather dimension %.
            valid_rows = []
            for i, row in df.iterrows():
                name_val = row.get("User Name", "")
                type_val = row.get("TF Type", "")
                e_val = row.get("TF E/I", "")
                n_val = row.get("TF N/S", "")
                f_val = row.get("TF F/T", "")
                j_val = row.get("TF J/P", "")

                # Convert to strings for name/type
                name_str = str(name_val).strip()
                tf_type = parse_tf_type(str(type_val))

                if name_str and tf_type:
                    # We'll try to parse the dimension percentages as floats
                    try:
                        e_float = float(e_val) if e_val != "" else None
                        n_float = float(n_val) if n_val != "" else None
                        f_float = float(f_val) if f_val != "" else None
                        j_float = float(j_val) if j_val != "" else None
                    except:
                        # If we can't parse them, set them to None
                        e_float = n_float = f_float = j_float = None
                    
                    valid_rows.append((name_str, tf_type, e_float, n_float, f_float, j_float))
            
            if not valid_rows:
                st.error("No valid TypeFinder data found in CSV. Nothing to report.")
            else:
                # Prepare dimension counts
                # We'll sum up E, I, S, N, T, F, J, P as float amounts
                # e.g. for E=55 => we add 0.55 to E, 0.45 to I
                dimension_sums = {
                    'E': 0.0, 'I': 0.0,
                    'S': 0.0, 'N': 0.0,
                    'T': 0.0, 'F': 0.0,
                    'J': 0.0, 'P': 0.0
                }
                count_dimension_rows = 0

                team_list_str = ""
                type_list = []

                for idx, (nm, tf, eF, nF, fF, jF) in enumerate(valid_rows, start=1):
                    team_list_str += f"{idx}. {nm}: {tf}\n"
                    type_list.append(tf)

                    # If we have dimension data, sum them
                    # e.g. eF=55 => E += 0.55, I += 0.45
                    # We'll do this for whichever columns are not None
                    if (eF is not None and
                        nF is not None and
                        fF is not None and
                        jF is not None):
                        # Convert to fraction
                        e_frac = eF / 100.0
                        i_frac = 1.0 - e_frac
                        dimension_sums['E'] += e_frac
                        dimension_sums['I'] += i_frac

                        n_frac = nF / 100.0
                        s_frac = 1.0 - n_frac
                        dimension_sums['N'] += n_frac
                        dimension_sums['S'] += s_frac

                        f_frac = fF / 100.0
                        t_frac = 1.0 - f_frac
                        dimension_sums['F'] += f_frac
                        dimension_sums['T'] += t_frac

                        j_frac = jF / 100.0
                        p_frac = 1.0 - j_frac
                        dimension_sums['J'] += j_frac
                        dimension_sums['P'] += p_frac

                        count_dimension_rows += 1

                total_members = len(valid_rows)

                # For dimension sums, let's interpret them as "equivalent count"
                # e.g. if dimension_sums['E'] = 5.3 => means 5.3 "people" favored E
                # We'll round at the end
                # If we had partial dimension data for only some rows, count_dimension_rows is how many had numeric columns
                # but the user wants a *team-wide proportion*, so let's keep it with total_members or dimension_rows
                # We'll assume if some rows are missing dimension data, we exclude them from dimension calcs
                # => dimension_sums['E'] / count_dimension_rows => average fraction
                # => multiply by count_dimension_rows to get "count"
                # Then the user sees partial data, but let's keep it simpler

                if count_dimension_rows > 0:
                    # We'll scale up to "out of total_members" for simplicity
                    # so fractionE = ( dimension_sums['E'] / count_dimension_rows ) * total_members
                    # That way, the final bar/pie is representative of all X members
                    scale_factor = float(total_members) / float(count_dimension_rows)
                else:
                    scale_factor = 1.0  # no dimension data at all => fallback

                for k in dimension_sums:
                    dimension_sums[k] = dimension_sums[k] * scale_factor

                # We'll convert these "counts" to integer or round them
                # Then we can compute preference_counts
                preference_counts = {}
                for dkey in dimension_sums:
                    preference_counts[dkey] = round(dimension_sums[dkey])

                # Now let's get percentages
                # sum of E + I might not match total_members if some didn't have dimension data
                # We'll do a safe approach for each pair
                # E vs I
                # S vs N
                # T vs F
                # J vs P
                # We'll compute sum of each pair, then get percentage of total
                def pair_percent(a, b):
                    s = preference_counts[a] + preference_counts[b]
                    if s <= 0:
                        return (0, 0)
                    pa = round((preference_counts[a] / s) * 100)
                    pb = 100 - pa
                    return (pa, pb)

                # We'll build the final preference_count/percentage dict
                final_pref_counts = {}
                final_pref_pcts = {}
                for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                    a, b = pair[0], pair[1]
                    final_pref_counts[a] = preference_counts[a]
                    final_pref_counts[b] = preference_counts[b]
                    pa, pb = pair_percent(a, b)
                    final_pref_pcts[a] = pa
                    final_pref_pcts[b] = pb

                # We'll create a preference_breakdowns string
                preference_breakdowns = ""
                for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                    a, b = pair[0], pair[1]
                    preference_breakdowns += f"**{a} vs {b}**\n"
                    preference_breakdowns += f"- {a}: {final_pref_counts[a]} members ({final_pref_pcts[a]}%)\n"
                    preference_breakdowns += f"- {b}: {final_pref_counts[b]} members ({final_pref_pcts[b]}%)\n\n"

                # Type distribution
                type_counts = Counter(type_list)
                type_percentages = {
                    t: round((c / total_members) * 100)
                    for t, c in type_counts.items()
                }

                # Identify absent types
                absent_types = [t for t in typefinder_types if t not in type_counts]
                not_on_team_list_str = ""
                if absent_types:
                    for t in absent_types:
                        not_on_team_list_str += f"- {t} (0%)\n"
                else:
                    not_on_team_list_str = "None (All Types Represented)"

                # Build a type breakdown string
                type_breakdowns = ""
                for t, c in type_counts.items():
                    p = type_percentages[t]
                    type_breakdowns += f"- {t}: {c} members ({p}%)\n"

                # ---------------------
                # Build bar chart for types
                # ---------------------
                sns.set_style('whitegrid')
                plt.rcParams.update({'font.family':'serif'})
                plt.figure(figsize=(10, 6))
                sorted_types = sorted(type_counts.keys())
                sorted_counts = [type_counts[t] for t in sorted_types]
                sns.barplot(x=sorted_types, y=sorted_counts, palette='viridis')
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

                # For dimension pies, we now have final_pref_counts
                preference_plots = {}
                for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                    labels = [pair[0], pair[1]]
                    # get counts from final_pref_counts
                    c1 = final_pref_counts[pair[0]]
                    c2 = final_pref_counts[pair[1]]
                    plt.figure(figsize=(6,6))
                    plt.pie(
                        [c1, c2],
                        labels=labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=sns.color_palette('pastel'),
                        textprops={'fontsize':12, 'fontfamily':'serif'}
                    )
                    plt.tight_layout()
                    b = io.BytesIO()
                    plt.savefig(b, format='png')
                    b.seek(0)
                    preference_plots[''.join(pair)] = b.getvalue()
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
                    TEAM_SIZE=str(total_members),
                    TEAM_MEMBERS_LIST=team_list_str,
                    PREFERENCE_BREAKDOWNS=preference_breakdowns.strip(),
                    TYPE_BREAKDOWNS=type_breakdowns.strip()
                )

                # Generate sections
                report_sections = {}
                report_so_far = ""
                section_names = [
                    "Intro_and_Type_Distribution",
                    "Analysis of Dimension Preferences",
                    "Team Insights",
                    "Next Steps"
                ]

                for section_name in section_names:
                    if section_name == "Intro_and_Type_Distribution":
                        prompt_template = PromptTemplate.from_template(prompts[section_name])
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_initial_context.strip(),
                            "REPORT_SO_FAR": report_so_far.strip(),
                            "NOT_ON_TEAM_LIST": not_on_team_list_str
                        }
                    else:
                        prompt_template = PromptTemplate.from_template(prompts[section_name])
                        prompt_vars = {
                            "INITIAL_CONTEXT": formatted_initial_context.strip(),
                            "REPORT_SO_FAR": report_so_far.strip()
                        }

                    chain = LLMChain(prompt=prompt_template, llm=chat_model)
                    section_text = chain.run(**prompt_vars)
                    report_sections[section_name] = section_text.strip()
                    report_so_far += f"\n\n{section_text.strip()}"

                # Display final text in Streamlit
                for s_name in section_names:
                    st.markdown(report_sections[s_name])
                    if s_name == "Intro_and_Type_Distribution":
                        st.header("Type Distribution Plot")
                        st.image(type_distribution_plot, use_column_width=True)
                    if s_name == "Analysis of Dimension Preferences":
                        for pair in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
                            st.header(f"{pair[0]} vs {pair[1]} Preference Distribution")
                            st.image(preference_plots[''.join(pair)], use_column_width=True)

                # -------------------------------------------------------------------
                # PDF Generation with Cover Page
                # -------------------------------------------------------------------
                def build_cover_page(logo_path, type_system_name, company_name, team_name, date_str):
                    from reportlab.platypus import Spacer, Paragraph, Image as RepImage, HRFlowable, PageBreak
                    from reportlab.lib.enums import TA_CENTER
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib import colors

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

                    cov_elems.append(Spacer(1, 80))
                    try:
                        logo_img = RepImage(logo_path, width=140, height=52)
                        cov_elems.append(logo_img)
                    except:
                        pass

                    cov_elems.append(Spacer(1, 50))

                    title_para = Paragraph(f"{type_system_name} For The Workplace<br/>Team Report", cover_title_style)
                    cov_elems.append(title_para)

                    cov_elems.append(Spacer(1, 50))

                    sep = HRFlowable(width="70%", color=colors.darkgoldenrod)
                    cov_elems.append(sep)
                    cov_elems.append(Spacer(1, 20))

                    comp_para = Paragraph(company_name, cover_text_style)
                    cov_elems.append(comp_para)
                    tm_para = Paragraph(team_name, cover_text_style)
                    cov_elems.append(tm_para)
                    dt_para = Paragraph(date_str, cover_text_style)
                    cov_elems.append(dt_para)

                    cov_elems.append(Spacer(1, 60))
                    cov_elems.append(PageBreak())
                    return cov_elems

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
                    elems = []

                    # Add cover
                    cover_page = build_cover_page(
                        logo_path=logo_path,
                        type_system_name="TypeFinder",
                        company_name=company_name,
                        team_name=team_name,
                        date_str=date_str
                    )
                    elems.extend(cover_page)

                    # Normal styles
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
                                elems.append(Paragraph(elem.text, styleH1))
                                elems.append(Spacer(1,12))
                            elif elem.name == 'h2':
                                elems.append(Paragraph(elem.text, styleH2))
                                elems.append(Spacer(1,12))
                            elif elem.name == 'h3':
                                elems.append(Paragraph(elem.text, styleH3))
                                elems.append(Spacer(1,12))
                            elif elem.name == 'h4':
                                elems.append(Paragraph(elem.text, styleH4))
                                elems.append(Spacer(1,12))
                            elif elem.name == 'p':
                                elems.append(Paragraph(elem.decode_contents(), styleN))
                                elems.append(Spacer(1,12))
                            elif elem.name == 'ul':
                                for li in elem.find_all('li', recursive=False):
                                    elems.append(Paragraph('• ' + li.text, styleList))
                                    elems.append(Spacer(1,6))
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
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
                                        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
                                        ('BOTTOMPADDING', (0,0), (-1,0),12),
                                        ('GRID', (0,0), (-1,-1), 1, colors.black),
                                    ]))
                                    elems.append(t)
                                    elems.append(Spacer(1,12))
                            else:
                                elems.append(Paragraph(elem.get_text(strip=True), styleN))
                                elems.append(Spacer(1,12))

                    for s in ["Intro_and_Type_Distribution","Analysis of Dimension Preferences",
                              "Team Insights","Next Steps"]:
                        process_md(report_dict[s])
                        if s == "Intro_and_Type_Distribution":
                            elems.append(Spacer(1,12))
                            img_buf = io.BytesIO(distribution_plot)
                            img = ReportLabImage(img_buf, width=400, height=240)
                            elems.append(img)
                            elems.append(Spacer(1,12))
                        if s == "Analysis of Dimension Preferences":
                            for pair in [('E','I'),('S','N'),('T','F'),('J','P')]:
                                elems.append(Spacer(1,12))
                                elems.append(Paragraph(f"{pair[0]} vs {pair[1]} Preference Distribution", styleH2))
                                pimg_buf = io.BytesIO(preference_plots_dict[''.join(pair)])
                                pimg = ReportLabImage(pimg_buf, width=300, height=300)
                                elems.append(pimg)
                                elems.append(Spacer(1,12))

                    doc.build(elems)
                    pdf_buf.seek(0)
                    return pdf_buf

                # Build final PDF
                pdf_data = convert_markdown_to_pdf(
                    report_dict=report_sections,
                    distribution_plot=type_distribution_plot,
                    preference_plots_dict=preference_plots,
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
