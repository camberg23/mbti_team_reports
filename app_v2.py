import streamlit as st
import pandas as pd
import io
from collections import Counter
from openai import OpenAI

# =============================================================================
# Type Parsing Utilities
# =============================================================================

typefinder_types = [
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'INFJ', 'INFP', 'ENFJ', 'ENFP',
    'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
    'ISTP', 'ISFP', 'ESTP', 'ESFP'
]

def parse_tf_type(tf_str: str) -> str:
    if not tf_str:
        return ""
    tf_str = str(tf_str).strip().upper()
    return tf_str if tf_str in typefinder_types else ""

eg_map = {
    "one": "Type One", "two": "Type Two", "three": "Type Three",
    "four": "Type Four", "five": "Type Five", "six": "Type Six",
    "seven": "Type Seven", "eight": "Type Eight", "nine": "Type Nine",
}

def parse_eg_type(raw: str) -> str:
    if not raw:
        return ""
    lower = raw.strip().lower()
    result = eg_map.get(lower, "")
    if not result:
        result = eg_map.get(lower.replace("type ", "").strip(), "")
    return result

csv_to_disc_map = {"Drive": "D", "Influence": "I", "Support": "S", "Clarity": "C"}

def parse_disc_type(text: str) -> str:
    if not text:
        return ""
    parts = text.split('/')
    codes = []
    for p in parts:
        p = p.strip()
        if p in csv_to_disc_map:
            codes.append(csv_to_disc_map[p])
        elif p.upper() in ['D', 'I', 'S', 'C']:
            codes.append(p.upper())
        else:
            return ""
    if len(codes) == 1:
        return codes[0]
    elif len(codes) == 2:
        return f"{codes[0]}/{codes[1].lower()}"
    return ""


# =============================================================================
# Data Processing — one function per system
# =============================================================================

def extract_typefinder_data(df):
    """Extract TypeFinder data and build summary text."""
    members = []
    for _, row in df.iterrows():
        name = str(row.get("User Name", "")).strip()
        tf_type = parse_tf_type(str(row.get("TF Type", "")))
        if name and tf_type:
            members.append({"name": name, "type": tf_type})
    if not members:
        return None, []

    n = len(members)
    type_counts = Counter(m["type"] for m in members)
    dims = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    for m in members:
        t = m["type"]
        dims[t[0]] += 1; dims[t[1]] += 1; dims[t[2]] += 1; dims[t[3]] += 1

    lines = [f"Team Size: {n}", ""]
    lines.append("Team Members:")
    for i, m in enumerate(members, 1):
        lines.append(f"{i}. {m['name']}: {m['type']}")
    lines.append("")
    lines.append("Dimension Preferences:")
    for a, b in [('E','I'), ('S','N'), ('T','F'), ('J','P')]:
        pa = round((dims[a] / n) * 100)
        lines.append(f"- {a}: {dims[a]} ({pa}%) vs {b}: {dims[b]} ({100-pa}%)")
    lines.append("")
    lines.append("Types present: " + ", ".join(f"{t} ({c})" for t, c in type_counts.most_common()))
    absent = [t for t in typefinder_types if t not in type_counts]
    if absent:
        lines.append(f"Types absent: {', '.join(absent)}")
    return "\n".join(lines), members


def extract_enneagram_data(df):
    """Extract Enneagram data and build summary text."""
    members = []
    for _, row in df.iterrows():
        name = str(row.get("User Name", "")).strip()
        eg_type = parse_eg_type(str(row.get("EG Type", "")))
        if name and eg_type:
            members.append({"name": name, "type": eg_type})
    if not members:
        return None, []

    n = len(members)
    type_counts = Counter(m["type"] for m in members)
    all_eg = [f"Type {w}" for w in ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]]

    lines = [f"Team Size: {n}", ""]
    lines.append("Team Members:")
    for i, m in enumerate(members, 1):
        lines.append(f"{i}. {m['name']}: {m['type']}")
    lines.append("")
    lines.append("Type Distribution:")
    for t in all_eg:
        c = type_counts.get(t, 0)
        pct = round((c / n) * 100) if n > 0 else 0
        lines.append(f"- {t}: {c} ({pct}%)")
    lines.append("")
    heart = sum(type_counts.get(f"Type {t}", 0) for t in ["Two", "Three", "Four"])
    head = sum(type_counts.get(f"Type {t}", 0) for t in ["Five", "Six", "Seven"])
    body = sum(type_counts.get(f"Type {t}", 0) for t in ["Eight", "Nine", "One"])
    lines.append(f"Centers of Intelligence: Heart (2,3,4): {heart} | Head (5,6,7): {head} | Body (8,9,1): {body}")
    absent = [t for t in all_eg if type_counts.get(t, 0) == 0]
    if absent:
        lines.append(f"Types absent: {', '.join(absent)}")
    return "\n".join(lines), members


def extract_disc_data(df):
    """Extract DISC data and build summary text."""
    members = []
    for _, row in df.iterrows():
        name = str(row.get("User Name", "")).strip()
        disc_type = parse_disc_type(str(row.get("DISC Type", "")))
        if name and disc_type:
            members.append({"name": name, "type": disc_type})
    if not members:
        return None, []

    n = len(members)
    type_counts = Counter(m["type"] for m in members)
    style_counts = {'D': 0, 'I': 0, 'S': 0, 'C': 0}
    for m in members:
        primary = m["type"].split('/')[0] if '/' in m["type"] else m["type"]
        if primary in style_counts:
            style_counts[primary] += 1

    lines = [f"Team Size: {n}", ""]
    lines.append("Team Members:")
    for i, m in enumerate(members, 1):
        lines.append(f"{i}. {m['name']}: {m['type']}")
    lines.append("")
    lines.append("Primary Style Distribution:")
    for s in ['D', 'I', 'S', 'C']:
        full = {'D': 'Drive', 'I': 'Influence', 'S': 'Support', 'C': 'Clarity'}[s]
        pct = round((style_counts[s] / n) * 100)
        lines.append(f"- {full} ({s}): {style_counts[s]} ({pct}%)")
    lines.append("")
    lines.append("Detailed Types: " + ", ".join(f"{t} ({c})" for t, c in type_counts.most_common()))
    return "\n".join(lines), members


# =============================================================================
# System-Specific Prompts
# =============================================================================

TYPEFINDER_SYSTEM = """You are an expert organizational psychologist specializing in the TypeFinder personality framework.

TypeFinder has four dimensions:
1) Extraversion (E) vs. Introversion (I)
2) Sensing (S) vs. Intuition (N)
3) Thinking (T) vs. Feeling (F)
4) Judging (J) vs. Perceiving (P)

Important guidelines:
- Always call it "TypeFinder" (never "MBTI" or "Myers-Briggs").
- Use "dimension" for the four pairs, and "preference" only for one side of a dimension.
- Refer to it as "your team" or "the team," never "our team."
- Do NOT address the reader as "the manager" or assume any specific job title. Write so the content is useful whether the reader is a team lead, a peer, or the team itself.
- Be specific to THIS team's actual data. Don't write generic advice.
- Write in a warm but professional tone. No jargon-dumping."""

TYPEFINDER_USER = """{TEAM_DATA}

Write TWO sections:

**How This Team Works Together**
1-2 paragraphs identifying the 2-3 most important patterns in this team's TypeFinder composition and what they mean for communication, decision-making, and collaboration. Be specific: name the actual types, dimension splits, and dynamics at play. Focus on the most striking patterns rather than trying to cover everything.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

**Opportunities for Growth**
1-2 paragraphs identifying specific development opportunities based on the team's TypeFinder composition: blind spots from missing or under-represented preferences, communication gaps between different types, or friction points. Make recommendations actionable and tied to the actual type data.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

Use ONLY these two bold headers. No other headers or formatting."""


ENNEAGRAM_SYSTEM = """You are an expert organizational psychologist specializing in the Enneagram personality framework.

The Enneagram has nine types (Type One through Type Nine), grouped into three Centers of Intelligence:
- Body Center (Types Eight, Nine, One): instinct-driven, focused on autonomy and control
- Heart Center (Types Two, Three, Four): emotion-driven, focused on identity and connection
- Head Center (Types Five, Six, Seven): thinking-driven, focused on security and understanding

Important guidelines:
- Always spell out type numbers (Type One, Type Two, etc.).
- Refer to it as "your team" or "the team," never "our team."
- Do NOT address the reader as "the manager" or assume any specific job title. Write so the content is useful whether the reader is a team lead, a peer, or the team itself.
- Be specific to THIS team's actual data. Don't write generic advice.
- Write in a warm but professional tone. No jargon-dumping."""

ENNEAGRAM_USER = """{TEAM_DATA}

Write TWO sections:

**How This Team Works Together**
1-2 paragraphs identifying the 2-3 most important patterns in this team's Enneagram composition and what they mean for communication, decision-making, and collaboration. Reference the Centers of Intelligence balance, dominant types, and the specific interpersonal dynamics they create. Focus on the most striking patterns.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

**Opportunities for Growth**
1-2 paragraphs identifying specific development opportunities: blind spots from absent types or under-represented centers, tension points between types, or tendencies the team should watch for. Make recommendations actionable and tied to the actual type data.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

Use ONLY these two bold headers. No other headers or formatting."""


DISC_SYSTEM = """You are an expert organizational psychologist specializing in the DISC personality framework.

DISC has four primary styles:
- Drive (D): direct, results-oriented, competitive
- Influence (I): enthusiastic, collaborative, optimistic
- Support (S): patient, reliable, team-oriented
- Clarity (C): analytical, detail-oriented, systematic

Hybrid types combine two styles (e.g., D/i, I/s, S/c), where the second letter is lowercase and separated by a slash.

Important guidelines:
- Use Drive/Influence/Support/Clarity (NOT Dominance/Steadiness/Conscientiousness).
- For hybrid types, always use slash-lowercase (e.g., D/i, C/s).
- Refer to it as "your team" or "the team," never "our team."
- Do NOT address the reader as "the manager" or assume any specific job title. Write so the content is useful whether the reader is a team lead, a peer, or the team itself.
- Be specific to THIS team's actual data. Don't write generic advice.
- Write in a warm but professional tone. No jargon-dumping."""

DISC_USER = """{TEAM_DATA}

Write TWO sections:

**How This Team Works Together**
1-2 paragraphs identifying the 2-3 most important patterns in this team's DISC composition and what they mean for communication, decision-making, and collaboration. Reference the balance of primary styles, any dominant clusters, and how hybrid types add nuance. Focus on the most striking patterns.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

**Opportunities for Growth**
1-2 paragraphs identifying specific development opportunities: blind spots from missing or under-represented styles, pace or communication mismatches, or friction points between styles. Make recommendations actionable and tied to the actual type data.

Then 3-5 bullet points capturing the key takeaways. Format each as "• [point]".

Use ONLY these two bold headers. No other headers or formatting."""


# =============================================================================
# LLM Call
# =============================================================================

def generate_summary(api_key, team_data, system_prompt, user_prompt_template):
    client = OpenAI(api_key=api_key)
    user_content = user_prompt_template.replace("{TEAM_DATA}", team_data)
    response = client.chat.completions.create(
        model="gpt-5.3-chat-latest",
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


# =============================================================================
# Demo Datasets
# =============================================================================

TF_DEMOS = {
    "Mixed Team (10 people)": """User Name,TF Type
Alice Chen,ENTJ
Bob Martinez,ISFJ
Carol Washington,ENTP
Dave Kim,ISTJ
Eva Petrova,ENFP
Frank Okafor,INTJ
Grace Liu,ESFJ
Henry Park,ESTP
Ines Fernandez,INFP
James O'Brien,ESTJ""",

    "Introvert-Heavy Engineering Team (7 people)": """User Name,TF Type
Liam Zhang,INTJ
Sofia Petrov,INTP
Amir Hassan,ISTJ
Yuki Tanaka,INFJ
Ben Torres,ISTP
Mei-Lin Wu,INTJ
Nathan Cole,ENTP""",

    "Extravert Sales Team (6 people)": """User Name,TF Type
Marcus Rivera,ENTJ
Priya Sharma,ESTP
Tyler Brooks,ENTP
Sarah Kim,ESTJ
Jordan Wells,ENFJ
Olivia Grant,ESFP""",
}

EG_DEMOS = {
    "Mixed Team (8 people)": """User Name,EG Type
Hannah Berg,Nine
Isaac Voss,Three
Julia Reyes,Six
Kwame Asante,One
Lisa Moreau,Nine
Miguel Santos,Four
Nina Popov,Six
Derek Choi,Eight""",

    "Heart-Heavy Creative Team (6 people)": """User Name,EG Type
Amara Obi,Two
Chen Wei,Three
Rosa Delgado,Four
Tomás Silva,Four
Leah Goldstein,Three
Dante Moore,Two""",

    "Head-Heavy Analyst Team (7 people)": """User Name,EG Type
Sven Lindqvist,Five
Fatima Al-Rashid,Six
Kenji Yamamoto,Five
Priya Nair,Seven
Oscar Mendez,Six
Elin Strand,Six
Raj Kapoor,One""",
}

DISC_DEMOS = {
    "Mixed Team (8 people)": """User Name,DISC Type
Omar Farah,Drive
Paula Schmidt,Support
Quinn Nakamura,Influence/Support
Rosa Delgado,Clarity
Sam Okonkwo,Drive/Influence
Tara Singh,Support
Uma Johansson,Clarity/Drive
Victor Lam,Influence""",

    "Drive-Heavy Leadership Team (6 people)": """User Name,DISC Type
Alex Petrov,Drive
Morgan Chen,Drive/Influence
Casey Williams,Drive
Jordan Blake,Influence
Sam Torres,Drive/Clarity
Riley Okafor,Clarity""",

    "Support-Heavy Operations Team (7 people)": """User Name,DISC Type
Dana Kim,Support
Eli Washington,Support/Clarity
Fran Mbeki,Support
Greta Holm,Clarity
Harper Reyes,Support/Influence
Indra Patel,Support
Jules Fernandez,Influence""",
}

SYSTEM_CONFIGS = {
    "TypeFinder": {
        "demos": TF_DEMOS,
        "extract_fn": extract_typefinder_data,
        "system_prompt": TYPEFINDER_SYSTEM,
        "user_prompt": TYPEFINDER_USER,
        "type_col": "TF Type",
    },
    "Enneagram": {
        "demos": EG_DEMOS,
        "extract_fn": extract_enneagram_data,
        "system_prompt": ENNEAGRAM_SYSTEM,
        "user_prompt": ENNEAGRAM_USER,
        "type_col": "EG Type",
    },
    "DISC": {
        "demos": DISC_DEMOS,
        "extract_fn": extract_disc_data,
        "system_prompt": DISC_SYSTEM,
        "user_prompt": DISC_USER,
        "type_col": "DISC Type",
    },
}


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="Team Summary Generator", layout="wide")
st.title("Team Summary Generator")
st.caption("Single-system team summaries for Truity at Work")

system_choice = st.radio(
    "Assessment system",
    ["TypeFinder", "Enneagram", "DISC"],
    horizontal=True,
)

config = SYSTEM_CONFIGS[system_choice]

st.markdown("---")

data_source = st.radio(
    "Data source",
    ["Use a demo team", "Upload CSV"],
    horizontal=True,
)

df = None

if data_source == "Use a demo team":
    demo_choice = st.selectbox("Choose a demo team:", list(config["demos"].keys()))
    csv_text = config["demos"][demo_choice]
    df = pd.read_csv(io.StringIO(csv_text))
    st.markdown("**Preview:**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download this demo as CSV",
        csv_text,
        file_name=f"demo_{system_choice.lower()}_team.csv",
        mime="text/csv",
        key="dl_demo"
    )
else:
    type_col = config["type_col"]
    st.markdown(f"Upload a CSV with columns: `User Name` and `{type_col}`.")
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

if st.button("Generate Team Summary", type="primary"):
    if df is None:
        st.error("Please select a demo team or upload a CSV file first.")
    else:
        team_data, members = config["extract_fn"](df)

        if not team_data:
            st.error(f"No valid {system_choice} data found in the CSV.")
        else:
            with st.expander("View parsed team data", expanded=False):
                st.text(team_data)

            api_key = st.secrets['API_KEY']

            with st.spinner(f"Generating {system_choice} team summary..."):
                result = generate_summary(
                    api_key, team_data,
                    config["system_prompt"],
                    config["user_prompt"]
                )

            st.markdown(result)
            st.download_button(
                "Download summary",
                result,
                file_name=f"team_summary_{system_choice.lower()}.md",
                mime="text/markdown",
                key="dl_summary"
            )
