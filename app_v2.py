import streamlit as st
import pandas as pd
from collections import Counter
from openai import OpenAI

# =============================================================================
# Type Parsing Utilities
# =============================================================================

# --- TypeFinder ---
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

# --- Enneagram ---
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
        stripped = lower.replace("type ", "").strip()
        result = eg_map.get(stripped, "")
    return result

# --- DISC ---
csv_to_disc_map = {
    "Drive": "D", "Influence": "I", "Support": "S", "Clarity": "C"
}

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
# Data Processing
# =============================================================================

def process_csv(df):
    team_members = []
    for _, row in df.iterrows():
        name = str(row.get("User Name", "")).strip()
        if not name:
            continue

        member = {"name": name}

        # TypeFinder
        tf_type = parse_tf_type(str(row.get("TF Type", "")))
        if tf_type:
            member["tf_type"] = tf_type
            for col, key in [
                ("TF Extraversion", "tf_e"), ("TF Intuition", "tf_n"),
                ("TF Feeling", "tf_f"), ("TF Judging", "tf_j"),
                ("TF E/I", "tf_e"), ("TF N/S", "tf_n"),
                ("TF F/T", "tf_f"), ("TF J/P", "tf_j"),
            ]:
                if col in df.columns and key not in member:
                    try:
                        val = float(row.get(col, ""))
                        member[key] = val
                    except (ValueError, TypeError):
                        pass

        # Enneagram
        eg_raw = str(row.get("EG Type", "")).strip()
        eg_type = parse_eg_type(eg_raw)
        if eg_type:
            member["eg_type"] = eg_type

        # DISC
        disc_raw = str(row.get("DISC Type", "")).strip()
        disc_type = parse_disc_type(disc_raw)
        if disc_type:
            member["disc_type"] = disc_type

        if any(k in member for k in ["tf_type", "eg_type", "disc_type"]):
            team_members.append(member)

    return team_members


def build_team_summary_text(members):
    lines = []
    team_size = len(members)
    lines.append(f"**Team Size:** {team_size}")
    lines.append("")

    # Per-member listing
    lines.append("**Team Members and Assessment Results:**")
    for i, m in enumerate(members, 1):
        parts = [m["name"]]
        if "tf_type" in m:
            parts.append(f"TypeFinder: {m['tf_type']}")
        if "eg_type" in m:
            parts.append(f"Enneagram: {m['eg_type']}")
        if "disc_type" in m:
            parts.append(f"DISC: {m['disc_type']}")
        lines.append(f"{i}. {' | '.join(parts)}")
    lines.append("")

    # TypeFinder aggregates
    tf_types = [m["tf_type"] for m in members if "tf_type" in m]
    if tf_types:
        lines.append("**TypeFinder Distribution:**")
        tf_counts = Counter(tf_types)
        n_tf = len(tf_types)
        dims = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
        for t in tf_types:
            dims[t[0]] += 1
            dims[t[1]] += 1
            dims[t[2]] += 1
            dims[t[3]] += 1
        for pair in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
            a, b = pair
            pa = round((dims[a] / n_tf) * 100)
            lines.append(f"- {a}: {dims[a]} ({pa}%) vs {b}: {dims[b]} ({100-pa}%)")
        lines.append("Types present: " + ", ".join(
            f"{t} ({c})" for t, c in tf_counts.most_common()
        ))
        absent = [t for t in typefinder_types if t not in tf_counts]
        if absent:
            lines.append(f"Types absent: {', '.join(absent)}")
        lines.append("")

    # Enneagram aggregates
    eg_types = [m["eg_type"] for m in members if "eg_type" in m]
    if eg_types:
        lines.append("**Enneagram Distribution:**")
        eg_counts = Counter(eg_types)
        n_eg = len(eg_types)
        all_eg = [f"Type {w}" for w in ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]]
        for t in all_eg:
            c = eg_counts.get(t, 0)
            pct = round((c / n_eg) * 100) if n_eg > 0 else 0
            lines.append(f"- {t}: {c} ({pct}%)")
        heart = sum(eg_counts.get(f"Type {t}", 0) for t in ["Two", "Three", "Four"])
        head = sum(eg_counts.get(f"Type {t}", 0) for t in ["Five", "Six", "Seven"])
        body = sum(eg_counts.get(f"Type {t}", 0) for t in ["Eight", "Nine", "One"])
        lines.append(f"Centers: Heart (2,3,4): {heart} | Head (5,6,7): {head} | Body (8,9,1): {body}")
        lines.append("")

    # DISC aggregates
    disc_types = [m["disc_type"] for m in members if "disc_type" in m]
    if disc_types:
        lines.append("**DISC Distribution:**")
        disc_counts = Counter(disc_types)
        n_disc = len(disc_types)
        style_counts = {'D': 0, 'I': 0, 'S': 0, 'C': 0}
        for d in disc_types:
            primary = d.split('/')[0] if '/' in d else d
            if primary in style_counts:
                style_counts[primary] += 1
        for s in ['D', 'I', 'S', 'C']:
            full_name = {'D': 'Drive', 'I': 'Influence', 'S': 'Support', 'C': 'Clarity'}[s]
            pct = round((style_counts[s] / n_disc) * 100)
            lines.append(f"- {full_name} ({s}): {style_counts[s]} ({pct}%)")
        lines.append("Types: " + ", ".join(
            f"{t} ({c})" for t, c in disc_counts.most_common()
        ))
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert organizational psychologist. You have deep expertise in the TypeFinder (similar to MBTI, with four dimensions: E/I, S/N, T/F, J/P), the Enneagram (Types One through Nine), and DISC (Drive, Influence, Support, Clarity, plus hybrid types like D/i, I/s, etc.).

Important guidelines:
- Refer to the MBTI-style assessment as "TypeFinder" (never "MBTI").
- For DISC, use Drive/Influence/Support/Clarity (not Dominance/Steadiness/Conscientiousness).
- For Enneagram, spell out type numbers (Type One, Type Two, etc.).
- Refer to it as "your team" or "the team" — never "our team."
- Be specific to THIS team's actual data. Don't write generic advice that could apply to any team.
- Write in a warm but professional tone. No jargon-dumping."""

PARAGRAPH_USER_PROMPT = """{TEAM_DATA}

Write TWO sections (clearly separated by a blank line):

**Section 1 — How This Team Works Together (1-2 paragraphs)**
Look at the combined picture across all three assessment systems. Identify the 2-3 most important patterns and what they mean for how this team communicates, makes decisions, and collaborates. Be specific — name the actual types, dimensions, and dynamics at play. If there are interesting cross-system patterns (e.g., a team heavy on both DISC Drive types AND TypeFinder Thinking preference), call those out. Focus on the most striking or consequential patterns — don't try to cover everything.

Then provide 3-5 bullet points that capture the key takeaways from the paragraphs above. Format each bullet as "• [point]" on its own line.

**Section 2 — How the Manager Can Help This Team Grow (1-2 paragraphs)**
Based on the team composition, identify specific development opportunities — blind spots, communication gaps, under-represented perspectives, or friction points that the manager should be aware of. Make recommendations actionable and tied to the actual type data.

Then provide 3-5 bullet points that capture the key takeaways. Format each bullet as "• [point]" on its own line.

Start Section 1 with the header "**How This Team Works Together**" on its own line, then the paragraphs, then the bullets.
Start Section 2 with the header "**How the Manager Can Help This Team Grow**" on its own line, then the paragraphs, then the bullets.
Do not add any other headers or formatting beyond these two."""

BULLET_USER_PROMPT = """{TEAM_DATA}

Generate team-specific content for the following topic categories. For each topic, write 2-3 sentences that are grounded in this team's actual assessment data. Reference specific types, dimensions, and patterns — not generic platitudes.

Format your output exactly as follows (use these exact topic headers):

**Communication Style**
[2-3 sentences about how this team's personality composition shapes their communication patterns]

**Decision-Making**
[2-3 sentences about how the team approaches decisions based on their types]

**Collaboration & Work Style**
[2-3 sentences about how team members work together day-to-day]

**Potential Blind Spots**
[2-3 sentences about what perspectives or approaches may be under-represented]

**Team Strengths**
[2-3 sentences about the team's core advantages based on their composition]

**Tips for the Manager**
[2-3 sentences of actionable advice for getting the best out of this specific team]"""


# =============================================================================
# LLM Call
# =============================================================================

def generate_summary(api_key: str, team_data: str, user_prompt_template: str) -> str:
    client = OpenAI(api_key=api_key)
    user_content = user_prompt_template.replace("{TEAM_DATA}", team_data)

    response = client.chat.completions.create(
        model="gpt-5.3-chat-latest",
        messages=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


# =============================================================================
# Demo Datasets
# =============================================================================

import io

DEMO_TEAMS = {
    "All Three Systems (10 people, mixed types)": """User Name,TF Type,EG Type,DISC Type
Alice Chen,ENTJ,Eight,Drive
Bob Martinez,ISFJ,Two,Support
Carol Washington,ENTP,Seven,Influence
Dave Kim,ISTJ,One,Clarity
Eva Petrova,ENFP,Four,Influence/Support
Frank Okafor,INTJ,Five,Clarity/Drive
Grace Liu,ESFJ,Nine,Support
Henry Park,ESTP,Three,Drive/Influence
Ines Fernandez,INFP,Four,Support
James O'Brien,ESTJ,Eight,Drive""",

    "Drive-Heavy Sales Team (8 people)": """User Name,TF Type,EG Type,DISC Type
Marcus Rivera,ENTJ,Eight,Drive
Priya Sharma,ESTP,Three,Drive/Influence
Tyler Brooks,ENTP,Seven,Influence
Sarah Kim,ESTJ,Eight,Drive
Jordan Wells,ENFJ,Two,Influence/Support
Rachel Nguyen,ENTJ,Three,Drive
Damien Cole,ESTP,Seven,Drive/Influence
Olivia Grant,ESFP,Seven,Influence""",

    "Introvert-Heavy Engineering Team (7 people)": """User Name,TF Type,EG Type,DISC Type
Liam Zhang,INTJ,Five,Clarity
Sofia Petrov,INTP,Six,Clarity/Drive
Amir Hassan,ISTJ,One,Clarity
Yuki Tanaka,INFJ,Four,Support/Clarity
Ben Torres,ISTP,Five,Clarity
Mei-Lin Wu,INTJ,One,Clarity/Drive
Nathan Cole,ENTP,Seven,Drive/Influence""",

    "Small Startup Team (4 people)": """User Name,TF Type,EG Type,DISC Type
Zara Ahmed,ENFP,Seven,Influence
Leo Park,INTJ,Five,Clarity/Drive
Mia Johnson,ESFJ,Two,Support
Raj Patel,ENTJ,Three,Drive""",

    "TypeFinder Only (no Enneagram or DISC)": """User Name,TF Type,EG Type,DISC Type
Anna Lee,ENFP,,
Carlos Ruiz,ISTJ,,
Diana Osei,ENTJ,,
Eric Holm,ISFP,,
Fiona Chang,INTP,,
George Mbeki,ESFJ,,""",

    "Enneagram Only (no TypeFinder or DISC)": """User Name,TF Type,EG Type,DISC Type
Hannah Berg,,Nine,
Isaac Voss,,Three,
Julia Reyes,,Six,
Kwame Asante,,One,
Lisa Moreau,,Nine,
Miguel Santos,,Four,
Nina Popov,,Six,""",

    "DISC Only (no TypeFinder or Enneagram)": """User Name,TF Type,EG Type,DISC Type
Omar Farah,,,Drive
Paula Schmidt,,,Support
Quinn Nakamura,,,Influence/Support
Rosa Delgado,,,Clarity
Sam Okonkwo,,,Drive/Influence
Tara Singh,,,Support
Uma Johansson,,,Clarity/Drive
Victor Lam,,,Influence""",
}


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="Team Summary Generator", layout="wide")
st.title("Team Summary Generator")
st.caption("Combined TypeFinder + Enneagram + DISC — lightweight team summaries for Truity at Work")

output_mode = st.radio(
    "Output format",
    ["Paragraph mode", "Bullet mode", "Both (for comparison)"],
    help="Paragraph mode: AI picks the most relevant dynamics and writes about them. "
         "Bullet mode: fixed topic categories with team-specific content for each."
)

st.markdown("---")

data_source = st.radio(
    "Data source",
    ["Use a demo team", "Upload CSV"],
    horizontal=True,
)

df = None

if data_source == "Use a demo team":
    demo_choice = st.selectbox("Choose a demo team:", list(DEMO_TEAMS.keys()))
    csv_text = DEMO_TEAMS[demo_choice]
    df = pd.read_csv(io.StringIO(csv_text))
    st.markdown("**Preview:**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download this demo as CSV",
        csv_text,
        file_name="demo_team.csv",
        mime="text/csv",
        key="dl_demo"
    )
else:
    st.markdown(
        "Upload a CSV with columns: `User Name`, `TF Type`, `EG Type`, `DISC Type`. "
        "Not every member needs all three — the tool uses whatever data is available."
    )
    csv_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

if st.button("Generate Team Summary", type="primary"):
    if df is None:
        st.error("Please select a demo team or upload a CSV file first.")
    else:
        members = process_csv(df)

        if not members:
            st.error("No valid assessment data found. "
                     "Make sure you have columns like 'User Name', 'TF Type', 'EG Type', 'DISC Type'.")
        else:
            team_data = build_team_summary_text(members)

            with st.expander("View parsed team data", expanded=False):
                st.markdown(team_data)

            api_key = st.secrets['API_KEY']

            do_paragraph = output_mode in ["Paragraph mode", "Both (for comparison)"]
            do_bullet = output_mode in ["Bullet mode", "Both (for comparison)"]

            if do_paragraph:
                with st.spinner("Generating paragraph summary..."):
                    paragraph_result = generate_summary(api_key, team_data, PARAGRAPH_USER_PROMPT)

                st.subheader("📝 Paragraph Mode")
                st.markdown(paragraph_result)
                st.download_button(
                    "Download paragraph summary",
                    paragraph_result,
                    file_name="team_summary_paragraph.md",
                    mime="text/markdown",
                    key="dl_paragraph"
                )

            if do_bullet:
                with st.spinner("Generating bullet summary..."):
                    bullet_result = generate_summary(api_key, team_data, BULLET_USER_PROMPT)

                st.subheader("📋 Bullet Mode")
                st.markdown(bullet_result)
                st.download_button(
                    "Download bullet summary",
                    bullet_result,
                    file_name="team_summary_bullets.md",
                    mime="text/markdown",
                    key="dl_bullet"
                )
