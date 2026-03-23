import streamlit as st
import pandas as pd
from collections import Counter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
    # Handle both "Nine" and "Type Nine" formats
    result = eg_map.get(lower, "")
    if not result:
        # Try stripping "type " prefix
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
    """
    Parse the CSV and extract per-member data for all three systems.
    Returns a dict with team_members list and aggregate stats.
    """
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
            # Try to get dimension scores
            for col, key in [
                ("TF Extraversion", "tf_e"), ("TF Intuition", "tf_n"),
                ("TF Feeling", "tf_f"), ("TF Judging", "tf_j"),
                # Alternative column names from older exports
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

        # Only include members who have at least one valid assessment
        if any(k in member for k in ["tf_type", "eg_type", "disc_type"]):
            team_members.append(member)

    return team_members


def build_team_summary_text(members):
    """
    Build a structured text summary of team composition data to feed into the prompt.
    """
    lines = []
    team_size = len(members)
    lines.append(f"**Team Size:** {team_size}")
    lines.append("")

    # --- Per-member listing ---
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

    # --- TypeFinder aggregates ---
    tf_types = [m["tf_type"] for m in members if "tf_type" in m]
    if tf_types:
        lines.append("**TypeFinder Distribution:**")
        tf_counts = Counter(tf_types)
        n_tf = len(tf_types)

        # Dimension preference counts
        dims = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
        for t in tf_types:
            dims[t[0]] += 1  # E or I
            dims[t[1]] += 1  # S or N
            dims[t[2]] += 1  # T or F
            dims[t[3]] += 1  # J or P

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

    # --- Enneagram aggregates ---
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

        # Centers of intelligence
        heart = sum(eg_counts.get(f"Type {t}", 0) for t in ["Two", "Three", "Four"])
        head = sum(eg_counts.get(f"Type {t}", 0) for t in ["Five", "Six", "Seven"])
        body = sum(eg_counts.get(f"Type {t}", 0) for t in ["Eight", "Nine", "One"])
        lines.append(f"Centers: Heart (2,3,4): {heart} | Head (5,6,7): {head} | Body (8,9,1): {body}")
        lines.append("")

    # --- DISC aggregates ---
    disc_types = [m["disc_type"] for m in members if "disc_type" in m]
    if disc_types:
        lines.append("**DISC Distribution:**")
        disc_counts = Counter(disc_types)
        n_disc = len(disc_types)

        # Primary style counts (including hybrids)
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

PARAGRAPH_PROMPT = """You are an expert organizational psychologist. You have deep expertise in the TypeFinder (similar to MBTI, with four dimensions: E/I, S/N, T/F, J/P), the Enneagram (Types One through Nine), and DISC (Drive, Influence, Support, Clarity, plus hybrid types like D/i, I/s, etc.).

You are writing a brief team summary for a manager. The summary will appear in the Truity at Work platform.

**Important guidelines:**
- Refer to the MBTI-style assessment as "TypeFinder" (never "MBTI").
- For DISC, use Drive/Influence/Support/Clarity (not Dominance/Steadiness/Conscientiousness).
- For Enneagram, spell out type numbers (Type One, Type Two, etc.).
- Refer to it as "your team" or "the team" — never "our team."
- Be specific to THIS team's actual data. Don't write generic advice that could apply to any team.
- Focus on the 2-3 most striking or consequential patterns in the data. Don't try to cover everything.
- Write in a warm but professional tone. No jargon-dumping.
- Do NOT use headers or section titles in your output — just flowing prose.

{TEAM_DATA}

**Your task:**

Write TWO sections (clearly separated by a blank line):

**Section 1 — How This Team Works Together (1-2 paragraphs)**
Look at the combined picture across all three assessment systems. Identify the 2-3 most important patterns and what they mean for how this team communicates, makes decisions, and collaborates. Be specific — name the actual types, dimensions, and dynamics at play. If there are interesting cross-system patterns (e.g., a team heavy on both DISC Drive types AND TypeFinder Thinking preference), call those out.

Then provide 3-5 bullet points that capture the key takeaways from the paragraphs above. Format each bullet as "• [point]" on its own line.

**Section 2 — How the Manager Can Help This Team Grow (1-2 paragraphs)**
Based on the team composition, identify specific development opportunities — blind spots, communication gaps, under-represented perspectives, or friction points that the manager should be aware of. Make recommendations actionable and tied to the actual type data.

Then provide 3-5 bullet points that capture the key takeaways. Format each bullet as "• [point]" on its own line.

Begin your output now:
"""

BULLET_PROMPT = """You are an expert organizational psychologist. You have deep expertise in the TypeFinder (similar to MBTI, with four dimensions: E/I, S/N, T/F, J/P), the Enneagram (Types One through Nine), and DISC (Drive, Influence, Support, Clarity, plus hybrid types like D/i, I/s, etc.).

You are writing a brief team summary for a manager. The summary will appear in the Truity at Work platform.

**Important guidelines:**
- Refer to the MBTI-style assessment as "TypeFinder" (never "MBTI").
- For DISC, use Drive/Influence/Support/Clarity (not Dominance/Steadiness/Conscientiousness).
- For Enneagram, spell out type numbers (Type One, Type Two, etc.).
- Refer to it as "your team" or "the team" — never "our team."
- Be specific to THIS team's actual data. Don't write generic advice that could apply to any team.
- Write in a warm but professional tone. No jargon-dumping.

{TEAM_DATA}

**Your task:**

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
[2-3 sentences of actionable advice for getting the best out of this specific team]

Begin your output now:
"""


# =============================================================================
# Streamlit App
# =============================================================================

st.set_page_config(page_title="Team Summary Generator", layout="wide")
st.title("Team Summary Generator")
st.caption("Combined TypeFinder + Enneagram + DISC — lightweight team summaries for Truity at Work")

st.subheader("Upload CSV")
st.markdown(
    "Upload a CSV with columns: `User Name`, `TF Type`, `EG Type`, `DISC Type`. "
    "Not every member needs all three — the tool uses whatever data is available."
)

output_mode = st.radio(
    "Output format",
    ["Paragraph mode", "Bullet mode", "Both (for comparison)"],
    help="Paragraph mode: AI picks the most relevant dynamics and writes about them. "
         "Bullet mode: fixed topic categories with team-specific content for each."
)

csv_file = st.file_uploader("Choose a CSV file", type=["csv"])

if st.button("Generate Team Summary"):
    if not csv_file:
        st.error("Please upload a CSV file first.")
    else:
        with st.spinner("Processing..."):
            df = pd.read_csv(csv_file)
            members = process_csv(df)

            if not members:
                st.error("No valid assessment data found in the CSV. "
                         "Make sure you have columns like 'User Name', 'TF Type', 'EG Type', 'DISC Type'.")
            else:
                team_data = build_team_summary_text(members)

                st.subheader("Team Data Summary")
                with st.expander("View parsed team data", expanded=False):
                    st.markdown(team_data)

                chat_model = ChatOpenAI(
                    openai_api_key=st.secrets['API_KEY'],
                    model_name='gpt-4o-2024-08-06',
                    temperature=0.3
                )

                generate_paragraph = output_mode in ["Paragraph mode", "Both (for comparison)"]
                generate_bullet = output_mode in ["Bullet mode", "Both (for comparison)"]

                if generate_paragraph:
                    with st.spinner("Generating paragraph summary..."):
                        prompt = PromptTemplate.from_template(PARAGRAPH_PROMPT)
                        chain = LLMChain(prompt=prompt, llm=chat_model)
                        paragraph_result = chain.run(TEAM_DATA=team_data)

                    st.subheader("📝 Paragraph Mode")
                    st.markdown(paragraph_result)
                    st.download_button(
                        "Download paragraph summary",
                        paragraph_result,
                        file_name="team_summary_paragraph.md",
                        mime="text/markdown",
                        key="dl_paragraph"
                    )

                if generate_bullet:
                    with st.spinner("Generating bullet summary..."):
                        prompt = PromptTemplate.from_template(BULLET_PROMPT)
                        chain = LLMChain(prompt=prompt, llm=chat_model)
                        bullet_result = chain.run(TEAM_DATA=team_data)

                    st.subheader("📋 Bullet Mode")
                    st.markdown(bullet_result)
                    st.download_button(
                        "Download bullet summary",
                        bullet_result,
                        file_name="team_summary_bullets.md",
                        mime="text/markdown",
                        key="dl_bullet"
                    )

                # Also generate a sample CSV for testing
                st.markdown("---")
                st.caption("Need test data? Download this sample CSV to try the tool.")

SAMPLE_CSV = """User Name,TF Type,EG Type,DISC Type
Alice Chen,ENTJ,Eight,Drive
Bob Martinez,ISFJ,Two,Support
Carol Washington,ENTP,Seven,Influence
Dave Kim,ISTJ,One,Clarity
Eva Petrova,ENFP,Four,Influence/Support
Frank Okafor,INTJ,Five,Clarity/Drive
Grace Liu,ESFJ,Nine,Support
Henry Park,ESTP,Three,Drive/Influence
Ines Fernandez,INFP,Four,Support
James O'Brien,ESTJ,Eight,Drive
"""

st.download_button(
    "Download sample CSV",
    SAMPLE_CSV,
    file_name="sample_team_data.csv",
    mime="text/csv",
    key="dl_sample"
)
