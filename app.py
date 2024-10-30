import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain

generate_team_report = """
You are an expert organizational psychologist specializing in team dynamics and personality assessments based on the Myers-Briggs Type Indicator (MBTI) framework.

**Task:** Generate a comprehensive team personality report for a team whose members have the following MBTI types:
{TEAM_TYPES}

**Your report should include the following sections:**

1. **Team Profile:**
   - Determine the overall "team type" by identifying the most common traits among all team members (e.g., if the majority are Extraverted, Intuitive, Feeling, and Judging, the team type is ENFJ).
   - Provide an overview of this team type, including key characteristics and how it influences team dynamics.

2. **Type Distribution:**
   - Present a percentage breakdown of each MBTI type within the team.
   - Explain what the distribution suggests about the team's diversity in thinking and working styles.

3. **Team Insights:**
   - **Strengths:** Highlight the collective strengths of the team based on the prevalent personality traits.
   - **Blind Spots:** Identify potential blind spots or challenges the team may face due to less represented traits.

4. **Type Preference Breakdown:**
   - For each MBTI dimension (Extraversion vs. Introversion, Sensing vs. Intuition, Thinking vs. Feeling, Judging vs. Perceiving), calculate the team's percentage distribution.
   - Explain what these percentages mean for team communication, decision-making, and problem-solving.

5. **Actions and Next Steps:**
   - Provide actionable recommendations for team leaders to enhance collaboration and performance.
   - Suggest resources or strategies tailored to the team's unique personality composition.

**Formatting Requirements:**

- Use clear headings and subheadings for each section.
- Write in Markdown format for easy readability.
- Use bullet points and tables where appropriate.
- Ensure the content is specific to the provided team MBTI types and offers unique insights.

**Begin your report below:**
"""

st.title('Team Personality Report Generator')

# Input for team size
team_size = st.number_input('Enter the number of team members (up to 30)', min_value=1, max_value=30, value=5)

# Initialize list to store MBTI types
team_mbti_types = []

# Input for MBTI types of each team member
st.header('Enter MBTI types for each team member')
mbti_options = ['INTJ', 'INTP', 'ENTJ', 'ENTP',
                'INFJ', 'INFP', 'ENFJ', 'ENFP',
                'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
                'ISTP', 'ISFP', 'ESTP', 'ESFP']

for i in range(int(team_size)):
    mbti_type = st.selectbox(f'Team Member {i+1}', options=['Select MBTI Type'] + mbti_options, key=f'mbti_{i}')
    if mbti_type != 'Select MBTI Type':
        team_mbti_types.append(mbti_type)

# Submit button
if st.button('Generate Report'):
    if len(team_mbti_types) < team_size:
        st.error('Please select MBTI types for all team members.')
    else:
        with st.spinner('Generating report, please wait...'):
            # Prepare the team types as a string
            team_types_str = ', '.join(team_mbti_types)
            # Initialize the LLM
            chat_model = ChatOpenAI(openai_api_key=st.secrets['API_KEY'], model_name='gpt-4-1106-preview', temperature=0.2)
            chat_chain = LLMChain(prompt=PromptTemplate.from_template(generate_team_report), llm=chat_model)
            generated_report = chat_chain.run(TEAM_TYPES=team_types_str)
            
            # Display the report using markdown
            st.markdown(generated_report)
            
            # Download button for the report
            st.download_button(
                label="Download Report",
                data=generated_report,
                file_name="team_personality_report.md",
                mime="text/markdown"
            )
