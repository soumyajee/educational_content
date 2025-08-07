#!/usr/bin/env python3
"""
Streamlit UI for Content Sourcing Agent
"""

import streamlit as st
from datetime import datetime
import os
from dotenv import load_dotenv

# Import from custom module
try:
    from agant_updated_assement import ContentSourcingAgent, get_config
except ImportError as e:
    st.error(f"Failed to import ContentSourcingAgent: {e}")
    # Fallback mock implementation
    class ContentSourcingAgent:
        def __init__(self, config, api_key, model, base_url, max_tokens):
            self.config = config
            self.api_key = api_key
            self.model = model
            self.base_url = base_url
            self.max_tokens = max_tokens
            self.content_api = type('ContentAPI', (), {'storage': {}})()
            self.content_api.storage = {
                'item1': {
                    'id': 'item1',
                    'title': 'Sample Content',
                    'content': 'Sample content about AI in automotive systems...',
                    'source_url': 'http://example.com',
                    'category': 'AI',
                    'tags': ['AI', 'automotive'],
                    'quality_score': 0.9,
                    'bloom_level': 'understanding'
                }
            }

        def run(self, query, sources, trigger="manual"):
            return [
                type('Assessment', (), {
                    'question_type': 'mcq',
                    'question_text': f'What is the role of AI in {query}?',
                    'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                    'correct_answer': 'Option B',
                    'bloom_level': 'understanding',
                    'objective': f'Understand {query}',
                    'curriculum_standard': 'Implement relevant standards'
                })(),
                type('Assessment', (), {
                    'question_type': 'short_answer',
                    'question_text': f'Explain how {query} improves efficiency.',
                    'options': [],
                    'correct_answer': f'{query} enables efficient processing.',
                    'bloom_level': 'applying',
                    'objective': f'Apply {query} concepts',
                    'curriculum_standard': 'Implement relevant standards'
                })()
            ]

        def submit_assessment(self, index, user_answer, student_id, teacher_id):
            return {'score': 0.85, 'feedback': 'Good answer, but include more details.'}

        def get_student_report(self, student_id):
            return {
                'student_id': student_id,
                'name': 'John Doe',
                'total_assessments': 2,
                'average_score': 0.85,
                'skill_gaps': ['Advanced AI concepts'],
                'assessments': [{'question_text': 'Sample question', 'score': 0.85}]
            }

        def get_teacher_report(self, teacher_id):
            return {
                'teacher_id': teacher_id,
                'name': 'Jane Smith',
                'students': ['student_001', 'student_002'],
                'classwide_gaps': ['AI application'],
                'assessment_summary': 'Students need more practice with AI concepts.'
            }

    def get_config():
        return type('Config', (), {
            'STATIC_SOURCES': ['http://example.com'],
            'GROQ_API_KEY': 'mock_key',
            'LLM_MODEL': 'mock_model',
            'LLM_BASE_URL': 'http://mock.api',
            'MAX_TOKENS': 1000
        })()

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Assesment Agent Dashboard", layout="wide")

# Title and timestamp
st.title("Content Sourcing Agent Dashboard")
st.write("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"))

# Debug environment variables
st.write(f"GROQ_API_KEY: {'Set' if os.getenv('GROQ_API_KEY') else 'Not Set'}")
st.write(f"TEST_QUERY: {os.getenv('TEST_QUERY', 'Not Set')}")

# Sidebar for navigation and inputs
st.sidebar.header("Dashboard Controls")
selected_view = st.sidebar.selectbox("Select View", ["Overview", "Assessments", "Student Report", "Teacher Report"])
query = st.sidebar.text_input("Enter Query", value=os.getenv('TEST_QUERY', 'artificial intelligence in automotive systems'))
sources_input = st.sidebar.text_area("Enter Sources (comma-separated URLs)", value=",".join(get_config().STATIC_SOURCES))
sources = [url.strip() for url in sources_input.split(',') if url.strip()]

# Initialize or reuse agent
if 'agent' not in st.session_state:
    try:
        config = get_config()
        st.session_state.agent = ContentSourcingAgent(
            config=config,
            api_key=config.GROQ_API_KEY,
            model=config.LLM_MODEL,
            base_url=config.LLM_BASE_URL,
            max_tokens=config.MAX_TOKENS
        )
        st.write("Agent initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")

agent = st.session_state.agent

# Run agent when query or sources change
if st.sidebar.button("Run Agent"):
    with st.spinner("Processing content and generating assessments..."):
        try:
            assessments = agent.run(query, sources, trigger="manual")
            st.session_state.assessments = assessments
            st.success(f"Agent execution completed! Generated {len(assessments)} assessments")
        except Exception as e:
            st.error(f"Error running agent: {e}")

# Initialize assessments in session state
if 'assessments' not in st.session_state:
    st.session_state.assessments = []

# Display views based on selection
if selected_view == "Overview":
    st.header("Execution Summary")
    st.write(f"**Query:** {query}")
    st.write(f"**Sources Processed:** {len(sources)}")
    st.write(f"**Assessments Generated:** {len(st.session_state.assessments)}")
    if st.session_state.assessments:
        st.subheader("Generated Assessments")
        for i, assessment in enumerate(st.session_state.assessments, 1):
            with st.expander(f"Assessment {i}: {assessment.question_text[:50]}..."):
                st.write(f"**Type:** {assessment.question_type}")
                st.write(f"**Question:** {assessment.question_text}")
                if assessment.options:
                    st.write(f"**Options:** {', '.join(assessment.options)}")
                st.write(f"**Correct Answer:** {assessment.correct_answer or 'None'}")
                st.write(f"**Bloom Level:** {assessment.bloom_level}")
                st.write(f"**Objective:** {assessment.objective}")
                st.write(f"**Curriculum Standard:** {assessment.curriculum_standard}")
    else:
        st.warning("No assessments generated. Run the agent to generate assessments.")

elif selected_view == "Assessments":
    st.header("Generated Assessments")
    st.write(f"Total assessments: {len(st.session_state.assessments)}")
    if st.session_state.assessments:
        for i, assessment in enumerate(st.session_state.assessments, 1):
            with st.expander(f"Assessment {i}: {assessment.question_text[:50]}..."):
                st.write(f"**Type:** {assessment.question_type}")
                st.write(f"**Question:** {assessment.question_text}")
                if assessment.options:
                    st.write(f"**Options:** {', '.join(assessment.options)}")
                st.write(f"**Correct Answer:** {assessment.correct_answer or 'None'}")
                st.write(f"**Bloom Level:** {assessment.bloom_level}")
                st.write(f"**Objective:** {assessment.objective}")
                st.write(f"**Curriculum Standard:** {assessment.curriculum_standard}")

                # Input for student answer
                student_id = st.text_input(f"Student ID for Assessment {i}", value=os.getenv('STUDENT_ID', 'student_001'))
                teacher_id = st.text_input(f"Teacher ID for Assessment {i}", value=os.getenv('TEACHER_ID', 'teacher_001'))
                sample_answer = assessment.correct_answer or "AI improves ECU diagnostics by analyzing CAN bus data."
                user_answer = st.text_area(f"Enter answer for Assessment {i}", value=sample_answer, height=100)
                if st.button(f"Submit Answer for Assessment {i}", key=f"submit_{i}"):
                    try:
                        response = agent.submit_assessment(i-1, user_answer, student_id, teacher_id)
                        if "error" not in response:
                            st.success(f"Submitted! Score: {response['score']:.2f}, Feedback: {response['feedback']}")
                        else:
                            st.error(response["error"])
                    except Exception as e:
                        st.error(f"Error submitting assessment: {e}")
    else:
        st.warning("No assessments generated yet. Run the agent to generate assessments.")

elif selected_view == "Student Report":
    st.header("Student Report")
    student_id = st.text_input("Enter Student ID", value=os.getenv('STUDENT_ID', 'student_001'))
    try:
        student_report = agent.get_student_report(student_id)
        if "error" not in student_report:
            st.write(f"**Student ID:** {student_report['student_id']}")
            st.write(f"**Name:** {student_report['name']}")
            st.write(f"**Total Assessments:** {student_report['total_assessments']}")
            st.write(f"**Average Score:** {student_report['average_score']:.2f}")
            st.write(f"**Skill Gaps:** {', '.join(student_report['skill_gaps']) if student_report['skill_gaps'] else 'None'}")
            if student_report['assessments']:
                st.write("**Assessment History:**")
                for a in student_report['assessments']:
                    st.write(f"- Question: {a['question_text']}, Score: {a.get('score', 'N/A')}")
        else:
            st.error(student_report["error"])
    except Exception as e:
        st.error(f"Error generating student report: {e}")

elif selected_view == "Teacher Report":
    st.header("Teacher Report")
    teacher_id = st.text_input("Enter Teacher ID", value=os.getenv('TEACHER_ID', 'teacher_001'))
    try:
        teacher_report = agent.get_teacher_report(teacher_id)
        if "error" not in teacher_report:
            st.write(f"**Teacher ID:** {teacher_report['teacher_id']}")
            st.write(f"**Name:** {teacher_report['name']}")
            st.write(f"**Students:** {', '.join(teacher_report['students']) if teacher_report['students'] else 'None'}")
            st.write(f"**Class-wide Gaps:** {', '.join(teacher_report['classwide_gaps']) if teacher_report['classwide_gaps'] else 'None'}")
            st.write(f"**Assessment Summary:** {teacher_report['assessment_summary']}")
        else:
            st.error(teacher_report["error"])
    except Exception as e:
        st.error(f"Error generating teacher report: {e}")

st.sidebar.text("Powered by xAI Grok 3")

