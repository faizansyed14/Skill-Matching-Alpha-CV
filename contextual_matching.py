# enhanced_cv_matcher_custom_weights.py
# ------------------------------------------------------------
# JD ↔ CV matching with:
#    • Secure login authentication
#    • Multiple-file uploads for CVs and single JD upload
#    • Automatic text extraction (PDF/DOCX/TXT)
#    • LLM-powered structured extraction with strict prompts (CV + JD)
#    • API key loaded from .env or Streamlit secrets (OPENAI_API_KEY, GEMINI_API_KEY)
#    • Shows per-request time taken and token counts in UI
#    • Customizable weights + modern UI
#    • Matching logic identical to your app (with multi-CV index fix)
#    • Displays extracted data in UI and auto-fills input boxes
#    • 3 tabs: Customization • Upload & Extract • Results
#    • 🔽 NEW: Model dropdown + per-model temperature (shown and auto-applied)
#    • 🔽 NEW: Handles missing JD years by defaulting to 0
#    • 🔽 NEW: Refresh button to reset all data
#    • 🔽 NEW: Improved UI with JD on left and CVs on right for comparison
#    • 🔽 NEW: Simplified extraction logs showing only model, time, and tokens
#    • 🔽 NEW: Added Google Gemini models support
# ------------------------------------------------------------
# Requirements (pip):
# streamlit, numpy, pandas, sentence-transformers, qdrant-client, scipy, graphviz, plotly,
# pdfplumber, docx2txt, python-dotenv, openai>=1.0.0, google-generativeai
# ------------------------------------------------------------
import os
import io
import json
import time
import pdfplumber
import docx2txt
import numpy as np
import pandas as pd
import graphviz
import plotly.graph_objects as go
import streamlit as st
import openai  # ONLY this import for OpenAI SDK
import google.generativeai as genai  # Added for Gemini
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from scipy.optimize import linear_sum_assignment
# ---------------------------
# Config
# ---------------------------
COLLECTION_NAME = "cv_skills"
RESP_COLLECTION_NAME = "cv_responsibilities"
GOOD_THRESHOLD = 0.40  # Threshold for good matches
DEFAULT_MODEL = "gpt-4o-mini"
# 🔽 NEW: model presets (temperature per model) — tweak as you like
MODEL_PRESETS = {
    # 🔽 OpenAI Models
    "gpt-4o-mini": {"temperature": 0.0, "note": "Vision-optimized mini (deterministic)"},
    "gpt-4.1-mini": {"temperature": 0.0, "note": "Balanced mini"},
    "gpt-5-mini": {"temperature": 1.0, "note": "Small, slightly flexible"},
    "gpt-5-nano": {"temperature": 1.0, "note": "Deterministic, tiny"},
    "gpt-4.1-nano": {"temperature": 0.0, "note": "Ultra-tiny, deterministic"},
    # Gemini 2.5 family (GA)
    "gemini-2.5-flash": {"temperature": 0.0, "note": "2.5 Flash – balanced, GA"},
    "gemini-2.5-pro": {"temperature": 0.0, "note": "2.5 Pro – reasoning, GA"},
    "gemini-2.5-flash-lite": {"temperature": 0.0, "note": "2.5 Flash Lite – fastest, GA"},
    "gemini-2.0-flash": {"temperature": 0.0, "note": "2.0 Flash"},
}
st.set_page_config(
    page_title="Alpha CV Matcher",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# ---------------------------
# .env load + OpenAI/Gemini setup
# ---------------------------
ENV_PATH = find_dotenv(usecwd=True)
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
try:
    secret_openai_key = st.secrets.get("OPENAI_API_KEY", None)
    secret_gemini_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    secret_openai_key = None
    secret_gemini_key = None
if not OPENAI_API_KEY and secret_openai_key:
    OPENAI_API_KEY = secret_openai_key
if not GEMINI_API_KEY and secret_gemini_key:
    GEMINI_API_KEY = secret_gemini_key
OPENAI_AVAILABLE = True
OPENAI_IMPORT_ERROR = None
CLIENT_AVAILABLE = hasattr(openai, "Client")
LEGACY_AVAILABLE = hasattr(openai, "ChatCompletion")
GEMINI_AVAILABLE = True
GEMINI_IMPORT_ERROR = None
# ---------------------------
# Authentication Setup
# ---------------------------
# Load login credentials from environment or secrets
LOGIN_EMAIL = os.getenv("LOGIN_EMAIL")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD")
try:
    secret_email = st.secrets.get("LOGIN_EMAIL", None)
    secret_password = st.secrets.get("LOGIN_PASSWORD", None)
except Exception:
    secret_email = None
    secret_password = None
if not LOGIN_EMAIL and secret_email:
    LOGIN_EMAIL = secret_email
if not LOGIN_PASSWORD and secret_password:
    LOGIN_PASSWORD = secret_password
# Session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_error' not in st.session_state:
    st.session_state.login_error = None
# ---------------------------
# Login UI
# ---------------------------
def show_login():
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 0px auto;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .login-header h1 {
            color: #667eea;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .login-header p {
            color: #666;
            font-size: 1.1rem;
        }
        .login-form {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .login-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102,126,234,0.4);
        }
        .error-message {
            color: #e74c3c;
            text-align: center;
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: rgba(231,76,60,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>🔐 Secure Login</h1>
                <p>Please enter your credentials to access the CV Matcher</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit_button = st.form_submit_button("Login", type="primary")
            
            if submit_button:
                if LOGIN_EMAIL and LOGIN_PASSWORD:
                    if email == LOGIN_EMAIL and password == LOGIN_PASSWORD:
                        st.session_state.logged_in = True
                        st.session_state.login_error = None
                        st.rerun()
                    else:
                        st.session_state.login_error = "Invalid email or password"
                else:
                    st.session_state.login_error = "Login credentials not configured. Please check your environment variables or secrets."
        
        if st.session_state.login_error:
            st.markdown(f"""
            <div class="error-message">
                {st.session_state.login_error}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
# Check if user is logged in
if not st.session_state.logged_in:
    show_login()
    st.stop()  # Stop execution if not logged in
# ---------------------------
# Reset Function
# ---------------------------
def reset_session_state():
    """Reset all session state variables to initial values"""
    # Keep authentication state
    logged_in = st.session_state.logged_in
    
    # Clear all other session state
    for key in list(st.session_state.keys()):
        if key != 'logged_in':
            del st.session_state[key]
    
    # Reinitialize authentication state
    st.session_state.logged_in = logged_in
    
    # Initialize default values
    st.session_state.cv_data = []
    st.session_state.cv_names = []
    st.session_state.logs_cvs = []
    st.session_state.logs_jd = []
    st.session_state.jd_fields = {"skills": "", "responsibilities": "", "job_title": "", "years": 0}
    st.session_state.run_analysis = False
    st.session_state.show_extracted_jd = False
    st.session_state.show_extracted_cvs = False
    st.session_state.weights = {'skills': 80, 'responsibilities': 15, 'job_title': 2.5, 'experience': 2.5}
    st.session_state.prompts = {"cv": DEFAULT_CV_PROMPT, "jd": DEFAULT_JD_PROMPT}
    st.session_state.model_name = DEFAULT_MODEL
    st.session_state.model_temp = MODEL_PRESETS.get(DEFAULT_MODEL, {"temperature": 0.0})["temperature"]
    
    st.rerun()
# ---------------------------
# CSS (modern UI)
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem; border-radius: 20px; text-align: center; color: white;
        margin-bottom: 2rem; box-shadow: 0 20px 40px rgba(102,126,234,0.3); position: relative; overflow: hidden;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-header::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%); animation: rotate 20s linear infinite;
    }
    @keyframes rotate { 0% { transform: rotate(0deg);} 100% { transform: rotate(360deg);} }
    .hero-title { font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem; position: relative; z-index: 1; }
    .hero-subtitle { font-size: 1.2rem; opacity: 0.9; position: relative; z-index: 1; }
    @keyframes slideInUp { from { opacity: 0; transform: translateY(30px);} to { opacity: 1; transform: translateY(0);} }
    @keyframes pulse { 0%, 100% { transform: scale(1);} 50% { transform: scale(1.05);} }
    .score-card {
        animation: slideInUp 0.6s ease-out forwards; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden;
    }
    .score-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2); }
    .score-card:hover { transform: translateY(-8px) scale(1.02); box-shadow: 0 20px 60px rgba(0,0,0,0.15); }
    .high-score { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); border-left: 6px solid #2ecc71; }
    .medium-score { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-left: 6px solid #f39c12; }
    .low-score { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-left: 6px solid #e74c3c; }
    .score-value {
        font-size: 3.5rem; font-weight: 800; margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: pulse 2s infinite;
    }
    .rank-badge {
        display: inline-flex; align-items: center; justify-content: center; width: 50px; height: 50px; border-radius: 50%;
        color: white; font-weight: 700; font-size: 1.2rem; margin-right: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); box-shadow: 0 6px 20px rgba(255,215,0,0.4); }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0 0%, #A0A0A0 100%); box-shadow: 0 6px 20px rgba(192,192,192,0.4); }
    .rank-3 { background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%); box-shadow: 0 6px 20px rgba(205,127,50,0.4); }
    .metric-card {
        background: rgba(255,255,255,0.8); backdrop-filter: blur(10px); border-radius: 12px; padding: 1.5rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.3); transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .stProgress > div > div > div { background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; }
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-radius: 12px; border: 1px solid rgba(102,126,234,0.2);
    }
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        font-weight: 700; font-size: 1.8rem; margin: 2rem 0 1rem 0; text-align: center;
    }
    .assignment-card { background: rgba(255,255,255,0.9); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
        border-left: 4px solid; box-shadow: 0 4px 15px rgba(0,0,0,0.05); transition: all 0.3s ease; }
    .assignment-card:hover { transform: translateX(5px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
    .assignment-good {
        border-left-color: #2ecc71;
        background: linear-gradient(135deg, rgba(46,204,113,0.05) 0%, rgba(46,204,113,0.1) 100%);
    }
    .assignment-rejected {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, rgba(231,76,60,0.05) 0%, rgba(231,76,60,0.1) 100%);
    }
    .alternatives-container {
        background: rgba(255,255,255,0.7); border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
        border: 1px solid rgba(102,126,234,0.2);
    }
    .alternative-item { padding: 0.75rem; margin: 0.5rem 0; border-radius: 8px; background: rgba(255,255,255,0.5);
        border-left: 3px solid #667eea; }
    .alternative-rank {
        display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; margin-right: 0.5rem;
        font-size: 0.9rem;
    }
    .extracted-data-card {
        background: rgba(255,255,255,0.9); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08); border-left: 5px solid #667eea; transition: all 0.3s ease;
        height: 100%;
    }
    .extracted-data-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(0,0,0,0.12); }
    .extracted-title { font-size: 1.3rem; font-weight: 600; color: #667eea; margin-bottom: 1rem; display: flex; align-items: center; }
    .extracted-title .icon { margin-right: 0.5rem; font-size: 1.5rem; }
    .extracted-section { margin-bottom: 1.2rem; }
    .extracted-section h5 { color: #764ba2; margin-bottom: 0.5rem; font-weight: 600; }
    .extracted-list { list-style-type: none; padding-left: 0; }
    .extracted-list li { background: rgba(102,126,234,0.08); border-radius: 8px; padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem; border-left: 3px solid #667eea; }
    .logout-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        border: none;
        padding: 8px 15px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .logout-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(231,76,60,0.4);
    }
    .refresh-btn {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46,204,113,0.3);
    }
    .refresh-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46,204,113,0.4);
    }
    .comparison-container {
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .comparison-column {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .comparison-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px 12px 0 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .comparison-body {
        background: rgba(255,255,255,0.9);
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        flex-grow: 1;
    }
    .cv-scroll-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .cv-scroll-container::-webkit-scrollbar {
        width: 8px;
    }
    .cv-scroll-container::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 4px;
    }
    .cv-scroll-container::-webkit-scrollbar-thumb {
        background: rgba(102,126,234,0.5);
        border-radius: 4px;
    }
    .cv-scroll-container::-webkit-scrollbar-thumb:hover {
        background: rgba(102,126,234,0.7);
    }
    .log-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    .log-table th, .log-table td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    .log-table th {
        background-color: rgba(102,126,234,0.1);
        font-weight: 600;
    }
    .log-table tr:hover {
        background-color: rgba(102,126,234,0.05);
    }
    .model-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .model-openai {
        background-color: rgba(0, 123, 255, 0.1);
        color: #007bff;
    }
    .model-gemini {
        background-color: rgba(234, 67, 53, 0.1);
        color: #ea4335;
    }
</style>
""", unsafe_allow_html=True)
# Add logout button to the sidebar
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if st.session_state.logged_in:
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
else:
    st.write("Please log in to continue.")
# ---------------------------
# Default Prompts (can be edited from UI)
# ---------------------------
DEFAULT_CV_PROMPT = """You are an information-extraction engine. You will receive the full plain text of ONE resume/CV.
Your job is to output STRICT JSON with the following schema, extracting:
Candidate NAME
Exactly 20 SKILL PHRASES (by reviewing the full CV and understanding the skills possessed by this candidate; if fewer than 20 exist, leave the remaining slots as "").
Exactly 10 RESPONSIBILITY PHRASES (from WORK EXPERIENCE / PROFESSIONAL EXPERIENCE sections; if fewer than 10, derive the remaining from CERTIFICATIONS or other professional sections, but never from skills).
The most recent JOB TITLE.
YEARS OF EXPERIENCE (total professional experience by seeing the date the candidate started working and taking a general calculation from start to present. do not calculate using code and you may infer from the text in the CV if you find phrases such as "15 years of experience").
General Rules:
Output valid JSON only. No markdown, no comments, no trailing commas.
Use English only.
Do not invent facts. If something is missing, leave empty strings "" or null.
Arrays must be fixed length: skills_sentences = 20, responsibility_sentences = 10.
De-duplicate near-duplicates (case-insensitive). Keep the most informative version.
Each skill/responsibility must be a concise, descriptive phrase (not a full sentence).
Example: "active directory security assessments to strengthen authentication and access controls"
Avoid: "Performs active directory security assessments to strengthen authentication and access controls."
Remove filler verbs such as performs, provides, carries out, responsible for, manages, oversees.
Skills must be derived by reviewing the full document and understanding what skills the candidate possesses.
Responsibilities must come only from EXPERIENCE/WORK HISTORY sections (and CERTIFICATIONS if needed).
Expand acronyms into their full professional terms (e.g., AWS → Amazon Web Services, SQL → Structured Query Language). Apply consistently.
Ensure skill and responsibility lists are domain-specific phrases only without generic wording.
No duplication across skills and responsibilities.
Output Format:
{
  "doc_type": "resume",
  "name": string | null,
  "job_title": string | null,
  "years_of_experience": number | null,
  "skills_sentences": [
    "<Skill phrase 1>",
    "... (total 20 items)",
    ""
  ],
  "responsibility_sentences": [
    "<Responsibility phrase 1>",
    "... (total 10 items)",
    ""
  ]
}
"""
DEFAULT_JD_PROMPT = """You are an information-extraction engine. You will receive the full plain text of ONE job description (JD).
Your job is to output STRICT JSON with the following schema, extracting:
Exactly 20 SKILL PHRASES (by preferring SKILLS, REQUIREMENTS, QUALIFICATIONS, TECHNOLOGY STACK sections, however read the full document and suggest what skills are required for this position; if fewer than 20 exist, create additional descriptive phrases from related requirements until 20 are filled).
Exactly 10 RESPONSIBILITY PHRASES (from RESPONSIBILITIES, DUTIES, WHAT YOU’LL DO sections; if fewer than 10 exist, expand implied responsibilities until 10 are filled).
The JOB TITLE of the role.
YEARS OF EXPERIENCE (minimum required, if explicitly stated; if a range is given, use the minimum).
General Rules:
Output valid JSON only. No markdown, no comments, no trailing commas.
Use English only.
Do not invent facts. If something is missing, leave empty strings "" or null.
Arrays must be fixed length: skills_sentences = 20, responsibility_sentences = 10.
De-duplicate near-duplicates (case-insensitive). Keep the most informative version.
Each skill/responsibility must be a concise, descriptive phrase (not a full sentence).
Example: "structured query language database administration"
Avoid: "Uses Structured Query Language to administer relational databases."
Remove filler verbs such as develops, implements, provides, generates, manages, responsible for.
Skills should come from SKILLS/REQUIREMENTS/QUALIFICATIONS sections, however review the full document and suggest the skills required for this position.
Responsibilities should come from RESPONSIBILITIES/DUTIES/WHAT YOU’LL DO sections.
Expand acronyms into their full professional terms (e.g., CRM → Customer Relationship Management, API → Application Programming Interface). Apply consistently.
Ensure skills and responsibilities remain short, embedding-friendly phrases with no generic filler wording.
Skills and responsibilities must remain distinct, with no overlap.
Output Format:
{
  "doc_type": "job_description",
  "job_title": string | null,
  "years_of_experience": number | null,
  "skills_sentences": [
    "<Skill phrase 1>",
    "... (total 20 items)",
    ""
  ],
  "responsibility_sentences": [
    "<Responsibility phrase 1>",
    "... (total 10 items)",
    ""
  ]
}
"""
# ---------------------------
# Helpers
# ---------------------------
def truncate_text(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len-3] + "..."
def calculate_years_score(jd_years, cv_years):
    if jd_years == 0:
        return 1.0  # If JD doesn't specify years, any experience is acceptable
    if cv_years >= jd_years:
        return 1.0
    return cv_years / jd_years
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
def normalize_weights(weights):
    total = sum(weights)
    if total == 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [w / total for w in weights]
def read_upload_text(uploaded_file) -> str:
    """Read text from uploaded PDF/DOCX/TXT file."""
    suffix = os.path.splitext(uploaded_file.name.lower())[-1]
    data = uploaded_file.read()
    byts = io.BytesIO(data)
    if suffix in [".pdf"]:
        try:
            text_parts = []
            with pdfplumber.open(byts) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            return ""
    elif suffix in [".docx"]:
        try:
            tmp_path = f"/tmp/{time.time_ns()}_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(data)
            text = docx2txt.process(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return text or ""
        except Exception:
            return ""
    elif suffix in [".txt"]:
        try:
            return byts.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        try:
            return byts.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
def safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
    return None
def pad_or_trim_list(lst, target_len):
    lst = list(lst or [])
    if len(lst) < target_len:
        lst = lst + [""] * (target_len - len(lst))
    else:
        lst = lst[:target_len]
    return lst
def join_bullets(items):
    items = [x for x in (items or []) if isinstance(x, str)]
    return "\n".join([f"• {x.strip()}" for x in items if x.strip()])
def ensure_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0
# ---------------------------
# 🔽 NEW: session defaults for model + temperature
# ---------------------------
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL
if "model_temp" not in st.session_state:
    st.session_state.model_temp = MODEL_PRESETS.get(DEFAULT_MODEL, {"temperature": 0.0})["temperature"]
# ---------------------------
# OpenAI/Gemini call wrapper
# ---------------------------
def _create_chat_completion(model_name, messages, response_format_json=True, temperature=0):
    """
    Returns: (resp_obj_or_dict, duration_seconds, error)
    """
    t0 = time.perf_counter()
    
    # Check if it's a Gemini model
    if model_name.startswith("gemini-"):
        if not GEMINI_API_KEY:
            return None, 0.0, "GEMINI_API_KEY not found (set it in .env or Streamlit secrets)"
        
        try:
            # Configure Gemini
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(model_name)
            
            # Combine all messages into one content string
            full_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    full_content += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    full_content += f"User: {msg['content']}\n\n"
            
            # Generate content
            response = model.generate_content(
                full_content,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                )
            )
            
            dt = time.perf_counter() - t0
            
            # Create a response object similar to OpenAI's structure
            resp_obj = {
                "choices": [{"message": {"content": response.text}}],
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                    "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                }
            }
            
            return resp_obj, dt, None
            
        except Exception as e:
            dt = time.perf_counter() - t0
            return None, dt, str(e)
    
    # Otherwise, use OpenAI
    else:
        if not OPENAI_API_KEY:
            return None, 0.0, "OPENAI_API_KEY not found (set it in .env or Streamlit secrets)"
        
        try:
            if CLIENT_AVAILABLE:
                client = openai.Client(api_key=OPENAI_API_KEY)
                kwargs = {"model": model_name, "messages": messages, "temperature": float(temperature)}
                if response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                dt = time.perf_counter() - t0
                return resp, dt, None
            elif LEGACY_AVAILABLE:
                openai.api_key = OPENAI_API_KEY
                kwargs = {"model": model_name, "messages": messages, "temperature": float(temperature)}
                try:
                    if response_format_json:
                        kwargs["response_format"] = {"type": "json_object"}
                    # Some legacy versions might ignore response_format
                except Exception:
                    pass
                resp = openai.ChatCompletion.create(**kwargs)
                dt = time.perf_counter() - t0
                return resp, dt, None
            else:
                return None, 0.0, "openai SDK is not available in this environment"
        except Exception as e:
            dt = time.perf_counter() - t0
            return None, dt, str(e)
def _extract_usage(resp):
    """
    Normalize token usage for both OpenAI and Gemini responses.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if resp is None:
        return usage
    
    # Handle Gemini response (dict)
    if isinstance(resp, dict):
        u = resp.get("usage", {})
        usage["prompt_tokens"] = u.get("prompt_tokens", 0) or 0
        usage["completion_tokens"] = u.get("completion_tokens", 0) or 0
        usage["total_tokens"] = u.get("total_tokens", 0) or 0
        return usage
    
    # Handle OpenAI response (object)
    if hasattr(resp, "usage"):
        u = getattr(resp, "usage", None)
        usage["prompt_tokens"] = getattr(u, "prompt_tokens", 0) or 0
        usage["completion_tokens"] = getattr(u, "completion_tokens", 0) or 0
        usage["total_tokens"] = getattr(u, "total_tokens", 0) or 0
        return usage
    
    try:
        if isinstance(resp, dict) and "usage" in resp:
            u = resp["usage"] or {}
            usage["prompt_tokens"] = int(u.get("prompt_tokens", 0))
            usage["completion_tokens"] = int(u.get("completion_tokens", 0))
            usage["total_tokens"] = int(u.get("total_tokens", 0))
    except Exception:
        pass
    
    return usage
def _extract_content(resp):
    """
    Get message content from both OpenAI and Gemini responses.
    """
    if resp is None:
        return ""
    
    # Handle Gemini response (dict)
    if isinstance(resp, dict):
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return ""
    
    # Handle OpenAI response (object)
    if hasattr(resp, "choices"):
        try:
            return resp.choices[0].message.content
        except Exception:
            return ""
    
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return ""
# ---------------------------
# LLM extraction (STRICT JSON) with prompt override
# ---------------------------
def llm_extract(text: str, which: str, model_name: str, temperature: float, prompt_override: str = None):
    """
    which: "cv" or "jd"
    returns: (parsed_json, usage_dict, duration_seconds, raw_response_text, error)
    """
    if not OPENAI_AVAILABLE and not GEMINI_AVAILABLE:
        return None, {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}, 0.0, "", "Neither OpenAI nor Gemini package is available"
    
    if not model_name:
        return None, {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}, 0.0, "", "Model name is empty"
    
    if prompt_override and isinstance(prompt_override, str) and prompt_override.strip():
        prompt = prompt_override
    else:
        prompt = DEFAULT_CV_PROMPT if which == "cv" else DEFAULT_JD_PROMPT
    
    messages = [
        {"role": "system", "content": "You are a careful information-extraction engine. Output strict JSON only."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": f"---\nBEGIN DOCUMENT\n{text}\nEND DOCUMENT\n---"}
    ]
    
    resp, dt, err = _create_chat_completion(model_name, messages, response_format_json=True, temperature=temperature)
    if err:
        return None, {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}, dt, "", err
    
    raw = _extract_content(resp)
    usage = _extract_usage(resp)
    parsed = safe_json_loads(raw)
    if parsed is None:
        return None, usage, dt, raw, "Failed to parse JSON"
    
    return parsed, usage, dt, raw, None
# ---------------------------
# Hero Header
# ---------------------------
col1, _ = st.columns([1, 0.01])  # just keep one wide column
with col1:
    st.markdown("""
    <div class="hero-header" style="text-align:center;">
        <h1 class="hero-title">🎯 ALPHA CV Matcher</h1>
        <p class="hero-subtitle">BETA Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

    # 🔄 Refresh button centered below the header
    if st.button("🔄 Refresh", key="refresh_button"):
        reset_session_state()

# ---------------------------
# Session state init (rest)
# ---------------------------
if "cv_data" not in st.session_state:
    st.session_state.cv_data = []       # list of dicts: {skills, responsibilities, job_title, years}
if "cv_names" not in st.session_state:
    st.session_state.cv_names = []      # list of names for each candidate card
if "logs_cvs" not in st.session_state:
    st.session_state.logs_cvs = []      # list of dicts per CV extraction (name, time, tokens)
if "logs_jd" not in st.session_state:
    st.session_state.logs_jd = []       # list with a single dict for JD extraction
if "jd_fields" not in st.session_state:
    st.session_state.jd_fields = {"skills": "", "responsibilities": "", "job_title": "", "years": 0}
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "show_extracted_jd" not in st.session_state:
    st.session_state.show_extracted_jd = False
if "show_extracted_cvs" not in st.session_state:
    st.session_state.show_extracted_cvs = False
if "weights" not in st.session_state:
    st.session_state.weights = {'skills': 80, 'responsibilities': 15, 'job_title': 2.5, 'experience': 2.5}
if "prompts" not in st.session_state:
    st.session_state.prompts = {"cv": DEFAULT_CV_PROMPT, "jd": DEFAULT_JD_PROMPT}
# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["⚙️ Customization", "📥 Upload & Extract", "📊 Results"])

# =========================== TAB 1: CUSTOMIZATION ===========================
with tab1:
    st.markdown("## 🔐 API Status")

    # 🔽 NEW: Model dropdown + automatic temp
    model_options = list(MODEL_PRESETS.keys())
    default_index = model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0

    colA, colB = st.columns([1.5, 2])
    with colA:
        selected_model = st.selectbox("Select Model", model_options, index=default_index, key="model_name")
        st.session_state.model_temp = MODEL_PRESETS.get(selected_model, {"temperature": 0.0})["temperature"]

        # Add styled badge for provider
        model_type = "OpenAI" if not selected_model.startswith("gemini-") else "Gemini"
        badge_color = "#007bff" if model_type == "OpenAI" else "#ea4335"
        st.markdown(
            f"""
            <div style="margin-top:10px; display:inline-block; 
                        padding:4px 10px; border-radius:8px; 
                        background-color:{badge_color}; color:white; 
                        font-weight:600; font-size:0.85rem;">
                {model_type} Model
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("Temperature (auto)", f"{st.session_state.model_temp:.2f}")

    with colB:
        # API connection cards
        openai_ok_pkg = "✅" if OPENAI_AVAILABLE else "❌"
        openai_ok_key = "✅" if bool(OPENAI_API_KEY) else "❌"
        gemini_ok_pkg = "✅" if GEMINI_AVAILABLE else "❌"
        gemini_ok_key = "✅" if bool(GEMINI_API_KEY) else "❌"

        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem;">
                <div style="padding:12px; border-radius:10px; background:rgba(0,123,255,0.08);">
                    <h4 style="margin:0; color:#007bff;">OpenAI</h4>
                    <p style="margin:4px 0;">📦 Package: {openai_ok_pkg}</p>
                    <p style="margin:4px 0;">🔑 API Key: {openai_ok_key}</p>
                    <code style="font-size:0.8rem;">{(OPENAI_API_KEY[:2] + "..." + OPENAI_API_KEY[-1:]) if OPENAI_API_KEY else "—"}</code>
                </div>
                <div style="padding:12px; border-radius:10px; background:rgba(234,67,53,0.08);">
                    <h4 style="margin:0; color:#ea4335;">Gemini</h4>
                    <p style="margin:4px 0;">📦 Package: {gemini_ok_pkg}</p>
                    <p style="margin:4px 0;">🔑 API Key: {gemini_ok_key}</p>
                    <code style="font-size:0.8rem;">{(GEMINI_API_KEY[:2] + "..." + GEMINI_API_KEY[-1:]) if GEMINI_API_KEY else "—"}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


  

    # Test API call button
    st.markdown("### 🧪 Test API Connection")
    if st.button("▶️ Run Test"):
        model_type = "OpenAI" if not selected_model.startswith("gemini-") else "Gemini"
        if model_type == "OpenAI":
            if not OPENAI_AVAILABLE:
                st.error("❌ OpenAI package not importable. Run: `pip install --upgrade openai`")
            elif not OPENAI_API_KEY:
                st.error("❌ OPENAI_API_KEY missing in `.env` or Streamlit secrets.")
            else:
                try:
                    messages = [{"role":"system","content":"Reply with JSON {\"ok\":true} and nothing else."}]
                    resp, _, err = _create_chat_completion(
                        st.session_state.model_name,
                        messages,
                        response_format_json=True,
                        temperature=st.session_state.model_temp
                    )
                    if err:
                        st.error(f"❌ OpenAI call failed: {err}")
                    else:
                        st.success(f"✅ OpenAI is working! Usage: {_extract_usage(resp)}")
                except Exception as e:
                    st.error(f"❌ OpenAI call failed: {e}")
        else:  # Gemini
            if not GEMINI_AVAILABLE:
                st.error("❌ Gemini package not importable. Run: `pip install --upgrade google-generativeai`")
            elif not GEMINI_API_KEY:
                st.error("❌ GEMINI_API_KEY missing in `.env` or Streamlit secrets.")
            else:
                try:
                    messages = [{"role":"system","content":"Reply with JSON {\"ok\":true} and nothing else."}]
                    resp, _, err = _create_chat_completion(
                        st.session_state.model_name,
                        messages,
                        response_format_json=True,
                        temperature=st.session_state.model_temp
                    )
                    if err:
                        st.error(f"❌ Gemini call failed: {err}")
                    else:
                        st.success(f"✅ Gemini is working! Usage: {_extract_usage(resp)}")
                except Exception as e:
                    st.error(f"❌ Gemini call failed: {e}")

    st.markdown("---")
    st.subheader("📊 Weights")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        skills_weight = st.slider("🎯 Skills (%)", 0, 100, int(st.session_state.weights['skills']), 5)
    with c2:
        resp_weight = st.slider("📋 Responsibilities (%)", 0, 100, int(st.session_state.weights['responsibilities']), 5)
    with c3:
        title_weight = st.slider("💼 Job Title (%)", 0.0, 100.0, float(st.session_state.weights['job_title']), 2.5)
    with c4:
        exp_weight = st.slider("⏳ Experience (%)", 0.0, 100.0, float(st.session_state.weights['experience']), 2.5)
    total_weight = skills_weight + resp_weight + title_weight + exp_weight
    if total_weight == 100:
        st.success(f"✅ Total: {total_weight}%")
    elif total_weight == 0:
        st.error("❌ All weights are 0%. Set at least one.")
    else:
        st.warning(f"⚠️ Total: {total_weight}% (will be normalized)")
    st.session_state.weights = {
        'skills': skills_weight,
        'responsibilities': resp_weight,
        'job_title': title_weight,
        'experience': exp_weight
    }
    if st.button("🔄 Reset Default Weights"):
        st.session_state.weights = {'skills': 80, 'responsibilities': 15, 'job_title': 2.5, 'experience': 2.5}
        st.rerun()
    st.markdown("---")
    st.subheader("✍️ Prompts (Edit as needed)")
    colP1, colP2 = st.columns(2)
    with colP1:
        st.text_area("CV Extraction Prompt (STRICT JSON)", value=st.session_state.prompts["cv"], height=380, key="cv_prompt_ui")
    with colP2:
        st.text_area("JD Extraction Prompt (STRICT JSON)", value=st.session_state.prompts["jd"], height=380, key="jd_prompt_ui")
    st.caption("These prompts will be used when extracting CVs and JD.")
    st.session_state.prompts["cv"] = st.session_state.cv_prompt_ui
    st.session_state.prompts["jd"] = st.session_state.jd_prompt_ui
# =========================== TAB 2: UPLOAD & EXTRACT ===========================
with tab2:
    st.subheader("Upload Files")
    colU1, colU2 = st.columns([1,1])
    with colU1:
        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf","docx","txt"], accept_multiple_files=False, key="jd_upl")
    with colU2:
        cv_files = st.file_uploader("Upload CVs (Multiple)", type=["pdf","docx","txt"], accept_multiple_files=True, key="cv_upl")
    colB1, colB2, colB3, colB4 = st.columns([1,1,1,1])
    with colB1:
        extract_jd_btn = st.button("🧠 Extract JD")
    with colB2:
        clear_jd_btn = st.button("🧹 Clear JD")
    with colB3:
        extract_cvs_btn = st.button("🧠 Extract CVs")
    with colB4:
        clear_cvs_btn = st.button("🧹 Clear CVs")
    # Handle Clear buttons
    if clear_jd_btn:
        st.session_state.jd_fields = {"skills": "", "responsibilities": "", "job_title": "", "years": 0}
        st.session_state.logs_jd = []
        st.session_state.show_extracted_jd = False
        st.rerun()
    if clear_cvs_btn:
        st.session_state.cv_data = []
        st.session_state.cv_names = []
        st.session_state.logs_cvs = []
        st.session_state.show_extracted_cvs = False
        st.rerun()
    # Extract JD
    if extract_jd_btn:
        if jd_file is None:
            st.warning("Please upload a JD file first.")
        else:
            with st.spinner("Extracting JD with LLM..."):
                text = read_upload_text(jd_file)
                parsed, usage, dt, raw, err = llm_extract(
                    text, which="jd",
                    model_name=st.session_state.model_name,
                    temperature=st.session_state.model_temp,
                    prompt_override=st.session_state.prompts["jd"]
                )
                if err:
                    st.error(f"JD extraction error: {err}")
                elif not parsed:
                    st.error("JD extraction failed (no JSON parsed).")
                else:
                    job_title = parsed.get("job_title") or ""
                    years = parsed.get("years_of_experience")
                    years = int(ensure_float(years))
                    skills = pad_or_trim_list(parsed.get("skills_sentences", []), 20)
                    resps  = pad_or_trim_list(parsed.get("responsibility_sentences", []), 10)
                    st.session_state.jd_fields = {
                        "skills": join_bullets(skills),
                        "responsibilities": join_bullets(resps),
                        "job_title": job_title or "",
                        "years": years
                    }
                    st.session_state.logs_jd = [{
                        "model": st.session_state.model_name,
                        "time_sec": round(dt, 3),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }]
                    st.session_state.show_extracted_jd = True
                    st.success("JD extracted and fields filled.")
                    st.rerun()
    # Extract CVs
    if extract_cvs_btn:
        if not cv_files:
            st.warning("Please upload at least one CV file first.")
        else:
            logs = []
            new_cv_data = []
            new_cv_names = []
            with st.spinner("Extracting CVs with LLM..."):
                for f in cv_files:
                    text = read_upload_text(f)
                    parsed, usage, dt, raw, err = llm_extract(
                        text, which="cv",
                        model_name=st.session_state.model_name,
                        temperature=st.session_state.model_temp,
                        prompt_override=st.session_state.prompts["cv"]
                    )
                    if err or not parsed:
                        new_cv_data.append({"skills": "", "responsibilities": "", "job_title": "", "years": 0})
                        new_cv_names.append(os.path.splitext(f.name)[0] + " (extract error)")
                        logs.append({
                            "model": st.session_state.model_name,
                            "time_sec": round(dt, 3),
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        })
                        continue
                    name = parsed.get("name") or os.path.splitext(f.name)[0]
                    job_title = parsed.get("job_title") or ""
                    years = parsed.get("years_of_experience")
                    years = int(ensure_float(years))
                    skills = pad_or_trim_list(parsed.get("skills_sentences", []), 20)
                    resps  = pad_or_trim_list(parsed.get("responsibility_sentences", []), 10)
                    new_cv_data.append({
                        "skills": join_bullets(skills),
                        "responsibilities": join_bullets(resps),
                        "job_title": job_title,
                        "years": years
                    })
                    new_cv_names.append(str(name))
                    logs.append({
                        "model": st.session_state.model_name,
                        "time_sec": round(dt, 3),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    })
            st.session_state.cv_data = new_cv_data
            st.session_state.cv_names = new_cv_names
            st.session_state.logs_cvs = logs
            st.session_state.show_extracted_cvs = True
            st.success("CVs extracted and fields filled.")
            st.rerun()
    # Show extraction logs (inside tab) - Modified to show only model, time, and tokens
    st.markdown('<h2 class="section-header">🧾 LLM Extraction Logs</h2>', unsafe_allow_html=True)
    cols_logs = st.columns(2)
    with cols_logs[0]:
        st.subheader("JD")
        if st.session_state.logs_jd:
            # Create a simplified dataframe with only model, time, and tokens
            jd_logs_simplified = []
            for log in st.session_state.logs_jd:
                jd_logs_simplified.append({
                    "Model": log.get("model", ""),
                    "Time (sec)": log.get("time_sec", 0),
                    "Prompt Tokens": log.get("prompt_tokens", 0),
                    "Completion Tokens": log.get("completion_tokens", 0),
                    "Total Tokens": log.get("total_tokens", 0)
                })
            df_jd = pd.DataFrame(jd_logs_simplified)
            st.dataframe(df_jd, use_container_width=True)
        else:
            st.info("No JD extraction yet.")
    with cols_logs[1]:
        st.subheader("CVs")
        if st.session_state.logs_cvs:
            # Create a simplified dataframe with only model, time, and tokens
            cv_logs_simplified = []
            for log in st.session_state.logs_cvs:
                cv_logs_simplified.append({
                    "Model": log.get("model", ""),
                    "Time (sec)": log.get("time_sec", 0),
                    "Prompt Tokens": log.get("prompt_tokens", 0),
                    "Completion Tokens": log.get("completion_tokens", 0),
                    "Total Tokens": log.get("total_tokens", 0)
                })
            df_cvs = pd.DataFrame(cv_logs_simplified)
            st.dataframe(df_cvs, use_container_width=True)
        else:
            st.info("No CV extraction yet.")
    
    # Show extracted data (JD and CVs side by side)
    if st.session_state.show_extracted_jd or st.session_state.show_extracted_cvs:
        st.markdown('<h2 class="section-header">📋 Extracted Data Comparison</h2>', unsafe_allow_html=True)
        
        # Create a comparison container with two columns
        col_jd, col_cv = st.columns([1, 1])
        
        # JD Column (Left)
        with col_jd:
            if st.session_state.show_extracted_jd and st.session_state.jd_fields:
                jd = st.session_state.jd_fields
                skills_list = [item.strip().lstrip("• ").strip() for item in jd['skills'].split('\n') if item.strip()]
                resp_list = [item.strip().lstrip("• ").strip() for item in jd['responsibilities'].split('\n') if item.strip()]
                
                st.markdown("""
                <div class="comparison-container">
                    <div class="comparison-column">
                        <div class="comparison-header">📄 Job Description</div>
                        <div class="comparison-body">
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="extracted-title"><span class="icon">📄</span> Job Description</div>
                    <div class="extracted-info">
                        <div class="extracted-info-item"><strong>Job Title:</strong> {jd['job_title'] or 'Not specified'}</div>
                        <div class="extracted-info-item"><strong>Years Required:</strong> {jd['years'] if jd['years'] > 0 else 'Not specified'}</div>
                    </div>
                    <div class="extracted-section">
                        <h5>Skills ({len(skills_list)})</h5>
                        <ul class="extracted-list">
                            {"".join([f"<li>{skill}</li>" for skill in skills_list])}
                        </ul>
                    </div>
                    <div class="extracted-section">
                        <h5>Responsibilities ({len(resp_list)})</h5>
                        <ul class="extracted-list">
                            {"".join([f"<li>{resp}</li>" for resp in resp_list])}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No JD data extracted yet.")
        
        # CV Column (Right)
        with col_cv:
            if st.session_state.show_extracted_cvs and st.session_state.cv_data:
                st.markdown("""
                <div class="comparison-container">
                    <div class="comparison-column">
                        <div class="comparison-header">👥 Candidate CVs</div>
                        <div class="comparison-body cv-scroll-container">
                """, unsafe_allow_html=True)
                
                for i, (cv, name) in enumerate(zip(st.session_state.cv_data, st.session_state.cv_names)):
                    skills_list = [item.strip().lstrip("• ").strip() for item in cv['skills'].split('\n') if item.strip()]
                    resp_list = [item.strip().lstrip("• ").strip() for item in cv['responsibilities'].split('\n') if item.strip()]
                    
                    st.markdown(f"""
                        <div class="extracted-data-card">
                            <div class="extracted-title">
                                <span class="icon">👤</span> {name}
                            </div>
                            <div class="extracted-info">
                                <div class="extracted-info-item"><strong>Current Role:</strong> {cv['job_title'] or 'Not specified'}</div>
                                <div class="extracted-info-item"><strong>Experience:</strong> {cv['years']} years</div>
                            </div>
                            <div class="extracted-section">
                                <h5>Skills ({len(skills_list)})</h5>
                                <ul class="extracted-list">
                                    {"".join([f"<li>{skill}</li>" for skill in skills_list])}
                                </ul>
                            </div>
                            <div class="extracted-section">
                                <h5>Responsibilities ({len(resp_list)})</h5>
                                <ul class="extracted-list">
                                    {"".join([f"<li>{resp}</li>" for resp in resp_list])}
                                </ul>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No CV data extracted yet.")
    
    # Editable JD + CVs (auto-filled)
    st.markdown("---")
    st.subheader("📋 Job Description (editable)")
    colJ1, colJ2 = st.columns(2)
    with colJ1:
        st.text_input("Job Title", value=st.session_state.jd_fields.get("job_title", ""), key="jd_job_title_input")
    with colJ2:
        st.number_input("Years Required", min_value=0, value=int(st.session_state.jd_fields.get("years", 0)), step=1, key="jd_years_input")
    st.text_area(
        "Required Skills (one per line)",
        value=st.session_state.jd_fields.get("skills", ""),
        height=140,
        key="jd_skills_input",
        placeholder="• Python Programming\n• Machine Learning\n• Data Analysis\n• Team Leadership"
    )
    st.text_area(
        "Key Responsibilities (one per line)",
        value=st.session_state.jd_fields.get("responsibilities", ""),
        height=140,
        key="jd_responsibilities_input",
        placeholder="• Lead data science projects\n• Develop ML models\n• Mentor junior developers"
    )
    # Sync back to session
    st.session_state.jd_fields["job_title"] = st.session_state.jd_job_title_input
    st.session_state.jd_fields["years"] = int(st.session_state.jd_years_input)
    st.session_state.jd_fields["skills"] = st.session_state.jd_skills_input
    st.session_state.jd_fields["responsibilities"] = st.session_state.jd_responsibilities_input
    st.markdown("---")
    st.subheader("👥 Candidate CVs (editable)")
    def add_cv():
        st.session_state.cv_data.append({"skills": "", "responsibilities": "", "job_title": "", "years": 0})
        st.session_state.cv_names.append(f"Candidate {len(st.session_state.cv_data)}")
    def remove_cv(index):
        if len(st.session_state.cv_data) > 0:
            st.session_state.cv_data.pop(index)
            st.session_state.cv_names.pop(index)
    if not st.session_state.cv_data:
        add_cv()
    for i, (cv_data, cv_name) in enumerate(zip(st.session_state.cv_data, st.session_state.cv_names)):
        with st.expander(f"👤 {cv_name}", expanded=(i < 2)):
            new_name = st.text_input("Candidate Name", value=cv_name, key=f"cv_name_{i}")
            st.session_state.cv_names[i] = new_name
            cv_data["skills"] = st.text_area("Skills (one per line)", value=cv_data["skills"], height=100, key=f"cv_skills_{i}",
                                             placeholder="• Python\n• SQL\n• Machine Learning")
            cv_data["responsibilities"] = st.text_area("Experience (one per line)", value=cv_data["responsibilities"], height=100, key=f"cv_resp_{i}",
                                                       placeholder="• Built ML models\n• Analyzed datasets")
            colcv1, colcv2 = st.columns(2)
            with colcv1:
                cv_data["job_title"] = st.text_input("Current Role", value=cv_data["job_title"], key=f"cv_title_{i}")
            with colcv2:
                cv_data["years"] = st.number_input("Experience (years)", min_value=0, value=int(cv_data["years"]), step=1, key=f"cv_years_{i}")
            st.button("🗑️ Remove", key=f"remove_{i}", on_click=remove_cv, args=(i,), type="secondary")
    st.button("➕ Add CV", on_click=add_cv, type="secondary")
# =========================== TAB 3: RESULTS ===========================
with tab3:
    st.subheader("Run Matching")
    analyze_button = st.button("🚀 Analyze Matches")
    if analyze_button:
        st.session_state.run_analysis = True
    if st.session_state.get('run_analysis', False):
        # Read latest editable values
        jd_skills = st.session_state.jd_fields.get("skills", "")
        jd_responsibilities = st.session_state.jd_fields.get("responsibilities", "")
        jd_job_title = st.session_state.jd_fields.get("job_title", "")
        jd_years = int(st.session_state.jd_fields.get("years", 0))
        # Prepare lists
        jd_skills_list = [s.strip("• ").strip() for s in jd_skills.split("\n") if s.strip()]
        jd_resp_list   = [s.strip("• ").strip() for s in jd_responsibilities.split("\n") if s.strip()]
        # Validate weights/inputs
        w = st.session_state.weights
        skills_weight_norm, resp_weight_norm, title_weight_norm, exp_weight_norm = normalize_weights(
            [w['skills'], w['responsibilities'], w['job_title'], w['experience']]
        )
        validation_errors = []
        if skills_weight_norm > 0 and not jd_skills_list:
            validation_errors.append("Skills are weighted but no JD skills provided")
        if resp_weight_norm > 0 and not jd_resp_list:
            validation_errors.append("Responsibilities are weighted but no JD responsibilities provided")
        if title_weight_norm > 0 and not jd_job_title:
            validation_errors.append("Job title is weighted but no JD job title provided")
        # Removed validation for experience when jd_years is 0
        # If experience is weighted but JD years is 0, we'll proceed with 0 years
        if validation_errors:
            st.error("⚠️ Configuration issues:")
            for e in validation_errors:
                st.error(f"• {e}")
            st.stop()
        # Build CV list
        cv_data_list, cv_names_list = [], []
        for i, cv in enumerate(st.session_state.cv_data):
            cv_name = st.session_state.cv_names[i]
            cv_skills = [s.strip("• ").strip() for s in cv.get("skills","").split("\n") if s.strip()]
            cv_resp   = [s.strip("• ").strip() for s in cv.get("responsibilities","").split("\n") if s.strip()]
            cv_job_title = cv.get("job_title","")
            cv_years = int(ensure_float(cv.get("years", 0)))
            cv_data_list.append({"skills": cv_skills, "responsibilities": cv_resp, "job_title": cv_job_title, "years": cv_years})
            cv_names_list.append(cv_name)
        if not cv_data_list:
            st.error("⚠️ Please provide at least one candidate CV with relevant data")
            st.stop()
        # Progress + embeddings
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner(""):
            status_text.markdown("🔄 **Loading AI model...**")
            progress_bar.progress(10)
            model = load_model()
            status_text.markdown("🧠 **Processing embeddings...**")
            progress_bar.progress(30)
            jd_skills_embeddings = np.array([])
            jd_resp_embeddings   = np.array([])
            if skills_weight_norm > 0 and jd_skills_list:
                jd_skills_embeddings = model.encode(jd_skills_list, normalize_embeddings=True)
            if resp_weight_norm > 0 and jd_resp_list:
                jd_resp_embeddings   = model.encode(jd_resp_list, normalize_embeddings=True)
            progress_bar.progress(50)
            # Flatten CV entries with separate index lists for skills/resp
            all_cv_skills, cv_skill_indices, cv_indices_skills = [], [], []
            all_cv_resp,   cv_resp_indices,   cv_indices_resp   = [], [], []
            for cv_idx, cvd in enumerate(cv_data_list):
                if skills_weight_norm > 0:
                    for si, s in enumerate(cvd["skills"]):
                        all_cv_skills.append(s)
                        cv_skill_indices.append(si)
                        cv_indices_skills.append(cv_idx)
                if resp_weight_norm > 0:
                    for ri, r in enumerate(cvd["responsibilities"]):
                        all_cv_resp.append(r)
                        cv_resp_indices.append(ri)
                        cv_indices_resp.append(cv_idx)
            progress_bar.progress(70)
            cv_skills_embeddings = np.array([])
            cv_resp_embeddings   = np.array([])
            if skills_weight_norm > 0 and all_cv_skills:
                cv_skills_embeddings = model.encode(all_cv_skills, normalize_embeddings=True)
            if resp_weight_norm > 0 and all_cv_resp:
                cv_resp_embeddings   = model.encode(all_cv_resp, normalize_embeddings=True)
            # Qdrant in-memory indexes
            qdrant_skills = None
            if skills_weight_norm > 0 and len(cv_skills_embeddings) > 0:
                qdrant_skills = QdrantClient(":memory:")
                if qdrant_skills.collection_exists(COLLECTION_NAME):
                    qdrant_skills.delete_collection(COLLECTION_NAME)
                qdrant_skills.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=cv_skills_embeddings.shape[1], distance=models.Distance.COSINE),
                )
                points = []
                for i in range(len(all_cv_skills)):
                    points.append(models.PointStruct(
                        id=i,
                        vector=cv_skills_embeddings[i].tolist(),
                        payload={
                            "skill": all_cv_skills[i],
                            "cv_index": cv_indices_skills[i],
                            "cv_skill_index": cv_skill_indices[i],
                            "cv_name": cv_names_list[cv_indices_skills[i]]
                        }
                    ))
                qdrant_skills.upsert(collection_name=COLLECTION_NAME, points=points)
            qdrant_resp = None
            if resp_weight_norm > 0 and len(cv_resp_embeddings) > 0:
                qdrant_resp = QdrantClient(":memory:")
                if qdrant_resp.collection_exists(RESP_COLLECTION_NAME):
                    qdrant_resp.delete_collection(RESP_COLLECTION_NAME)
                qdrant_resp.create_collection(
                    collection_name=RESP_COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=cv_resp_embeddings.shape[1], distance=models.Distance.COSINE),
                )
                points = []
                for i in range(len(all_cv_resp)):
                    points.append(models.PointStruct(
                        id=i,
                        vector=cv_resp_embeddings[i].tolist(),
                        payload={
                            "responsibility": all_cv_resp[i],
                            "cv_index": cv_indices_resp[i],
                            "cv_resp_index": cv_resp_indices[i],
                            "cv_name": cv_names_list[cv_indices_resp[i]]
                        }
                    ))
                qdrant_resp.upsert(collection_name=RESP_COLLECTION_NAME, points=points)
            progress_bar.progress(90)
            status_text.markdown("⚡ **Calculating match scores...**")
        # Title embeddings once
        jd_title_embedding = None
        if title_weight_norm > 0 and jd_job_title:
            jd_title_embedding = load_model().encode([jd_job_title], normalize_embeddings=True)
        # Aggregate results
        all_results = []
        cv_overall_scores = {}
        for cv_idx, (cv_name, cvd) in enumerate(zip(cv_names_list, cv_data_list)):
            cv_skills = cvd["skills"]
            cv_resp   = cvd["responsibilities"]
            cv_title  = cvd["job_title"]
            cv_years  = cvd["years"]
            skills_score = 0.0
            resp_score   = 0.0
            title_score  = 0.0
            years_score  = 0.0
            assignments, resp_assignments = [], []
            skills_top_sorted, resp_top_sorted = {}, {}
            # Skills matching
            if skills_weight_norm > 0 and jd_skills_list and cv_skills and qdrant_skills is not None:
                M, N = len(jd_skills_list), len(cv_skills)
                sim = np.zeros((M, N), dtype=np.float32)
                for j, jd_vec in enumerate(jd_skills_embeddings):
                    res = qdrant_skills.query_points(
                        collection_name=COLLECTION_NAME,
                        query=jd_vec.tolist(),
                        limit=len(cv_skills_embeddings),
                        with_payload=True,
                    )
                    filt = [p for p in res.points if p.payload.get("cv_index") == cv_idx]
                    row = []
                    for p in filt:
                        si = p.payload["cv_skill_index"]
                        sc = float(p.score)
                        if 0 <= si < N:
                            sim[j, si] = sc
                            row.append((si, cv_skills[si], sc))
                    row.sort(key=lambda x: x[2], reverse=True)
                    skills_top_sorted[j] = row
                if sim.size > 0:
                    r, c = linear_sum_assignment(-sim)
                    for rr, cc in zip(r, c):
                        assignments.append({
                            "type": "skill",
                            "jd_index": rr,
                            "jd_item": jd_skills_list[rr],
                            "cv_index": cc,
                            "cv_item": cv_skills[cc],
                            "score": float(sim[rr, cc])
                        })
                    skills_score = float(np.mean([a["score"] for a in assignments])) if assignments else 0.0
            # Responsibilities matching
            if resp_weight_norm > 0 and jd_resp_list and cv_resp and qdrant_resp is not None:
                M, N = len(jd_resp_list), len(cv_resp)
                simr = np.zeros((M, N), dtype=np.float32)
                for j, jd_vec in enumerate(jd_resp_embeddings):
                    res = qdrant_resp.query_points(
                        collection_name=RESP_COLLECTION_NAME,
                        query=jd_vec.tolist(),
                        limit=len(cv_resp_embeddings),
                        with_payload=True,
                    )
                    filt = [p for p in res.points if p.payload.get("cv_index") == cv_idx]
                    row = []
                    for p in filt:
                        ri = p.payload["cv_resp_index"]
                        sc = float(p.score)
                        if 0 <= ri < N:
                            simr[j, ri] = sc
                            row.append((ri, cv_resp[ri], sc))
                    row.sort(key=lambda x: x[2], reverse=True)
                    resp_top_sorted[j] = row
                if simr.size > 0:
                    r, c = linear_sum_assignment(-simr)
                    for rr, cc in zip(r, c):
                        resp_assignments.append({
                            "type": "responsibility",
                            "jd_index": rr,
                            "jd_item": jd_resp_list[rr],
                            "cv_index": cc,
                            "cv_item": cv_resp[cc],
                            "score": float(simr[rr, cc])
                        })
                    resp_score = float(np.mean([a["score"] for a in resp_assignments])) if resp_assignments else 0.0
            # Job title
            if title_weight_norm > 0 and jd_title_embedding is not None and cv_title:
                cv_title_emb = load_model().encode([cv_title], normalize_embeddings=True)
                title_score = float(np.dot(jd_title_embedding, cv_title_emb.T)[0][0])
            # Years
            if exp_weight_norm > 0:
                years_score = calculate_years_score(jd_years, cv_years)
            overall = (skills_weight_norm * skills_score +
                       resp_weight_norm   * resp_score +
                       title_weight_norm  * title_score +
                       exp_weight_norm    * years_score)
            cv_overall_scores[cv_name] = {
                "overall_score": overall,
                "skills_score": skills_score,
                "resp_score": resp_score,
                "job_title_score": title_score,
                "years_score": years_score,
                "jd_job_title": jd_job_title,
                "cv_job_title": cv_title,
                "jd_years": jd_years,
                "cv_years": cv_years
            }
            all_results.append({
                "cv_name": cv_name,
                "cv_idx": cv_idx,
                "cv_data": cvd,
                "skills_assignments": assignments,
                "resp_assignments": resp_assignments,
                "skills_top_sorted_lists": skills_top_sorted,
                "resp_top_sorted_lists": resp_top_sorted,
                "overall_score": overall,
                "skills_score": skills_score,
                "resp_score": resp_score,
                "job_title_score": title_score,
                "years_score": years_score
            })
        progress_bar.progress(100)
        status_text.markdown("✅ **Analysis complete!**")
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        # ---------------------------
        # Results UI
        # ---------------------------
        st.markdown('<h2 class="section-header">🏆 Executive Summary</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <h4>📊 Current Matching Weights</h4>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin:1rem 0;">
                <div style="padding:1rem;background:rgba(102,126,234,0.1);border-radius:8px;"><strong>🎯 Skills: {skills_weight_norm*100:.1f}%</strong></div>
                <div style="padding:1rem;background:rgba(102,126,234,0.1);border-radius:8px;"><strong>📋 Responsibilities: {resp_weight_norm*100:.1f}%</strong></div>
                <div style="padding:1rem;background:rgba(102,126,234,0.1);border-radius:8px;"><strong>💼 Job Title: {title_weight_norm*100:.1f}%</strong></div>
                <div style="padding:1rem;background:rgba(102,126,234,0.1);border-radius:8px;"><strong>⏳ Experience: {exp_weight_norm*100:.1f}%</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        sorted_cv_scores = sorted(cv_overall_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Candidates", len(cv_data_list))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            best_score = sorted_cv_scores[0][1]['overall_score'] if sorted_cv_scores else 0
            st.metric("Best Match Score", f"{best_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_score = np.mean([s[1]['overall_score'] for s in sorted_cv_scores]) if sorted_cv_scores else 0.0
            st.metric("Average Score", f"{avg_score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            scores = [s[1]['overall_score'] for s in sorted_cv_scores] if sorted_cv_scores else [0.0]
            st.metric("Score Std Dev", f"{np.std(scores):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🏅 Candidate Rankings</h3>', unsafe_allow_html=True)
        cols = st.columns(min(3, max(1, len(sorted_cv_scores))))
        for i, (cv_name, sd) in enumerate(sorted_cv_scores):
            col_idx = i % len(cols)
            with cols[col_idx]:
                overall = sd['overall_score']
                ssc, rsc, tsc, ysc = sd['skills_score'], sd['resp_score'], sd['job_title_score'], sd['years_score']
                rank = i + 1
                if overall >= 0.7: card_class = "score-card high-score"
                elif overall >= 0.5: card_class = "score-card medium-score"
                else: card_class = "score-card low-score"
                rank_class = f"rank-{rank}" if rank <= 3 else "rank-badge"
                st.markdown(f"""
                <div class="{card_class}" style="animation-delay:{i*0.1}s;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <span class="rank-badge {rank_class}">#{rank}</span>
                        <h3 style="margin:0;font-weight:600;">{cv_name}</h3>
                    </div>
                    <div class="score-value">{overall:.3f}</div>
                    <div style="font-size:1.1rem;color:#666;margin-bottom:1rem;">Overall Match Score</div>
                    <hr style="margin:1rem 0;border:none;height:1px;background:rgba(0,0,0,0.1);">
                    <div style="font-size:0.95rem;line-height:1.6;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;"><span>🎯 Skills ({skills_weight_norm*100:.1f}%)</span><strong>{ssc:.3f}</strong></div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;"><span>📋 Responsibilities ({resp_weight_norm*100:.1f}%)</span><strong>{rsc:.3f}</strong></div>
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;"><span>💼 Job Title ({title_weight_norm*100:.1f}%)</span><strong>{tsc:.3f}</strong></div>
                        <div style="display:flex;justify-content:space-between;"><span>⏳ Experience ({exp_weight_norm*100:.1f}%)</span><strong>{ysc:.3f}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        # Detailed view
        st.markdown('<h2 class="section-header">🔍 Detailed Analysis</h2>', unsafe_allow_html=True)
        selected_candidate = st.selectbox("Select candidate:", options=cv_names_list, index=0)
        sel = next((r for r in all_results if r["cv_name"] == selected_candidate), None)
        if sel:
            cv_name = sel["cv_name"]
            cvd = sel["cv_data"]
            skills_assign = sel["skills_assignments"]
            resp_assign   = sel["resp_assignments"]
            skills_top    = sel["skills_top_sorted_lists"]
            resp_top      = sel["resp_top_sorted_lists"]
            overall       = sel["overall_score"]
            colA, colB = st.columns([1, 2])
            with colA:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.metric("Overall Score", f"{overall:.3f}")
                st.progress(min(max(overall,0.0),1.0))
                if skills_weight_norm > 0: st.metric("Skills Match", f"{sel['skills_score']:.3f}", help=f"Weight: {skills_weight_norm*100:.1f}%")
                if resp_weight_norm   > 0: st.metric("Responsibilities", f"{sel['resp_score']:.3f}", help=f"Weight: {resp_weight_norm*100:.1f}%")
                if title_weight_norm  > 0: st.metric("Job Title", f"{sel['job_title_score']:.3f}", help=f"Weight: {title_weight_norm*100:.1f}%")
                if exp_weight_norm    > 0: st.metric("Experience", f"{sel['years_score']:.3f}", help=f"Weight: {exp_weight_norm*100:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            with colB:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("**Candidate Profile:**")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Current Role:** {cvd['job_title'] or 'Not specified'}")
                    st.markdown(f"**Experience:** {cvd['years']} years")
                    st.markdown(f"**Skills Count:** {len(cvd['skills'])}")
                with c2:
                    st.markdown(f"**Target Role:** {jd_job_title or 'Not specified'}")
                    st.markdown(f"**Required Experience:** {jd_years if jd_years > 0 else 'Not specified'}")
                    st.markdown(f"**Required Skills:** {len(jd_skills_list)}")
                st.markdown('</div>', unsafe_allow_html=True)
            if skills_weight_norm > 0:
                with st.expander("🎯 Skills Analysis & Matching", expanded=True):
                    if skills_assign:
                        st.markdown(f"#### Skills Matching Results (Weight: {skills_weight_norm*100:.1f}%)")
                        for a in skills_assign:
                            if a["score"] >= GOOD_THRESHOLD:
                                card_class, status = "assignment-card assignment-good", "✅ Good Match"
                            else:
                                card_class, status = "assignment-card assignment-rejected", "⚠️ Weak Match"
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <div style="flex:1;">
                                        <strong>JD:</strong> {a["jd_item"]}<br>
                                        <strong>CV:</strong> {a["cv_item"]}
                                    </div>
                                    <div style="text-align:right;">
                                        <div style="font-size:1.2rem;font-weight:bold;">{a["score"]:.3f}</div>
                                        <div style="font-size:0.9rem;">{status}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        if len(skills_assign) > 1:
                            fig_sk = go.Figure(data=[
                                go.Bar(
                                    x=[f"Skill {i+1}" for i in range(len(skills_assign))],
                                    y=[a["score"] for a in skills_assign],
                                    text=[f"{a['score']:.3f}" for a in skills_assign],
                                    textposition='auto',
                                    marker_color=['rgba(46,204,113,0.8)' if a["score"] >= GOOD_THRESHOLD else 'rgba(231,76,60,0.8)' for a in skills_assign],
                                    hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>"
                                )
                            ])
                            fig_sk.update_layout(title="Skills Matching Scores", xaxis_title="Skill Pairs", yaxis_title="Match Score",
                                                template="plotly_white", height=400)
                            st.plotly_chart(fig_sk, use_container_width=True)
                        st.markdown("#### Top-3 Alternatives per JD Skill")
                        jd_to_cv = {a["jd_index"]: a["cv_index"] for a in skills_assign}
                        for jd_i in range(len(jd_skills_list)):
                            if jd_i in skills_top:
                                assigned_idx = jd_to_cv.get(jd_i)
                                alts = []
                                for (cv_idx2, cv_text2, sc2) in skills_top[jd_i]:
                                    if assigned_idx is not None and cv_idx2 == assigned_idx:
                                        continue
                                    alts.append((cv_text2, sc2))
                                    if len(alts) == 3:
                                        break
                                st.markdown(f"**JD Skill {jd_i+1}: {jd_skills_list[jd_i]}**")
                                if alts:
                                    for rank, (t, sc) in enumerate(alts, start=1):
                                        st.markdown(f"""
                                        <div class="alternative-item">
                                            <span class="alternative-rank">Top {rank}</span>
                                            <strong>{t}</strong> | Score: {sc:.3f}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown("- No alternatives found")
                                st.markdown("---")
                    else:
                        st.info("No skills data available for comparison or skills weight is 0%")
            if resp_weight_norm > 0:
                with st.expander("📋 Responsibilities Analysis", expanded=False):
                    if resp_assign:
                        st.markdown(f"#### Responsibilities Matching Results (Weight: {resp_weight_norm*100:.1f}%)")
                        for a in resp_assign:
                            if a["score"] >= GOOD_THRESHOLD:
                                card_class, status = "assignment-card assignment-good", "✅ Good Match"
                            else:
                                card_class, status = "assignment-card assignment-rejected", "⚠️ Weak Match"
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <div style="flex:1;">
                                        <strong>JD:</strong> {a["jd_item"]}<br>
                                        <strong>CV:</strong> {a["cv_item"]}
                                    </div>
                                    <div style="text-align:right;">
                                        <div style="font-size:1.2rem;font-weight:bold;">{a["score"]:.3f}</div>
                                        <div style="font-size:0.9rem;">{status}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No responsibilities data available for comparison or responsibilities weight is 0%")
            with st.expander("📊 Visual Matching Graph", expanded=False):
                if skills_weight_norm > 0 and skills_assign:
                    def create_enhanced_bipartite_graph(assignments_list, jd_list, cv_list, graph_type="Skills"):
                        dot = graphviz.Digraph(comment=f'JD to {cv_name} {graph_type} Matching')
                        dot.attr(rankdir='LR', splines='curved', overlap='false', nodesep='0.8', ranksep='3', bgcolor='transparent')
                        with dot.subgraph(name='cluster_jd') as c:
                            c.attr(label=f'Job Description {graph_type}', style='filled,rounded', color='lightblue', fontsize='16', fontname='Arial Bold')
                            c.attr('node', shape='box', style='rounded,filled', fillcolor='#667eea', fontcolor='white', fontname='Arial', fontsize='12')
                            for i, it in enumerate(jd_list):
                                c.node(f'jd_{i}', f'JD{i+1}\\n{truncate_text(it, 25)}')
                        with dot.subgraph(name='cluster_cv') as c:
                            c.attr(label=f'{cv_name} {graph_type}', style='filled,rounded', color='lightgreen', fontsize='16', fontname='Arial Bold')
                            c.attr('node', shape='box', style='rounded,filled', fillcolor='#2ecc71', fontcolor='white', fontname='Arial', fontsize='12')
                            for i, it in enumerate(cv_list):
                                c.node(f'cv_{i}', f'CV{i+1}\\n{truncate_text(it, 25)}')
                        for a in assignments_list:
                            jd_i, cv_i, score = a['jd_index'], a['cv_index'], a['score']
                            color, style = ("#2ecc71","solid") if score >= GOOD_THRESHOLD else ("#e74c3c","dashed")
                            penwidth = str(1 + 4 * score)
                            dot.edge(f'jd_{jd_i}', f'cv_{cv_i}', label=f'{score:.2f}', fontcolor=color, color=color, penwidth=penwidth, style=style, fontname='Arial Bold', fontsize='10')
                        return dot
                    st.graphviz_chart(create_enhanced_bipartite_graph(skills_assign, jd_skills_list, cvd["skills"]), use_container_width=True)
                else:
                    st.info("No matching data available for visualization")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <h2>🚀 Ready to Find Your Perfect Match?</h2>
            <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
                Go to <strong>Upload & Extract</strong> to extract structured fields with AI, then come back here to run the matching with your custom weights and prompts.
            </p>
            <div style="margin: 2rem 0;">
                <h3>🎯 Components:</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                    <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <strong>🎯 Skills</strong><br>Technical and soft skills matching<br><em>Default: 80%</em>
                    </div>
                    <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <strong>📋 Responsibilities</strong><br>Experience alignment<br><em>Default: 15%</em>
                    </div>
                    <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <strong>💼 Job Title</strong><br>Role alignment<br><em>Default: 2.5%</em>
                    </div>
                    <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <strong>⏳ Experience Years</strong><br>Seniority fit<br><em>Default: 2.5%</em>
                    </div>
                </div>
            </div>
            <p style="font-size: 1rem; color: #888; margin-top: 2rem;">
                💡 Tip: Set a component to 0% to ignore it. We normalize to 100% automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)

