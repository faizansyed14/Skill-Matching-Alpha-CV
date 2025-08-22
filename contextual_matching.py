# enhanced_cv_matcher_custom_weights.py
# ------------------------------------------------------------
# Enhanced JD ↔ CV matching with customizable weights and modern UI
#    • Customizable weights for Skills, Responsibilities, Job Title, Years of Exp
#    • Modern glassmorphism design with animations
#    • Interactive visualizations and dynamic charts
#    • Enhanced user experience with better navigation
#    • Top-3 alternatives per JD skill
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from scipy.optimize import linear_sum_assignment
import streamlit as st
import graphviz
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import re
# ---------------------------
# Config
# ---------------------------
COLLECTION_NAME = "cv_skills"
RESP_COLLECTION_NAME = "cv_responsibilities"
GOOD_THRESHOLD = 0.50  # Threshold for good matches
st.set_page_config(
    page_title="AI-Powered CV Matcher", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)
# Enhanced CSS with modern design principles
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    .css-1d391kg, .css-18e3th9, .css-1y0tads {
        padding: 0;
    }
    
    /* Main container styling */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Score Cards with Animations */
    @keyframes slideInUp {
        from { 
            opacity: 0; 
            transform: translateY(30px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .score-card {
        animation: slideInUp 0.6s ease-out forwards;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .score-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    .high-score {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 6px solid #2ecc71;
    }
    
    .medium-score {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 6px solid #f39c12;
    }
    
    .low-score {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 6px solid #e74c3c;
    }
    
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: pulse 2s infinite;
    }
    
    /* Rank Badges */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        margin-right: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .rank-1 {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    .rank-2 {
        background: linear-gradient(135deg, #C0C0C0 0%, #A0A0A0 100%);
        box-shadow: 0 6px 20px rgba(192, 192, 192, 0.4);
    }
    
    .rank-3 {
        background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%);
        box-shadow: 0 6px 20px rgba(205, 127, 50, 0.4);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9ff 0%, #e8ecff 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Assignment Cards */
    .assignment-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .assignment-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }
    
    .assignment-good {
        border-left-color: #2ecc71;
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.05) 0%, rgba(46, 204, 113, 0.1) 100%);
    }
    
    .assignment-rejected {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.05) 0%, rgba(231, 76, 60, 0.1) 100%);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Weight Sliders */
    .weight-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .score-value {
            font-size: 2.5rem;
        }
        
        .main .block-container {
            padding: 1rem;
        }
    }
    
    /* Top Alternatives Styling */
    .alternatives-container {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .alternative-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.5);
        border-left: 3px solid #667eea;
    }
    
    .alternative-rank {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        margin-right: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)
# ---------------------------
# Helper functions
# ---------------------------
def truncate_text(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len-3] + "..."
def extract_years(text):
    """Extract years of experience from text"""
    if not text:
        return 0
    patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*yrs?',
        r'experience:\s*(\d+)',
        r'exp:\s*(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))
    return 0
def calculate_years_score(jd_years, cv_years):
    """Calculate years of experience matching score"""
    if jd_years == 0:
        return 1.0
    if cv_years >= jd_years:
        return 1.0
    return cv_years / jd_years
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
def normalize_weights(weights):
    """Normalize weights to sum to 1.0"""
    total = sum(weights)
    if total == 0:
        return [0.25, 0.25, 0.25, 0.25]  # Equal weights if all are 0
    return [w / total for w in weights]
# ---------------------------
# Hero Header
# ---------------------------
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🎯 ALPHA CV, AI-CV Matcher</h1>
    <p class="hero-subtitle">Advanced JD-CV matching with customizable weights and semantic analysis</p>
</div>
""", unsafe_allow_html=True)
# ---------------------------
# Initialize session state
# ---------------------------
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = []
    st.session_state.cv_names = []
# ---------------------------
# Modern Sidebar with Enhanced UI
# ---------------------------
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## ⚖️ Matching Weights Configuration")
    
    # Weight sliders with validation
    st.markdown("### 📊 Component Weights")
    st.markdown("*Adjust the importance of each matching component (must sum to 100%)*")
    
    # Initialize default weights
    if 'weights' not in st.session_state:
        st.session_state.weights = {
            'skills': 80,
            'responsibilities': 15,
            'job_title': 2.5,
            'experience': 2.5
        }
    
    st.markdown('<div class="weight-container">', unsafe_allow_html=True)
    
    # Skills weight
    skills_weight = st.slider(
        "🎯 Skills Weight (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.weights['skills']),
        step=5,
        help="Importance of technical and soft skills matching"
    )
    
    # Responsibilities weight
    resp_weight = st.slider(
        "📋 Responsibilities Weight (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state.weights['responsibilities']),
        step=5,
        help="Importance of work experience and achievements"
    )
    
    # Job title weight
    title_weight = st.slider(
        "💼 Job Title Weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.weights['job_title']),
        step=2.5,
        help="Importance of role alignment and career progression"
    )
    
    # Experience years weight
    exp_weight = st.slider(
        "⏳ Experience Years Weight (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.weights['experience']),
        step=2.5,
        help="Importance of years of experience requirement"
    )
    
    # Calculate total and show validation
    total_weight = skills_weight + resp_weight + title_weight + exp_weight
    
    if total_weight == 100:
        st.success(f"✅ Total: {total_weight}% (Perfect!)")
    elif total_weight == 0:
        st.error("❌ All weights are 0%. Please set at least one weight.")
    else:
        st.warning(f"⚠️ Total: {total_weight}% (Will be normalized to 100%)")
    
    # Update session state
    st.session_state.weights = {
        'skills': skills_weight,
        'responsibilities': resp_weight,
        'job_title': title_weight,
        'experience': exp_weight
    }
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset to default button
    if st.button("🔄 Reset to Default Weights", type="secondary"):
        st.session_state.weights = {
            'skills': 80,
            'responsibilities': 15,
            'job_title': 2.5,
            'experience': 2.5
        }
        st.rerun()  # Updated for new Streamlit versions
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # JD Input Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 📋 Job Description")
    
    jd_skills = st.text_area(
        "Required Skills",
        height=120,
        placeholder="• Python Programming\n• Machine Learning\n• Data Analysis\n• Team Leadership",
        help="Enter each skill on a new line"
    )
    jd_responsibilities = st.text_area(
        "Key Responsibilities",
        height=120,
        placeholder="• Lead data science projects\n• Develop ML models\n• Mentor junior developers",
        help="Enter each responsibility on a new line"
    )
    col1, col2 = st.columns(2)
    with col1:
        jd_job_title = st.text_input("Job Title", placeholder="Senior Data Scientist")
    with col2:
        jd_years = st.number_input("Years Required", min_value=0, value=0, step=1)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # CV Input Section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("## 👥 Candidate CVs")
    
    def add_cv():
        st.session_state.cv_data.append({
            "skills": "",
            "responsibilities": "",
            "job_title": "",
            "years": 0
        })
        st.session_state.cv_names.append(f"Candidate {len(st.session_state.cv_data)}")
    
    def remove_cv(index):
        if len(st.session_state.cv_data) > 1:
            st.session_state.cv_data.pop(index)
            st.session_state.cv_names.pop(index)
    
    if not st.session_state.cv_data:
        add_cv()
    
    # Display CV inputs in expandable sections
    for i, (cv_data, cv_name) in enumerate(zip(st.session_state.cv_data, st.session_state.cv_names)):
        with st.expander(f"👤 {cv_name}", expanded=i < 2):
            new_name = st.text_input("Candidate Name", value=cv_name, key=f"cv_name_{i}")
            st.session_state.cv_names[i] = new_name
            
            cv_data["skills"] = st.text_area(
                "Skills",
                value=cv_data["skills"],
                height=100,
                key=f"cv_skills_{i}",
                placeholder="• Python\n• SQL\n• Machine Learning"
            )
            cv_data["responsibilities"] = st.text_area(
                "Experience",
                value=cv_data["responsibilities"],
                height=100,
                key=f"cv_resp_{i}",
                placeholder="• Built ML models\n• Analyzed datasets"
            )
            col1, col2 = st.columns(2)
            with col1:
                cv_data["job_title"] = st.text_input(
                    "Current Role",
                    value=cv_data["job_title"],
                    key=f"cv_title_{i}"
                )
            with col2:
                cv_data["years"] = st.number_input(
                    "Experience (years)",
                    min_value=0,
                    value=cv_data["years"],
                    step=1,
                    key=f"cv_years_{i}"
                )
            
            if len(st.session_state.cv_data) > 1:
                st.button("🗑️ Remove", key=f"remove_cv_{i}", on_click=remove_cv, args=(i,), type="secondary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("➕ Add CV", on_click=add_cv, type="secondary", use_container_width=True)
    with col2:
        analyze_button = st.button("🚀 Analyze Matches", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
# ---------------------------
# Main Analysis Section
# ---------------------------
if analyze_button or st.session_state.get('run_analysis', False):
    st.session_state.run_analysis = True
    
    # Validate total weight
    total_weight = sum(st.session_state.weights.values())
    if total_weight == 0:
        st.error("⚠️ Please set at least one weight greater than 0%")
        st.stop()
    
    # Normalize weights
    normalized_weights = normalize_weights(list(st.session_state.weights.values()))
    skills_weight_norm, resp_weight_norm, title_weight_norm, exp_weight_norm = normalized_weights
    
    # Display normalized weights
    if total_weight != 100:
        st.info(f"🔄 Weights normalized: Skills {skills_weight_norm*100:.1f}%, Responsibilities {resp_weight_norm*100:.1f}%, Job Title {title_weight_norm*100:.1f}%, Experience {exp_weight_norm*100:.1f}%")
    
    # Process inputs
    jd_skills_list = [s.strip() for s in jd_skills.split("\n") if s.strip()] if jd_skills else []
    jd_resp_list = [s.strip() for s in jd_responsibilities.split("\n") if s.strip()] if jd_responsibilities else []
    
    # Validation based on weights
    validation_errors = []
    if skills_weight_norm > 0 and not jd_skills_list:
        validation_errors.append("Skills are weighted but no JD skills provided")
    if resp_weight_norm > 0 and not jd_resp_list:
        validation_errors.append("Responsibilities are weighted but no JD responsibilities provided")
    if title_weight_norm > 0 and not jd_job_title:
        validation_errors.append("Job title is weighted but no JD job title provided")
    if exp_weight_norm > 0 and jd_years == 0:
        validation_errors.append("Experience is weighted but no JD years requirement provided")
    
    if validation_errors:
        st.error("⚠️ Configuration issues:")
        for error in validation_errors:
            st.error(f"• {error}")
        st.stop()
    
    cv_data_list = []
    cv_names_list = []
    
    for i, cv_data in enumerate(st.session_state.cv_data):
        cv_name = st.session_state.cv_names[i]
        cv_skills = [s.strip() for s in cv_data["skills"].split("\n") if s.strip()] if cv_data["skills"] else []
        cv_resp = [s.strip() for s in cv_data["responsibilities"].split("\n") if s.strip()] if cv_data["responsibilities"] else []
        cv_job_title = cv_data["job_title"]
        cv_years = cv_data["years"]
        
        # Check if CV has required data based on weights
        cv_valid = True
        if skills_weight_norm > 0 and not cv_skills:
            cv_valid = False
        if resp_weight_norm > 0 and not cv_resp:
            cv_valid = False
        if title_weight_norm > 0 and not cv_job_title:
            cv_valid = False
        # Note: CV years can be 0 (entry level), so we don't validate this
        
        if cv_valid or cv_skills or cv_resp or cv_job_title or cv_years:
            cv_data_list.append({
                "skills": cv_skills,
                "responsibilities": cv_resp,
                "job_title": cv_job_title,
                "years": cv_years
            })
            cv_names_list.append(cv_name)
    
    if not cv_data_list:
        st.error("⚠️ Please provide at least one candidate CV with relevant data")
        st.stop()
    
    # Processing with enhanced loading experience
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner(""):
        status_text.markdown("🔄 **Loading AI model...**")
        progress_bar.progress(10)
        model = load_model()
        
        status_text.markdown("🧠 **Processing embeddings...**")
        progress_bar.progress(30)
        
        # Process embeddings only for components with non-zero weights
        jd_skills_embeddings = np.array([])
        jd_resp_embeddings = np.array([])
        
        if skills_weight_norm > 0 and jd_skills_list:
            jd_skills_embeddings = model.encode(jd_skills_list, normalize_embeddings=True)
        
        if resp_weight_norm > 0 and jd_resp_list:
            jd_resp_embeddings = model.encode(jd_resp_list, normalize_embeddings=True)
        
        progress_bar.progress(50)
        
        # Setup CV data
        all_cv_skills = []
        cv_skill_indices = []
        all_cv_resp = []
        cv_resp_indices = []
        cv_indices = []
        
        for cv_idx, cv_data in enumerate(cv_data_list):
            if skills_weight_norm > 0:
                for skill_idx, skill in enumerate(cv_data["skills"]):
                    all_cv_skills.append(skill)
                    cv_skill_indices.append(skill_idx)
                    cv_indices.append(cv_idx)
            
            if resp_weight_norm > 0:
                for resp_idx, resp in enumerate(cv_data["responsibilities"]):
                    all_cv_resp.append(resp)
                    cv_resp_indices.append(resp_idx)
                    cv_indices.append(cv_idx)
        
        progress_bar.progress(70)
        
        # Process CV embeddings
        cv_skills_embeddings = np.array([])
        cv_resp_embeddings = np.array([])
        
        if skills_weight_norm > 0 and all_cv_skills:
            cv_skills_embeddings = model.encode(all_cv_skills, normalize_embeddings=True)
        
        if resp_weight_norm > 0 and all_cv_resp:
            cv_resp_embeddings = model.encode(all_cv_resp, normalize_embeddings=True)
        
        # Setup Qdrant for skills
        qdrant_skills = None
        if skills_weight_norm > 0 and len(cv_skills_embeddings) > 0:
            qdrant_skills = QdrantClient(":memory:")
            if qdrant_skills.collection_exists(COLLECTION_NAME):
                qdrant_skills.delete_collection(COLLECTION_NAME)
            
            qdrant_skills.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=cv_skills_embeddings.shape[1], distance=models.Distance.COSINE),
            )
            
            points_data = []
            cv_idx_counter = 0
            for i in range(len(all_cv_skills)):
                points_data.append(models.PointStruct(
                    id=i,
                    vector=cv_skills_embeddings[i].tolist(),
                    payload={
                        "skill": all_cv_skills[i],
                        "cv_index": cv_indices[i],
                        "cv_skill_index": cv_skill_indices[i],
                        "cv_name": cv_names_list[cv_indices[i]]
                    }
                ))
            
            qdrant_skills.upsert(collection_name=COLLECTION_NAME, points=points_data)
        
        # Setup Qdrant for responsibilities
        qdrant_resp = None
        if resp_weight_norm > 0 and len(cv_resp_embeddings) > 0:
            qdrant_resp = QdrantClient(":memory:")
            if qdrant_resp.collection_exists(RESP_COLLECTION_NAME):
                qdrant_resp.delete_collection(RESP_COLLECTION_NAME)
            
            qdrant_resp.create_collection(
                collection_name=RESP_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=cv_resp_embeddings.shape[1], distance=models.Distance.COSINE),
            )
            
            points_data = []
            for i in range(len(all_cv_resp)):
                points_data.append(models.PointStruct(
                    id=i,
                    vector=cv_resp_embeddings[i].tolist(),
                    payload={
                        "responsibility": all_cv_resp[i],
                        "cv_index": cv_indices[i],
                        "cv_resp_index": cv_resp_indices[i],
                        "cv_name": cv_names_list[cv_indices[i]]
                    }
                ))
            
            qdrant_resp.upsert(collection_name=RESP_COLLECTION_NAME, points=points_data)
        
        progress_bar.progress(90)
        status_text.markdown("⚡ **Calculating match scores...**")
    
    # Process results with custom weights
    all_results = []
    cv_overall_scores = {}
    
    for cv_idx, (cv_name, cv_data) in enumerate(zip(cv_names_list, cv_data_list)):
        cv_skills = cv_data["skills"]
        cv_resp = cv_data["responsibilities"]
        cv_job_title = cv_data["job_title"]
        cv_years = cv_data["years"]
        
        # Initialize scores
        skills_score = 0.0
        resp_score = 0.0
        job_title_score = 0.0
        years_score = 0.0
        
        assignments = []
        resp_assignments = []
        top_sorted_lists = {}
        resp_top_sorted_lists = {}
        
        # Skills matching (only if weight > 0)
        if skills_weight_norm > 0 and jd_skills_list and cv_skills and qdrant_skills:
            M, N = len(jd_skills_list), len(cv_skills)
            similarity_matrix = np.zeros((M, N), dtype=np.float32)
            
            for j, jd_vec in enumerate(jd_skills_embeddings):
                res = qdrant_skills.query_points(
                    collection_name=COLLECTION_NAME,
                    query=jd_vec.tolist(),
                    limit=len(all_cv_skills),
                    with_payload=True,
                )
                cv_res = [p for p in res.points if p.payload["cv_index"] == cv_idx]
                sorted_rows = []
                for p in cv_res:
                    skill_idx = p.payload["cv_skill_index"]
                    score = float(p.score)
                    similarity_matrix[j, skill_idx] = score
                    sorted_rows.append((skill_idx, cv_skills[skill_idx], score))
                sorted_rows.sort(key=lambda x: x[2], reverse=True)
                top_sorted_lists[j] = sorted_rows
            
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
            for r, c in zip(row_ind, col_ind):
                assignments.append({
                    "type": "skill",
                    "jd_index": r,
                    "jd_item": jd_skills_list[r],
                    "cv_index": c,
                    "cv_item": cv_skills[c],
                    "score": float(similarity_matrix[r, c]),
                })
            skills_score = float(np.mean([a["score"] for a in assignments])) if assignments else 0.0
        
        # Responsibilities matching (only if weight > 0)
        if resp_weight_norm > 0 and jd_resp_list and cv_resp and qdrant_resp:
            M, N = len(jd_resp_list), len(cv_resp)
            resp_similarity_matrix = np.zeros((M, N), dtype=np.float32)
            
            for j, jd_vec in enumerate(jd_resp_embeddings):
                res = qdrant_resp.query_points(
                    collection_name=RESP_COLLECTION_NAME,
                    query=jd_vec.tolist(),
                    limit=len(all_cv_resp),
                    with_payload=True,
                )
                cv_res = [p for p in res.points if p.payload["cv_index"] == cv_idx]
                sorted_rows = []
                for p in cv_res:
                    resp_idx = p.payload["cv_resp_index"]
                    score = float(p.score)
                    resp_similarity_matrix[j, resp_idx] = score
                    sorted_rows.append((resp_idx, cv_resp[resp_idx], score))
                sorted_rows.sort(key=lambda x: x[2], reverse=True)
                resp_top_sorted_lists[j] = sorted_rows
            
            row_ind, col_ind = linear_sum_assignment(-resp_similarity_matrix)
            for r, c in zip(row_ind, col_ind):
                resp_assignments.append({
                    "type": "responsibility",
                    "jd_index": r,
                    "jd_item": jd_resp_list[r],
                    "cv_index": c,
                    "cv_item": cv_resp[c],
                    "score": float(resp_similarity_matrix[r, c]),
                })
            resp_score = float(np.mean([a["score"] for a in resp_assignments])) if resp_assignments else 0.0
        
        # Job title matching (only if weight > 0)
        if title_weight_norm > 0 and jd_job_title and cv_job_title:
            jd_title_embedding = model.encode([jd_job_title], normalize_embeddings=True)
            cv_title_embedding = model.encode([cv_job_title], normalize_embeddings=True)
            job_title_score = float(np.dot(jd_title_embedding, cv_title_embedding.T)[0][0])
        
        # Years matching (only if weight > 0)
        if exp_weight_norm > 0:
            years_score = calculate_years_score(jd_years, cv_years)
        
        # Calculate overall weighted score using normalized weights
        overall_score = (
            skills_weight_norm * skills_score +
            resp_weight_norm * resp_score +
            title_weight_norm * job_title_score +
            exp_weight_norm * years_score
        )
        
        cv_overall_scores[cv_name] = {
            "overall_score": overall_score,
            "skills_score": skills_score,
            "resp_score": resp_score,
            "job_title_score": job_title_score,
            "years_score": years_score,
            "jd_job_title": jd_job_title,
            "cv_job_title": cv_job_title,
            "jd_years": jd_years,
            "cv_years": cv_years
        }
        
        all_results.append({
            "cv_name": cv_name,
            "cv_idx": cv_idx,
            "cv_data": cv_data,
            "skills_assignments": assignments,
            "resp_assignments": resp_assignments,
            "skills_top_sorted_lists": top_sorted_lists,
            "resp_top_sorted_lists": resp_top_sorted_lists,
            "overall_score": overall_score,
            "skills_score": skills_score,
            "resp_score": resp_score,
            "job_title_score": job_title_score,
            "years_score": years_score
        })
    
    progress_bar.progress(100)
    status_text.markdown("✅ **Analysis complete!**")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # ---------------------------
    # Enhanced Results Display
    # ---------------------------
    
    # Executive Summary with Interactive Charts
    st.markdown('<h2 class="section-header">🏆 Executive Summary</h2>', unsafe_allow_html=True)
    
    # Show current weights
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <h4>📊 Current Matching Weights</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                <strong>🎯 Skills: {skills_weight_norm*100:.1f}%</strong>
            </div>
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                <strong>📋 Responsibilities: {resp_weight_norm*100:.1f}%</strong>
            </div>
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                <strong>💼 Job Title: {title_weight_norm*100:.1f}%</strong>
            </div>
            <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                <strong>⏳ Experience: {exp_weight_norm*100:.1f}%</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create summary metrics
    sorted_cv_scores = sorted(cv_overall_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Candidates", len(cv_data_list))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        best_match_score = sorted_cv_scores[0][1]['overall_score'] if sorted_cv_scores else 0
        st.metric("Best Match Score", f"{best_match_score:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_score = np.mean([score[1]['overall_score'] for score in sorted_cv_scores])
        st.metric("Average Score", f"{avg_score:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        scores = [score[1]['overall_score'] for score in sorted_cv_scores]
        std_dev = np.std(scores)
        st.metric("Score Std Dev", f"{std_dev:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Candidate Comparison Cards
    st.markdown('<h3 class="section-header">🏅 Candidate Rankings</h3>', unsafe_allow_html=True)
    
    # Create animated score cards
    cols = st.columns(min(3, len(sorted_cv_scores)))  # Max 3 columns
    
    for i, (cv_name, score_data) in enumerate(sorted_cv_scores):
        col_idx = i % len(cols)
        with cols[col_idx]:
            overall_score = score_data['overall_score']
            skills_score = score_data['skills_score']
            resp_score = score_data['resp_score']
            job_title_score = score_data['job_title_score']
            years_score = score_data['years_score']
            rank = i + 1
            
            # Determine card class based on score
            if overall_score >= 0.7:
                card_class = "score-card high-score"
            elif overall_score >= 0.5:
                card_class = "score-card medium-score"
            else:
                card_class = "score-card low-score"
            
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-badge"
            
            st.markdown(f"""
            <div class="{card_class}" style="animation-delay: {i * 0.1}s;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span class="rank-badge {rank_class}">#{rank}</span>
                    <h3 style="margin: 0; font-weight: 600;">{cv_name}</h3>
                </div>
                <div class="score-value">{overall_score:.3f}</div>
                <div style="font-size: 1.1rem; color: #666; margin-bottom: 1rem;">Overall Match Score</div>
                <hr style="margin: 1rem 0; border: none; height: 1px; background: rgba(0,0,0,0.1);">
                <div style="font-size: 0.95rem; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>🎯 Skills ({skills_weight_norm*100:.1f}%)</span>
                        <strong>{skills_score:.3f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>📋 Responsibilities ({resp_weight_norm*100:.1f}%)</span>
                        <strong>{resp_score:.3f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span>💼 Job Title ({title_weight_norm*100:.1f}%)</span>
                        <strong>{job_title_score:.3f}</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>⏳ Experience ({exp_weight_norm*100:.1f}%)</span>
                        <strong>{years_score:.3f}</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed Analysis Section
    st.markdown('<h2 class="section-header">🔍 Detailed Analysis</h2>', unsafe_allow_html=True)
    
    # Candidate selector
    selected_candidate = st.selectbox(
        "Select candidate for detailed analysis:",
        options=cv_names_list,
        index=0,
        help="Choose a candidate to view detailed matching analysis"
    )
    
    # Find selected candidate's results
    selected_result = next((r for r in all_results if r["cv_name"] == selected_candidate), None)
    
    if selected_result:
        cv_name = selected_result["cv_name"]
        cv_data = selected_result["cv_data"]
        skills_assignments = selected_result["skills_assignments"]
        resp_assignments = selected_result["resp_assignments"]
        skills_top_sorted_lists = selected_result["skills_top_sorted_lists"]
        resp_top_sorted_lists = selected_result["resp_top_sorted_lists"]
        overall_score = selected_result["overall_score"]
        
        # Candidate overview
        st.markdown(f"### 👤 {cv_name} - Detailed Analysis")
        
        # Create two columns for overview
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("Overall Score", f"{overall_score:.3f}")
            st.progress(overall_score)
            
            # Component scores with weights
            st.markdown("**Weighted Component Breakdown:**")
            if skills_weight_norm > 0:
                st.metric("Skills Match", f"{selected_result['skills_score']:.3f}", help=f"Weight: {skills_weight_norm*100:.1f}%")
            if resp_weight_norm > 0:
                st.metric("Responsibilities", f"{selected_result['resp_score']:.3f}", help=f"Weight: {resp_weight_norm*100:.1f}%")
            if title_weight_norm > 0:
                st.metric("Job Title", f"{selected_result['job_title_score']:.3f}", help=f"Weight: {title_weight_norm*100:.1f}%")
            if exp_weight_norm > 0:
                st.metric("Experience", f"{selected_result['years_score']:.3f}", help=f"Weight: {exp_weight_norm*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("**Candidate Profile:**")
            
            profile_col1, profile_col2 = st.columns(2)
            with profile_col1:
                st.markdown(f"**Current Role:** {cv_data['job_title'] or 'Not specified'}")
                st.markdown(f"**Experience:** {cv_data['years']} years")
                st.markdown(f"**Skills Count:** {len(cv_data['skills'])}")
            
            with profile_col2:
                st.markdown(f"**Target Role:** {jd_job_title or 'Not specified'}")
                st.markdown(f"**Required Experience:** {jd_years} years")
                st.markdown(f"**Required Skills:** {len(jd_skills_list)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Skills Analysis (only show if weight > 0)
        if skills_weight_norm > 0:
            with st.expander("🎯 Skills Analysis & Matching", expanded=True):
                if skills_assignments:
                    # Create skills matching visualization
                    st.markdown(f"#### Skills Matching Results (Weight: {skills_weight_norm*100:.1f}%)")
                    
                    # Color-coded display
                    for a in skills_assignments:
                        if a["score"] >= GOOD_THRESHOLD:  # Use the GOOD_THRESHOLD constant
                            card_class = "assignment-card assignment-good"
                            status = "✅ Good Match"
                        else:
                            card_class = "assignment-card assignment-rejected"
                            status = "⚠️ Weak Match"
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <strong>JD:</strong> {a["jd_item"]}<br>
                                    <strong>CV:</strong> {a["cv_item"]}
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.2rem; font-weight: bold;">{a["score"]:.3f}</div>
                                    <div style="font-size: 0.9rem;">{status}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Skills matching chart
                    if len(skills_assignments) > 1:
                        fig_skills = go.Figure(data=[
                            go.Bar(
                                x=[f"Skill {i+1}" for i in range(len(skills_assignments))],
                                y=[a["score"] for a in skills_assignments],
                                text=[f"{a['score']:.3f}" for a in skills_assignments],
                                textposition='auto',
                                marker_color=[
                                    'rgba(46, 204, 113, 0.8)' if a["score"] >= GOOD_THRESHOLD 
                                    else 'rgba(231, 76, 60, 0.8)' 
                                    for a in skills_assignments
                                ],
                                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>"
                            )
                        ])
                        
                        fig_skills.update_layout(
                            title="Skills Matching Scores",
                            xaxis_title="Skill Pairs",
                            yaxis_title="Match Score",
                            template="plotly_white",
                            height=400
                        )
                        
                        st.plotly_chart(fig_skills, use_container_width=True)
                    
                    # All Assignments Table
                    st.markdown("#### All Assignments Table")
                    df_skills = pd.DataFrame(skills_assignments)
                    if not df_skills.empty:
                        df_skills["JD #"] = df_skills["jd_index"] + 1
                        df_skills["CV #"] = df_skills["cv_index"] + 1
                        df_skills["Status"] = df_skills["score"].apply(lambda s: "GOOD" if s >= GOOD_THRESHOLD else "REJECTED")
                        df_skills = df_skills[["JD #", "jd_item", "CV #", "cv_item", "score", "Status"]].rename(
                            columns={"jd_item": "JD Skill", "cv_item": "CV Skill", "score": "Score"}
                        )
                        st.dataframe(df_skills.style.format({"Score": "{:.3f}"}), use_container_width=True)
                    
                    # Top-3 Alternatives per JD Skill
                    st.markdown("#### Top-3 Alternatives per JD Skill")
                    jd_to_assigned_cv = {}
                    for a in skills_assignments:
                        jd_to_assigned_cv[a["jd_index"]] = a["cv_index"]
                    
                    for jd_idx in range(len(jd_skills_list)):
                        if jd_idx in skills_top_sorted_lists:
                            assigned_cv_skill_index = jd_to_assigned_cv.get(jd_idx)
                            alternatives = []
                            for (cv_skill_index, cv_skill_text, score) in skills_top_sorted_lists[jd_idx]:
                                if assigned_cv_skill_index is not None and cv_skill_index == assigned_cv_skill_index:
                                    continue
                                alternatives.append((cv_skill_text, score))
                                if len(alternatives) == 3:
                                    break
                            
                            st.markdown(f"**JD Skill {jd_idx+1}: {jd_skills_list[jd_idx]}**")
                            if alternatives:
                                for rank, (alt_text, alt_score) in enumerate(alternatives, start=1):
                                    st.markdown(f"""
                                    <div class="alternative-item">
                                        <span class="alternative-rank">Top {rank}</span>
                                        <strong>{alt_text}</strong> | Score: {alt_score:.3f}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown("- No alternatives found")
                            st.markdown("---")
                        
                else:
                    st.info("No skills data available for comparison or skills weight is 0%")
        
        # Responsibilities Analysis (only show if weight > 0)
        if resp_weight_norm > 0:
            with st.expander("📋 Responsibilities Analysis", expanded=False):
                if resp_assignments:
                    st.markdown(f"#### Responsibilities Matching Results (Weight: {resp_weight_norm*100:.1f}%)")
                    
                    for a in resp_assignments:
                        if a["score"] >= GOOD_THRESHOLD:
                            card_class = "assignment-card assignment-good"
                            status = "✅ Good Match"
                        else:
                            card_class = "assignment-card assignment-rejected"
                            status = "⚠️ Weak Match"
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <strong>JD:</strong> {a["jd_item"]}<br>
                                    <strong>CV:</strong> {a["cv_item"]}
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 1.2rem; font-weight: bold;">{a["score"]:.3f}</div>
                                    <div style="font-size: 0.9rem;">{status}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                else:
                    st.info("No responsibilities data available for comparison or responsibilities weight is 0%")
        
        # Interactive Bipartite Graph (only for components with weight > 0)
        with st.expander("📊 Visual Matching Graph", expanded=False):
            if skills_weight_norm > 0 and skills_assignments:
                def create_enhanced_bipartite_graph(assignments_list, jd_list, cv_list, graph_type="Skills"):
                    dot = graphviz.Digraph(comment=f'JD to {cv_name} {graph_type} Matching')
                    dot.attr(rankdir='LR', splines='curved', overlap='false', 
                            nodesep='0.8', ranksep='3', bgcolor='transparent')
                    
                    # JD nodes
                    with dot.subgraph(name='cluster_jd') as c:
                        c.attr(label=f'Job Description {graph_type}', 
                              style='filled,rounded', color='lightblue', 
                              fontsize='16', fontname='Arial Bold')
                        c.attr('node', shape='box', style='rounded,filled', 
                              fillcolor='#667eea', fontcolor='white', 
                              fontname='Arial', fontsize='12')
                        for i, item in enumerate(jd_list):
                            c.node(f'jd_{i}', f'JD{i+1}\\n{truncate_text(item, 25)}')
                    
                    # CV nodes
                    with dot.subgraph(name='cluster_cv') as c:
                        c.attr(label=f'{cv_name} {graph_type}', 
                              style='filled,rounded', color='lightgreen', 
                              fontsize='16', fontname='Arial Bold')
                        c.attr('node', shape='box', style='rounded,filled', 
                              fillcolor='#2ecc71', fontcolor='white', 
                              fontname='Arial', fontsize='12')
                        for i, item in enumerate(cv_list):
                            c.node(f'cv_{i}', f'CV{i+1}\\n{truncate_text(item, 25)}')
                    
                    # Edges
                    for a in assignments_list:
                        jd_idx, cv_idx, score = a['jd_index'], a['cv_index'], a['score']
                        if score >= GOOD_THRESHOLD:
                            color = "#2ecc71"
                            style = "solid"
                        else:
                            color = "#e74c3c"
                            style = "dashed"
                        
                        penwidth = str(1 + 4 * score)
                        dot.edge(f'jd_{jd_idx}', f'cv_{cv_idx}', 
                                label=f'{score:.2f}', fontcolor=color, 
                                color=color, penwidth=penwidth, style=style,
                                fontname='Arial Bold', fontsize='10')
                    
                    return dot
                
                st.graphviz_chart(
                    create_enhanced_bipartite_graph(skills_assignments, jd_skills_list, cv_data["skills"]), 
                    use_container_width=True
                )
            else:
                st.info("No matching data available for visualization")
    
else:
    # Welcome screen when no analysis is running
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h2>🚀 Ready to Find Your Perfect Match?</h2>
        <p style="font-size: 1.2rem; color: #666; margin: 2rem 0;">
            Use our advanced AI-powered matching system with customizable weights to find the best candidates for your job openings.
            Adjust the component weights in the sidebar and input your job description and candidate CVs to get started.
        </p>
        <div style="margin: 2rem 0;">
            <h3>🎯 Customizable Analysis Components:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                    <strong>🎯 Skills</strong><br>
                    Technical and soft skills matching<br>
                    <em>Default: 80%</em>
                </div>
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                    <strong>📋 Responsibilities</strong><br>
                    Work experience and achievements<br>
                    <em>Default: 15%</em>
                </div>
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                    <strong>💼 Job Title</strong><br>
                    Role alignment and career progression<br>
                    <em>Default: 2.5%</em>
                </div>
                <div style="padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                    <strong>⏳ Experience Years</strong><br>
                    Years of experience requirements<br>
                    <em>Default: 2.5%</em>
                </div>
            </div>
        </div>
        <div style="margin: 2rem 0;">
            <h3>✨ Key Features:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 8px;">
                    <strong>⚖️ Custom Weights</strong><br>
                    Adjust importance of each component with 5% increments
                </div>
                <div style="padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 8px;">
                    <strong>🧠 AI-Powered</strong><br>
                    Advanced semantic similarity using transformer models
                </div>
                <div style="padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 8px;">
                    <strong>📊 Interactive Charts</strong><br>
                    Visual analysis with dynamic charts and graphs
                </div>
                <div style="padding: 1rem; background: rgba(118, 75, 162, 0.1); border-radius: 8px;">
                    <strong>🎯 Flexible Analysis</strong><br>
                    Focus on skills only, responsibilities only, or any combination
                </div>
            </div>
        </div>
        <p style="font-size: 1rem; color: #888; margin-top: 2rem;">
            💡 <strong>Pro Tip:</strong> Set weights to 0% for components you want to ignore completely. 
            The system will automatically normalize your weights to 100%.
        </p>
    </div>
    """, unsafe_allow_html=True)



# # contextual_matching_hungarian_ui.py
# # ------------------------------------------------------------
# # JD ↔ CV skill matching with:
# #    • User input boxes for JD and multiple CV skill lists
# #    • Qdrant vector search (cosine similarity)
# #    • Hungarian algorithm (optimal assignment, one-to-one)
# #    • Top-3 alternatives per JD (excluding its own assigned CV)
# #    • Tables:
# #        1) All Assignments (Accepted + Rejected)
# #        2) Ignored Top Match
# #        3) CV Comparison Summary
# #    • Visualizations:
# #        1) Bipartite graph of assignments (Graphviz) - Collapsible
# #        2) Histogram of assignment scores (Plotly)
# #        3) Animated score cards for CV comparison
# #    • Color coding: >= 0.50 GOOD (green), < 0.50 REJECTED (red)
# # ------------------------------------------------------------
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from scipy.optimize import linear_sum_assignment
# import streamlit as st
# import graphviz
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import time

# # ---------------------------
# # Config
# # ---------------------------
# GOOD_THRESHOLD = 0.50
# COLLECTION_NAME = "cv_skills"
# st.set_page_config(layout="wide")

# # Custom CSS for animations and styling
# st.markdown("""
# <style>
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(10px); }
#         to { opacity: 1; transform: translateY(0); }
#     }
    
#     .score-card {
#         animation: fadeIn 0.8s ease-out forwards;
#         border-radius: 10px;
#         padding: 15px;
#         margin-bottom: 15px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#     }
    
#     .score-card:hover {
#         transform: translateY(-5px);
#         box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
#     }
    
#     .high-score {
#         background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
#         border-left: 5px solid #2ecc71;
#     }
    
#     .medium-score {
#         background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
#         border-left: 5px solid #f39c12;
#     }
    
#     .low-score {
#         background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
#         border-left: 5px solid #e74c3c;
#     }
    
#     .score-value {
#         font-size: 2.5rem;
#         font-weight: bold;
#         margin: 10px 0;
#     }
    
#     .rank-badge {
#         display: inline-block;
#         padding: 5px 10px;
#         border-radius: 20px;
#         color: white;
#         font-weight: bold;
#         margin-right: 10px;
#     }
    
#     .rank-1 {
#         background-color: #FFD700;
#         color: #333;
#     }
    
#     .rank-2 {
#         background-color: #C0C0C0;
#         color: #333;
#     }
    
#     .rank-3 {
#         background-color: #CD7F32;
#         color: #333;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper function
# # ---------------------------
# def truncate_text(text, max_len=40):
#     return text if len(text) <= max_len else text[:max_len-3] + "..."

# @st.cache_resource
# def load_model():
#     return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# # ---------------------------
# # Dashboard Layout
# # ---------------------------
# st.title("🔎 Alpha CV Skill Matching")
# st.markdown("This tool uses the **Hungarian algorithm** to find the optimal one-to-one assignment between job description (JD) skills and CV skills, based on their semantic similarity.")
# st.markdown("---")

# # ---------------------------
# # Input Boxes in Sidebar
# # ---------------------------
# with st.sidebar:
#     st.header("📥 Input Skills")
#     # JD Skills Input
#     jd_input = st.text_area(
#         "Paste JD skills (one per line)",
#         height=200,
#         placeholder="Enter each JD skill on a new line..."
#     )
    
#     # Multiple CV Skills Input
#     st.write("### CV Skills")
#     if 'cv_inputs' not in st.session_state:
#         st.session_state.cv_inputs = [""]
#         st.session_state.cv_names = [f"CV {i+1}" for i in range(len(st.session_state.cv_inputs))]
    
#     def add_cv_input():
#         st.session_state.cv_inputs.append("")
#         st.session_state.cv_names.append(f"CV {len(st.session_state.cv_inputs)}")
    
#     def remove_cv_input(index):
#         if len(st.session_state.cv_inputs) > 1:
#             st.session_state.cv_inputs.pop(index)
#             st.session_state.cv_names.pop(index)
    
#     # Display CV input fields
#     for i, (cv_text, cv_name) in enumerate(zip(st.session_state.cv_inputs, st.session_state.cv_names)):
#         col1, col2 = st.columns([5, 1])
#         with col1:
#             new_name = st.text_input(f"CV Name", value=cv_name, key=f"cv_name_{i}")
#             st.session_state.cv_names[i] = new_name
#             st.session_state.cv_inputs[i] = st.text_area(
#                 f"Paste CV skills for {new_name} (one per line)",
#                 value=cv_text,
#                 height=150,
#                 key=f"cv_text_{i}",
#                 placeholder="Enter each CV skill on a new line..."
#             )
#         with col2:
#             if len(st.session_state.cv_inputs) > 1:
#                 st.button("🗑️", key=f"remove_cv_{i}", on_click=remove_cv_input, args=(i,))
    
#     st.button("➕ Add Another CV", on_click=add_cv_input)
    
#     st.markdown("---")
#     if st.button("🔎 Get Scores", use_container_width=True):
#         st.session_state.run_analysis = True
#     else:
#         st.session_state.run_analysis = False

# # ---------------------------
# # Main Content Area
# # ---------------------------
# if st.session_state.run_analysis:
#     jd_skills = [s.strip() for s in jd_input.split("\n") if s.strip()]
#     cv_skills_list = []
#     cv_names_list = []
    
#     for i, cv_text in enumerate(st.session_state.cv_inputs):
#         cv_skills = [s.strip() for s in cv_text.split("\n") if s.strip()]
#         if cv_skills:
#             cv_skills_list.append(cv_skills)
#             cv_names_list.append(st.session_state.cv_names[i])
    
#     if not jd_skills or not cv_skills_list:
#         st.error("⚠️ Please enter both JD and at least one CV skills.")
#         st.stop()
    
#     # ---------------------------
#     # Embeddings & Qdrant Setup
#     # ---------------------------
#     with st.spinner("⏳ Encoding skills and setting up Qdrant..."):
#         model = load_model()
#         jd_embeddings = model.encode(jd_skills, normalize_embeddings=True)
#         all_cv_skills = []
#         cv_indices = []
#         cv_skill_indices = []
        
#         for cv_idx, cv_skills in enumerate(cv_skills_list):
#             for skill_idx, skill in enumerate(cv_skills):
#                 all_cv_skills.append(skill)
#                 cv_indices.append(cv_idx)
#                 cv_skill_indices.append(skill_idx)
        
#         cv_embeddings = model.encode(all_cv_skills, normalize_embeddings=True)
#         qdrant = QdrantClient(":memory:")
        
#         if qdrant.collection_exists(COLLECTION_NAME):
#             qdrant.delete_collection(COLLECTION_NAME)
        
#         qdrant.create_collection(
#             collection_name=COLLECTION_NAME,
#             vectors_config=models.VectorParams(size=cv_embeddings.shape[1], distance=models.Distance.COSINE),
#         )
        
#         qdrant.upsert(
#             collection_name=COLLECTION_NAME,
#             points=[
#                 models.PointStruct(
#                     id=i,
#                     vector=cv_embeddings[i].tolist(),
#                     payload={
#                         "skill": all_cv_skills[i],
#                         "cv_index": cv_indices[i],
#                         "cv_skill_index": cv_skill_indices[i],
#                         "cv_name": cv_names_list[cv_indices[i]]
#                     }
#                 )
#                 for i in range(len(all_cv_skills))
#             ],
#         )
    
#     st.success("✅ Analysis setup complete!")
    
#     # ---------------------------
#     # Process each CV
#     # ---------------------------
#     all_results = []
#     cv_overall_scores = {}
    
#     for cv_idx, (cv_name, cv_skills) in enumerate(zip(cv_names_list, cv_skills_list)):
#         # Similarity Matrix for this CV
#         M, N = len(jd_skills), len(cv_skills)
#         similarity_matrix = np.zeros((M, N), dtype=np.float32)
#         top_sorted_lists = {}
        
#         for j, jd_vec in enumerate(jd_embeddings):
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=jd_vec.tolist(),
#                 limit=len(all_cv_skills),
#                 with_payload=True,
#             )
#             cv_res = [p for p in res.points if p.payload["cv_index"] == cv_idx]
#             sorted_rows = []
#             for p in cv_res:
#                 skill_idx = p.payload["cv_skill_index"]
#                 score = float(p.score)
#                 similarity_matrix[j, skill_idx] = score
#                 sorted_rows.append((skill_idx, cv_skills[skill_idx], score))
#             sorted_rows.sort(key=lambda x: x[2], reverse=True)
#             top_sorted_lists[j] = sorted_rows
        
#         # Hungarian Algorithm for this CV
#         row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
#         assignments = []
#         for r, c in zip(row_ind, col_ind):
#             assignments.append({
#                 "cv_index": cv_idx,
#                 "cv_name": cv_name,
#                 "jd_index": r,
#                 "jd_skill": jd_skills[r],
#                 "cv_skill_index": c,
#                 "cv_skill": cv_skills[c],
#                 "score": float(similarity_matrix[r, c]),
#             })
#         assignments = sorted(assignments, key=lambda x: x['jd_index'])
#         avg_score = float(np.mean([a["score"] for a in assignments]))
#         # Calculate good and rejected counts
#         good_count = sum(1 for a in assignments if a["score"] >= GOOD_THRESHOLD)
#         rejected_count = len(assignments) - good_count
#         # Store results
#         cv_overall_scores[cv_name] = {"score": avg_score, "good": good_count, "rejected": rejected_count}
#         all_results.append({
#             "cv_name": cv_name,
#             "cv_idx": cv_idx,
#             "cv_skills": cv_skills,
#             "similarity_matrix": similarity_matrix,
#             "assignments": assignments,
#             "avg_score": avg_score,
#             "top_sorted_lists": top_sorted_lists,
#             "good_count": good_count,
#             "rejected_count": rejected_count
#         })
    
#     # ---------------------------
#     # CV Comparison Summary
#     # ---------------------------
#     st.subheader("🏆 CV Comparison Summary")
#     sorted_cv_scores = sorted(cv_overall_scores.items(), key=lambda x: x[1]['score'], reverse=True)
#     cols = st.columns(len(sorted_cv_scores))
    
#     for i, (cv_name, score_data) in enumerate(sorted_cv_scores):
#         with cols[i]:
#             score = score_data['score']
#             good_count = score_data['good']
#             rejected_count = score_data['rejected']
#             rank = i + 1
#             rank_class = f"rank-{rank}" if rank <= 3 else ""
            
#             if score >= 0.7:
#                 card_class = "score-card high-score"
#             elif score >= 0.5:
#                 card_class = "score-card medium-score"
#             else:
#                 card_class = "score-card low-score"
            
#             st.markdown(f"""
#             <div class="{card_class}">
#                 <div style="display: flex; justify-content: space-between; align-items: center;">
#                     <span class="rank-badge {rank_class}">#{rank}</span>
#                     <h3>{cv_name}</h3>
#                 </div>
#                 <div class="score-value">{score:.3f}</div>
#                 <div>Average Match Score</div>
#                 <hr>
#                 <div style="display: flex; justify-content: space-between; font-weight: bold;">
#                     <span>✅ Good: {good_count}</span>
#                     <span>❌ Rejected: {rejected_count}</span>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
    
#     st.divider()
    
#     # ---------------------------
#     # Detailed Results per CV
#     # ---------------------------
#     for result in all_results:
#         cv_name = result["cv_name"]
#         cv_idx = result["cv_idx"]
#         cv_skills = result["cv_skills"]
#         assignments = result["assignments"]
#         avg_score = result["avg_score"]
#         top_sorted_lists = result["top_sorted_lists"]
#         good_count = result["good_count"]
#         rejected_count = result["rejected_count"]
        
#         with st.container():
#             st.subheader(f"📊 Detailed Analysis for {cv_name}")
            
#             col_a, col_b = st.columns([1, 2])
#             with col_a:
#                 st.metric(label="Overall Match Score", value=f"{avg_score:.3f}")
#                 st.progress(avg_score)
#                 st.metric(label="Total JD Skills", value=len(jd_skills))
#                 st.metric(label=f"Total {cv_name} Skills", value=len(cv_skills))
#             with col_b:
#                 st.metric(label="GOOD Matches (>= 0.50)", value=good_count, delta=f"{100*good_count/len(jd_skills):.1f}%")
#                 st.metric(label="REJECTED Matches (< 0.50)", value=rejected_count, delta=f"{100*rejected_count/len(jd_skills):.1f}%", delta_color="inverse")
            
#             # ---------------------------
#             # Collapsible Graph Visualization
#             # ---------------------------
#             with st.expander("📈 Bipartite Assignment Graph (Click to expand)", expanded=False):
#                 def create_bipartite_graph(assignments_list, jd_list, cv_list):
#                     dot = graphviz.Digraph(comment=f'JD to {cv_name} Skill Matching')
#                     dot.attr(rankdir='LR', splines='true', overlap='false', nodesep='0.5', ranksep='2')
#                     with dot.subgraph(name='cluster_jd') as c:
#                         c.attr(label='Job Description Skills', style='filled', color='lightgrey')
#                         c.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
#                         for i, skill in enumerate(jd_list):
#                             c.node(f'jd_{i}', f'JD {i+1}: {truncate_text(skill)}')
#                     with dot.subgraph(name='cluster_cv') as c:
#                         c.attr(label=f'{cv_name} Skills', style='filled', color='lightgrey')
#                         c.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
#                         for i, skill in enumerate(cv_list):
#                             c.node(f'cv_{i}', f'CV {i+1}: {truncate_text(skill)}')
#                     for a in assignments_list:
#                         jd_idx, cv_idx, score = a['jd_index'], a['cv_skill_index'], a['score']
#                         color = "darkgreen" if score >= GOOD_THRESHOLD else "red"
#                         penwidth = str(0.5 + 3.5 * score)
#                         dot.edge(f'jd_{jd_idx}', f'cv_{cv_idx}', label=f' {score:.3f} ', fontcolor=color, color=color, penwidth=penwidth)
#                     return dot
#                 st.graphviz_chart(create_bipartite_graph(assignments, jd_skills, cv_skills), use_container_width=True)
            
#             st.markdown("---")
            
#             # ---------------------------
#             # Collapsible Detailed View + Top-3 Alternatives
#             # ---------------------------
#             with st.expander("✅ Optimal Assignments & Top Alternatives (Click to expand)", expanded=False):
#                 jd_to_assigned_cv = {a["jd_index"]: a["cv_skill_index"] for a in assignments}
#                 cv_assigned_to_jd = {a["cv_skill_index"]: a["jd_index"] for a in assignments}
#                 for a in assignments:
#                     jd_idx, jd, cv, s = a["jd_index"], a["jd_skill"], a["cv_skill"], a["score"]
#                     if s >= GOOD_THRESHOLD:
#                         st.markdown(f"""
#                         <div class="score-card high-score">
#                             <h4>JD Skill: {jd}</h4>
#                             <p>→ ✅ <strong>Matched CV Skill:</strong> {cv} | <strong>Score:</strong> {s:.3f}</p>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     else:
#                         st.markdown(f"""
#                         <div class="score-card low-score">
#                             <h4>JD Skill: {jd}</h4>
#                             <p>→ ❌ <strong>Matched CV Skill:</strong> {cv} | <strong>Score:</strong> {s:.3f} (Rejected)</p>
#                         </div>
#                         """, unsafe_allow_html=True)
#                     alts = []
#                     for (alt_cv_id, alt_cv_text, alt_score) in top_sorted_lists[jd_idx]:
#                         if alt_cv_id == jd_to_assigned_cv[jd_idx]:
#                             continue
#                         tag = ""
#                         if alt_cv_id in cv_assigned_to_jd and cv_assigned_to_jd[alt_cv_id] != jd_idx:
#                             tag = " (in use)"
#                         alts.append((alt_cv_text + tag, alt_score))
#                         if len(alts) == 3:
#                             break
#                     st.write("Top-3 Alternatives:")
#                     if alts:
#                         for rank, (alt_text, alt_s) in enumerate(alts, start=1):
#                             st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ (Top {rank}) {alt_text} | Score: {alt_s:.3f}")
#                     else:
#                         st.write("  —")
#                     st.divider()
            
#             # ---------------------------
#             # Table: All Assignments
#             # ---------------------------
#             st.subheader(f"📋 All Assignments for {cv_name} (Tabular Summary)")
#             df_matched = pd.DataFrame(assignments)
#             df_matched["JD #"] = df_matched["jd_index"] + 1
#             df_matched["CV #"] = df_matched["cv_skill_index"] + 1
#             df_matched["Status"] = df_matched["score"].apply(lambda s: "GOOD" if s >= GOOD_THRESHOLD else "REJECTED")
#             df_matched = df_matched[["JD #", "jd_skill", "CV #", "cv_skill", "score", "Status"]].rename(
#                 columns={"jd_skill": "JD Skill", "cv_skill": "CV Skill", "score": "Score"}
#             )
#             st.dataframe(df_matched.style.format({"Score": "{:.3f}"}), use_container_width=True)
#             st.divider()


