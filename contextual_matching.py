# contextual_matching_hungarian_ui.py
# ------------------------------------------------------------
# JD ↔ CV skill matching with:
#    • User input boxes for JD and multiple CV skill lists
#    • Qdrant vector search (cosine similarity)
#    • Hungarian algorithm (optimal assignment, one-to-one)
#    • Top-3 alternatives per JD (excluding its own assigned CV)
#    • Tables:
#        1) All Assignments (Accepted + Rejected)
#        2) Ignored Top Match
#        3) CV Comparison Summary
#    • Visualizations:
#        1) Bipartite graph of assignments (Graphviz) - Collapsible
#        2) Histogram of assignment scores (Plotly)
#        3) Animated score cards for CV comparison
#    • Color coding: >= 0.50 GOOD (green), < 0.50 REJECTED (red)
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

# ---------------------------
# Config
# ---------------------------
GOOD_THRESHOLD = 0.50
COLLECTION_NAME = "cv_skills"
st.set_page_config(layout="wide")

# Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .score-card {
        animation: fadeIn 0.8s ease-out forwards;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .score-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .high-score {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 5px solid #2ecc71;
    }
    
    .medium-score {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #f39c12;
    }
    
    .low-score {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        border-left: 5px solid #e74c3c;
    }
    
    .score-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .rank-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .rank-1 {
        background-color: #FFD700;
        color: #333;
    }
    
    .rank-2 {
        background-color: #C0C0C0;
        color: #333;
    }
    
    .rank-3 {
        background-color: #CD7F32;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helper function
# ---------------------------
def truncate_text(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len-3] + "..."

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---------------------------
# Dashboard Layout
# ---------------------------
st.title("🔎 Skill Matching Dashboard")
st.markdown("This tool uses the **Hungarian algorithm** to find the optimal one-to-one assignment between job description (JD) skills and CV skills, based on their semantic similarity.")
st.markdown("---")

# ---------------------------
# Input Boxes in Sidebar
# ---------------------------
with st.sidebar:
    st.header("📥 Input Skills")
    # JD Skills Input
    jd_input = st.text_area(
        "Paste JD skills (one per line)",
        height=200,
        placeholder="Enter each JD skill on a new line..."
    )
    
    # Multiple CV Skills Input
    st.write("### CV Skills")
    if 'cv_inputs' not in st.session_state:
        st.session_state.cv_inputs = [""]
        st.session_state.cv_names = [f"CV {i+1}" for i in range(len(st.session_state.cv_inputs))]
    
    def add_cv_input():
        st.session_state.cv_inputs.append("")
        st.session_state.cv_names.append(f"CV {len(st.session_state.cv_inputs)}")
    
    def remove_cv_input(index):
        if len(st.session_state.cv_inputs) > 1:
            st.session_state.cv_inputs.pop(index)
            st.session_state.cv_names.pop(index)
    
    # Display CV input fields
    for i, (cv_text, cv_name) in enumerate(zip(st.session_state.cv_inputs, st.session_state.cv_names)):
        col1, col2 = st.columns([5, 1])
        with col1:
            new_name = st.text_input(f"CV Name", value=cv_name, key=f"cv_name_{i}")
            st.session_state.cv_names[i] = new_name
            st.session_state.cv_inputs[i] = st.text_area(
                f"Paste CV skills for {new_name} (one per line)",
                value=cv_text,
                height=150,
                key=f"cv_text_{i}",
                placeholder="Enter each CV skill on a new line..."
            )
        with col2:
            if len(st.session_state.cv_inputs) > 1:
                st.button("🗑️", key=f"remove_cv_{i}", on_click=remove_cv_input, args=(i,))
    
    st.button("➕ Add Another CV", on_click=add_cv_input)
    
    st.markdown("---")
    if st.button("🔎 Get Scores", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

# ---------------------------
# Main Content Area
# ---------------------------
if st.session_state.run_analysis:
    jd_skills = [s.strip() for s in jd_input.split("\n") if s.strip()]
    cv_skills_list = []
    cv_names_list = []
    
    for i, cv_text in enumerate(st.session_state.cv_inputs):
        cv_skills = [s.strip() for s in cv_text.split("\n") if s.strip()]
        if cv_skills:
            cv_skills_list.append(cv_skills)
            cv_names_list.append(st.session_state.cv_names[i])
    
    if not jd_skills or not cv_skills_list:
        st.error("⚠️ Please enter both JD and at least one CV skills.")
        st.stop()
    
    # ---------------------------
    # Embeddings & Qdrant Setup
    # ---------------------------
    with st.spinner("⏳ Encoding skills and setting up Qdrant..."):
        model = load_model()
        jd_embeddings = model.encode(jd_skills, normalize_embeddings=True)
        all_cv_skills = []
        cv_indices = []
        cv_skill_indices = []
        
        for cv_idx, cv_skills in enumerate(cv_skills_list):
            for skill_idx, skill in enumerate(cv_skills):
                all_cv_skills.append(skill)
                cv_indices.append(cv_idx)
                cv_skill_indices.append(skill_idx)
        
        cv_embeddings = model.encode(all_cv_skills, normalize_embeddings=True)
        qdrant = QdrantClient(":memory:")
        
        if qdrant.collection_exists(COLLECTION_NAME):
            qdrant.delete_collection(COLLECTION_NAME)
        
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=cv_embeddings.shape[1], distance=models.Distance.COSINE),
        )
        
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=i,
                    vector=cv_embeddings[i].tolist(),
                    payload={
                        "skill": all_cv_skills[i],
                        "cv_index": cv_indices[i],
                        "cv_skill_index": cv_skill_indices[i],
                        "cv_name": cv_names_list[cv_indices[i]]
                    }
                )
                for i in range(len(all_cv_skills))
            ],
        )
    
    st.success("✅ Analysis setup complete!")
    
    # ---------------------------
    # Process each CV
    # ---------------------------
    all_results = []
    cv_overall_scores = {}
    
    for cv_idx, (cv_name, cv_skills) in enumerate(zip(cv_names_list, cv_skills_list)):
        # Similarity Matrix for this CV
        M, N = len(jd_skills), len(cv_skills)
        similarity_matrix = np.zeros((M, N), dtype=np.float32)
        top_sorted_lists = {}
        
        for j, jd_vec in enumerate(jd_embeddings):
            res = qdrant.query_points(
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
        
        # Hungarian Algorithm for this CV
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        assignments = []
        for r, c in zip(row_ind, col_ind):
            assignments.append({
                "cv_index": cv_idx,
                "cv_name": cv_name,
                "jd_index": r,
                "jd_skill": jd_skills[r],
                "cv_skill_index": c,
                "cv_skill": cv_skills[c],
                "score": float(similarity_matrix[r, c]),
            })
        assignments = sorted(assignments, key=lambda x: x['jd_index'])
        avg_score = float(np.mean([a["score"] for a in assignments]))
        # Calculate good and rejected counts
        good_count = sum(1 for a in assignments if a["score"] >= GOOD_THRESHOLD)
        rejected_count = len(assignments) - good_count
        # Store results
        cv_overall_scores[cv_name] = {"score": avg_score, "good": good_count, "rejected": rejected_count}
        all_results.append({
            "cv_name": cv_name,
            "cv_idx": cv_idx,
            "cv_skills": cv_skills,
            "similarity_matrix": similarity_matrix,
            "assignments": assignments,
            "avg_score": avg_score,
            "top_sorted_lists": top_sorted_lists,
            "good_count": good_count,
            "rejected_count": rejected_count
        })
    
    # ---------------------------
    # CV Comparison Summary
    # ---------------------------
    st.subheader("🏆 CV Comparison Summary")
    sorted_cv_scores = sorted(cv_overall_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    cols = st.columns(len(sorted_cv_scores))
    
    for i, (cv_name, score_data) in enumerate(sorted_cv_scores):
        with cols[i]:
            score = score_data['score']
            good_count = score_data['good']
            rejected_count = score_data['rejected']
            rank = i + 1
            rank_class = f"rank-{rank}" if rank <= 3 else ""
            
            if score >= 0.7:
                card_class = "score-card high-score"
            elif score >= 0.5:
                card_class = "score-card medium-score"
            else:
                card_class = "score-card low-score"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="rank-badge {rank_class}">#{rank}</span>
                    <h3>{cv_name}</h3>
                </div>
                <div class="score-value">{score:.3f}</div>
                <div>Average Match Score</div>
                <hr>
                <div style="display: flex; justify-content: space-between; font-weight: bold;">
                    <span>✅ Good: {good_count}</span>
                    <span>❌ Rejected: {rejected_count}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ---------------------------
    # Detailed Results per CV
    # ---------------------------
    for result in all_results:
        cv_name = result["cv_name"]
        cv_idx = result["cv_idx"]
        cv_skills = result["cv_skills"]
        assignments = result["assignments"]
        avg_score = result["avg_score"]
        top_sorted_lists = result["top_sorted_lists"]
        good_count = result["good_count"]
        rejected_count = result["rejected_count"]
        
        with st.container():
            st.subheader(f"📊 Detailed Analysis for {cv_name}")
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric(label="Overall Match Score", value=f"{avg_score:.3f}")
                st.progress(avg_score)
                st.metric(label="Total JD Skills", value=len(jd_skills))
                st.metric(label=f"Total {cv_name} Skills", value=len(cv_skills))
            with col_b:
                st.metric(label="GOOD Matches (>= 0.50)", value=good_count, delta=f"{100*good_count/len(jd_skills):.1f}%")
                st.metric(label="REJECTED Matches (< 0.50)", value=rejected_count, delta=f"{100*rejected_count/len(jd_skills):.1f}%", delta_color="inverse")
            
            # ---------------------------
            # Collapsible Graph Visualization
            # ---------------------------
            with st.expander("📈 Bipartite Assignment Graph (Click to expand)", expanded=False):
                def create_bipartite_graph(assignments_list, jd_list, cv_list):
                    dot = graphviz.Digraph(comment=f'JD to {cv_name} Skill Matching')
                    dot.attr(rankdir='LR', splines='true', overlap='false', nodesep='0.5', ranksep='2')
                    with dot.subgraph(name='cluster_jd') as c:
                        c.attr(label='Job Description Skills', style='filled', color='lightgrey')
                        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
                        for i, skill in enumerate(jd_list):
                            c.node(f'jd_{i}', f'JD {i+1}: {truncate_text(skill)}')
                    with dot.subgraph(name='cluster_cv') as c:
                        c.attr(label=f'{cv_name} Skills', style='filled', color='lightgrey')
                        c.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
                        for i, skill in enumerate(cv_list):
                            c.node(f'cv_{i}', f'CV {i+1}: {truncate_text(skill)}')
                    for a in assignments_list:
                        jd_idx, cv_idx, score = a['jd_index'], a['cv_skill_index'], a['score']
                        color = "darkgreen" if score >= GOOD_THRESHOLD else "red"
                        penwidth = str(0.5 + 3.5 * score)
                        dot.edge(f'jd_{jd_idx}', f'cv_{cv_idx}', label=f' {score:.3f} ', fontcolor=color, color=color, penwidth=penwidth)
                    return dot
                st.graphviz_chart(create_bipartite_graph(assignments, jd_skills, cv_skills), use_container_width=True)
            
            st.markdown("---")
            
            # ---------------------------
            # Collapsible Detailed View + Top-3 Alternatives
            # ---------------------------
            with st.expander("✅ Optimal Assignments & Top Alternatives (Click to expand)", expanded=False):
                jd_to_assigned_cv = {a["jd_index"]: a["cv_skill_index"] for a in assignments}
                cv_assigned_to_jd = {a["cv_skill_index"]: a["jd_index"] for a in assignments}
                for a in assignments:
                    jd_idx, jd, cv, s = a["jd_index"], a["jd_skill"], a["cv_skill"], a["score"]
                    if s >= GOOD_THRESHOLD:
                        st.markdown(f"""
                        <div class="score-card high-score">
                            <h4>JD Skill: {jd}</h4>
                            <p>→ ✅ <strong>Matched CV Skill:</strong> {cv} | <strong>Score:</strong> {s:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="score-card low-score">
                            <h4>JD Skill: {jd}</h4>
                            <p>→ ❌ <strong>Matched CV Skill:</strong> {cv} | <strong>Score:</strong> {s:.3f} (Rejected)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    alts = []
                    for (alt_cv_id, alt_cv_text, alt_score) in top_sorted_lists[jd_idx]:
                        if alt_cv_id == jd_to_assigned_cv[jd_idx]:
                            continue
                        tag = ""
                        if alt_cv_id in cv_assigned_to_jd and cv_assigned_to_jd[alt_cv_id] != jd_idx:
                            tag = " (in use)"
                        alts.append((alt_cv_text + tag, alt_score))
                        if len(alts) == 3:
                            break
                    st.write("Top-3 Alternatives:")
                    if alts:
                        for rank, (alt_text, alt_s) in enumerate(alts, start=1):
                            st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;→ (Top {rank}) {alt_text} | Score: {alt_s:.3f}")
                    else:
                        st.write("  —")
                    st.divider()
            
            # ---------------------------
            # Table: All Assignments
            # ---------------------------
            st.subheader(f"📋 All Assignments for {cv_name} (Tabular Summary)")
            df_matched = pd.DataFrame(assignments)
            df_matched["JD #"] = df_matched["jd_index"] + 1
            df_matched["CV #"] = df_matched["cv_skill_index"] + 1
            df_matched["Status"] = df_matched["score"].apply(lambda s: "GOOD" if s >= GOOD_THRESHOLD else "REJECTED")
            df_matched = df_matched[["JD #", "jd_skill", "CV #", "cv_skill", "score", "Status"]].rename(
                columns={"jd_skill": "JD Skill", "cv_skill": "CV Skill", "score": "Score"}
            )
            st.dataframe(df_matched.style.format({"Score": "{:.3f}"}), use_container_width=True)
            st.divider()
