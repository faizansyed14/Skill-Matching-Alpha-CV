# contextual_matching_hungarian_ui.py
# ------------------------------------------------------------
# JD ↔ CV skill matching with:
#   • User input boxes for JD and CV skill lists
#   • Qdrant vector search (cosine similarity)
#   • Hungarian algorithm (optimal assignment, one-to-one)
#   • Top-3 alternatives per JD (excluding its own assigned CV)
#   • Tables:
#       1) All Assignments (Accepted + Rejected)
#       2) Ignored Top Match
#   • Visualizations:
#       1) Bipartite graph of assignments (Graphviz)
#       2) Heatmap of the full similarity matrix (Plotly)
#       3) Histogram of assignment scores (Plotly)
#   • Color coding: >= 0.50 GOOD (green), < 0.50 REJECTED (red)
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

# ---------------------------
# Config
# ---------------------------
GOOD_THRESHOLD = 0.50
COLLECTION_NAME = "cv_skills"

st.set_page_config(layout="wide")
st.title("🔎 JD ↔ CV Skill Matching (Hungarian Algorithm + Qdrant)")

# ---------------------------
# Helper function
# ---------------------------
def truncate_text(text, max_len=40):
    return text if len(text) <= max_len else text[:max_len-3] + "..."

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ---------------------------
# Input Boxes
# ---------------------------
st.subheader("📥 Enter Skills")

jd_input = st.text_area(
    "Paste JD skills (one per line)",
    height=200,
    placeholder="Enter each JD skill on a new line..."
)

cv_input = st.text_area(
    "Paste CV skills (one per line)",
    height=200,
    placeholder="Enter each CV skill on a new line..."
)

if st.button("🔎 Get Scores"):
    jd_skills = [s.strip() for s in jd_input.split("\n") if s.strip()]
    cv_skills = [s.strip() for s in cv_input.split("\n") if s.strip()]

    if not jd_skills or not cv_skills:
        st.error("⚠️ Please enter both JD and CV skills.")
        st.stop()

    # ---------------------------
    # Embeddings
    # ---------------------------
    st.write("⏳ Encoding skills...")
    model = load_model()
    cv_embeddings = model.encode(cv_skills, normalize_embeddings=True)
    jd_embeddings = model.encode(jd_skills, normalize_embeddings=True)

    # ---------------------------
    # Setup Qdrant
    # ---------------------------
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
            models.PointStruct(id=i, vector=cv_embeddings[i].tolist(), payload={"skill": cv_skills[i], "cv_index": i})
            for i in range(len(cv_skills))
        ],
    )

    # ---------------------------
    # Similarity Matrix
    # ---------------------------
    M, N = len(jd_skills), len(cv_skills)
    similarity_matrix = np.zeros((M, N), dtype=np.float32)
    top_sorted_lists = {}

    for j, jd_vec in enumerate(jd_embeddings):
        res = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=jd_vec.tolist(),
            limit=N,
            with_payload=True,
        )

        sorted_rows = []
        for p in res.points:
            cv_id = int(p.id)
            score = float(p.score)
            similarity_matrix[j, cv_id] = score
            sorted_rows.append((cv_id, cv_skills[cv_id], score))
        top_sorted_lists[j] = sorted_rows

    # ---------------------------
    # Hungarian Algorithm
    # ---------------------------
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    assignments = []
    for r, c in zip(row_ind, col_ind):
        assignments.append({
            "jd_index": r,
            "jd_skill": jd_skills[r],
            "cv_index": c,
            "cv_skill": cv_skills[c],
            "score": float(similarity_matrix[r, c]),
        })
    assignments = sorted(assignments, key=lambda x: x['jd_index'])
    avg_score = float(np.mean([a["score"] for a in assignments]))
    jd_to_assigned_cv = {a["jd_index"]: a["cv_index"] for a in assignments}
    cv_assigned_to_jd = {a["cv_index"]: a["jd_index"] for a in assignments}

    # ---------------------------
    # Graph Visualization
    # ---------------------------
    st.subheader("📊 Assignment Graph (Optimal Matches)")

    def create_bipartite_graph(assignments_list, jd_list, cv_list):
        dot = graphviz.Digraph(comment='JD to CV Skill Matching')
        dot.attr(rankdir='LR', splines='true', overlap='false', nodesep='0.5', ranksep='2')

        with dot.subgraph(name='cluster_jd') as c:
            c.attr(label='Job Description Skills', style='filled', color='lightgrey')
            c.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
            for i, skill in enumerate(jd_list):
                c.node(f'jd_{i}', f'JD {i+1}: {truncate_text(skill)}')

        with dot.subgraph(name='cluster_cv') as c:
            c.attr(label='CV / Candidate Skills', style='filled', color='lightgrey')
            c.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
            for i, skill in enumerate(cv_list):
                c.node(f'cv_{i}', f'CV {i+1}: {truncate_text(skill)}')

        for a in assignments_list:
            jd_idx, cv_idx, score = a['jd_index'], a['cv_index'], a['score']
            color = "darkgreen" if score >= GOOD_THRESHOLD else "red"
            penwidth = str(0.5 + 3.5 * score)
            dot.edge(f'jd_{jd_idx}', f'cv_{cv_idx}', label=f' {score:.3f} ', fontcolor=color, color=color, penwidth=penwidth)
        return dot

    st.graphviz_chart(create_bipartite_graph(assignments, jd_skills, cv_skills), use_container_width=True)
    st.divider()

    # ---------------------------
    # Detailed View + Top-3 Alternatives
    # ---------------------------
    st.subheader("✅ Optimal JD ↔ CV Assignments (Detailed View)")
    for a in assignments:
        jd_idx, jd, cv, s = a["jd_index"], a["jd_skill"], a["cv_skill"], a["score"]

        if s >= GOOD_THRESHOLD:
            st.markdown(f"**JD:** {jd}<br/>→ ✅ <span style='color:green'>**CV:** {cv} | **Score:** {s:.3f}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**JD:** {jd}<br/>→ ❌ <span style='color:red'>**CV:** {cv} | **Score:** {s:.3f} (Rejected)</span>", unsafe_allow_html=True)

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
    # Table 1: All Assignments
    # ---------------------------
    df_matched = pd.DataFrame(assignments)
    df_matched["JD #"] = df_matched["jd_index"] + 1
    df_matched["CV #"] = df_matched["cv_index"] + 1
    df_matched["Status"] = df_matched["score"].apply(lambda s: "GOOD" if s >= GOOD_THRESHOLD else "REJECTED")
    df_matched = df_matched[["JD #", "jd_skill", "CV #", "cv_skill", "score", "Status"]].rename(
        columns={"jd_skill":"JD Skill", "cv_skill":"CV Skill", "score":"Score"}
    )
    st.subheader("📋 All Assignments (Tabular Summary)")
    st.dataframe(df_matched.style.format({"Score": "{:.3f}"}), use_container_width=True)
    st.divider()

    # ---------------------------
    # Table 2: Ignored Top Match
    # ---------------------------
    ignored_rows = []
    for j in range(M):
        best_c = int(np.argmax(similarity_matrix[j]))
        best_score = float(similarity_matrix[j, best_c])
        chosen_c = jd_to_assigned_cv[j]
        chosen_score = float(similarity_matrix[j, chosen_c])
        if best_c != chosen_c:
            ignored_rows.append({
                "JD #": j + 1,
                "JD Skill": jd_skills[j],
                "Top CV # (Ignored)": best_c + 1,
                "Top CV Skill (Ignored)": cv_skills[best_c],
                "Top Score": best_score,
                "Assigned CV #": chosen_c + 1,
                "Assigned CV Skill": cv_skills[chosen_c],
                "Assigned Score": chosen_score,
                "Reason": "Best CV was used by another JD"
            })

    st.subheader("🧭 Ignored Top Match (Per JD)")
    if ignored_rows:
        st.dataframe(pd.DataFrame(ignored_rows).style.format({"Top Score": "{:.3f}", "Assigned Score": "{:.3f}"}), use_container_width=True)
    else:
        st.write("For every JD, the top local CV was also chosen by the Hungarian algorithm.")
    st.divider()

    # ---------------------------
    # Overall Stats
    # ---------------------------
    good = (df_matched["Status"] == "GOOD").sum()
    rejected = (df_matched["Status"] == "REJECTED").sum()
    st.subheader("📈 Overall Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Average Hungarian Score", value=f"{avg_score:.3f}")
        st.metric(label="Total JD Skills", value=len(jd_skills))
        st.metric(label="Total CV Skills", value=len(cv_skills))
        st.metric(label="GOOD Matches (>= 0.50)", value=good, delta=f"{100*good/M:.1f}%")
        st.metric(label="REJECTED Matches (< 0.50)", value=rejected, delta=f"{100*rejected/M:.1f}%", delta_color="inverse")
