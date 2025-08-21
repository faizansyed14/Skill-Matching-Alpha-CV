🔎 Step-by-Step Matching Process
1. Convert text into vectors (embeddings)

Both CV skills and JD skills are plain English sentences.

We pass each sentence through a pretrained transformer model (all-mpnet-base-v2).

The model outputs a 768-dimensional vector (embedding) for each sentence.

Example:

"Writing PL/SQL stored procedures"
→ [0.11, -0.34, 0.07, ..., 0.56] (768 numbers)


👉 These embeddings capture semantic meaning, not just keywords.
So “SQL query tuning” and “Optimize SQL performance” will be close together in vector space.

----------------------------------------------------------------
2. Store JD skill embeddings in Qdrant

We create a Qdrant collection (jd_skills).

Each JD skill embedding is stored with:

vector = 768-dim array

Qdrant builds an HNSW index (Hierarchical Navigable Small World graph) for fast nearest-neighbor search.
That means instead of brute-force comparing every vector, it organizes them in a graph for quick lookup.

----------------------------------------------------------
3. Match each CV skill against JD skills

For each CV skill:

Its embedding is computed.

We run a vector similarity search in Qdrant against all JD embeddings.

Distance metric = cosine similarity (closer to 1 = better).

Qdrant returns the Top-K nearest JD vectors (here K=3).

Example:

CV Skill: "Managing incident tickets using ServiceNow"
Embedding → query vector

Search against all JD vectors →
    JD: "Manage support tickets using ServiceNow"  → Score 0.89
    JD: "Apply ITIL incident management"           → Score 0.73
    JD: "Monitor application performance"          → Score 0.32


So you see how semantically related skills bubble up to the top.

-----------------------------------------------------
4. Compute Top-1 and Top-3

For reporting:

Top-3 JD skills are shown for each CV skill.

Top-1 score is saved for overall scoring.

-----------------------------------------------
5. Compute Overall Skill Match Score

Take the average of all Top-1 scores across CV skills.

This gives a single number = “how well this CV aligns with the JD skills”.

Also count:

GOOD matches (≥ 0.50)

WEAK matches (< 0.50)

🧠 Intuition

You’re basically measuring semantic overlap between CV skill statements and JD skill requirements.

Transformer embeddings + cosine similarity ensure that even if wording is different, but meaning is close, they’ll match.

Examples:

“SQL tuning” ↔ “Optimize SQL performance” → high score

“Docker & Kubernetes” ↔ “Unix/Linux scripting” → lower score