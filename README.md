# Semantic Search System with Fuzzy Clustering and Semantic Cache

This project implements a lightweight **semantic search system** over the **20 Newsgroups dataset (~20k documents)**. The system supports **vector-based semantic retrieval**, **probabilistic fuzzy clustering**, and a **semantic cache built from scratch** that recognizes similar queries even when phrased differently.

The system exposes a **FastAPI service** with endpoints for querying the corpus, inspecting cache statistics, and resetting the cache.

---

# System Architecture

The project is designed as a two-stage system:

Offline Preprocessing Pipeline  
→ Embedding Generation  
→ Vector Index (FAISS)  
→ Fuzzy Clustering (GMM)  
→ Artifacts Storage  
→ FastAPI Runtime Service  
→ Semantic Cache + Query Engine

The preprocessing stage runs once to prepare artifacts.  
The FastAPI service loads these artifacts and serves queries.

                    ┌──────────────────────────┐
                    │  Offline Preprocessing   │
                    └─────────────┬────────────┘
                                  │
                        Clean & preprocess dataset
                                  │
                                  ▼
                     Generate document embeddings
                                  │
                                  ▼
                        Build FAISS vector index
                                  │
                                  ▼
                     Train GMM fuzzy clustering
                                  │
                                  ▼
                           Save artifacts
            (embeddings.npy, faiss.index, cluster_model.pkl)

────────────────────────────────────────────────────────────

                     ┌──────────────────────────┐
                     │     FastAPI Service      │
                     └─────────────┬────────────┘
                                   │
                             Incoming Query
                                   │
                                   ▼
                           Embed the query
                                   │
                                   ▼
                       Determine dominant cluster
                                   │
                                   ▼
                         Semantic Cache Lookup
                         │                  │
                         │                  │
                      Cache Hit          Cache Miss
                         │                  │
                         ▼                  ▼
                    Return result       FAISS search
                                            │
                                            ▼
                                   Store in cache
                                            │
                                            ▼
                                       Return result

---

# Dataset

The system uses the **20 Newsgroups dataset**, which contains approximately **20,000 documents across 20 categories**.

Challenges in the dataset include:

- noisy formatting
- quoted replies
- email headers
- mixed topic boundaries

To address this, preprocessing removes:

- headers
- footers
- quoted replies
- extremely short documents

This ensures embeddings capture **semantic content rather than formatting noise**.

---

# Embedding Model

The system uses the **SentenceTransformers model**:

all-MiniLM-L6-v2

Key properties:

Embedding dimension: 384  
Architecture: MiniLM Transformer  
Similarity metric: Cosine similarity  
Performance: Fast and high-quality semantic embeddings  

Embeddings are **normalized**, allowing cosine similarity to be computed using **inner product**, which enables efficient FAISS indexing.

---

# Vector Database

The system uses **FAISS** for vector similarity search.

Index type used:

IndexFlatIP

Reasons for this choice:

- Exact nearest neighbor search
- Efficient for datasets under ~1M vectors
- Works directly with normalized embeddings
- No recall loss

With normalized embeddings:

cosine_similarity(a, b) = dot_product(a, b)

This allows FAISS to perform fast cosine similarity retrieval.

---

# Fuzzy Clustering

Instead of hard clustering, the system uses **Gaussian Mixture Models (GMM)**.

Each document receives a **probability distribution over clusters**:

Document → [P(cluster1), P(cluster2), ..., P(clusterK)]

Example:

Doc_42 → [0.62 politics, 0.28 firearms, 0.10 law]

This reflects the reality that documents may belong to multiple semantic themes.

## Cluster Count Selection

The number of clusters is determined using **Bayesian Information Criterion (BIC)**.

BIC balances:

- model likelihood
- model complexity

Lower BIC indicates a better model.

This prevents both:

- too few clusters (underfitting)
- too many clusters (overfitting)

---

# Semantic Cache

Traditional caches only work when queries are identical.

Example:

Query 1: "how to install linux drivers"  
Query 2: "install drivers on linux"

A traditional cache would treat these as different queries.

The semantic cache instead:

1. Embeds the query
2. Finds its dominant cluster
3. Searches cached queries within that cluster
4. Uses cosine similarity to detect matches

If similarity exceeds a threshold, the cached result is returned.

## Similarity Threshold

The cache uses a configurable similarity threshold:

similarity ≥ 0.85 → cache hit

Typical cosine similarity ranges:

0.95+ → nearly identical  
0.85 → strong semantic match  
0.70 → related  
0.50 → weak relation  

---

# Cluster-Aware Cache Optimization

Cache entries are partitioned by cluster:

cluster_id  
• FAISS index of cached queries  
• query embeddings  
• cached results  

This reduces lookup complexity from:

O(N)

to approximately:

O(N / clusters)

This significantly improves cache lookup efficiency as the cache grows.

Each cluster maintains its own FAISS index for cached query embeddings.

---

# Query Execution Flow

When a query is received:

Query  
→ Text cleaning  
→ Query embedding  
→ Cluster probability estimation  
→ Semantic cache lookup  
→ Cache hit → return cached result  
→ Cache miss → vector search  
→ Store result in cache  
→ Return response  

---

# API Endpoints

## POST /query

Accepts a natural language query.

Example request:

{
  "query": "how to install linux drivers"
}

Example response:

{
  "query": "how to install linux drivers",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "dominant_cluster": 4,
  "cluster_probability": 0.71,
  "result": [
    {
      "text": "linux device drivers can be installed...",
      "score": 0.87
    }
  ]
}

If a semantic match is found in cache:

{
  "cache_hit": true,
  "matched_query": "install linux drivers",
  "similarity_score": 0.92
}

---

## GET /cache/stats

Returns cache statistics.

Example:

{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}

---

## DELETE /cache

Clears the semantic cache and resets statistics.

---

## GET /health

Service health endpoint.

Example response:

{
  "status": "ok",
  "documents_loaded": 17886,
  "clusters": 27,
  "cache_entries": 12
}

---

# Project Structure

semantic-search/

app/
- main.py
- routes.py
- search.py
- cache.py
- config.py

preprocessing/
- run_pipeline.py
- load_dataset.py
- generate_embeddings.py
- build_index.py
- clustering.py

utils/
- embeddings.py
- similarity.py
- text_cleaning.py

artifacts/
- documents.json
- embeddings.npy
- faiss.index
- cluster_model.pkl
- cluster_probs.npy
- metadata.json

Dockerfile  
docker-compose.yml  
requirements.txt  

---

# Running the Project

## 1. Create virtual environment

python -m venv venv

Activate:

venv\Scripts\activate

---

## 2. Install dependencies

pip install -r requirements.txt

---

## 3. Run preprocessing pipeline

python -m preprocessing.run_pipeline

This generates artifacts required for the API.

---

## 4. Start FastAPI server

uvicorn app.main:app --reload

API documentation:

http://127.0.0.1:8000/docs

---

# Docker

To run using Docker:

docker build -t semantic-search .

docker run -p 8000:8000 semantic-search

---

# Key Design Decisions

Embedding model → MiniLM (fast semantic embeddings)  
Vector DB → FAISS (efficient similarity search)  
Clustering → Gaussian Mixture Model (probabilistic clusters)  
Cluster selection → BIC (prevents overfitting)  
Cache design → semantic similarity matching  
Cache optimization → cluster-aware partitioning  

---

# Future Improvements

Possible extensions:

- hybrid search (BM25 + embeddings)
- cluster summaries
- result reranking
- persistent cache storage
- query analytics dashboard

---

# Summary

This project demonstrates a complete **semantic retrieval pipeline**, combining:

- transformer-based embeddings
- vector similarity search
- probabilistic clustering
- semantic caching
- API-based deployment

The design emphasizes **correctness, efficiency, and explainability**, similar to real-world semantic search systems.
