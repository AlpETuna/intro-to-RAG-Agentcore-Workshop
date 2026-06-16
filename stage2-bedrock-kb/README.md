# Stage 2 — Bedrock Knowledge Base

**Time:** 25 minutes | **Cost:** ~$0.48/hr while active (OpenSearch Serverless)

---

## What You'll Build

A fully managed Bedrock Knowledge Base backed by Amazon OpenSearch Serverless. You'll replace every manual step from Stage 1 with a managed equivalent.

```
S3 Bucket (docs) → Bedrock KB (ingestion job) → OpenSearch Serverless (HNSW)
                                                          ↓
                                      Retrieve API ← Query → Retrieve & Generate API
```

---

## Install Dependencies

```bash
cd stage2-bedrock-kb
uv sync
```

---

## Scripts (run in order)

### 01 — Create Infrastructure (`~3–4 min`)

Creates the S3 bucket, uploads the 5 workshop documents, creates the IAM service role, and provisions the **OpenSearch Serverless collection + vector index** that the KB writes embeddings into.

```bash
uv run 01_create_infrastructure.py
```

**Watch for:**
- The IAM trust policy structure (why Bedrock can assume the role)
- The inline policy granting S3 read and OpenSearch write access
- Resource names with a random suffix (ensures uniqueness across accounts)
- The three OpenSearch Serverless policies (encryption / network / data access) that must exist before a collection can be created
- The kNN vector index (`embedding` field, 1024-dim, HNSW + faiss)

> **Why here?** The Bedrock `CreateKnowledgeBase` API does **not** auto-create the vector store — the collection and index must already exist and be passed in by ARN. This script creates them and saves `OPENSEARCH_COLLECTION_ARN` to `.env`.

---

### 02 — Create Knowledge Base (`~2–3 min`)

Creates the Bedrock Knowledge Base pointed at the OpenSearch Serverless collection from step 01.

```bash
uv run 02_create_knowledge_base.py
```

**Watch for:**
- The `storageConfiguration` referencing the real `collectionArn` + `vectorIndexName`
- The `embeddingModelArn` — same Titan model as Stage 1
- KB status progression: CREATING → ACTIVE
- What Bedrock handles vs what you set up in Stage 1

---

### 03 — Sync and Query (`~3 min`)

Triggers the ingestion job (Bedrock chunks, embeds, and indexes the S3 documents), waits for completion, then runs queries using both APIs.

```bash
uv run 03_sync_and_query.py

# If the KB was already synced in a previous run:
uv run 03_sync_and_query.py --skip-sync

# Custom query:
uv run 03_sync_and_query.py --query "What is hybrid search?"
```

**Watch for:**
- Ingestion statistics: documents scanned vs indexed
- `retrieve()` output — same shape as FAISS results from Stage 1
- `retrieve_and_generate()` output — answer + citations with S3 URIs
- `overrideSearchType: "HYBRID"` — combining dense + sparse retrieval

---

### 04 — Compare Approaches (`~3 min`)

Runs the same questions through both Stage 1 (FAISS) and Stage 2 (Bedrock KB) and renders side-by-side panels.

```bash
uv run 04_compare_approaches.py
```

**Watch for:**
- Cases where hybrid search finds chunks that FAISS dense search misses
- Citation differences — KB gives S3 URIs, FAISS gives chunk metadata
- Latency differences (managed vs local)

---

## Architecture Deep Dive

### What the KB Chunking Does vs Stage 1

| Stage 1 (manual) | Stage 2 (KB config) |
|---|---|
| `chunk_text()` in Python | `FIXED_SIZE` chunking strategy |
| 400 chars, 80 char overlap | 300 tokens, 20% overlap |
| Character-boundary splitting | Token-boundary splitting |
| Triggered by running script | Triggered by ingestion job |

### The Hybrid Search Advantage

When you query "Lambda cold start performance":
- **Dense (Stage 1)**: Finds chunks about Lambda initialization by semantic similarity
- **Hybrid (Stage 2)**: Also finds chunks containing the exact phrase "cold start" via BM25

Hybrid search consistently outperforms dense-only on queries with specific technical terms.

### Citations

The `retrieve_and_generate()` API returns citations — mappings from answer sentences back to the source S3 objects. This enables:
- Audit trails for compliance
- "Show your sources" UI components
- Debugging hallucinations

---

## Cleanup

**Run this when done** to stop OpenSearch Serverless charges:

```bash
uv run cleanup.py
```

---

## Troubleshooting

**`ValidationException: Collection already exists`**
The OpenSearch collection from a previous run wasn't deleted. Run `cleanup.py` first.

**Ingestion job shows 0 documents indexed**
Check the IAM role has S3 read access for your bucket. The bucket prefix in `dataSourceConfiguration` must match where your files actually are.

**`retrieve_and_generate` returns empty citations**
The KB sync may be incomplete. Re-run `03_sync_and_query.py` without `--skip-sync`.
