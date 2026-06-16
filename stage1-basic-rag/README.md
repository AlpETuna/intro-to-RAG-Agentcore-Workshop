# Stage 1 — Basic RAG from Scratch

**Time:** 20 minutes | **Cost:** < $0.05 (Bedrock API calls only)

---

## What You'll Build

A complete RAG pipeline using only Python and Bedrock — no managed vector store, no orchestration framework. Every step is explicit and visible.

```
Documents → Chunking → Titan Embed V2 → FAISS Index
                                              ↓
User Query → Titan Embed V2 → Similarity Search → Top-K Chunks
                                                         ↓
                              Chunks + Query → Claude 3 Haiku → Answer
```

---

## Install Dependencies

```bash
cd stage1-basic-rag
pip install -r requirements.txt
```

---

## Scripts (run in order)

### 01 — Chunk and Embed (`~3 min`)

Loads all 5 documents from `stage0-setup/data/`, splits them into overlapping chunks, embeds each chunk with Titan Embed V2, and saves a FAISS index to disk.

```bash
python 01_chunk_and_embed.py
```

**Watch for:**
- Chunk counts per document (how many pieces each doc becomes)
- Sample chunk preview (see the overlap in action)
- Embedding call rate (Titan is fast — you'll see chunks/second)
- FAISS index size vs document size

**Key concepts introduced:**
- Fixed-size chunking with overlap
- Normalized embeddings (unit length → cosine similarity via dot product)
- FAISS `IndexFlatIP` — exact inner product search

---

### 02 — Retrieve (`~2 min`)

Runs 5 demo queries against the FAISS index and shows ranked results with similarity scores.

```bash
python 02_retrieve.py

# Or run a custom query
python 02_retrieve.py --query "What is hybrid search?" --top-k 5
```

**Watch for:**
- How scores differ between relevant and less-relevant chunks
- Same-document chunks appearing at different ranks
- How semantic similarity handles paraphrased questions

**Discussion prompt:** What score threshold would you use to decide "not enough information"?

---

### 03 — Full RAG Pipeline (`~5 min`)

Runs 5 test questions through the complete pipeline: embed → retrieve → prompt → generate. Prints every step.

```bash
python 03_rag_pipeline.py

# Show the full constructed prompt (important!)
python 03_rag_pipeline.py --show-prompt

# Single question
python 03_rag_pipeline.py --query "What is chunking?" --show-prompt
```

**Watch for:**
- The context block inserted into the prompt (see the [Context N | Source: ...] headers)
- How the model uses specific chunk content in its answer
- Token counts (input tokens = prompt length, output tokens = answer length)
- Latency breakdown (embedding is fast; generation dominates)

---

### 04 — Interactive Chat (`open-ended`)

Multi-turn RAG conversation. Each message triggers fresh retrieval. History is included in the prompt for follow-up questions.

```bash
python 04_interactive_chat.py

# Retrieve more chunks per turn
python 04_interactive_chat.py --top-k 5
```

**Try these conversation flows:**

Flow 1 — Follow-up questions:
```
You: What is RAG?
You: What are its main failure modes?
You: How do you fix the "lost in the middle" problem?
```

Flow 2 — Cross-document questions:
```
You: Compare FAISS and OpenSearch Serverless for vector search
You: Which one does Bedrock Knowledge Bases use?
```

Type `stats` after any answer to see which chunks were retrieved.

---

## What to Notice

| Behavior | Why it happens |
|---|---|
| Very short answers for vague questions | The system prompt says "be concise" — adjust it |
| Occasional hallucination | This system has no guardrails — Stage 2 adds them |
| Slow first embedding call | Bedrock cold start — subsequent calls are faster |
| Same chunk retrieved twice | Chunk boundaries create duplicates — deduplicate in production |

---

## Key Limitations (fixed in Stage 2)

- **Single machine only** — FAISS is in-memory, not distributed
- **No keyword search** — "Lambda cold start" doesn't match if the chunk says "initialization latency"
- **No metadata filtering** — can't filter by date, department, or version
- **Manual re-indexing** — add a document → re-run Script 1
- **No access control** — any question gets any document

Stage 2 uses Amazon Bedrock Knowledge Bases to solve all of these.
