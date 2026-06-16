# Stage 4 — Production Hardening

**Time:** 15 minutes | **Cost:** ~$0.05 (Lambda + CloudWatch)

---

## What You'll Add

Four production essentials that every real RAG deployment needs:

```
Stage 3 Agent
    + Memory        → agent remembers users across sessions
    + Observability → every step is traced in X-Ray + CloudWatch
    + Evaluation    → automated quality scoring (RAG Triad)
    + Gateway       → KB exposed as MCP tool for any agent
```

---

## Install Dependencies

```bash
cd stage4-production
pip install -r requirements.txt
```

---

## Scripts (run in any order — they're independent)

### 01 — AgentCore Memory (`~5 min`)

Creates an AgentCore Memory resource, populates it with a simulated conversation, triggers a memory extraction job, and demonstrates recall in a new session.

```bash
python 01_add_memory.py
```

**Watch for:**
- The extraction job processing the transcript into durable facts
- User preferences and project context being stored as memory entries
- A new session "recalling" facts from the prior session — no user re-explaining

**Key concept:** Extraction jobs run asynchronously. In production, schedule them nightly on completed sessions.

---

### 02 — Observability (`~3 min`)

Instruments the RAG pipeline with OpenTelemetry spans and shows what X-Ray traces look like. Optionally creates a CloudWatch dashboard.

```bash
python 02_observability.py

# Create a CloudWatch dashboard (requires Stage 3 runtime to be deployed)
python 02_observability.py --create-dashboard
```

**Watch for:**
- Span hierarchy: `rag.query` → `rag.embed_query` + `rag.retrieve` + `rag.generate`
- Attributes on each span (model ID, token counts, latency, similarity scores)
- The AgentCore auto-instrumentation diagram — what you get for free

**Key concept:** Instrument from day one. A production incident in an unobservable agent is impossible to debug.

---

### 03 — Evaluate RAG Quality (`~5 min`)

Runs 5 ground-truth questions through the RAG pipeline and scores them on the RAG Triad using Claude 3.5 Sonnet as the LLM judge.

```bash
# Evaluate Stage 1 FAISS pipeline
python 03_evaluate_rag.py --pipeline faiss

# Evaluate Stage 2 Bedrock KB pipeline
python 03_evaluate_rag.py --pipeline bedrock-kb
```

**Watch for:**
- Context Relevance: are the right chunks being retrieved?
- Faithfulness: is the answer grounded in context (no hallucination)?
- Answer Relevance: does the answer address what was asked?
- Element Coverage: does the answer contain the expected facts?
- The "worst result" panel — this is where to focus optimization

**Key concept:** Define a quality threshold (e.g., faithfulness > 0.80) and run this in your CI/CD pipeline. Fail the build if quality drops.

---

### 04 — Gateway Tool (`~5 min`)

Creates an AgentCore Gateway that exposes the Bedrock KB as an MCP-compatible tool. Any agent — regardless of framework — can now discover and call the KB through a standard interface.

```bash
python 04_gateway_tool.py
```

**Watch for:**
- The Lambda function wrapping the Bedrock KB retrieve API
- The Gateway registering the Lambda with an MCP-compatible tool schema
- The Gateway test call showing the tool in action

**Key concept:** Gateway + Policy = centralized control. Attach a Cedar policy to restrict which agents can search which topics.

---

## Production Checklist

Use this before going live:

```
[ ] Memory extraction jobs scheduled nightly
[ ] Observability spans cover every RAG step
[ ] Evaluation suite runs in CI/CD with quality thresholds
[ ] Gateway attached with Policy engine
[ ] Bedrock Guardrails added for content filtering
[ ] CloudWatch alarms on error rate, latency, and faithfulness
[ ] Session TTLs configured (avoid idle compute charges)
[ ] IAM roles follow least-privilege (no wildcard * actions)
[ ] S3 data source versioning enabled (rollback capability)
[ ] Cleanup scripts documented for cost control
```

---

## Cleanup

```bash
python cleanup.py
```

Then clean up stages 2 and 3:
```bash
cd ../stage2-bedrock-kb && python cleanup.py
cd ../stage3-agentcore-agent && python cleanup.py
```
