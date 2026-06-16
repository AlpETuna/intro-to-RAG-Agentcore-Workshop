# Intro to RAG with Amazon Bedrock AgentCore

**Duration:** 1.5 hours | **Level:** Intermediate | **AWS Services:** Bedrock, AgentCore, S3, IAM, ECR

Build a production-ready RAG system from scratch — starting with raw embeddings and ending with a deployed AgentCore agent with memory and observability.

---

## Workshop Map

```
Stage 0 — Setup              (15 min)   AWS credentials, model access, env check
    │
Stage 1 — Basic RAG          (20 min)   Titan embeddings + FAISS + Claude (all local)
    │
Stage 2 — Bedrock KB         (25 min)   Managed knowledge base, S3 data source, hybrid search
    │
Stage 3 — AgentCore Agent    (25 min)   Strands agent → ECR → AgentCore Runtime
    │
Stage 4 — Production         (15 min)   Memory + Observability + Evaluation + Gateway
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Managed by uv |
| uv | latest | Package/venv manager — install instructions in Stage 0 |
| AWS CLI | v2 | Required for login — install instructions in Stage 0 |
| Docker | 24+ | Optional — only for `agentcore deploy --local-build` (Stage 3 uses CodeBuild by default) |
| AWS Region | `us-east-1` | Recommended for model availability |

### AWS Login

Log in to AWS with:

```bash
aws login
```

Make sure your default region is `us-east-1`. Verify it worked:

```bash
aws sts get-caller-identity
```

### Required AWS Permissions

Your IAM role/user needs:
- `AmazonBedrockFullAccess`
- `AmazonS3FullAccess`
- `IAMFullAccess` (for creating KB service roles)
- `AmazonOpenSearchServiceFullAccess` (Stage 2)
- `AmazonECR_FullAccess` (Stage 3)

### Bedrock Model Access

As of late 2025, the **Model access** page is retired — serverless foundation models are **automatically enabled** in every account/region. Access is now governed by IAM (`bedrock:InvokeModel`), which `AmazonBedrockFullAccess` already grants.

The workshop uses:
- `Amazon → Titan Text Embeddings V2` — ready to use, no action needed
- `Anthropic → Claude Haiku 4.5`
- `Anthropic → Claude Sonnet 4.6`

**Anthropic models require a one-time usage form before first use.** Submit it once by opening the [Bedrock Chat / Playground](https://us-east-1.console.aws.amazon.com/bedrock/home#/chat-playground), selecting a Claude model, and completing the short form it prompts — access is granted immediately.

---

## Quick Start

```bash
# 1. Install uv and AWS CLI v2 (see stage0-setup/README.md for OS-specific instructions)
curl -LsSf https://astral.sh/uv/install.sh | sh
aws --version   # should show aws-cli/2.x.x

# 2. Log in to AWS
aws login

# 3. Clone the repo
git clone <this-repo>
cd intro-to-RAG-Agentcore-Workshop

# 4. Copy config file
cp .env.example .env   # Windows: copy .env.example .env

# 5. Install Stage 0 deps and run the prerequisite check
cd stage0-setup
uv sync
uv run 00_check_prerequisites.py

# 6. Work through each stage in order (each is its own uv project)
cd ../stage1-basic-rag && uv sync && uv run 01_chunk_and_embed.py
```

---

## Stage Overview

### Stage 0 — Setup & Prerequisites
Check your environment, verify AWS credentials, confirm Bedrock model access, and understand the sample dataset we'll use throughout the workshop.

**You build:** Nothing yet — just make sure everything works.

### Stage 1 — Basic RAG from Scratch
No managed services. You embed documents yourself with Amazon Titan Embed, store them in a local FAISS index, retrieve with cosine similarity, and generate answers with Claude 3 Haiku. Every step is visible.

**You build:** A working RAG pipeline in ~100 lines of Python.

### Stage 2 — Bedrock Knowledge Base
Replace the DIY pipeline with Amazon Bedrock Knowledge Bases — a fully managed service that handles chunking, embedding, indexing (OpenSearch Serverless), and retrieval. Compare results with Stage 1.

**You build:** An S3-backed knowledge base with hybrid search and metadata filtering.

### Stage 3 — AgentCore Agent
Wrap your RAG into a Strands agent, containerize it, push to ECR, and deploy to AgentCore Runtime. Your agent now has session isolation, auto-scaling, and WebSocket streaming.

**You build:** A deployed, invokable RAG agent on AgentCore.

### Stage 4 — Production
Add AgentCore Memory (conversation continuity + long-term learning), instrument with OpenTelemetry for traces in CloudWatch X-Ray, run a RAG evaluation suite, and expose the KB as a Gateway MCP tool.

**You build:** A production-hardened agent with full observability.

---

## Cost Estimate

| Stage | Services | Est. Cost |
|---|---|---|
| Stage 0–1 | Bedrock model invocations only | < $0.10 |
| Stage 2 | + OpenSearch Serverless (2 OCU) | ~$0.70/hr |
| Stage 3 | + ECR storage + AgentCore Runtime | ~$0.10/hr |
| Stage 4 | + CloudWatch logs | ~$0.05/hr |

**Run `cleanup.py` in each stage when done to avoid ongoing charges.**

---

## Dataset

Five documents in `stage0-setup/data/` are used throughout the workshop:

| File | Topic |
|---|---|
| `rag_fundamentals.txt` | How RAG works — concepts and patterns |
| `bedrock_models.txt` | Bedrock model families and selection guide |
| `aws_agentcore.txt` | AgentCore services overview |
| `vector_databases.txt` | Vector database concepts and trade-offs |
| `serverless_aws.txt` | AWS serverless architecture patterns |

---

## Troubleshooting

**`AccessDeniedException` on Bedrock:** For Anthropic (Claude) models, submit the one-time usage form via the Bedrock Chat / Playground. Otherwise, check that your IAM role has `bedrock:InvokeModel`. (The old Model access page is retired — there's nothing to "enable" there anymore.)

**`ResourceNotFoundException` in Stage 2:** The KB sync may still be in progress — run `03_sync_and_query.py` with the `--wait` flag.

**Stage 3 deploy fails:** `agentcore deploy` builds with AWS CodeBuild (no local Docker needed) — check the CodeBuild log link it prints. To build locally instead, run `agentcore deploy --local-build` (or `uv run 02_deploy_agent.py --local-build`), which does require Docker Desktop running.

**`ModuleNotFoundError`:** Each stage has its own `pyproject.toml` — run `uv sync` inside that stage's folder, and run scripts with `uv run <script>.py`.
