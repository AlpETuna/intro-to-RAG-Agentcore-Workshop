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
| Python | 3.10+ | |
| AWS CLI | v2 | Required for login — install instructions in Stage 0 |
| Docker | 24+ | Stage 3 only |
| AWS Region | `us-east-1` | Recommended for model availability |

### AWS Login

This workshop uses **AWS IAM Identity Center (SSO)** for authentication — no static access keys needed.

```bash
# First time: set up SSO
aws configure sso

# Each session: log in
aws sso login --profile <your-profile>

# Or let Stage 0 handle it:
python stage0-setup/00_check_prerequisites.py --login
```

If your organisation uses IAM user access keys instead, `aws configure` also works.

### Required AWS Permissions

Your IAM role/user needs:
- `AmazonBedrockFullAccess`
- `AmazonS3FullAccess`
- `IAMFullAccess` (for creating KB service roles)
- `AmazonOpenSearchServiceFullAccess` (Stage 2)
- `AmazonECR_FullAccess` (Stage 3)

### Bedrock Model Access

Enable these models in the [Bedrock Console → Model access](https://us-east-1.console.aws.amazon.com/bedrock/home#/modelaccess):
- `Amazon → Titan Text Embeddings V2`
- `Anthropic → Claude 3 Haiku`
- `Anthropic → Claude 3.5 Sonnet v2`

---

## Quick Start

```bash
# 1. Install AWS CLI v2 (see stage0-setup/README.md for OS-specific instructions)
aws --version   # should show aws-cli/2.x.x

# 2. Log in to AWS
aws sso login --profile <your-profile>
# or: aws configure  (for static access keys)

# 3. Clone and install deps
git clone <this-repo>
cd intro-to-RAG-Agentcore-Workshop
pip install -r requirements.txt
pip install -r stage1-basic-rag/requirements.txt

# 4. Copy config file
cp .env.example .env   # Windows: copy .env.example .env

# 5. Run the prerequisite check (--login triggers aws sso login for you)
cd stage0-setup
python 00_check_prerequisites.py

# 6. Work through each stage in order
cd ../stage1-basic-rag && python 01_chunk_and_embed.py
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

**`AccessDeniedException` on Bedrock:** Enable model access in the console under Bedrock → Model access.

**`ResourceNotFoundException` in Stage 2:** The KB sync may still be in progress — run `03_sync_and_query.py` with the `--wait` flag.

**Docker build fails in Stage 3:** Ensure Docker Desktop is running and you have logged in via `aws ecr get-login-password`.

**`ModuleNotFoundError`:** Each stage has its own `requirements.txt` — run `pip install -r requirements.txt` inside that stage's folder.
