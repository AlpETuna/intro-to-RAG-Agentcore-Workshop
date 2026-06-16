# Stage 3 — AgentCore Agent

**Time:** 25 minutes | **Cost:** ~$0.10/hr (AgentCore Runtime + ECR)

---

## What You'll Build

A Strands agent containerized and deployed to AgentCore Runtime. The agent autonomously decides when to call the knowledge base, maintains conversation state across turns, and runs at scale without infrastructure management.

```
You (HTTPS) → AgentCore Runtime → agent.py (Strands Agent)
                                       ↓ tool call
                                  Bedrock KB (retrieve)
                                       ↓ tool call  
                                  Bedrock Claude (generate)
```

---

## Prerequisites

- Docker Desktop running (`docker info` should succeed)
- Stage 2 complete (KNOWLEDGE_BASE_ID in `.env`)
- AWS credentials with ECR and AgentCore permissions

---

## Install Dependencies

```bash
cd stage3-agentcore-agent
pip install -r requirements.txt
```

---

## Scripts (run in order)

### 01 — Setup IAM (`~2 min`)

Creates the IAM execution role for AgentCore Runtime and an ECR repository for your container image.

```bash
python 01_setup_iam.py
```

**Watch for:**
- The trust policy: `bedrock-agentcore.amazonaws.com` is the principal (not `lambda` or `ecs`)
- The inline policy: Bedrock model + KB access + ECR pull + CloudWatch logs
- No `AmazonBedrockFullAccess` — least-privilege, scoped permissions only

---

### 02 — Build and Deploy (`~8 min`)

Builds the Docker image, pushes to ECR, creates the AgentCore Runtime, and waits for it to become READY.

```bash
python 02_deploy_agent.py

# If image already pushed (re-deploy only):
python 02_deploy_agent.py --skip-build
```

**Watch for:**
- Docker image size (strands-agents + boto3 = ~200MB compressed)
- ECR push layers — Docker layer caching means re-deploys are fast
- Runtime provisioning: typically 2-4 minutes for cold start
- `networkConfiguration: PUBLIC` — the runtime gets a public HTTPS endpoint

---

### 03 — Chat with Agent (`open-ended`)

Sends messages to the deployed runtime and renders responses.

```bash
# Interactive mode
python 03_chat_with_agent.py

# Pre-scripted demo (good for presentations)
python 03_chat_with_agent.py --demo
```

**Watch for (demo mode):**
- Turn 1: Agent uses `search_knowledge_base` tool before answering
- Turn 2: Follow-up uses session context without re-retrieving
- Latency: first call ~5-10s (container warm-up), subsequent calls ~2-4s

**Try in interactive mode:**
```
You: What is RAG?
You: What chunking strategies are available?
You: Summarize the AgentCore Gateway for me
You: How is what you just described different from API Gateway?
```

---

## The Agent Code

`agent/agent.py` defines two tools:

```python
@tool
def search_knowledge_base(query: str, num_results: int = 5) -> str:
    # Calls bedrock-agent-runtime.retrieve() with HYBRID search
    # Returns formatted chunks with source attribution

@tool
def summarize_topic(topic: str) -> str:
    # Calls bedrock-agent-runtime.retrieve_and_generate()
    # Returns a structured overview
```

The Strands `@tool` decorator auto-generates the JSON schema from the docstring and type hints, making the tool available to the LLM.

---

## What AgentCore Runtime Adds vs Running Locally

| Feature | Local (`python agent.py`) | AgentCore Runtime |
|---|---|---|
| Scale | Single process | Auto-scales per session |
| Session isolation | Shared state | Each session isolated |
| Uptime | Manual | Managed |
| Auth | Your credentials | IAM execution role |
| Endpoint | localhost only | HTTPS, globally routable |
| Monitoring | stdout only | CloudWatch + X-Ray |
| Cost | EC2/compute always on | Pay per invocation |

---

## Cleanup

```bash
python cleanup.py
```

Deletes the AgentCore Runtime, ECR repository, and IAM role.
