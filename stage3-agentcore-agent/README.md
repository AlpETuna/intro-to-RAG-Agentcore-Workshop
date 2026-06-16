# Stage 3 — AgentCore Agent

**Time:** 25 minutes | **Cost:** ~$0.10/hr (AgentCore Runtime + ECR)

---

## What You'll Build

A Strands agent deployed to AgentCore Runtime using the **`agentcore` CLI** (the `bedrock-agentcore-starter-toolkit`). The CLI builds the container, pushes it to ECR, and creates the runtime for you — no Dockerfile and no local Docker required. The agent autonomously decides when to call the knowledge base, maintains conversation state across turns, and runs at scale without infrastructure management.

```
You (HTTPS) → AgentCore Runtime → agent.py (Strands Agent)
                                       ↓ tool call
                                  Bedrock KB (retrieve)
                                       ↓ tool call  
                                  Bedrock Claude (generate)
```

---

## Prerequisites

- Stage 2 complete (`KNOWLEDGE_BASE_ID` in `.env`)
- AWS credentials with ECR, CodeBuild, and AgentCore permissions
- Docker is **optional** — only needed if you deploy with `--local-build` instead of CodeBuild

---

## Install Dependencies

```bash
cd stage3-agentcore-agent
uv sync
```

This installs the `agentcore` CLI (`bedrock-agentcore-starter-toolkit`). Verify it (the CLI has no `--version` flag, so use `--help`):

```bash
uv run agentcore --help
```

---

## Scripts (run in order)

### 01 — Setup IAM (`~2 min`)

Creates the IAM execution role AgentCore Runtime assumes. (The ECR repository is created for you later by `agentcore deploy` — no need to make one here.)

```bash
uv run 01_setup_iam.py
```

**Watch for:**
- The trust policy: `bedrock-agentcore.amazonaws.com` is the principal (not `lambda` or `ecs`)
- The inline policy: Bedrock model + KB access + ECR pull + CloudWatch logs
- No `AmazonBedrockFullAccess` — least-privilege, scoped permissions only

---

### 02 — Configure and Deploy (`~5 min`)

Drives the `agentcore` CLI: `agentcore configure` generates `agent/.bedrock_agentcore.yaml`, then `agentcore deploy` builds the image with CodeBuild, pushes to ECR, and creates the runtime. The script then reads the runtime ARN/ID back into `.env`.

```bash
uv run 02_deploy_agent.py

# Build locally with Docker instead of CodeBuild:
uv run 02_deploy_agent.py --local-build
```

Equivalent manual commands (run inside `agent/`):

```bash
agentcore configure --entrypoint agent.py --name rag_workshop_agent \
    --region us-east-1 --execution-role <AGENTCORE_EXECUTION_ROLE_ARN> --disable-memory
agentcore deploy
agentcore invoke '{"prompt": "What is RAG?"}'
```

**Watch for:**
- `agentcore configure` writes `agent/.bedrock_agentcore.yaml` — the deployment config
- `agentcore deploy` runs an AWS CodeBuild job (no local Docker) — it prints a log link
- Runtime provisioning: typically 2-4 minutes for cold start
- The runtime gets a public HTTPS endpoint managed by AgentCore

---

### 03 — Chat with Agent (`open-ended`)

Sends messages to the deployed runtime and renders responses. Uses `AGENTCORE_RUNTIME_ARN` from `.env` and sends a `{"prompt": ...}` payload.

```bash
# Interactive mode
uv run 03_chat_with_agent.py

# Pre-scripted demo (good for presentations)
uv run 03_chat_with_agent.py --demo
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

The agent is wrapped with `BedrockAgentCoreApp`:

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    user_message = payload.get("prompt", "")
    return {"result": str(agent(user_message))}

if __name__ == "__main__":
    app.run()
```

The `agentcore` CLI detects `@app.entrypoint` / `app.run()` and generates the container automatically — that's why there's no Dockerfile. Run it locally with `uv run agent/agent.py` and POST to `http://localhost:8080/invocations`.

Dependencies live in `agent/pyproject.toml`; `agent/requirements.txt` is the export the CLI's container build consumes (regenerate with `uv export --no-hashes -o requirements.txt`).

---

## What AgentCore Runtime Adds vs Running Locally

| Feature | Local (`uv run agent/agent.py`) | AgentCore Runtime |
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
uv run cleanup.py
```

Runs `agentcore destroy` (removes the AgentCore Runtime and the ECR repository the CLI created), then deletes the IAM execution role and clears the Stage 3 values from `.env`.
