# Stage 0 — Setup & Prerequisites

**Time:** 15 minutes

---

## What You'll Do

1. Install the AWS CLI
2. Log in with `aws login`
3. Install Python dependencies
4. Verify Bedrock model access
5. Understand the five sample documents used throughout the workshop

---

## Steps

### 1. Install the AWS CLI

The AWS CLI is required for authentication. Install v2 for your OS:

**Windows**
```powershell
# Download and run the MSI installer:
# https://awscli.amazonaws.com/AWSCLIV2.msi

# Verify:
aws --version
# aws-cli/2.x.x ...
```

**macOS**
```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o AWSCLIV2.pkg
sudo installer -pkg AWSCLIV2.pkg -target /
aws --version
```

**Linux**
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install
aws --version
```

---

### 2. Log in to AWS

Log in to AWS with:

```bash
aws login
```

Make sure your default region is set to `us-east-1`. Verify the login worked:
```bash
aws sts get-caller-identity
# {
#   "Account": "123456789012",
#   "Arn": "arn:aws:sts::123456789012:assumed-role/..."
# }
```

---

### 3. Install uv

This workshop uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. Each stage is its own `pyproject.toml` project.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

uv creates and manages the virtual environment for you — no `python -m venv` needed.

---

### 4. Install Python dependencies

```bash
# Stage 0 deps (from this directory)
uv sync

# uv created a .venv here. Run scripts with `uv run` so they use it:
uv run 00_check_prerequisites.py
```

Each stage has its own `pyproject.toml`; `cd` into a stage and run `uv sync` before working in it.

---

### 5. Bedrock model access

The old **Model access** page is retired — serverless foundation models are now **automatically enabled** in every account/region. Access is controlled by IAM (`bedrock:InvokeModel`), which `AmazonBedrockFullAccess` grants.

The workshop uses:
- **Amazon → Titan Text Embeddings V2** — ready to use, no action needed
- **Anthropic → Claude Haiku 4.5**
- **Anthropic → Claude Sonnet 4.6**

**One thing to do:** Anthropic (Claude) models require a one-time usage form before first use. Open the [Bedrock Chat / Playground](https://us-east-1.console.aws.amazon.com/bedrock/home#/chat-playground), pick a Claude model, and complete the short form it prompts. Access is granted immediately.

---

### 6. Copy the env file

```bash
# From the repo root
cp .env.example .env   # Windows: copy .env.example .env
```

---

### 7. Run the prerequisite check

```bash
cd stage0-setup
uv run 00_check_prerequisites.py
```

If credentials are missing, the script prints the exact login command (`aws login`).

---

## Expected Output

```
╭──────────────────────────────────────────────╮
│  Intro to RAG with AgentCore                 │
│  Stage 0 — Prerequisites Check              │
╰──────────────────────────────────────────────╯

Environment Check
┌──────────────────────────────────────┬────────┬───────────────────────────┐
│ Check                                │ Status │ Detail                    │
├──────────────────────────────────────┼────────┼───────────────────────────┤
│ Python >= 3.11                       │  PASS  │ Python 3.12.0             │
│ boto3: AWS SDK                       │  PASS  │ installed                 │
│ rich: Terminal output                │  PASS  │ installed                 │
│ python-dotenv: config management     │  PASS  │ installed                 │
├──────────────────────────────────────┼────────┼───────────────────────────┤
│ AWS credentials                      │  PASS  │ Account 123456789012      │
│ AWS region configured                │  PASS  │ us-east-1                 │
├──────────────────────────────────────┼────────┼───────────────────────────┤
│ Model: amazon.titan-embed-text-v2:0  │  PASS  │ Stage 1–4 embeddings      │
│ Model: anthropic.claude-3-haiku-...  │  PASS  │ Stage 1 generation        │
│ Model: anthropic.claude-3-5-sonnet.. │  PASS  │ Stage 3–4 agent reasoning │
...
```

---

## The Workshop Dataset

Five documents live in `stage0-setup/data/`. They're used as the knowledge base for all stages.

| Document | What it teaches about RAG |
|---|---|
| `rag_fundamentals.txt` | Good for testing: "What is RAG?", "What is chunking?" |
| `bedrock_models.txt` | Good for testing: "Which model should I use for embeddings?" |
| `aws_agentcore.txt` | Good for testing: "What does AgentCore Gateway do?" |
| `vector_databases.txt` | Good for testing: "What is HNSW?", "What is hybrid search?" |
| `serverless_aws.txt` | Good for testing: "What is Lambda's maximum duration?" |

---

## Troubleshooting

**`AWS CLI v2` check fails**
The CLI is not installed or on PATH. Follow Step 1 above, then open a new terminal.

**`AWS login` check fails — "No credentials"**
You haven't logged in yet. Run `aws login` (Step 2 above).

**`AWS region configured` is FAIL**
Add `AWS_DEFAULT_REGION=us-east-1` to your `.env` file, or set your default region to `us-east-1`.

**Model status is `FAIL` for a Claude model**
You likely haven't submitted the one-time Anthropic usage form yet — open the Bedrock Chat / Playground, select a Claude model, complete the form, then re-run. For non-Anthropic models, a `FAIL` means your IAM role is missing `bedrock:InvokeModel`.

**`ModuleNotFoundError: No module named 'faiss'`**
This is normal if you haven't installed Stage 1 deps yet — run `uv sync` inside `stage1-basic-rag/`.
