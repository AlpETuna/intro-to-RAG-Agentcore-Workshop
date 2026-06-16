# Stage 0 — Setup & Prerequisites

**Time:** 15 minutes

---

## What You'll Do

1. Install the AWS CLI
2. Log in with `aws sso login`
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

The recommended approach is AWS IAM Identity Center (SSO). If your account is already configured for SSO:

```bash
# If you have an SSO profile already set up:
aws sso login --profile <your-profile-name>

# Or let the prerequisite script trigger it for you:
python 00_check_prerequisites.py --login
```

**First time setting up SSO?** Run the guided setup:
```bash
aws configure sso
# SSO session name: workshop
# SSO start URL:    https://<your-org>.awsapps.com/start
# SSO region:       us-east-1
# Follow the browser prompt to authenticate
```

**Alternative — static access keys** (if your organisation uses IAM users):
```bash
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region name:   us-east-1
# Default output format: json
```

Verify the login worked:
```bash
aws sts get-caller-identity
# {
#   "Account": "123456789012",
#   "Arn": "arn:aws:sts::123456789012:assumed-role/..."
# }
```

---

### 3. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

### 4. Install Python dependencies

```bash
# From the repo root
pip install -r requirements.txt

# Stage 1 deps (install now so they're ready)
pip install -r stage1-basic-rag/requirements.txt
```

---

### 5. Enable Bedrock model access

Open [Bedrock Console → Model access](https://us-east-1.console.aws.amazon.com/bedrock/home#/modelaccess) and enable:
- **Amazon → Titan Text Embeddings V2**
- **Anthropic → Claude 3 Haiku**
- **Anthropic → Claude 3.5 Sonnet v2**

Model access propagates within 1–2 minutes.

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
python 00_check_prerequisites.py
```

If credentials are missing, the script detects your SSO profile and prints the exact login command. You can also pass `--login` to trigger it automatically:

```bash
python 00_check_prerequisites.py --login
```

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
│ Python >= 3.10                       │  PASS  │ Python 3.12.0             │
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

**`AWS login` check fails — "Token expired"**
Your SSO session has expired. Re-run: `aws sso login --profile <your-profile>` or `python 00_check_prerequisites.py --login`

**`AWS login` check fails — "No credentials"**
You haven't logged in yet. Follow Step 2 above. If you don't have SSO configured, run `aws configure` with static access keys.

**`AWS region configured` is FAIL**
Add `AWS_DEFAULT_REGION=us-east-1` to your `.env` file, or set it in your SSO profile (`~/.aws/config`).

**Model status is `FAIL` after enabling access**
Wait 2 minutes and re-run. Model access propagation can take a moment.

**`ModuleNotFoundError: No module named 'faiss'`**
Run `pip install faiss-cpu` — this is normal if you haven't installed Stage 1 deps yet.
