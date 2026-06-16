FROM python:3.12-slim

# Security: run as non-root
RUN adduser --disabled-password --gecos "" agentuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY agent.py .

# AgentCore Runtime expects the container to serve HTTP on port 8080
# The bedrock-agentcore SDK handles this automatically when the app is started
EXPOSE 8080

USER agentuser

# The bedrock-agentcore package wraps agent.py and starts an HTTP server
# Environment variables KNOWLEDGE_BASE_ID and AWS_REGION are injected at runtime
CMD ["python", "-m", "bedrock_agentcore.runtime", "--handler", "agent:handle_request", "--port", "8080"]
