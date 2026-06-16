"""
Stage 3 — RAG Agent (runs inside AgentCore Runtime)

This is the agent code that runs INSIDE the container the `agentcore` CLI
builds and deploys to AgentCore Runtime. It uses Strands Agents for the agent
framework and exposes two tools:
  - search_knowledge_base: retrieves chunks from the Bedrock KB
  - summarize_topic: retrieves and summarizes a topic in a structured format

`BedrockAgentCoreApp` wraps this agent to handle HTTP requests, session
management, and the runtime protocol. The `agentcore` CLI detects the
`@app.entrypoint` function and `app.run()` automatically — no Dockerfile needed.

Environment variables (injected by AgentCore Runtime):
  KNOWLEDGE_BASE_ID   — set in the runtime configuration
  AWS_REGION          — set in the runtime configuration
"""

import logging
import os

import boto3
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent, tool
from strands.models import BedrockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KB_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

bedrock_agent_rt = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


@tool
def search_knowledge_base(query: str, num_results: int = 5) -> str:
    """
    Search the workshop knowledge base for information relevant to the query.

    Use this tool whenever you need factual information about:
    - RAG concepts and patterns
    - Amazon Bedrock models and APIs
    - AgentCore services
    - Vector databases
    - AWS serverless architecture

    Args:
        query: The search query — make it specific and descriptive
        num_results: Number of chunks to retrieve (1-10, default 5)

    Returns:
        Formatted string with retrieved chunks and their source documents
    """
    if not KB_ID:
        return "Error: KNOWLEDGE_BASE_ID environment variable not set."

    try:
        response = bedrock_agent_rt.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": min(max(num_results, 1), 10),
                    "overrideSearchType": "HYBRID",
                }
            },
        )

        results = response.get("retrievalResults", [])
        if not results:
            return f"No results found for: {query}"

        parts = []
        for i, r in enumerate(results):
            score = r.get("score", 0)
            text = r.get("content", {}).get("text", "")
            uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
            source = uri.split("/")[-1] if uri else "unknown"
            parts.append(f"[Result {i+1} | Source: {source} | Score: {score:.3f}]\n{text}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.error(f"KB search error: {e}")
        return f"Search failed: {str(e)}"


@tool
def summarize_topic(topic: str) -> str:
    """
    Retrieve information about a topic and return it in a structured summary format.
    Good for "explain X" or "what is X" requests where a comprehensive overview is needed.

    Args:
        topic: The topic to summarize (e.g., "HNSW algorithm", "AgentCore Gateway")

    Returns:
        A structured summary with definition, key points, and use cases
    """
    if not KB_ID:
        return "Error: KNOWLEDGE_BASE_ID not set."

    try:
        response = bedrock_agent_rt.retrieve_and_generate(
            input={"text": f"Give a comprehensive overview of: {topic}"},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KB_ID,
                    "modelArn": (
                        f"arn:aws:bedrock:{AWS_REGION}::foundation-model/"
                        "anthropic.claude-3-haiku-20240307-v1:0"
                    ),
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5,
                            "overrideSearchType": "HYBRID",
                        }
                    },
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": (
                                "Based on the context below, provide a structured overview of: "
                                "{topic}\n\nContext:\n$search_results$\n\n"
                                "Format your response as:\n"
                                "**Definition:** one sentence\n"
                                "**Key Points:** bullet list\n"
                                "**When to Use:** brief guidance"
                            ).replace("{topic}", topic)
                        },
                        "inferenceConfig": {
                            "textInferenceConfig": {"maxTokens": 400, "temperature": 0.0}
                        },
                    },
                },
            },
        )
        return response.get("output", {}).get("text", "No summary generated.")
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"Summarize failed: {str(e)}"


SYSTEM_PROMPT = f"""You are a knowledgeable assistant for an AWS workshop on RAG and AgentCore.

You have access to a knowledge base containing information about:
- Retrieval-Augmented Generation (RAG) concepts and patterns
- Amazon Bedrock models, APIs, and services
- Amazon Bedrock AgentCore platform
- Vector databases and similarity search
- AWS serverless architecture

INSTRUCTIONS:
1. Always use search_knowledge_base before answering factual questions
2. Use summarize_topic for broad "explain X" or "what is X" questions
3. Be specific and cite which document your information comes from
4. If the knowledge base doesn't have the answer, say so clearly
5. You are running inside AgentCore Runtime — mention this context when relevant

Knowledge Base ID: {KB_ID}
Region: {AWS_REGION}"""


def create_agent() -> Agent:
    model = BedrockModel(
        model_id=AGENT_MODEL,
        region_name=AWS_REGION,
        max_tokens=1024,
        temperature=0.1,
    )
    return Agent(
        model=model,
        tools=[search_knowledge_base, summarize_topic],
        system_prompt=SYSTEM_PROMPT,
    )


app = BedrockAgentCoreApp()
agent = create_agent()


@app.entrypoint
def invoke(payload: dict) -> dict:
    """Entry point called by AgentCore Runtime for each invocation.

    The runtime delivers the request body as `payload`; the convention used
    by the agentcore CLI (`agentcore invoke '{"prompt": "..."}'`) is the
    `prompt` key. We also accept `inputText`/`input` for compatibility.
    """
    user_message = (
        payload.get("prompt")
        or payload.get("inputText")
        or payload.get("input")
        or ""
    )
    if not user_message:
        return {"error": "No input provided. Send a JSON body with a 'prompt' field."}

    logger.info(f"Processing: {user_message[:100]}")

    try:
        result = agent(user_message)
        return {"result": str(result)}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Starts the local HTTP server (also how the runtime launches the agent).
    # Test locally with:
    #   curl -X POST http://localhost:8080/invocations \
    #        -H 'Content-Type: application/json' -d '{"prompt": "What is RAG?"}'
    app.run()
