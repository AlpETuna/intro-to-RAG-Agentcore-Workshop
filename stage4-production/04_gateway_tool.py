#!/usr/bin/env python3
"""
Stage 4, Script 4 — AgentCore Gateway

Exposes the Bedrock Knowledge Base as an MCP-compatible tool via
AgentCore Gateway. Other agents can now discover and call your KB
without knowing it's backed by Bedrock — just an MCP tool endpoint.

Architecture:
  External Agent  →  AgentCore Gateway  →  Lambda Function  →  Bedrock KB
                      (MCP protocol)       (your wrapper)

Steps:
  1. Create a Lambda function that wraps the Bedrock KB retrieve API
  2. Create an AgentCore Gateway
  3. Register the Lambda as a Gateway tool target
  4. Test the tool via the Gateway API

Usage:
    python 04_gateway_tool.py
    python 04_gateway_tool.py --test-only  (skip creation, test existing gateway)
"""

import argparse
import base64
import json
import os
import time
import zipfile
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LAMBDA_NAME = "rag-workshop-kb-tool"
GATEWAY_NAME = "rag-workshop-gateway"

console = Console()

LAMBDA_CODE = '''
import json
import os
import boto3

KB_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")

client = boto3.client("bedrock-agent-runtime", region_name=REGION)

def handler(event, context):
    """
    MCP tool wrapper for Bedrock Knowledge Base retrieval.
    Accepts: {"query": str, "num_results": int}
    Returns: {"results": [...], "query": str}
    """
    body = event.get("body") or "{}"
    if isinstance(body, str):
        body = json.loads(body)

    query = body.get("query") or event.get("query", "")
    num_results = int(body.get("num_results", 5))

    if not query:
        return {"statusCode": 400, "body": json.dumps({"error": "query is required"})}

    if not KB_ID:
        return {"statusCode": 500, "body": json.dumps({"error": "KNOWLEDGE_BASE_ID not configured"})}

    try:
        response = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": min(num_results, 10),
                    "overrideSearchType": "HYBRID",
                }
            },
        )
        results = []
        for r in response.get("retrievalResults", []):
            uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
            results.append({
                "text": r.get("content", {}).get("text", ""),
                "source": uri.split("/")[-1] if uri else "unknown",
                "score": r.get("score", 0),
            })
        return {
            "statusCode": 200,
            "body": json.dumps({"results": results, "query": query, "count": len(results)}),
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
'''


def load_config():
    load_dotenv(ENV_FILE)
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "")
    if not kb_id:
        console.print("[yellow]KNOWLEDGE_BASE_ID not set — Gateway tool will work but return empty results.[/yellow]")
    return kb_id


def get_account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def create_lambda_zip() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", LAMBDA_CODE)
    return buf.getvalue()


def create_lambda_role(iam, account_id: str) -> str:
    role_name = "RAGWorkshopLambdaRole"
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    }
    inline = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
                "Resource": f"arn:aws:bedrock:{AWS_REGION}:{account_id}:knowledge-base/*",
            },
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": f"arn:aws:logs:{AWS_REGION}:{account_id}:log-group:/aws/lambda/{LAMBDA_NAME}:*",
            },
        ],
    }
    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
        )
        role_arn = role["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        else:
            raise
    iam.put_role_policy(RoleName=role_name, PolicyName="KBAccess", PolicyDocument=json.dumps(inline))
    time.sleep(10)
    return role_arn


def deploy_lambda(lambda_client, role_arn: str, kb_id: str) -> str:
    console.print(f"\n[bold]Deploying Lambda function:[/bold] {LAMBDA_NAME}")
    zip_bytes = create_lambda_zip()
    try:
        response = lambda_client.create_function(
            FunctionName=LAMBDA_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.handler",
            Code={"ZipFile": zip_bytes},
            Description="AgentCore Gateway tool: KB retrieval wrapper",
            Timeout=30,
            MemorySize=256,
            Environment={"Variables": {"KNOWLEDGE_BASE_ID": kb_id, "AWS_REGION": AWS_REGION}},
        )
        arn = response["FunctionArn"]
        console.print(f"  [green]✓ Created:[/green] {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            resp = lambda_client.update_function_code(FunctionName=LAMBDA_NAME, ZipFile=zip_bytes)
            arn = resp["FunctionArn"]
            console.print(f"  [yellow]Updated existing function:[/yellow] {arn}")
        else:
            raise
    # Wait for function to be active
    time.sleep(5)
    return arn


def create_gateway(agentcore_client, account_id: str, lambda_arn: str) -> dict:
    console.print(f"\n[bold]Creating AgentCore Gateway:[/bold] {GATEWAY_NAME}")
    console.print("[dim]  Gateway exposes your Lambda as an MCP-compatible tool.[/dim]")

    try:
        response = agentcore_client.create_gateway(
            name=GATEWAY_NAME,
            description="RAG Workshop — Knowledge Base search tool",
            roleArn=f"arn:aws:iam::{account_id}:role/AgentCoreRAGWorkshopRole",
        )
        gateway = response["gateway"]
        gateway_id = gateway["gatewayId"]
        console.print(f"  [green]✓ Gateway ID:[/green] {gateway_id}")

        # Register the Lambda as a tool target
        agentcore_client.create_gateway_target(
            gatewayId=gateway_id,
            name="knowledge-base-search",
            description="Search the workshop knowledge base via Bedrock KB",
            targetConfiguration={
                "lambda": {
                    "lambdaArn": lambda_arn,
                    "toolSchema": {
                        "name": "search_knowledge_base",
                        "description": (
                            "Search the workshop knowledge base for information about "
                            "RAG, Bedrock, AgentCore, vector databases, and serverless AWS."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query"},
                                "num_results": {
                                    "type": "integer",
                                    "description": "Number of results (1-10)",
                                    "default": 5,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            },
        )
        console.print(f"  [green]✓ Tool target registered:[/green] search_knowledge_base")
        return gateway
    except ClientError as e:
        if "already exists" in str(e).lower():
            gateways = agentcore_client.list_gateways()["gateways"]
            for gw in gateways:
                if gw.get("name") == GATEWAY_NAME:
                    console.print(f"  [yellow]Already exists:[/yellow] {gw['gatewayId']}")
                    return gw
        console.print(f"  [yellow]Gateway creation note:[/yellow] {e}")
        return {}


def test_gateway_tool(agentcore_client, gateway_id: str):
    console.print(f"\n[bold]Testing Gateway tool[/bold]")
    query = "What is HNSW in vector databases?"
    console.print(f"  Query: [cyan]{query}[/cyan]")

    try:
        response = agentcore_client.invoke_gateway(
            gatewayId=gateway_id,
            toolName="search_knowledge_base",
            toolInput=json.dumps({"query": query, "num_results": 3}),
        )
        result = json.loads(response.get("toolResult", "{}"))
        results = result.get("results", [])

        console.print(f"\n  [green]✓ Gateway returned {len(results)} results[/green]")
        for i, r in enumerate(results[:2]):
            console.print(Panel(
                f"[dim]Source: {r.get('source')} | Score: {r.get('score', 0):.3f}[/dim]\n\n"
                + r.get("text", "")[:300] + "…",
                title=f"Result {i+1}",
                border_style="green" if i == 0 else "dim",
            ))
    except Exception as e:
        console.print(f"  [yellow]Gateway test note:[/yellow] {e}")
        console.print("  [dim](Gateway may take a moment to become fully active)[/dim]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Test existing gateway without creating")
    args = parser.parse_args()

    kb_id = load_config()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 4 — AgentCore Gateway[/bold cyan]\n"
        "[dim]Expose the KB as an MCP tool for any agent to discover and use[/dim]",
        border_style="cyan",
    ))

    console.print()
    console.print(Panel(
        "[bold]Why AgentCore Gateway?[/bold]\n\n"
        "  Without Gateway:\n"
        "    Each agent needs custom boto3 code to call Bedrock KB\n"
        "    No centralized auth, logging, or policy enforcement\n"
        "    Tool changes require updating every agent\n\n"
        "  With Gateway:\n"
        "    KB is an MCP-standard tool — any framework can call it\n"
        "    Centralized auth and audit logging\n"
        "    Attach a Policy engine to control what agents can search\n"
        "    Tool schema auto-discovered by agents\n\n"
        "  This is the production pattern for multi-agent RAG systems.",
        border_style="blue",
    ))

    sts = boto3.client("sts", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    agentcore = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)

    account_id = get_account_id(sts)

    gateway_id = os.getenv("AGENTCORE_GATEWAY_ID")

    if not args.test_only:
        role_arn = create_lambda_role(iam, account_id)
        lambda_arn = deploy_lambda(lambda_client, role_arn, kb_id)
        gateway = create_gateway(agentcore, account_id, lambda_arn)
        gateway_id = gateway.get("gatewayId", "")
        if gateway_id:
            set_key(str(ENV_FILE), "AGENTCORE_GATEWAY_ID", gateway_id)

    if gateway_id:
        test_gateway_tool(agentcore, gateway_id)

    console.print()
    console.print(Panel(
        "[green]Workshop Complete![/green]\n\n"
        "You've built the full stack:\n\n"
        "  Stage 1: DIY RAG — Titan Embed + FAISS + Claude (raw Python)\n"
        "  Stage 2: Managed RAG — Bedrock Knowledge Base + OpenSearch Serverless\n"
        "  Stage 3: Agentic RAG — Strands Agent on AgentCore Runtime\n"
        "  Stage 4: Production — Memory + Observability + Evaluation + Gateway\n\n"
        "Cleanup commands:\n"
        "  [bold]cd ../stage2-bedrock-kb && python cleanup.py[/bold]\n"
        "  [bold]cd ../stage3-agentcore-agent && python cleanup.py[/bold]\n"
        "  [bold]cd ../stage4-production && python cleanup.py[/bold]",
        title="Congratulations",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
