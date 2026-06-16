#!/usr/bin/env python3
"""
Stage 3, Script 1 — Setup IAM for AgentCore

Creates the IAM execution role that the AgentCore Runtime assumes when
running your agent container. This role needs:
  - Bedrock invocation permissions (for the LLM and KB)
  - ECR pull permissions (to pull the agent container image)
  - CloudWatch Logs permissions (for runtime logging)

Also creates an ECR repository for the agent image.

Usage:
    python 01_setup_iam.py
"""

import json
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ECR_REPO_NAME = "rag-workshop-agent"

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    kb_id = os.getenv("KNOWLEDGE_BASE_ID")
    if not kb_id:
        console.print("[yellow]Note: KNOWLEDGE_BASE_ID not set — Stage 2 KB will not be available in Stage 3.[/yellow]")
    return kb_id


def get_account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def create_agentcore_execution_role(iam, account_id: str) -> str:
    role_name = "AgentCoreRAGWorkshopRole"
    console.print(f"\n[bold]Creating AgentCore execution role:[/bold] {role_name}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account_id}
                },
            }
        ],
    }

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "BedrockAccess",
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{AWS_REGION}::foundation-model/*",
                    f"arn:aws:bedrock:{AWS_REGION}:*:inference-profile/*",
                ],
            },
            {
                "Sid": "BedrockKBAccess",
                "Effect": "Allow",
                "Action": [
                    "bedrock:Retrieve",
                    "bedrock:RetrieveAndGenerate",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{AWS_REGION}:{account_id}:knowledge-base/*"
                ],
            },
            {
                "Sid": "ECRAccess",
                "Effect": "Allow",
                "Action": [
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                    "ecr:GetAuthorizationToken",
                ],
                "Resource": "*",
            },
            {
                "Sid": "CloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                "Resource": [
                    f"arn:aws:logs:{AWS_REGION}:{account_id}:log-group:/aws/bedrock-agentcore/*"
                ],
            },
        ],
    }

    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="AgentCore Runtime execution role for RAG workshop",
        )
        role_arn = role["Role"]["Arn"]
        console.print(f"  [green]✓ Created:[/green] {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            console.print(f"  [yellow]Already exists:[/yellow] {role_arn}")
        else:
            raise

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="AgentCoreRAGPolicy",
        PolicyDocument=json.dumps(inline_policy),
    )
    console.print("  [green]✓ Inline policy attached[/green]")

    console.print("  [dim]Waiting 15s for IAM propagation...[/dim]")
    time.sleep(15)
    return role_arn


def create_ecr_repo(ecr, account_id: str) -> str:
    console.print(f"\n[bold]Creating ECR repository:[/bold] {ECR_REPO_NAME}")
    try:
        response = ecr.create_repository(
            repositoryName=ECR_REPO_NAME,
            imageScanningConfiguration={"scanOnPush": True},
            encryptionConfiguration={"encryptionType": "AES256"},
        )
        repo_uri = response["repository"]["repositoryUri"]
        console.print(f"  [green]✓ Created:[/green] {repo_uri}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryAlreadyExistsException":
            repo_uri = ecr.describe_repositories(
                repositoryNames=[ECR_REPO_NAME]
            )["repositories"][0]["repositoryUri"]
            console.print(f"  [yellow]Already exists:[/yellow] {repo_uri}")
        else:
            raise
    return repo_uri


def save_env(role_arn: str, repo_uri: str):
    env_path = str(ENV_FILE)
    if not Path(ENV_FILE).exists():
        Path(ENV_FILE).write_text("")
    set_key(env_path, "AGENTCORE_EXECUTION_ROLE_ARN", role_arn)
    set_key(env_path, "ECR_REPO_URI", repo_uri)
    console.print(f"\n  [green]✓ Saved to .env:[/green] AGENTCORE_EXECUTION_ROLE_ARN, ECR_REPO_URI")


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 3 — Setup IAM & ECR[/bold cyan]\n"
        "[dim]Execution role + container registry for AgentCore Runtime[/dim]",
        border_style="cyan",
    ))

    load_config()

    sts = boto3.client("sts", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)
    ecr = boto3.client("ecr", region_name=AWS_REGION)

    account_id = get_account_id(sts)
    console.print(f"\n  AWS Account: [bold]{account_id}[/bold]")
    console.print(f"  Region:      [bold]{AWS_REGION}[/bold]")

    # Explain the architecture
    console.print()
    console.print(Panel(
        "[bold]AgentCore Runtime Architecture:[/bold]\n\n"
        "  Your agent code (agent.py) runs in a Docker container.\n"
        "  AgentCore Runtime manages the container lifecycle:\n\n"
        "  You          →  AgentCore Runtime  →  Your Container\n"
        "  (HTTPS)         (scales, isolates)     (agent.py)\n"
        "                                              ↓\n"
        "                                       Bedrock KB (search)\n"
        "                                       Bedrock Claude (generate)\n\n"
        "  The execution role grants the container permission to call Bedrock.",
        border_style="blue",
    ))

    role_arn = create_agentcore_execution_role(iam, account_id)
    repo_uri = create_ecr_repo(ecr, account_id)
    save_env(role_arn, repo_uri)

    table = Table("Resource", "Value", header_style="bold magenta")
    table.add_row("Execution Role", role_arn)
    table.add_row("ECR Repository", repo_uri)
    console.print()
    console.print(table)

    console.print()
    console.print(Panel(
        "[green]IAM and ECR ready![/green]\n\n"
        "Next step — build and push the agent Docker image:\n\n"
        "  [bold]python 02_deploy_agent.py[/bold]\n\n"
        "[dim]This step requires Docker to be running.[/dim]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
