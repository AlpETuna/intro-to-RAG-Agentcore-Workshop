#!/usr/bin/env python3
"""
Stage 3 — Cleanup

Deletes AgentCore Runtime, ECR repository, and IAM role.

Usage:
    python cleanup.py
    python cleanup.py --yes
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ECR_REPO_NAME = "rag-workshop-agent"

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    return {
        "runtime_id": os.getenv("AGENTCORE_RUNTIME_ID"),
        "role_arn": os.getenv("AGENTCORE_EXECUTION_ROLE_ARN"),
        "repo_uri": os.getenv("ECR_REPO_URI"),
    }


def delete_runtime(client, runtime_id: str):
    if not runtime_id:
        console.print("[dim]No AGENTCORE_RUNTIME_ID — skipping[/dim]")
        return
    try:
        client.delete_agent_runtime(agentRuntimeId=runtime_id)
        console.print(f"  [green]✓ Deleted AgentCore Runtime:[/green] {runtime_id}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceNotFoundException",):
            console.print(f"  [dim]Runtime already gone[/dim]")
        else:
            console.print(f"  [yellow]Error:[/yellow] {e}")


def delete_ecr_repo(ecr):
    try:
        ecr.delete_repository(repositoryName=ECR_REPO_NAME, force=True)
        console.print(f"  [green]✓ Deleted ECR repository:[/green] {ECR_REPO_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("RepositoryNotFoundException",):
            console.print(f"  [dim]ECR repo already gone[/dim]")
        else:
            console.print(f"  [yellow]Error:[/yellow] {e}")


def delete_iam_role(iam, role_arn: str):
    if not role_arn:
        return
    role_name = role_arn.split("/")[-1]
    try:
        for policy in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy)
        iam.delete_role(RoleName=role_name)
        console.print(f"  [green]✓ Deleted IAM role:[/green] {role_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            console.print(f"  [dim]Role already gone[/dim]")
        else:
            console.print(f"  [yellow]Error:[/yellow] {e}")


def clear_env():
    env_path = str(ENV_FILE)
    for key in ["AGENTCORE_RUNTIME_ID", "AGENTCORE_RUNTIME_ARN", "AGENTCORE_ENDPOINT",
                 "AGENTCORE_EXECUTION_ROLE_ARN", "ECR_REPO_URI"]:
        set_key(env_path, key, "")
    console.print("  [green]✓ Cleared Stage 3 values from .env[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    config = load_config()

    console.print()
    console.print(Panel(
        "[bold red]Stage 3 Cleanup[/bold red]\n\n"
        "Resources to delete:\n\n"
        f"  • AgentCore Runtime:  {config['runtime_id'] or '(not set)'}\n"
        f"  • ECR Repository:     {ECR_REPO_NAME}\n"
        f"  • IAM Role:           {(config['role_arn'] or '').split('/')[-1] or '(not set)'}\n",
        border_style="red",
    ))

    if not args.yes:
        confirm = console.input("Type [bold]yes[/bold] to confirm: ").strip().lower()
        if confirm != "yes":
            console.print("[dim]Aborted.[/dim]")
            return

    agentcore = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
    ecr = boto3.client("ecr", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)

    delete_runtime(agentcore, config["runtime_id"])
    delete_ecr_repo(ecr)
    delete_iam_role(iam, config["role_arn"])
    clear_env()

    console.print()
    console.print("[green]Stage 3 cleanup complete.[/green]")


if __name__ == "__main__":
    main()
