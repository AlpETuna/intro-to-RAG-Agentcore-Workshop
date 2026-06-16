#!/usr/bin/env python3
"""
Stage 4 — Cleanup

Deletes Stage 4 resources:
  - AgentCore Memory
  - AgentCore Gateway + tool targets
  - Lambda function (KB tool wrapper)
  - IAM role for Lambda
  - CloudWatch dashboard

Usage:
    uv run cleanup.py
    uv run cleanup.py --yes
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
LAMBDA_NAME = "rag-workshop-kb-tool"
GATEWAY_NAME = "rag-workshop-gateway"

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    return {
        "memory_id": os.getenv("AGENTCORE_MEMORY_ID"),
        "gateway_id": os.getenv("AGENTCORE_GATEWAY_ID"),
    }


def delete_memory(client, memory_id: str):
    if not memory_id:
        console.print("[dim]No AGENTCORE_MEMORY_ID — skipping[/dim]")
        return
    try:
        client.delete_memory(memoryId=memory_id)
        console.print(f"  [green]✓ Deleted memory:[/green] {memory_id}")
    except ClientError as e:
        console.print(f"  [dim]Memory: {e.response['Error']['Code']}[/dim]")


def delete_gateway(client, gateway_id: str):
    if not gateway_id:
        console.print("[dim]No AGENTCORE_GATEWAY_ID — skipping[/dim]")
        return
    try:
        # Delete targets first
        targets = client.list_gateway_targets(gatewayIdentifier=gateway_id).get("items", [])
        for t in targets:
            client.delete_gateway_target(gatewayIdentifier=gateway_id, targetId=t["targetId"])
        client.delete_gateway(gatewayIdentifier=gateway_id)
        console.print(f"  [green]✓ Deleted Gateway:[/green] {gateway_id}")
    except ClientError as e:
        console.print(f"  [dim]Gateway: {e.response['Error']['Code']}[/dim]")


def delete_lambda(lambda_client):
    try:
        lambda_client.delete_function(FunctionName=LAMBDA_NAME)
        console.print(f"  [green]✓ Deleted Lambda:[/green] {LAMBDA_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            console.print(f"  [dim]Lambda already gone[/dim]")
        else:
            console.print(f"  [yellow]Lambda error:[/yellow] {e}")


def delete_iam_role(iam, role_name: str):
    try:
        for p in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=p)
        iam.delete_role(RoleName=role_name)
        console.print(f"  [green]✓ Deleted IAM role:[/green] {role_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            console.print(f"  [dim]Role already gone: {role_name}[/dim]")
        else:
            console.print(f"  [yellow]IAM error:[/yellow] {e}")


def delete_dashboard(cw):
    try:
        cw.delete_dashboards(DashboardNames=["RAGWorkshopAgent"])
        console.print(f"  [green]✓ Deleted CloudWatch dashboard[/green]")
    except ClientError:
        console.print(f"  [dim]Dashboard not found[/dim]")


def clear_env():
    env_path = str(ENV_FILE)
    for key in ["AGENTCORE_MEMORY_ID", "AGENTCORE_GATEWAY_ID"]:
        set_key(env_path, key, "")
    console.print("  [green]✓ Cleared Stage 4 values from .env[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    config = load_config()

    console.print()
    console.print(Panel(
        "[bold red]Stage 4 Cleanup[/bold red]\n\n"
        f"  Memory:    {config['memory_id'] or '(not set)'}\n"
        f"  Gateway:   {config['gateway_id'] or '(not set)'}\n"
        f"  Lambda:    {LAMBDA_NAME}\n"
        f"  IAM Roles: RAGWorkshopLambdaRole, RAGWorkshopGatewayRole\n"
        f"  Dashboard: RAGWorkshopAgent\n",
        border_style="red",
    ))

    if not args.yes:
        confirm = console.input("Type [bold]yes[/bold] to confirm: ").strip().lower()
        if confirm != "yes":
            console.print("[dim]Aborted.[/dim]")
            return

    agentcore = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)
    cw = boto3.client("cloudwatch", region_name=AWS_REGION)

    delete_memory(agentcore, config["memory_id"])
    delete_gateway(agentcore, config["gateway_id"])
    delete_lambda(lambda_client)
    delete_iam_role(iam, "RAGWorkshopLambdaRole")
    delete_iam_role(iam, "RAGWorkshopGatewayRole")
    delete_dashboard(cw)
    clear_env()

    console.print()
    console.print("[green]Stage 4 cleanup complete.[/green]")
    console.print("\nRemember to also run cleanup in stages 2 and 3 if you haven't:")
    console.print("  [bold]cd ../stage2-bedrock-kb && uv run cleanup.py[/bold]")
    console.print("  [bold]cd ../stage3-agentcore-agent && uv run cleanup.py[/bold]")


if __name__ == "__main__":
    main()
