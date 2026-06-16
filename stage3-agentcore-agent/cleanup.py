#!/usr/bin/env python3
"""
Stage 3 — Cleanup

Tears down everything Stage 3 created:
  1. `agentcore destroy` — deletes the AgentCore Runtime and the ECR repository
     the CLI created (run from the agent/ directory, uses .bedrock_agentcore.yaml)
  2. Deletes the IAM execution role we made in 01_setup_iam.py
  3. Clears the Stage 3 values from .env

Usage:
    python cleanup.py
    python cleanup.py --yes
"""

import argparse
import os
import subprocess
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel

ENV_FILE = Path(__file__).parent.parent / ".env"
AGENT_DIR = Path(__file__).parent / "agent"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    return {
        "runtime_arn": os.getenv("AGENTCORE_RUNTIME_ARN"),
        "role_arn": os.getenv("AGENTCORE_EXECUTION_ROLE_ARN"),
    }


def destroy_with_cli() -> bool:
    """Run `agentcore destroy` in the agent dir. Returns True if it ran."""
    if not (AGENT_DIR / ".bedrock_agentcore.yaml").exists():
        console.print("[dim]No .bedrock_agentcore.yaml — skipping agentcore destroy[/dim]")
        return False
    try:
        result = subprocess.run(["agentcore", "destroy"], cwd=str(AGENT_DIR))
        if result.returncode == 0:
            console.print("  [green]✓ agentcore destroy completed[/green]")
            return True
        console.print(f"  [yellow]agentcore destroy exited {result.returncode}[/yellow]")
    except FileNotFoundError:
        console.print("[yellow]`agentcore` CLI not found — install Stage 3 deps to use it.[/yellow]")
    return False


def delete_runtime_fallback(runtime_arn: str):
    """If the CLI couldn't run, delete the runtime via the control-plane API."""
    if not runtime_arn:
        return
    runtime_id = runtime_arn.split("/")[-1]
    try:
        client = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)
        client.delete_agent_runtime(agentRuntimeId=runtime_id)
        console.print(f"  [green]✓ Deleted AgentCore Runtime:[/green] {runtime_id}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceNotFoundException",):
            console.print("  [dim]Runtime already gone[/dim]")
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
            console.print("  [dim]Role already gone[/dim]")
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
        f"  • AgentCore Runtime + ECR repo (via agentcore destroy)\n"
        f"  • IAM Role:  {(config['role_arn'] or '').split('/')[-1] or '(not set)'}\n",
        border_style="red",
    ))

    if not args.yes:
        confirm = console.input("Type [bold]yes[/bold] to confirm: ").strip().lower()
        if confirm != "yes":
            console.print("[dim]Aborted.[/dim]")
            return

    if not destroy_with_cli():
        delete_runtime_fallback(config["runtime_arn"])

    iam = boto3.client("iam", region_name=AWS_REGION)
    delete_iam_role(iam, config["role_arn"])
    clear_env()

    console.print()
    console.print("[green]Stage 3 cleanup complete.[/green]")


if __name__ == "__main__":
    main()
