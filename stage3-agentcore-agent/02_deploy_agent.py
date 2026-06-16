#!/usr/bin/env python3
"""
Stage 3, Script 2 — Build, Push, and Deploy Agent

1. Builds the Docker image from agent/Dockerfile
2. Authenticates with ECR and pushes the image
3. Creates an AgentCore Runtime pointing at the ECR image
4. Waits for the runtime to become READY
5. Saves the runtime ID and ARN to .env

Prerequisites: Docker must be running.

Usage:
    python 02_deploy_agent.py
    python 02_deploy_agent.py --skip-build   (reuse existing ECR image)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ENV_FILE = Path(__file__).parent.parent / ".env"
AGENT_DIR = Path(__file__).parent / "agent"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
RUNTIME_NAME = "rag-workshop-agent"

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    repo_uri = os.getenv("ECR_REPO_URI")
    role_arn = os.getenv("AGENTCORE_EXECUTION_ROLE_ARN")
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "")

    if not repo_uri or not role_arn:
        console.print("[red]Missing ECR_REPO_URI or AGENTCORE_EXECUTION_ROLE_ARN. Run 01_setup_iam.py first.[/red]")
        raise SystemExit(1)
    return repo_uri, role_arn, kb_id


def run_cmd(cmd: list[str], description: str) -> subprocess.CompletedProcess:
    console.print(f"  [dim]$ {' '.join(cmd[:4])}{'...' if len(cmd) > 4 else ''}[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"  [red]Command failed:[/red] {result.stderr[:500]}")
        raise SystemExit(1)
    return result


def build_and_push(repo_uri: str, account_id: str) -> str:
    image_tag = f"{repo_uri}:latest"

    console.print(f"\n[bold]Step 1/3 — Building Docker image[/bold]")
    console.print(f"  From: {AGENT_DIR}")
    run_cmd(
        ["docker", "build", "-t", image_tag, str(AGENT_DIR)],
        "Building container",
    )
    console.print(f"  [green]✓ Image built:[/green] {image_tag}")

    console.print(f"\n[bold]Step 2/3 — Authenticating with ECR[/bold]")
    ecr = boto3.client("ecr", region_name=AWS_REGION)
    auth = ecr.get_authorization_token()
    token = auth["authorizationData"][0]["authorizationToken"]
    import base64
    username, password = base64.b64decode(token).decode().split(":", 1)
    registry = repo_uri.split("/")[0]

    run_cmd(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        "ECR login",
    )
    # Pass password via stdin
    result = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "Login Succeeded" not in result.stdout + result.stderr:
        console.print(f"  [yellow]ECR login note:[/yellow] {result.stderr[:200]}")
    console.print(f"  [green]✓ Authenticated with ECR[/green]")

    console.print(f"\n[bold]Step 3/3 — Pushing image to ECR[/bold]")
    run_cmd(["docker", "push", image_tag], "Pushing image")
    console.print(f"  [green]✓ Pushed:[/green] {image_tag}")

    return image_tag


def create_agentcore_runtime(
    bedrock_agentcore, repo_uri: str, role_arn: str, kb_id: str, account_id: str
) -> dict:
    image_tag = f"{repo_uri}:latest"
    console.print(f"\n[bold]Creating AgentCore Runtime[/bold]")
    console.print(f"  Name:      {RUNTIME_NAME}")
    console.print(f"  Image:     {image_tag}")
    console.print(f"  Role:      {role_arn}")

    try:
        response = bedrock_agentcore.create_agent_runtime(
            agentRuntimeName=RUNTIME_NAME,
            description="RAG Workshop agent with Bedrock KB search",
            agentRuntimeArtifact={
                "containerConfiguration": {
                    "containerUri": image_tag,
                }
            },
            roleArn=role_arn,
            networkConfiguration={"networkMode": "PUBLIC"},
            environmentVariables={
                "KNOWLEDGE_BASE_ID": kb_id,
                "AWS_REGION": AWS_REGION,
            },
            protocolConfiguration={"serverProtocol": "HTTP"},
        )
        runtime = response["agentRuntime"]
        console.print(f"  [green]✓ Runtime ID:[/green] {runtime['agentRuntimeId']}")
        console.print(f"  [green]✓ Status:[/green] {runtime['status']}")
        return runtime
    except ClientError as e:
        if "already exists" in str(e).lower():
            console.print("  [yellow]Runtime already exists — updating...[/yellow]")
            response = bedrock_agentcore.update_agent_runtime(
                agentRuntimeId=RUNTIME_NAME,
                agentRuntimeArtifact={
                    "containerConfiguration": {"containerUri": image_tag}
                },
                roleArn=role_arn,
                environmentVariables={
                    "KNOWLEDGE_BASE_ID": kb_id,
                    "AWS_REGION": AWS_REGION,
                },
            )
            return response["agentRuntime"]
        raise


def wait_for_runtime(bedrock_agentcore, runtime_id: str, timeout: int = 600):
    console.print(f"\n[bold]Waiting for runtime to become READY[/bold] (up to {timeout}s)...")
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Provisioning...", total=None)
        while time.time() - start < timeout:
            response = bedrock_agentcore.get_agent_runtime(agentRuntimeId=runtime_id)
            runtime = response["agentRuntime"]
            status = runtime["status"]
            progress.update(task, description=f"Status: {status}")
            if status == "READY":
                console.print(f"  [green]✓ Runtime is READY[/green]")
                return runtime
            if "FAILED" in status or "ERROR" in status:
                console.print(f"  [red]Runtime failed: {status}[/red]")
                console.print(f"  {runtime.get('statusReasons', [])}")
                raise SystemExit(1)
            time.sleep(15)
    raise TimeoutError("Runtime did not become READY in time")


def save_env(runtime_id: str, runtime_arn: str, endpoint: str = ""):
    env_path = str(ENV_FILE)
    set_key(env_path, "AGENTCORE_RUNTIME_ID", runtime_id)
    set_key(env_path, "AGENTCORE_RUNTIME_ARN", runtime_arn)
    if endpoint:
        set_key(env_path, "AGENTCORE_ENDPOINT", endpoint)
    console.print(f"\n  [green]✓ Saved to .env[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true", help="Skip Docker build/push")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 3 — Deploy Agent to AgentCore Runtime[/bold cyan]\n"
        "[dim]Build → Push → Deploy → Wait for READY[/dim]",
        border_style="cyan",
    ))

    repo_uri, role_arn, kb_id = load_config()

    sts = boto3.client("sts", region_name=AWS_REGION)
    account_id = sts.get_caller_identity()["Account"]

    bedrock_agentcore = boto3.client("bedrock-agentcore-control", region_name=AWS_REGION)

    if not args.skip_build:
        build_and_push(repo_uri, account_id)
    else:
        console.print("[dim]Skipping Docker build (--skip-build)[/dim]")

    runtime = create_agentcore_runtime(bedrock_agentcore, repo_uri, role_arn, kb_id, account_id)
    runtime_id = runtime["agentRuntimeId"]
    runtime_arn = runtime["agentRuntimeArn"]

    runtime = wait_for_runtime(bedrock_agentcore, runtime_id)

    endpoint = runtime.get("agentRuntimeEndpoint", "")
    save_env(runtime_id, runtime_arn, endpoint)

    console.print()
    console.print(Panel(
        "[green]Agent deployed to AgentCore Runtime![/green]\n\n"
        f"  Runtime ID:   {runtime_id}\n"
        f"  Runtime ARN:  {runtime_arn[:70]}…\n"
        f"  Endpoint:     {endpoint or '(check console)'}\n\n"
        "What AgentCore Runtime provides:\n"
        "  ✓ Session isolation (each user gets their own execution context)\n"
        "  ✓ Auto-scaling (new container per concurrent session)\n"
        "  ✓ Secure execution (IAM role, no shared state)\n"
        "  ✓ HTTPS endpoint (no API Gateway needed)\n\n"
        "Next step:\n"
        "  [bold]python 03_chat_with_agent.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
