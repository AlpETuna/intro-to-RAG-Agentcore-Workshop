#!/usr/bin/env python3
"""
Stage 3, Script 2 — Configure and Deploy the Agent with the agentcore CLI

Instead of hand-building a Docker image and calling the AgentCore control-plane
API directly, this uses the `agentcore` CLI (bedrock-agentcore-starter-toolkit):

  1. `agentcore configure` — generates agent/.bedrock_agentcore.yaml
  2. `agentcore deploy`    — builds the image (AWS CodeBuild), pushes to ECR,
                             and creates/updates the AgentCore Runtime
                             (this command was formerly called `launch`)
  3. Reads the runtime ARN/ID back out of .bedrock_agentcore.yaml into .env

No local Docker required — `agentcore deploy` builds with CodeBuild by default.

Usage:
    uv run 02_deploy_agent.py
    uv run 02_deploy_agent.py --local-build   (build locally with Docker instead of CodeBuild)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel

ENV_FILE = Path(__file__).parent.parent / ".env"
AGENT_DIR = Path(__file__).parent / "agent"
AGENT_CONFIG = AGENT_DIR / ".bedrock_agentcore.yaml"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AGENT_NAME = "rag_workshop_agent"

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    role_arn = os.getenv("AGENTCORE_EXECUTION_ROLE_ARN")
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "")
    if not role_arn:
        console.print("[red]Missing AGENTCORE_EXECUTION_ROLE_ARN. Run 01_setup_iam.py first.[/red]")
        raise SystemExit(1)
    return role_arn, kb_id


def ensure_agentcore_cli():
    try:
        subprocess.run(["agentcore", "--help"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print(
            "[red]`agentcore` CLI not found.[/red] Install Stage 3 deps:\n"
            "  [bold]uv sync[/bold]   (or: uv pip install bedrock-agentcore-starter-toolkit)"
        )
        raise SystemExit(1)


def run_cli(cmd: list[str], description: str):
    console.print(f"\n[bold]{description}[/bold]")
    console.print(f"  [dim]$ {' '.join(cmd)}[/dim]")
    # Stream output so the user sees the CodeBuild/launch progress live.
    result = subprocess.run(cmd, cwd=str(AGENT_DIR))
    if result.returncode != 0:
        console.print(f"  [red]Command failed (exit {result.returncode}).[/red]")
        raise SystemExit(1)


def configure(role_arn: str):
    cmd = [
        "agentcore", "configure",
        "--entrypoint", "agent.py",
        "--name", AGENT_NAME,
        "--region", AWS_REGION,
        "--execution-role", role_arn,
        "--requirements-file", "requirements.txt",
        # Stage 4 adds AgentCore Memory explicitly; keep it off for now.
        "--disable-memory",
    ]
    run_cli(cmd, "Step 1/2 — agentcore configure (generates .bedrock_agentcore.yaml)")


def launch(local_build: bool):
    cmd = ["agentcore", "deploy"]
    if local_build:
        cmd.append("--local-build")
    run_cli(cmd, "Step 2/2 — agentcore deploy (build → push → deploy → READY)")


def read_runtime_from_config() -> tuple[str, str]:
    """Pull the deployed runtime ARN/ID out of .bedrock_agentcore.yaml."""
    if not AGENT_CONFIG.exists():
        console.print(f"[red]{AGENT_CONFIG} not found — did configure run?[/red]")
        raise SystemExit(1)

    data = yaml.safe_load(AGENT_CONFIG.read_text()) or {}
    agents = data.get("agents", {})
    agent = agents.get(AGENT_NAME) or (next(iter(agents.values())) if agents else {})
    bac = (agent or {}).get("bedrock_agentcore", {}) or {}

    runtime_arn = bac.get("agent_arn", "")
    runtime_id = bac.get("agent_id", "")
    if not runtime_arn:
        console.print(
            "[yellow]Could not find agent_arn in .bedrock_agentcore.yaml.[/yellow]\n"
            "Run [bold]agentcore status[/bold] in the agent/ directory to inspect it."
        )
    return runtime_arn, runtime_id


def save_env(runtime_arn: str, runtime_id: str, kb_id: str):
    env_path = str(ENV_FILE)
    if runtime_arn:
        set_key(env_path, "AGENTCORE_RUNTIME_ARN", runtime_arn)
    if runtime_id:
        set_key(env_path, "AGENTCORE_RUNTIME_ID", runtime_id)
    # Re-record the KB id the agent should use (passed as a runtime env var below).
    console.print("\n  [green]✓ Saved runtime details to .env[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-build", action="store_true",
        help="Build the image locally with Docker instead of AWS CodeBuild",
    )
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 3 — Deploy Agent via agentcore CLI[/bold cyan]\n"
        "[dim]configure → launch → read runtime ARN[/dim]",
        border_style="cyan",
    ))

    role_arn, kb_id = load_config()
    ensure_agentcore_cli()

    if not kb_id:
        console.print("[yellow]Note: KNOWLEDGE_BASE_ID is empty — the agent's KB tools "
                      "will return an error until Stage 2 is complete.[/yellow]")

    # The agent reads KNOWLEDGE_BASE_ID and AWS_REGION from its environment.
    # `agentcore deploy` passes through env vars set in the current shell.
    os.environ["KNOWLEDGE_BASE_ID"] = kb_id
    os.environ.setdefault("AWS_REGION", AWS_REGION)

    configure(role_arn)
    launch(args.local_build)

    runtime_arn, runtime_id = read_runtime_from_config()
    save_env(runtime_arn, runtime_id, kb_id)

    console.print()
    console.print(Panel(
        "[green]Agent deployed to AgentCore Runtime![/green]\n\n"
        f"  Runtime ARN:  {runtime_arn or '(check: agentcore status)'}\n"
        f"  Runtime ID:   {runtime_id or '(check: agentcore status)'}\n\n"
        "What the agentcore CLI did for you:\n"
        "  ✓ Generated agent/.bedrock_agentcore.yaml (deployment config)\n"
        "  ✓ Built the container with CodeBuild and pushed to ECR\n"
        "  ✓ Created the AgentCore Runtime with your execution role\n\n"
        "Quick test:\n"
        "  [bold]cd agent && agentcore invoke '{\"prompt\": \"What is RAG?\"}'[/bold]\n\n"
        "Next step:\n"
        "  [bold]uv run 03_chat_with_agent.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
