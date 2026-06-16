#!/usr/bin/env python3
"""
Stage 3, Script 3 — Chat with the AgentCore Agent

Sends messages to the deployed AgentCore Runtime and renders responses.
Each conversation turn uses a consistent session ID so the runtime
maintains state across turns.

Shows what's different from Stages 1 and 2:
  - The agent (not you) decides when to call the KB
  - The agent can use multiple tools in sequence
  - Session isolation: your conversation is separate from others
  - The agent reasons about when retrieval is necessary

Usage:
    python 03_chat_with_agent.py
    python 03_chat_with_agent.py --demo          (runs pre-set demo questions)
    python 03_chat_with_agent.py --session-id MY_SESSION
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

DEMO_SCRIPT = [
    "What is RAG and why does it exist?",
    "Based on what you just told me, what chunking strategy would you recommend for a document with lots of tables?",
    "What Bedrock model should I use for the embeddings and what dimension does it produce?",
    "How does AgentCore Gateway differ from what we built in Stage 1?",
]


def new_session_id() -> str:
    # AgentCore Runtime requires runtimeSessionId to be at least 33 characters.
    return f"workshop-{uuid.uuid4().hex}{uuid.uuid4().hex[:8]}"


def load_config():
    load_dotenv(ENV_FILE)
    runtime_arn = os.getenv("AGENTCORE_RUNTIME_ARN")
    if not runtime_arn:
        console.print("[red]AGENTCORE_RUNTIME_ARN not set. Run 02_deploy_agent.py first.[/red]")
        raise SystemExit(1)
    return runtime_arn


def invoke_agent(client, runtime_arn: str, session_id: str, message: str) -> str:
    """Invoke the AgentCore Runtime and return the response text."""
    payload = json.dumps({"prompt": message}).encode()

    t0 = time.time()
    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=session_id,
        payload=payload,
        contentType="application/json",
        accept="application/json",
    )
    latency = time.time() - t0

    # Response can be a stream or a direct body depending on configuration
    body = response.get("response") or response.get("body")
    if hasattr(body, "read"):
        raw = body.read()
    else:
        raw = body or b"{}"

    try:
        result = json.loads(raw)
        text = result.get("result") or result.get("response") or result.get("output") or str(result)
    except (json.JSONDecodeError, TypeError):
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    return text, latency


def run_interactive(client, runtime_arn: str, session_id: str):
    console.print()
    console.print(Panel(
        "[bold cyan]AgentCore RAG Agent — Interactive Chat[/bold cyan]\n\n"
        f"Runtime ARN: {runtime_arn}\n"
        f"Session ID:  {session_id}\n\n"
        "The agent will use its tools (search_knowledge_base, summarize_topic)\n"
        "autonomously. You don't need to trigger retrieval explicitly.\n\n"
        "[dim]Commands: 'quit' to exit, 'new' to start a fresh session[/dim]",
        border_style="cyan",
    ))

    turn = 0
    while True:
        console.print()
        try:
            user_input = console.input("[bold cyan]You → [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_input.lower() == "new":
            session_id = new_session_id()
            console.print(f"[dim]New session: {session_id}[/dim]")
            turn = 0
            continue

        turn += 1
        console.print(f"[dim]  Invoking AgentCore Runtime (session={session_id}, turn={turn})...[/dim]")

        try:
            response, latency = invoke_agent(client, runtime_arn, session_id, user_input)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            continue

        console.print()
        console.print(Panel(
            response,
            title=f"[green]Agent[/green] [dim](turn {turn} | {latency:.2f}s)[/dim]",
            border_style="green",
        ))


def run_demo(client, runtime_arn: str, session_id: str):
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AgentCore Agent — Demo Mode[/bold cyan]\n"
        "[dim]Pre-scripted conversation demonstrating tool use and memory[/dim]",
        border_style="cyan",
    ))
    console.print(f"\n  Runtime ARN: {runtime_arn}")
    console.print(f"  Session ID: {session_id}")
    console.print(f"  Questions:  {len(DEMO_SCRIPT)}")

    for i, question in enumerate(DEMO_SCRIPT):
        console.print()
        console.print(Rule(f"[bold]Turn {i+1}/{len(DEMO_SCRIPT)}[/bold]", style="cyan"))
        console.print(Panel(f"[bold cyan]{question}[/bold cyan]", title="Question", border_style="cyan"))

        console.print("[dim]  Invoking...[/dim]")
        try:
            response, latency = invoke_agent(client, runtime_arn, session_id, question)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            continue

        console.print()
        console.print(Panel(
            response,
            title=f"[green]Agent[/green] [dim]({latency:.2f}s)[/dim]",
            border_style="green",
        ))

        # Pause between turns for readability
        if i < len(DEMO_SCRIPT) - 1:
            console.input("\n[dim]Press Enter for next question...[/dim]")

    console.print()
    console.print(Panel(
        "[bold]What you just observed:[/bold]\n\n"
        "  Turn 1: Agent called search_knowledge_base to answer the RAG question\n"
        "  Turn 2: Follow-up — agent used session context (no re-retrieval needed)\n"
        "  Turn 3: Agent retrieved specific embedding model info from the KB\n"
        "  Turn 4: Agent connected concepts across documents\n\n"
        "[bold]Key differences from Stages 1 & 2:[/bold]\n\n"
        "  • YOU chose when to retrieve → AGENT decides when to retrieve\n"
        "  • Single-step pipeline → Multi-step reasoning with tool use\n"
        "  • Stateless → Stateful (session memory across turns)\n"
        "  • Local process → Cloud-deployed, auto-scaled runtime\n\n"
        "Next: Stage 4 adds Memory, Observability, and Evaluation.\n\n"
        "  [bold]cd ../stage4-production[/bold]",
        title="Summary",
        border_style="blue",
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run the pre-scripted demo")
    parser.add_argument("--session-id", type=str, default=None, help="Custom session ID")
    args = parser.parse_args()

    runtime_arn = load_config()
    session_id = args.session_id or new_session_id()

    client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    if args.demo:
        run_demo(client, runtime_arn, session_id)
    else:
        run_interactive(client, runtime_arn, session_id)


if __name__ == "__main__":
    main()
