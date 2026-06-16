#!/usr/bin/env python3
"""
Stage 4, Script 1 — AgentCore Memory

Demonstrates AgentCore Memory — the service that gives your agent
conversation continuity and long-term learning.

Two memory types:
  Short-term: conversation history within a session (built-in to Runtime)
  Long-term:  facts extracted from past sessions and recalled in new ones

This script:
  1. Creates an AgentCore Memory resource
  2. Runs a multi-turn conversation that populates memory
  3. Starts a NEW session and shows the agent recalling prior context
  4. Shows the extracted memory entries

Usage:
    python 01_add_memory.py
"""

import json
import os
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

SESSION_A_CONVERSATION = [
    ("user", "I'm a Python developer and I prefer concise answers with code examples."),
    ("assistant", "Got it — Python examples, concise format. What would you like to know?"),
    ("user", "What is the Titan Embed V2 model ID in Bedrock?"),
    ("assistant", "The model ID is `amazon.titan-embed-text-v2:0`. It produces 1024-dimensional normalized vectors."),
    ("user", "And I'm building a RAG system for legal documents — long, dense PDFs."),
    ("assistant", "For legal PDFs I'd recommend larger chunks (600-800 tokens) with semantic chunking to preserve clause boundaries."),
]


def load_config():
    load_dotenv(ENV_FILE)
    runtime_id = os.getenv("AGENTCORE_RUNTIME_ID")
    if not runtime_id:
        console.print("[yellow]Note: AGENTCORE_RUNTIME_ID not set — memory demo will run without live agent.[/yellow]")
    return runtime_id


def create_memory(client) -> str:
    console.print("\n[bold]Creating AgentCore Memory resource...[/bold]")

    try:
        response = client.create_memory(
            name="rag-workshop-memory",
            description="Long-term memory for the RAG workshop agent",
            memoryConfiguration={
                "strategies": [
                    {
                        "type": "SEMANTIC",
                        "semanticMemoryStrategy": {
                            "name": "user-preferences",
                            "description": "Stores user preferences, technical background, and project context",
                        },
                    }
                ]
            },
        )
        memory_id = response["memory"]["memoryId"]
        console.print(f"  [green]✓ Memory ID:[/green] {memory_id}")
        return memory_id
    except ClientError as e:
        if "already exists" in str(e).lower():
            existing = client.list_memories()["memories"]
            for m in existing:
                if m.get("name") == "rag-workshop-memory":
                    mid = m["memoryId"]
                    console.print(f"  [yellow]Already exists:[/yellow] {mid}")
                    return mid
        raise


def save_memory_to_session(client, memory_id: str, session_id: str, messages: list):
    console.print(f"\n[bold]Saving conversation to memory[/bold] (session={session_id})")

    client.create_memory_session(
        memoryId=memory_id,
        sessionId=session_id,
        memorySessionConfiguration={
            "maxCompletionMessages": 20,
        },
    )

    for msg in messages:
        client.put_memory_session_message(
            memoryId=memory_id,
            sessionId=session_id,
            messageContent={"role": msg[0], "content": msg[1]},
        )
        console.print(f"  [dim]{msg[0][:8]}:[/dim] {msg[1][:80]}")


def trigger_extraction(client, memory_id: str, session_id: str):
    console.print(f"\n[bold]Triggering memory extraction job...[/bold]")
    console.print("[dim]  This processes the session transcript and extracts memorable facts.[/dim]")

    response = client.create_memory_extraction_job(
        memoryId=memory_id,
        sessionId=session_id,
    )
    job_id = response["extractionJob"]["extractionJobId"]
    console.print(f"  [green]✓ Job started:[/green] {job_id}")

    start = time.time()
    while time.time() - start < 120:
        job = client.get_memory_extraction_job(
            memoryId=memory_id,
            extractionJobId=job_id,
        )["extractionJob"]
        status = job["status"]
        console.print(f"  [dim]Status: {status}[/dim]", end="\r")
        if status == "COMPLETED":
            console.print(f"\n  [green]✓ Extraction complete![/green]")
            return
        if status == "FAILED":
            console.print(f"\n  [red]Extraction failed[/red]")
            return
        time.sleep(5)
    console.print("\n  [yellow]Extraction still running (taking longer than expected)[/yellow]")


def list_memory_entries(client, memory_id: str):
    console.print(f"\n[bold]Extracted memory entries:[/bold]")

    response = client.list_memory_entries(memoryId=memory_id)
    entries = response.get("entries", [])

    if not entries:
        console.print("  [dim]No entries yet (extraction may still be processing)[/dim]")
        return

    table = Table("Type", "Content", "Confidence", show_header=True, header_style="bold magenta")
    for entry in entries[:10]:
        content = entry.get("content", {}).get("text", "")[:80]
        confidence = entry.get("confidence", 0)
        entry_type = entry.get("type", "FACT")
        table.add_row(entry_type, content, f"{confidence:.2f}")

    console.print(table)


def demonstrate_recall(client, memory_id: str):
    console.print(f"\n[bold]Retrieving memory for a NEW session[/bold]")
    console.print("[dim]  Simulates: a user returns the next day and the agent remembers them.[/dim]")

    new_session_id = f"new-session-{uuid.uuid4().hex[:8]}"

    response = client.retrieve_memory(
        memoryId=memory_id,
        sessionId=new_session_id,
        retrievalQuery={"text": "What do you know about this user?"},
    )

    recalled = response.get("retrievalResults", [])
    if recalled:
        console.print()
        for r in recalled[:3]:
            text = r.get("content", {}).get("text", "")
            score = r.get("score", 0)
            console.print(Panel(
                text,
                title=f"[green]Recalled[/green] (confidence={score:.3f})",
                border_style="green",
            ))
    else:
        console.print("  [dim]No memories recalled yet — extraction may still be processing.[/dim]")


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 4 — AgentCore Memory[/bold cyan]\n"
        "[dim]Short-term session memory + long-term fact extraction[/dim]",
        border_style="cyan",
    ))

    load_config()

    console.print()
    console.print(Panel(
        "[bold]Memory Architecture:[/bold]\n\n"
        "  Short-term memory (automatic):\n"
        "    • Conversation transcript within a session\n"
        "    • Managed by AgentCore Runtime automatically\n"
        "    • Lives for the session duration\n\n"
        "  Long-term memory (this script):\n"
        "    • Extracted facts from past sessions\n"
        "    • Persists across sessions indefinitely\n"
        "    • Recalled at the start of new sessions\n"
        "    • Example: user preferences, project context, past decisions",
        border_style="blue",
    ))

    memory_client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

    memory_id = create_memory(memory_client)
    set_key(str(ENV_FILE), "AGENTCORE_MEMORY_ID", memory_id)

    session_id = f"workshop-session-{uuid.uuid4().hex[:8]}"
    save_memory_to_session(memory_client, memory_id, session_id, SESSION_A_CONVERSATION)
    trigger_extraction(memory_client, memory_id, session_id)
    list_memory_entries(memory_client, memory_id)
    demonstrate_recall(memory_client, memory_id)

    console.print()
    console.print(Panel(
        "[green]Memory demo complete![/green]\n\n"
        "What AgentCore Memory enables:\n\n"
        "  Without memory → user must re-explain their context every session\n"
        "  With memory    → agent recalls preferences, past decisions, project context\n\n"
        "Production pattern:\n"
        "  1. Associate memory_id with the AgentCore Runtime\n"
        "  2. Run extraction jobs nightly on completed sessions\n"
        "  3. Agent automatically enriches its context from recalled memories\n\n"
        "Next step:\n"
        "  [bold]python 02_observability.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
