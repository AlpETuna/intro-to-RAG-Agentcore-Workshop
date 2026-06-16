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
    uv run 01_add_memory.py
"""

import os
import time
import uuid
from pathlib import Path

from bedrock_agentcore.memory import MemoryClient
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

MEMORY_NAME = "rag_workshop_memory"        # must match [a-zA-Z][a-zA-Z0-9_]*
ACTOR_ID = "workshop-user"                 # the "who" memories are scoped to
STRATEGY_NAMESPACE = f"/users/{ACTOR_ID}"  # where the semantic strategy stores facts

console = Console()

# (role, text) — sent to create_event as (text, role) tuples with UPPERCASE roles.
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


def create_memory(client: MemoryClient) -> str:
    """Create (or reuse) a Memory with a semantic long-term strategy.

    create_memory_and_wait blocks until the resource is ACTIVE (a few minutes).
    """
    console.print("\n[bold]Creating AgentCore Memory resource...[/bold]")
    console.print("[dim]  Waiting for it to become ACTIVE (this can take a few minutes)...[/dim]")

    strategies = [{
        "semanticMemoryStrategy": {
            "name": "userPreferences",
            "description": "User preferences, technical background, and project context",
            "namespaces": [STRATEGY_NAMESPACE],
        }
    }]

    try:
        memory = client.create_memory_and_wait(
            name=MEMORY_NAME,
            strategies=strategies,
            description="Long-term memory for the RAG workshop agent",
            event_expiry_days=30,
            max_wait=300,
            poll_interval=10,
        )
        memory_id = memory.get("id") or memory.get("memoryId")
        console.print(f"  [green]✓ Memory ID:[/green] {memory_id}")
        return memory_id
    except Exception as e:
        # If a memory with this name already exists, reuse it.
        if "already" in str(e).lower() or "conflict" in str(e).lower() or "exist" in str(e).lower():
            for m in client.list_memories():
                if m.get("name") == MEMORY_NAME or m.get("id", "").startswith(MEMORY_NAME):
                    mid = m.get("id") or m.get("memoryId")
                    console.print(f"  [yellow]Already exists:[/yellow] {mid}")
                    return mid
        raise


def save_conversation(client: MemoryClient, memory_id: str, session_id: str, conversation: list):
    """Store a multi-turn conversation as a single event in the session."""
    console.print(f"\n[bold]Saving conversation to memory[/bold] (session={session_id})")

    messages = [(text, role.upper()) for role, text in conversation]
    client.create_event(
        memory_id=memory_id,
        actor_id=ACTOR_ID,
        session_id=session_id,
        messages=messages,
    )
    for role, text in conversation:
        console.print(f"  [dim]{role[:9]}:[/dim] {text[:80]}")


def wait_for_extraction(client: MemoryClient, memory_id: str, timeout: int = 150):
    """Long-term extraction runs asynchronously after events are stored.

    There is no manual 'extraction job' to trigger — the semantic strategy
    extracts facts in the background. Poll until records appear.
    """
    console.print(f"\n[bold]Waiting for long-term extraction...[/bold]")
    console.print("[dim]  The semantic strategy extracts facts from the transcript in the background.[/dim]")
    start = time.time()
    while time.time() - start < timeout:
        records = client.retrieve_memories(
            memory_id=memory_id,
            namespace=STRATEGY_NAMESPACE,
            query="user preferences and project context",
            actor_id=ACTOR_ID,
            top_k=10,
        )
        if records:
            console.print(f"  [green]✓ {len(records)} memory record(s) extracted[/green]")
            return records
        console.print(f"  [dim]Status: extracting... ({int(time.time()-start)}s)[/dim]", end="\r")
        time.sleep(10)
    console.print("\n  [yellow]No records yet — extraction can take a few minutes. Try recall again later.[/yellow]")
    return []


def _record_text(record: dict) -> str:
    content = record.get("content", record)
    if isinstance(content, dict):
        return content.get("text", "") or str(content)
    return str(content)


def display_memory_records(records: list):
    console.print(f"\n[bold]Extracted memory records:[/bold]")
    if not records:
        console.print("  [dim]No records yet (extraction may still be processing)[/dim]")
        return
    table = Table("Content", "Score", show_header=True, header_style="bold magenta")
    for r in records[:10]:
        score = r.get("score", 0) or 0
        table.add_row(_record_text(r)[:90], f"{score:.3f}")
    console.print(table)


def demonstrate_recall(client: MemoryClient, memory_id: str):
    console.print(f"\n[bold]Retrieving memory for a NEW session[/bold]")
    console.print("[dim]  Simulates: a user returns the next day and the agent remembers them.[/dim]")

    recalled = client.retrieve_memories(
        memory_id=memory_id,
        namespace=STRATEGY_NAMESPACE,
        query="What do you know about this user and their project?",
        actor_id=ACTOR_ID,
        top_k=3,
    )
    if recalled:
        console.print()
        for r in recalled[:3]:
            score = r.get("score", 0) or 0
            console.print(Panel(
                _record_text(r),
                title=f"[green]Recalled[/green] (score={score:.3f})",
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

    memory_client = MemoryClient(region_name=AWS_REGION)

    memory_id = create_memory(memory_client)
    set_key(str(ENV_FILE), "AGENTCORE_MEMORY_ID", memory_id)

    session_id = f"workshop-session-{uuid.uuid4().hex[:8]}"
    save_conversation(memory_client, memory_id, session_id, SESSION_A_CONVERSATION)
    records = wait_for_extraction(memory_client, memory_id)
    display_memory_records(records)
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
        "  [bold]uv run 02_observability.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
