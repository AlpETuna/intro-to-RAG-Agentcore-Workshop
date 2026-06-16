#!/usr/bin/env python3
"""
Stage 1, Script 4 — Interactive RAG Chat

A multi-turn chat interface that runs RAG on every user message.
Conversation history is included in each prompt so Claude can refer
back to previous turns — but retrieval is always fresh.

Type a question and press Enter. Type 'quit', 'exit', or Ctrl-C to stop.
Type 'stats' to see retrieval statistics for the last query.
Type 'clear' to reset conversation history.

Usage:
    python 04_interactive_chat.py
    python 04_interactive_chat.py --top-k 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import boto3
import faiss
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
EMBEDDING_DIM = 1024
MAX_HISTORY_TURNS = 6
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

SYSTEM_PROMPT = """You are a helpful technical assistant for this AWS workshop.
Answer questions based on the provided context snippets.
If the context is insufficient, say so rather than guessing.
Keep answers concise. You may refer to previous messages in the conversation."""


def load_index():
    if not INDEX_DIR.exists():
        console.print("[red]Index not found. Run 01_chunk_and_embed.py first.[/red]")
        raise SystemExit(1)
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def embed(client, text: str) -> np.ndarray:
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return np.array([json.loads(response["body"].read())["embedding"]], dtype="float32")


def retrieve(index, chunks, query_vec, top_k) -> list[dict]:
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
    return results


def build_rag_message(question: str, context_results: list[dict]) -> str:
    context_parts = []
    for i, r in enumerate(context_results):
        context_parts.append(
            f"[Source: {r['doc']} | similarity={r['score']:.3f}]\n{r['text']}"
        )
    context = "\n\n---\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(client, messages: list[dict]) -> tuple[str, dict]:
    t0 = time.time()
    response = client.invoke_model(
        modelId=GENERATION_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }),
        contentType="application/json",
        accept="application/json",
    )
    latency = time.time() - t0
    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]
    usage = result.get("usage", {})
    usage["latency_s"] = round(latency, 2)
    return answer, usage


def print_welcome():
    console.print()
    console.print(Panel(
        "[bold cyan]Stage 1 — Interactive RAG Chat[/bold cyan]\n\n"
        "Ask anything about the workshop topic:\n"
        "  RAG fundamentals, Bedrock models, AgentCore, vector databases, serverless\n\n"
        "[dim]Commands:[/dim]\n"
        "  [bold]quit[/bold] / [bold]exit[/bold] — stop\n"
        "  [bold]clear[/bold]         — reset conversation history\n"
        "  [bold]stats[/bold]         — show last retrieval details",
        border_style="cyan",
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print_welcome()

    index, chunks = load_index()
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    history: list[dict] = []
    last_retrieved: list[dict] = []
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

        if user_input.lower() == "clear":
            history.clear()
            turn = 0
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if user_input.lower() == "stats":
            if not last_retrieved:
                console.print("[dim]No retrieval yet.[/dim]")
            else:
                for i, r in enumerate(last_retrieved):
                    console.print(
                        f"  [{i+1}] [green]{r['score']:.4f}[/green] — "
                        f"{r['doc']} chunk {r['chunk_index']+1}"
                    )
            continue

        turn += 1

        # Embed + retrieve
        console.print(f"[dim]  Retrieving (top_k={args.top_k})...[/dim]", end="")
        query_vec = embed(bedrock, user_input)
        last_retrieved = retrieve(index, chunks, query_vec, args.top_k)
        sources = ", ".join(f"{r['doc']} ({r['score']:.3f})" for r in last_retrieved)
        console.print(f"\r[dim]  Retrieved: {sources}[/dim]")

        # Build RAG message
        rag_user_message = build_rag_message(user_input, last_retrieved)

        # Add to history (trim to last N turns)
        history.append({"role": "user", "content": rag_user_message})
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]

        # Generate
        answer, usage = generate(bedrock, history)
        history.append({"role": "assistant", "content": answer})

        console.print()
        console.print(Panel(
            answer,
            title=f"[green]Assistant[/green] [dim](turn {turn} | "
                  f"{usage.get('output_tokens','?')} tokens | {usage['latency_s']}s)[/dim]",
            border_style="green",
        ))


if __name__ == "__main__":
    main()
