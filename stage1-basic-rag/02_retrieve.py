#!/usr/bin/env python3
"""
Stage 1, Script 2 — Retrieve

Loads the FAISS index built in Script 1 and runs a set of demonstration
queries. For each query, it shows the top-k retrieved chunks with their
similarity scores and source documents — making the retrieval step visible.

What you'll see:
  - How query embedding maps to vector space
  - Similarity scores and what they mean
  - Which chunks are retrieved and why

Usage:
    python 02_retrieve.py
    python 02_retrieve.py --query "What is hybrid search?"
    python 02_retrieve.py --top-k 5
"""

import argparse
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

DEMO_QUERIES = [
    "What is retrieval-augmented generation and why does it exist?",
    "Which Amazon Bedrock model should I use for text embeddings?",
    "What does AgentCore Gateway do?",
    "How does HNSW work in vector databases?",
    "What is the maximum duration for an AWS Lambda function?",
]


def load_index() -> tuple[faiss.Index, list[dict]]:
    if not INDEX_DIR.exists():
        console.print("[red]Index not found. Run 01_chunk_and_embed.py first.[/red]")
        raise SystemExit(1)
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def embed_query(client, query: str) -> np.ndarray:
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({
            "inputText": query,
            "dimensions": EMBEDDING_DIM,
            "normalize": True,
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return np.array([result["embedding"]], dtype="float32")


def retrieve(index: faiss.Index, chunks: list[dict], query_vec: np.ndarray, top_k: int) -> list[dict]:
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def display_results(query: str, results: list[dict]) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]Query:[/bold cyan] {query}", style="cyan"))

    score_table = Table("Rank", "Score", "Document", "Chunk", show_header=True, header_style="bold")
    for i, r in enumerate(results):
        bar = "█" * int(r["score"] * 30)
        score_table.add_row(
            str(i + 1),
            f"{r['score']:.4f} {bar}",
            r["doc"],
            f"{r['chunk_index'] + 1}/{r['total_chunks']}",
        )
    console.print(score_table)

    console.print()
    for i, r in enumerate(results[:2]):  # Show top-2 chunk text
        snippet = r["text"][:350].replace("\n", " ") + ("…" if len(r["text"]) > 350 else "")
        console.print(Panel(
            snippet,
            title=f"[green]Rank {i + 1}[/green] — {r['doc']} (score={r['score']:.4f})",
            border_style="green" if i == 0 else "dim",
        ))


def main():
    parser = argparse.ArgumentParser(description="Retrieve chunks from the FAISS index")
    parser.add_argument("--query", type=str, default=None, help="Single query to run")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 1 — Retrieve[/bold cyan]\n"
        "[dim]Semantic search over the FAISS index[/dim]",
        border_style="cyan",
    ))

    console.print(f"\n[bold]Loading FAISS index from[/bold] {INDEX_DIR}")
    index, chunks = load_index()
    console.print(f"  Index contains [green]{index.ntotal}[/green] vectors "
                  f"({EMBEDDING_DIM}-dim, IndexFlatIP)")

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    queries = [args.query] if args.query else DEMO_QUERIES

    console.print(f"\n[bold]Running {len(queries)} queries (top_k={args.top_k})[/bold]")
    console.print("[dim]Embedding each query, then searching the vector index...[/dim]\n")

    for query in queries:
        query_vec = embed_query(bedrock, query)
        results = retrieve(index, chunks, query_vec, args.top_k)
        display_results(query, results)

    console.print()
    console.print(Panel(
        "[bold]Key observations:[/bold]\n\n"
        "• Scores close to 1.0 indicate near-perfect semantic alignment.\n"
        "• The top result usually comes from the most relevant document.\n"
        "• Notice how results span different chunks of the same document — the\n"
        "  overlap ensures the best matching passage is found even if a fact\n"
        "  sits near a chunk boundary.\n\n"
        "Next step:\n"
        "  [bold]python 03_rag_pipeline.py[/bold]",
        title="What's Happening",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
