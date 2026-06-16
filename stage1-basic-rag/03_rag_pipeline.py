#!/usr/bin/env python3
"""
Stage 1, Script 3 — Full RAG Pipeline

Combines retrieval (Script 2) and generation (Claude 3 Haiku) into a
complete RAG pipeline. Runs a suite of test questions and shows every
step: the query, retrieved context, the constructed prompt, and the answer.

This is the "show your work" script — everything is made visible.

Usage:
    python 03_rag_pipeline.py
    python 03_rag_pipeline.py --query "What is chunking in RAG?"
    python 03_rag_pipeline.py --top-k 5 --show-prompt
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
from rich.syntax import Syntax

INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
EMBEDDING_DIM = 1024
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

TEST_QUESTIONS = [
    "What are the three main reasons why RAG exists?",
    "Which Bedrock embedding model should I use and what dimension does it produce?",
    "What is the difference between short-term and long-term memory in AgentCore?",
    "What is the difference between HNSW and IVF in vector databases?",
    "How much does an AWS Lambda invocation cost?",
]

SYSTEM_PROMPT = """You are a helpful technical assistant. Answer questions based ONLY on the provided context.
If the context does not contain enough information to answer the question, say so clearly.
Be concise and precise. Cite the source document when relevant."""

PROMPT_TEMPLATE = """Here is relevant context retrieved from the knowledge base:

{context}

Based on this context, answer the following question:
{question}"""


def load_index() -> tuple[faiss.Index, list[dict]]:
    if not INDEX_DIR.exists():
        console.print("[red]Index not found. Run 01_chunk_and_embed.py first.[/red]")
        raise SystemExit(1)
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def embed_query(bedrock_rt, query: str) -> np.ndarray:
    response = bedrock_rt.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": query, "dimensions": EMBEDDING_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return np.array([json.loads(response["body"].read())["embedding"]], dtype="float32")


def retrieve(index: faiss.Index, chunks: list[dict], query_vec: np.ndarray, top_k: int) -> list[dict]:
    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
    return results


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results):
        parts.append(
            f"[Context {i+1} | Source: {r['doc']} | Chunk {r['chunk_index']+1} | "
            f"Similarity: {r['score']:.3f}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate(bedrock_rt, prompt: str) -> tuple[str, dict]:
    t0 = time.time()
    response = bedrock_rt.invoke_model(
        modelId=GENERATION_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
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


def run_rag(
    index: faiss.Index,
    chunks: list[dict],
    bedrock_rt,
    question: str,
    top_k: int,
    show_prompt: bool,
) -> None:
    console.print()
    console.print(Rule(style="cyan"))
    console.print(Panel(f"[bold cyan]{question}[/bold cyan]", title="Question", border_style="cyan"))

    # Step 1: Embed
    console.print("\n[dim]Step 1 → Embedding query with Titan Embed V2...[/dim]")
    query_vec = embed_query(bedrock_rt, question)
    console.print(f"  Query vector: [{query_vec[0][0]:.4f}, {query_vec[0][1]:.4f}, … "
                  f"{query_vec[0][-1]:.4f}]  (1024-dim)")

    # Step 2: Retrieve
    console.print(f"[dim]Step 2 → Searching FAISS index (top_k={top_k})...[/dim]")
    results = retrieve(index, chunks, query_vec, top_k)
    for i, r in enumerate(results):
        console.print(
            f"  [{i+1}] score=[green]{r['score']:.4f}[/green]  "
            f"[bold]{r['doc']}[/bold] chunk {r['chunk_index']+1}"
        )

    # Step 3: Build context + prompt
    context = build_context(results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    if show_prompt:
        console.print("\n[dim]Step 3 → Constructed prompt:[/dim]")
        console.print(Panel(prompt[:1200] + ("…" if len(prompt) > 1200 else ""),
                            title="Full Prompt (truncated)", border_style="dim"))

    # Step 4: Generate
    console.print(f"[dim]Step 4 → Generating with {GENERATION_MODEL}...[/dim]")
    answer, usage = generate(bedrock_rt, prompt)

    console.print()
    console.print(Panel(
        answer,
        title="[green]Answer[/green]",
        border_style="green",
    ))
    console.print(
        f"  [dim]Tokens: {usage.get('input_tokens', '?')} in / "
        f"{usage.get('output_tokens', '?')} out | "
        f"Latency: {usage['latency_s']}s[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(description="Full RAG pipeline: retrieve + generate")
    parser.add_argument("--query", type=str, default=None, help="Single question to answer")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks to retrieve")
    parser.add_argument("--show-prompt", action="store_true", help="Print the full prompt before generation")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 1 — Full RAG Pipeline[/bold cyan]\n"
        "[dim]Embed → Retrieve → Prompt → Generate[/dim]",
        border_style="cyan",
    ))
    console.print(f"\n  Embedding model:  [bold]{EMBEDDING_MODEL}[/bold]")
    console.print(f"  Generation model: [bold]{GENERATION_MODEL}[/bold]")
    console.print(f"  Top-k chunks:     [bold]{args.top_k}[/bold]")

    index, chunks = load_index()
    console.print(f"\n  Index: {index.ntotal} vectors loaded from {INDEX_DIR}")

    bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    questions = [args.query] if args.query else TEST_QUESTIONS

    for question in questions:
        run_rag(index, chunks, bedrock_rt, question, args.top_k, args.show_prompt)

    console.print()
    console.print(Panel(
        "[bold]What just happened:[/bold]\n\n"
        "  1. Query was embedded into a 1024-dim vector\n"
        "  2. FAISS searched for the {k} most similar chunk vectors\n"
        "  3. Retrieved chunks were formatted as context in the prompt\n"
        "  4. Claude generated an answer grounded in that context\n\n"
        "[bold]Limitations of this DIY approach:[/bold]\n\n"
        "  • FAISS is local only — not scalable beyond a single machine\n"
        "  • No hybrid search (keyword + semantic combined)\n"
        "  • No metadata filtering\n"
        "  • Must re-index when documents change\n\n"
        "  → Stage 2 solves all of these with Bedrock Knowledge Bases.\n\n"
        "Next step:\n"
        "  [bold]python 04_interactive_chat.py[/bold]",
        title="Summary",
        border_style="blue",
    ).renderable.format(k=args.top_k))


if __name__ == "__main__":
    main()
