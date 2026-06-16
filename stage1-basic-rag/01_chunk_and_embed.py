#!/usr/bin/env python3
"""
Stage 1, Script 1 — Chunk and Embed

Loads the five workshop documents, splits them into overlapping chunks,
embeds each chunk with Amazon Titan Text Embeddings V2, and saves the
resulting FAISS index to disk.

What you'll see:
  - Each document loaded and chunked with live statistics
  - Embedding API calls batched and tracked
  - Final FAISS index saved and verified

Usage:
    uv run 01_chunk_and_embed.py

Output files:
    faiss_index/index.faiss      — the vector index
    faiss_index/chunks.json      — chunk text + metadata (document, position)
"""

import json
import os
import sys
import time
from pathlib import Path

import boto3
import faiss
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import print as rprint

DATA_DIR = Path(__file__).parent.parent / "stage0-setup" / "data"
INDEX_DIR = Path(__file__).parent / "faiss_index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024
CHUNK_SIZE = 400          # characters
CHUNK_OVERLAP = 80        # characters — ~20% overlap
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"filename": path.name, "text": text})
        console.print(f"  [green]✓[/green] Loaded [bold]{path.name}[/bold] ({len(text):,} chars)")
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size character chunking with overlap. Splits on whitespace when possible."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at a paragraph or sentence boundary
            for sep in ["\n\n", "\n", ". ", " "]:
                boundary = text.rfind(sep, start, end)
                if boundary > start + chunk_size // 2:
                    end = boundary + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def embed_text(client, text: str) -> list[float]:
    """Call Titan Embed V2 for a single text. Returns a 1024-dim vector."""
    response = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIM,
            "normalize": True,       # unit vectors → cosine similarity via dot product
        }),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 1 — Chunk and Embed[/bold cyan]\n"
        "[dim]Building a FAISS vector index from the workshop documents[/dim]",
        border_style="cyan",
    ))

    # ── 1. Load documents ─────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/4 — Loading documents[/bold]")
    console.print(f"  Reading from: [dim]{DATA_DIR}[/dim]")
    docs = load_documents(DATA_DIR)
    if not docs:
        console.print("[red]No .txt files found in data/. Run from stage1-basic-rag/.[/red]")
        sys.exit(1)

    # ── 2. Chunk documents ────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 2/4 — Chunking[/bold]  "
                  f"[dim](size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars)[/dim]")

    all_chunks = []
    chunk_table = Table("Document", "Chunks", "Avg Chunk Size", show_header=True, header_style="bold magenta")
    for doc in docs:
        chunks = chunk_text(doc["text"])
        avg_len = int(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0
        chunk_table.add_row(doc["filename"], str(len(chunks)), f"{avg_len} chars")
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": len(all_chunks),
                "doc": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
            })

    console.print(chunk_table)
    console.print(f"\n  [green]Total chunks:[/green] {len(all_chunks)}")

    # Peek at one chunk so the audience can see what we're working with
    sample = all_chunks[len(all_chunks) // 2]
    console.print(Panel(
        f"[dim]Document:[/dim] {sample['doc']}  "
        f"[dim]Chunk:[/dim] {sample['chunk_index'] + 1}/{sample['total_chunks']}\n\n"
        + sample["text"][:300] + "…",
        title="Sample Chunk",
        border_style="dim",
    ))

    # ── 3. Embed chunks ───────────────────────────────────────────────────────
    console.print(f"\n[bold]Step 3/4 — Embedding with {EMBEDDING_MODEL}[/bold]")
    console.print(f"  Region: [dim]{AWS_REGION}[/dim]")

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    embeddings = []
    failed = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[rate]} chunks/s[/dim]"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Embedding...", total=len(all_chunks), rate="0.0"
        )
        for i, chunk in enumerate(all_chunks):
            try:
                vec = embed_text(bedrock, chunk["text"])
                embeddings.append(vec)
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] chunk {i} failed: {e}")
                embeddings.append([0.0] * EMBEDDING_DIM)
                failed += 1

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            progress.update(task, advance=1, rate=f"{rate:.1f}")
            time.sleep(0.02)  # avoid throttling at high chunk counts

    elapsed = time.time() - start_time
    console.print(f"\n  [green]Embedded {len(embeddings) - failed}/{len(all_chunks)} chunks[/green] "
                  f"in {elapsed:.1f}s ({(len(embeddings)/elapsed):.1f} chunks/s)")
    console.print(f"  Vector shape: {len(embeddings)} × {EMBEDDING_DIM}")
    console.print(f"  Embedding storage: {(len(embeddings) * EMBEDDING_DIM * 4) / 1024:.1f} KB")

    # Sanity-check: show the embedding range of the first vector
    first_vec = embeddings[0]
    console.print(f"  Sample vector: [{first_vec[0]:.4f}, {first_vec[1]:.4f}, "
                  f"... {first_vec[-1]:.4f}] (L2 norm={np.linalg.norm(first_vec):.4f})")

    # ── 4. Build and save FAISS index ─────────────────────────────────────────
    console.print(f"\n[bold]Step 4/4 — Building FAISS index[/bold]")

    vectors = np.array(embeddings, dtype="float32")

    # IndexFlatIP = exact inner product search. Since vectors are normalized (unit length),
    # dot product == cosine similarity. For < 100k vectors, exact search is fast enough.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    console.print(f"  Index type: IndexFlatIP (exact cosine search)")
    console.print(f"  Vectors indexed: {index.ntotal}")

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    console.print(f"  [green]✓ Saved FAISS index:[/green]  {INDEX_DIR}/index.faiss")
    console.print(f"  [green]✓ Saved chunk metadata:[/green] {INDEX_DIR}/chunks.json")

    console.print()
    console.print(Panel(
        "[green]Indexing complete![/green]\n\n"
        f"  Documents loaded:  {len(docs)}\n"
        f"  Chunks created:    {len(all_chunks)}\n"
        f"  Embedding model:   {EMBEDDING_MODEL}\n"
        f"  Vector dimensions: {EMBEDDING_DIM}\n"
        f"  Index type:        IndexFlatIP (exact cosine)\n\n"
        "Next step:\n"
        "  [bold]uv run 02_retrieve.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
