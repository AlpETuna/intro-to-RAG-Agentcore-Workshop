#!/usr/bin/env python3
"""
Stage 2, Script 4 — Compare Stage 1 vs Stage 2

Runs the same questions through both the Stage 1 FAISS pipeline and the
Stage 2 Bedrock Knowledge Base, then renders a side-by-side comparison.

Shows concretely where managed RAG improves over DIY RAG.

Usage:
    uv run 04_compare_approaches.py
"""

import functools
import json
import os
import time
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
FAISS_INDEX_DIR = Path(__file__).parent.parent / "stage1-basic-rag" / "faiss_index"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBEDDING_DIM = 1024
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()


@functools.lru_cache(maxsize=1)
def generation_model_arn() -> str:
    """Inference-profile ARN for retrieve_and_generate (needs an ARN, not an ID)."""
    account = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()["Account"]
    return f"arn:aws:bedrock:{AWS_REGION}:{account}:inference-profile/{GENERATION_MODEL}"

COMPARISON_QUESTIONS = [
    "What is the difference between dense and sparse retrieval?",
    "How does AWS Lambda pricing work?",
    "What is AgentCore Policy and how does it differ from Guardrails?",
]


def load_faiss():
    if not FAISS_INDEX_DIR.exists():
        return None, None
    index = faiss.read_index(str(FAISS_INDEX_DIR / "index.faiss"))
    chunks = json.loads((FAISS_INDEX_DIR / "chunks.json").read_text())
    return index, chunks


def embed(client, text: str) -> np.ndarray:
    r = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text, "dimensions": EMBEDDING_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return np.array([json.loads(r["body"].read())["embedding"]], dtype="float32")


def faiss_rag(index, chunks, bedrock_rt, question: str) -> tuple[str, float, list[str]]:
    t0 = time.time()
    vec = embed(bedrock_rt, question)
    scores, indices = index.search(vec, 3)
    retrieved = [chunks[i] for i in indices[0] if i >= 0]
    sources = [f"{r['doc']} ({scores[0][j]:.3f})" for j, r in enumerate(retrieved)]

    context = "\n\n---\n".join(
        f"[{r['doc']}]\n{r['text']}" for r in retrieved
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer concisely based only on the context:\n{question}"
    )
    resp = bedrock_rt.invoke_model(
        modelId=GENERATION_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    answer = json.loads(resp["body"].read())["content"][0]["text"]
    latency = time.time() - t0
    return answer, latency, sources


def bedrock_kb_rag(bedrock_rt_agent, kb_id: str, question: str) -> tuple[str, float, list[str]]:
    t0 = time.time()
    resp = bedrock_rt_agent.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": generation_model_arn(),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 3,
                        "overrideSearchType": "HYBRID",
                    }
                },
            },
        },
    )
    answer = resp.get("output", {}).get("text", "")
    latency = time.time() - t0
    citations = resp.get("citations", [])
    sources = []
    for cit in citations:
        for ref in cit.get("retrievedReferences", []):
            uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
            sources.append(uri.split("/")[-1] if uri else "unknown")

    return answer, latency, sources


def main():
    load_dotenv(ENV_FILE)
    kb_id = os.getenv("KNOWLEDGE_BASE_ID")
    if not kb_id:
        console.print("[red]KNOWLEDGE_BASE_ID not set. Run 02_create_knowledge_base.py first.[/red]")
        raise SystemExit(1)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 2 — Compare Approaches[/bold cyan]\n"
        "[dim]Stage 1 (FAISS) vs Stage 2 (Bedrock Knowledge Base)[/dim]",
        border_style="cyan",
    ))

    bedrock_rt = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    bedrock_agent_rt = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    index, chunks = load_faiss()
    faiss_available = index is not None
    if not faiss_available:
        console.print("[yellow]Stage 1 FAISS index not found — skipping FAISS comparison.[/yellow]")

    summary_table = Table(
        "Question", "Stage 1 Latency", "Stage 2 Latency", "Stage 2 Sources",
        show_header=True, header_style="bold magenta",
    )

    for question in COMPARISON_QUESTIONS:
        console.print()
        console.print(Rule(f"[bold]{question}[/bold]", style="cyan"))

        s1_answer, s1_latency, s1_sources = ("N/A (no FAISS index)", 0.0, [])
        if faiss_available:
            s1_answer, s1_latency, s1_sources = faiss_rag(index, chunks, bedrock_rt, question)

        s2_answer, s2_latency, s2_sources = bedrock_kb_rag(bedrock_agent_rt, kb_id, question)

        panels = []
        if faiss_available:
            panels.append(Panel(
                s1_answer,
                title=f"[yellow]Stage 1 — FAISS[/yellow] ({s1_latency:.2f}s)",
                subtitle=", ".join(s1_sources[:2]),
                border_style="yellow",
                width=60,
            ))
        panels.append(Panel(
            s2_answer,
            title=f"[green]Stage 2 — Bedrock KB[/green] ({s2_latency:.2f}s)",
            subtitle=", ".join(set(s2_sources[:2])),
            border_style="green",
            width=60,
        ))

        if len(panels) > 1:
            console.print(Columns(panels))
        else:
            console.print(panels[0])

        summary_table.add_row(
            question[:50] + "…",
            f"{s1_latency:.2f}s" if faiss_available else "N/A",
            f"{s2_latency:.2f}s",
            ", ".join(set(s2_sources[:2])),
        )

    console.print()
    console.print(summary_table)
    console.print()
    console.print(Panel(
        "[bold]Key differences you should observe:[/bold]\n\n"
        "  • Bedrock KB uses HYBRID search — catches keyword matches FAISS misses\n"
        "  • Bedrock KB returns citation URIs — makes answers auditable\n"
        "  • Bedrock KB latency includes managed retrieval + managed generation\n"
        "  • Answer quality is often similar — the advantage is in operations,\n"
        "    not raw accuracy for well-formed queries\n\n"
        "Next: Stage 3 wraps this KB in an AgentCore agent with memory.\n\n"
        "  [bold]cd ../stage3-agentcore-agent[/bold]\n"
        "  [bold]uv run 01_setup_iam.py[/bold]",
        title="Takeaways",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
