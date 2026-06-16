#!/usr/bin/env python3
"""
Stage 2, Script 3 — Sync Data Source and Query

Triggers the ingestion job (syncs S3 docs into the KB index), waits for
completion, then runs queries using two Bedrock APIs:

  retrieve()               → returns raw chunks with scores (like Stage 1 FAISS)
  retrieve_and_generate()  → returns a generated answer + citations (managed RAG)

Also demonstrates metadata filtering and hybrid search.

Usage:
    python 03_sync_and_query.py
    python 03_sync_and_query.py --skip-sync   (if already synced)
    python 03_sync_and_query.py --query "What is HNSW?"
"""

import argparse
import json
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

DEMO_QUERIES = [
    "What are the main chunking strategies for RAG?",
    "How does AgentCore Memory differ from session history?",
    "What is the cost model for AWS Lambda?",
]


def load_config():
    load_dotenv(ENV_FILE)
    kb_id = os.getenv("KNOWLEDGE_BASE_ID")
    ds_id = os.getenv("KB_DATA_SOURCE_ID")
    if not kb_id or not ds_id:
        console.print("[red]Missing KNOWLEDGE_BASE_ID or KB_DATA_SOURCE_ID. Run 02_create_knowledge_base.py first.[/red]")
        raise SystemExit(1)
    return kb_id, ds_id


def start_ingestion(bedrock_agent, kb_id: str, ds_id: str) -> str:
    console.print(f"\n[bold]Starting ingestion job[/bold]")
    console.print(f"  KB ID:     {kb_id}")
    console.print(f"  Source ID: {ds_id}")

    response = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description="Initial sync of workshop documents",
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    console.print(f"  [green]✓ Job started:[/green] {job_id}")
    return job_id


def wait_for_ingestion(bedrock_agent, kb_id: str, ds_id: str, job_id: str, timeout: int = 300):
    console.print(f"\n[bold]Waiting for ingestion to complete...[/bold]")
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting documents...", total=None)
        while time.time() - start < timeout:
            job = bedrock_agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"]
            status = job["status"]
            stats = job.get("statistics", {})
            progress.update(task, description=(
                f"Status: {status} | "
                f"Scanned: {stats.get('numberOfDocumentsScanned', 0)} | "
                f"Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)}"
            ))
            if status == "COMPLETE":
                console.print(f"\n  [green]✓ Ingestion complete![/green]")
                console.print(f"  Documents scanned:  {stats.get('numberOfDocumentsScanned', 0)}")
                console.print(f"  Documents indexed:  {stats.get('numberOfNewDocumentsIndexed', 0)}")
                console.print(f"  Documents failed:   {stats.get('numberOfDocumentsFailed', 0)}")
                return
            if status in ("FAILED", "STOPPED"):
                console.print(f"  [red]Ingestion {status}[/red]")
                for failure in job.get("failureReasons", []):
                    console.print(f"  Reason: {failure}")
                raise SystemExit(1)
            time.sleep(5)
    raise TimeoutError("Ingestion timed out")


def retrieve_chunks(bedrock_rt, kb_id: str, query: str, top_k: int = 5) -> list[dict]:
    response = bedrock_rt.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": "HYBRID",  # semantic + keyword
            }
        },
    )
    return response.get("retrievalResults", [])


def retrieve_and_generate(bedrock_rt, kb_id: str, query: str, session_id: str = None) -> dict:
    kwargs = dict(
        input={"text": query},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": (
                    f"arn:aws:bedrock:{AWS_REGION}::foundation-model/"
                    "anthropic.claude-3-haiku-20240307-v1:0"
                ),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                        "overrideSearchType": "HYBRID",
                    }
                },
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": (
                            "You are a helpful assistant. Answer the question based ONLY on "
                            "the provided context. Be concise.\n\nContext:\n$search_results$\n\n"
                            "Question: $query$"
                        )
                    },
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": 512,
                            "temperature": 0.0,
                        }
                    },
                },
            },
        },
    )
    if session_id:
        kwargs["sessionId"] = session_id
    return bedrock_rt.retrieve_and_generate(**kwargs)


def display_retrieve_results(query: str, results: list) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]Retrieve:[/bold cyan] {query}", style="cyan"))

    table = Table("Rank", "Score", "Source", "Preview", show_header=True, header_style="bold")
    for i, r in enumerate(results):
        score = r.get("score", 0)
        location = r.get("location", {})
        source = location.get("s3Location", {}).get("uri", "unknown").split("/")[-1]
        text = r.get("content", {}).get("text", "")[:100].replace("\n", " ")
        bar = "█" * int(score * 20)
        table.add_row(str(i + 1), f"{score:.3f} {bar}", source, text + "…")
    console.print(table)


def display_rag_result(query: str, response: dict) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]Retrieve & Generate:[/bold cyan] {query}", style="cyan"))

    output = response.get("output", {}).get("text", "")
    citations = response.get("citations", [])

    console.print(Panel(output, title="[green]Generated Answer[/green]", border_style="green"))

    if citations:
        console.print(f"\n  [dim]Citations ({len(citations)} source(s)):[/dim]")
        for i, cit in enumerate(citations):
            for ref in cit.get("retrievedReferences", []):
                location = ref.get("location", {})
                uri = location.get("s3Location", {}).get("uri", "")
                source_name = uri.split("/")[-1] if uri else "unknown"
                text = ref.get("content", {}).get("text", "")[:120].replace("\n", " ")
                console.print(f"    [{i+1}] [bold]{source_name}[/bold] — {text}…")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sync", action="store_true", help="Skip ingestion, go straight to queries")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 2 — Sync and Query[/bold cyan]\n"
        "[dim]Ingest documents → Retrieve chunks → Retrieve & Generate[/dim]",
        border_style="cyan",
    ))

    kb_id, ds_id = load_config()
    bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
    bedrock_rt = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    if not args.skip_sync:
        job_id = start_ingestion(bedrock_agent, kb_id, ds_id)
        wait_for_ingestion(bedrock_agent, kb_id, ds_id, job_id)
    else:
        console.print("[dim]Skipping ingestion (--skip-sync)[/dim]")

    queries = [args.query] if args.query else DEMO_QUERIES

    console.print()
    console.print(Panel(
        "[bold]Two Bedrock RAG APIs:[/bold]\n\n"
        "  [cyan]retrieve()[/cyan]              → raw chunks + scores (like FAISS in Stage 1)\n"
        "  [cyan]retrieve_and_generate()[/cyan] → managed RAG: chunks → Claude → answer + citations\n\n"
        "Both use HYBRID search (semantic + keyword), which often outperforms dense-only.",
        border_style="blue",
    ))

    for query in queries:
        # Show raw retrieval
        results = retrieve_chunks(bedrock_rt, kb_id, query)
        display_retrieve_results(query, results)

        # Show managed RAG
        response = retrieve_and_generate(bedrock_rt, kb_id, query)
        display_rag_result(query, response)

    console.print()
    console.print(Panel(
        "[bold]Stage 1 vs Stage 2 comparison:[/bold]\n\n"
        "  Stage 1 (FAISS)              Stage 2 (Bedrock KB)\n"
        "  ─────────────────────────────────────────────────\n"
        "  Manual chunking              Managed chunking (configurable)\n"
        "  Manual embedding loop        Automatic embedding on sync\n"
        "  Local FAISS index            OpenSearch Serverless (distributed)\n"
        "  Dense search only            Hybrid search (semantic + keyword)\n"
        "  No citations                 Citations with source URIs\n"
        "  Re-run script to update      Call StartIngestionJob API\n\n"
        "Next step:\n"
        "  [bold]python 04_compare_approaches.py[/bold]",
        title="What Changed",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
