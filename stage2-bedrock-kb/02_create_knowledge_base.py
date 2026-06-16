#!/usr/bin/env python3
"""
Stage 2, Script 2 — Create Bedrock Knowledge Base

Creates a Bedrock Knowledge Base backed by Amazon OpenSearch Serverless.
Bedrock handles chunking, embedding (Titan Embed V2), and indexing automatically.

⚠️  OpenSearch Serverless costs ~$0.48/hr while active (2 OCU minimum).
    Run cleanup.py when done to avoid ongoing charges.

What Bedrock does for you (vs Stage 1 where you did this manually):
  • Chunking: configurable strategy and size
  • Embedding: Titan Embed V2 (same model as Stage 1)
  • Indexing: OpenSearch Serverless with HNSW
  • Updates: just sync the data source — no manual re-indexing

Usage:
    python 02_create_knowledge_base.py
"""

import json
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    bucket = os.getenv("S3_BUCKET_NAME")
    role_arn = os.getenv("KB_IAM_ROLE_ARN")
    if not bucket or not role_arn:
        console.print("[red]Missing S3_BUCKET_NAME or KB_IAM_ROLE_ARN. Run 01_create_infrastructure.py first.[/red]")
        raise SystemExit(1)
    return bucket, role_arn


def create_knowledge_base(bedrock_agent, role_arn: str) -> dict:
    console.print("\n[bold]Creating Bedrock Knowledge Base...[/bold]")
    console.print("  [dim]This creates the KB definition — vector store comes next.[/dim]")

    response = bedrock_agent.create_knowledge_base(
        name="rag-workshop-kb",
        description="Workshop knowledge base — RAG, Bedrock, AgentCore, serverless, vector DBs",
        roleArn=role_arn,
        knowledgeBaseConfiguration={
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": (
                    f"arn:aws:bedrock:{AWS_REGION}::foundation-model/"
                    "amazon.titan-embed-text-v2:0"
                ),
                "embeddingModelConfiguration": {
                    "bedrockEmbeddingModelConfiguration": {
                        "dimensions": 1024,
                    }
                },
            },
        },
        storageConfiguration={
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": "",  # Bedrock creates and manages the collection
                "vectorIndexName": "rag-workshop-index",
                "fieldMapping": {
                    "vectorField": "embedding",
                    "textField": "text",
                    "metadataField": "metadata",
                },
            },
        },
    )
    kb = response["knowledgeBase"]
    console.print(f"  [green]✓ Knowledge Base ID:[/green] {kb['knowledgeBaseId']}")
    console.print(f"  [green]✓ Status:[/green] {kb['status']}")
    return kb


def wait_for_kb(bedrock_agent, kb_id: str, timeout: int = 300) -> str:
    console.print(f"\n[bold]Waiting for KB to become ACTIVE[/bold] (up to {timeout}s)...")
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Provisioning OpenSearch Serverless collection...", total=None)
        while time.time() - start < timeout:
            kb = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]
            status = kb["status"]
            progress.update(task, description=f"Status: {status}")
            if status == "ACTIVE":
                console.print(f"  [green]✓ KB is ACTIVE[/green]")
                return status
            if status == "FAILED":
                console.print(f"  [red]KB creation FAILED:[/red] {kb.get('failureReasons', [])}")
                raise SystemExit(1)
            time.sleep(10)
    raise TimeoutError(f"KB did not become ACTIVE within {timeout}s")


def create_data_source(bedrock_agent, kb_id: str, bucket_name: str) -> str:
    console.print(f"\n[bold]Creating S3 data source[/bold] → s3://{bucket_name}/docs/")

    response = bedrock_agent.create_data_source(
        knowledgeBaseId=kb_id,
        name="workshop-s3-source",
        description="Workshop documents from S3",
        dataSourceConfiguration={
            "type": "S3",
            "s3Configuration": {
                "bucketArn": f"arn:aws:s3:::{bucket_name}",
                "inclusionPrefixes": ["docs/"],
            },
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": {
                "chunkingStrategy": "FIXED_SIZE",
                "fixedSizeChunkingConfiguration": {
                    "maxTokens": 300,
                    "overlapPercentage": 20,
                },
            },
        },
    )
    ds = response["dataSource"]
    ds_id = ds["dataSourceId"]
    console.print(f"  [green]✓ Data Source ID:[/green] {ds_id}")
    console.print(f"  [green]✓ Chunking:[/green] FIXED_SIZE (300 tokens, 20% overlap)")
    return ds_id


def save_env(kb_id: str, kb_arn: str, ds_id: str) -> None:
    env_path = str(ENV_FILE)
    set_key(env_path, "KNOWLEDGE_BASE_ID", kb_id)
    set_key(env_path, "KNOWLEDGE_BASE_ARN", kb_arn)
    set_key(env_path, "KB_DATA_SOURCE_ID", ds_id)
    console.print(f"\n  [green]✓ Saved to .env:[/green] KNOWLEDGE_BASE_ID, KNOWLEDGE_BASE_ARN, KB_DATA_SOURCE_ID")


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 2 — Create Knowledge Base[/bold cyan]\n"
        "[dim]Managed chunking, embedding, and indexing via Bedrock[/dim]",
        border_style="cyan",
    ))
    console.print()
    console.print(Panel(
        "[yellow]Cost notice:[/yellow] OpenSearch Serverless costs ~$0.48/hr (2 OCU minimum).\n"
        "Run [bold]python cleanup.py[/bold] when done to delete all Stage 2 resources.",
        border_style="yellow",
    ))

    bucket, role_arn = load_config()
    bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)

    # Explain what Bedrock is doing vs Stage 1
    console.print()
    console.print(Panel(
        "[bold]What Bedrock handles automatically (vs Stage 1):[/bold]\n\n"
        "  Stage 1 (manual)          →  Stage 2 (managed)\n"
        "  ─────────────────────────────────────────────────\n"
        "  chunk_text() function     →  FIXED_SIZE chunking strategy\n"
        "  Titan Embed V2 loop       →  Automatic embedding on ingest\n"
        "  faiss.IndexFlatIP         →  OpenSearch Serverless (HNSW)\n"
        "  Manual re-index           →  StartIngestionJob API call\n"
        "  Local file                →  S3 + distributed, scalable",
        border_style="blue",
    ))

    kb = create_knowledge_base(bedrock_agent, role_arn)
    kb_id = kb["knowledgeBaseId"]
    kb_arn = kb["knowledgeBaseArn"]

    wait_for_kb(bedrock_agent, kb_id)

    ds_id = create_data_source(bedrock_agent, kb_id, bucket)
    save_env(kb_id, kb_arn, ds_id)

    console.print()
    console.print(Panel(
        "[green]Knowledge Base created![/green]\n\n"
        f"  Knowledge Base ID:  {kb_id}\n"
        f"  Data Source ID:     {ds_id}\n"
        f"  Embedding model:    amazon.titan-embed-text-v2:0 (1024-dim)\n"
        f"  Vector store:       OpenSearch Serverless (HNSW)\n"
        f"  Chunking:           FIXED_SIZE, 300 tokens, 20% overlap\n\n"
        "The KB exists but no documents are indexed yet.\n"
        "Next step syncs the S3 data source:\n"
        "  [bold]python 03_sync_and_query.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
