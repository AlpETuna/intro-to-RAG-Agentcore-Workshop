#!/usr/bin/env python3
"""
Stage 4, Script 2 — Observability

Shows how to instrument a RAG pipeline with OpenTelemetry and send
traces to AWS X-Ray via CloudWatch. Also creates a CloudWatch dashboard
for monitoring your RAG agent in production.

Two approaches demonstrated:
  A. Manual instrumentation — explicit spans around each RAG step
  B. What AgentCore Runtime does automatically (auto-instrumentation)

You don't need the deployed agent for this script — it instruments the
Stage 1 FAISS pipeline locally to show the concepts.

Usage:
    python 02_observability.py
    python 02_observability.py --create-dashboard
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
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
FAISS_INDEX_DIR = Path(__file__).parent.parent / "stage1-basic-rag" / "faiss_index"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def setup_tracer(service_name: str = "rag-workshop"):
    """Configure OpenTelemetry with console export (for demo) and X-Ray."""
    if not OTEL_AVAILABLE:
        return None

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "workshop",
    })

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


def demo_manual_instrumentation(tracer):
    """Run a RAG query with explicit OpenTelemetry spans around each step."""
    console.print("\n[bold]Demonstrating manual OpenTelemetry instrumentation[/bold]")
    console.print("[dim]  Each RAG step gets its own span — visible in X-Ray as a trace tree.[/dim]")

    if not OTEL_AVAILABLE or not tracer:
        console.print("[yellow]opentelemetry not installed — showing conceptual spans only.[/yellow]")
        _show_conceptual_trace()
        return

    import faiss
    import numpy as np
    import boto3

    try:
        index = faiss.read_index(str(FAISS_INDEX_DIR / "index.faiss"))
        chunks = json.loads((FAISS_INDEX_DIR / "chunks.json").read_text())
        bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    except Exception:
        console.print("[yellow]FAISS index not found — showing conceptual trace only.[/yellow]")
        _show_conceptual_trace()
        return

    question = "What is the HNSW algorithm used for?"
    session_id = uuid.uuid4().hex[:8]

    with tracer.start_as_current_span("rag.query") as root_span:
        root_span.set_attribute("session.id", session_id)
        root_span.set_attribute("query.text", question)

        # Span 1: Embedding
        with tracer.start_as_current_span("rag.embed_query") as embed_span:
            t0 = time.time()
            response = bedrock.invoke_model(
                modelId="amazon.titan-embed-text-v2:0",
                body=json.dumps({"inputText": question, "dimensions": 1024, "normalize": True}),
                contentType="application/json",
                accept="application/json",
            )
            vec = json.loads(response["body"].read())["embedding"]
            embed_latency = time.time() - t0
            embed_span.set_attribute("model.id", "amazon.titan-embed-text-v2:0")
            embed_span.set_attribute("embedding.dimensions", 1024)
            embed_span.set_attribute("latency.ms", int(embed_latency * 1000))
            console.print(f"  [green]✓[/green] embed span  ({embed_latency*1000:.0f}ms)")

        # Span 2: Retrieval
        with tracer.start_as_current_span("rag.retrieve") as retrieve_span:
            t0 = time.time()
            query_vec = np.array([vec], dtype="float32")
            scores, indices = index.search(query_vec, 3)
            results = [(chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]
            retrieve_latency = time.time() - t0
            retrieve_span.set_attribute("retrieval.top_k", 3)
            retrieve_span.set_attribute("retrieval.top_score", results[0][1] if results else 0)
            retrieve_span.set_attribute("retrieval.method", "faiss.IndexFlatIP")
            retrieve_span.set_attribute("latency.ms", int(retrieve_latency * 1000))
            console.print(f"  [green]✓[/green] retrieve span ({retrieve_latency*1000:.0f}ms, "
                          f"top score={results[0][1]:.3f})")

        # Span 3: Generation
        with tracer.start_as_current_span("rag.generate") as gen_span:
            context = "\n\n".join(f"[{r['doc']}]\n{r['text']}" for r, _ in results)
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
            t0 = time.time()
            gen_response = bedrock.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                }),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(gen_response["body"].read())
            answer = result["content"][0]["text"]
            usage = result.get("usage", {})
            gen_latency = time.time() - t0
            gen_span.set_attribute("model.id", "anthropic.claude-3-haiku-20240307-v1:0")
            gen_span.set_attribute("tokens.input", usage.get("input_tokens", 0))
            gen_span.set_attribute("tokens.output", usage.get("output_tokens", 0))
            gen_span.set_attribute("latency.ms", int(gen_latency * 1000))
            console.print(f"  [green]✓[/green] generate span ({gen_latency*1000:.0f}ms, "
                          f"{usage.get('input_tokens','?')}in/{usage.get('output_tokens','?')}out tokens)")

        root_span.set_attribute("rag.answer_length", len(answer))
        console.print(Panel(answer, title="[green]Answer[/green]", border_style="green"))


def _show_conceptual_trace():
    """Print a visual representation of what X-Ray traces look like."""
    console.print()
    console.print(Panel(
        "[bold]X-Ray Trace Structure for a RAG Query:[/bold]\n\n"
        "  rag.query  [████████████████████████████████████]  320ms\n"
        "  ├─ rag.embed_query  [██]  45ms\n"
        "  │    model.id: amazon.titan-embed-text-v2:0\n"
        "  │    embedding.dimensions: 1024\n"
        "  │\n"
        "  ├─ rag.retrieve  [█]  8ms\n"
        "  │    retrieval.top_k: 3\n"
        "  │    retrieval.top_score: 0.847\n"
        "  │    retrieval.method: faiss.IndexFlatIP\n"
        "  │\n"
        "  └─ rag.generate  [████████████████████████]  267ms\n"
        "       model.id: anthropic.claude-3-haiku\n"
        "       tokens.input: 1247\n"
        "       tokens.output: 89\n\n"
        "In X-Ray, each span is a timeline bar. You can click into any span\n"
        "to see its attributes, find slow steps, and debug failures.",
        border_style="blue",
    ))


def show_agentcore_auto_instrumentation():
    console.print()
    console.print(Panel(
        "[bold]AgentCore Runtime Auto-Instrumentation:[/bold]\n\n"
        "When you deploy to AgentCore Runtime, traces are generated automatically\n"
        "via the AWS Distro for OpenTelemetry (ADOT) — no code changes needed.\n\n"
        "Automatic spans include:\n"
        "  • agent.session.start — session initialization\n"
        "  • agent.turn — each user turn\n"
        "  • agent.tool_call — every tool invocation (search_knowledge_base, etc.)\n"
        "  • bedrock.invoke_model — LLM calls with token counts\n"
        "  • bedrock.retrieve — KB retrieval calls\n\n"
        "Enable it in your AgentCore Runtime config:\n\n"
        '  observabilityConfiguration: {\n'
        '    "enabled": True,\n'
        '    "destinations": [{"cloudWatch": {}}, {"xRay": {}}]\n'
        '  }',
        border_style="blue",
    ))


def create_cloudwatch_dashboard(cloudwatch, runtime_id: str):
    console.print("\n[bold]Creating CloudWatch Dashboard...[/bold]")

    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "title": "RAG Agent — Invocations",
                    "metrics": [
                        ["AWS/BedrockAgentCore", "Invocations",
                         "AgentRuntimeId", runtime_id, {"stat": "Sum", "period": 300}]
                    ],
                    "period": 300,
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "properties": {
                    "title": "RAG Agent — Latency (P99)",
                    "metrics": [
                        ["AWS/BedrockAgentCore", "InvocationLatency",
                         "AgentRuntimeId", runtime_id,
                         {"stat": "p99", "period": 300, "label": "P99"}],
                        ["AWS/BedrockAgentCore", "InvocationLatency",
                         "AgentRuntimeId", runtime_id,
                         {"stat": "p50", "period": 300, "label": "P50"}],
                    ],
                    "view": "timeSeries",
                    "yAxis": {"left": {"label": "ms"}},
                },
            },
            {
                "type": "metric",
                "properties": {
                    "title": "Bedrock — Token Usage",
                    "metrics": [
                        ["AWS/Bedrock", "InputTokenCount", {"stat": "Sum"}],
                        ["AWS/Bedrock", "OutputTokenCount", {"stat": "Sum"}],
                    ],
                    "view": "timeSeries",
                },
            },
            {
                "type": "metric",
                "properties": {
                    "title": "RAG Agent — Errors",
                    "metrics": [
                        ["AWS/BedrockAgentCore", "Errors",
                         "AgentRuntimeId", runtime_id, {"stat": "Sum", "color": "#d13212"}]
                    ],
                    "view": "timeSeries",
                },
            },
        ]
    }

    cloudwatch.put_dashboard(
        DashboardName="RAGWorkshopAgent",
        DashboardBody=json.dumps(dashboard_body),
    )
    dashboard_url = (
        f"https://{AWS_REGION}.console.aws.amazon.com/cloudwatch/home"
        f"?region={AWS_REGION}#dashboards:name=RAGWorkshopAgent"
    )
    console.print(f"  [green]✓ Dashboard created:[/green]")
    console.print(f"  {dashboard_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dashboard", action="store_true", help="Create a CloudWatch dashboard")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    runtime_id = os.getenv("AGENTCORE_RUNTIME_ID", "rag-workshop-agent")

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 4 — Observability[/bold cyan]\n"
        "[dim]OpenTelemetry tracing + CloudWatch dashboard[/dim]",
        border_style="cyan",
    ))

    tracer = setup_tracer()

    # Part 1: Manual instrumentation demo
    console.print()
    console.print(Rule("[bold]Part 1: Manual OpenTelemetry Spans[/bold]", style="cyan"))
    demo_manual_instrumentation(tracer)

    # Part 2: What AgentCore does automatically
    console.print()
    console.print(Rule("[bold]Part 2: AgentCore Auto-Instrumentation[/bold]", style="cyan"))
    show_agentcore_auto_instrumentation()

    # Part 3: Dashboard (optional)
    if args.create_dashboard:
        console.print()
        console.print(Rule("[bold]Part 3: CloudWatch Dashboard[/bold]", style="cyan"))
        cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)
        create_cloudwatch_dashboard(cloudwatch, runtime_id)

    # Key metrics reference
    console.print()
    console.print(Panel(
        "[bold]Key metrics to monitor in production:[/bold]\n\n"
        "  Metric                  | Target              | Alert if...\n"
        "  ────────────────────────┼─────────────────────┼──────────────────\n"
        "  Retrieval latency       | < 500ms             | > 1s p99\n"
        "  Generation latency      | < 3s                | > 8s p99\n"
        "  Total turn latency      | < 5s                | > 10s p99\n"
        "  Retrieval score (top-1) | > 0.7               | < 0.5 (bad retrieval)\n"
        "  Answer faithfulness     | > 0.85              | < 0.7 (hallucination risk)\n"
        "  Error rate              | < 0.1%              | > 1%\n"
        "  Token cost per turn     | Baseline + 20%      | > baseline + 50%\n\n"
        "Next step:\n"
        "  [bold]python 03_evaluate_rag.py[/bold]",
        title="Production Metrics",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
