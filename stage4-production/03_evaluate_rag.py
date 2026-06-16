#!/usr/bin/env python3
"""
Stage 4, Script 3 — RAG Evaluation

Evaluates the quality of the RAG pipeline using the RAG Triad:
  - Context Relevance:   Are the retrieved chunks relevant to the query?
  - Faithfulness:        Is the answer grounded in the retrieved context?
  - Answer Relevance:    Does the answer actually address the question?

Uses Claude as the LLM judge (LLM-as-judge pattern).
Can evaluate both the Stage 1 (FAISS) and Stage 2 (Bedrock KB) pipelines.

Outputs an evaluation report with scores and failure examples.

Usage:
    uv run 03_evaluate_rag.py
    uv run 03_evaluate_rag.py --pipeline faiss
    uv run 03_evaluate_rag.py --pipeline bedrock-kb
    uv run 03_evaluate_rag.py --pipeline both
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

ENV_FILE = Path(__file__).parent.parent / ".env"
FAISS_INDEX_DIR = Path(__file__).parent.parent / "stage1-basic-rag" / "faiss_index"
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"
GENERATION_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


@functools.lru_cache(maxsize=1)
def generation_model_arn() -> str:
    """Inference-profile ARN for retrieve_and_generate (needs an ARN, not an ID)."""
    account = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()["Account"]
    return f"arn:aws:bedrock:{AWS_REGION}:{account}:inference-profile/{GENERATION_MODEL}"

console = Console()

# Ground-truth evaluation set — question + expected answer elements
EVAL_SET = [
    {
        "question": "What are the three main reasons RAG exists?",
        "expected_elements": ["knowledge cutoff", "hallucination", "private data"],
        "source_doc": "rag_fundamentals.txt",
    },
    {
        "question": "What embedding model should I use for Bedrock Knowledge Bases?",
        "expected_elements": ["Titan", "1024", "normalize"],
        "source_doc": "bedrock_models.txt",
    },
    {
        "question": "What is the minimum OCU cost for OpenSearch Serverless?",
        "expected_elements": ["2", "OCU", "0.24", "0.48"],
        "source_doc": "vector_databases.txt",
    },
    {
        "question": "What is the maximum duration of an AWS Lambda function?",
        "expected_elements": ["15 minutes", "15 min"],
        "source_doc": "serverless_aws.txt",
    },
    {
        "question": "What does AgentCore Policy use for enforcement?",
        "expected_elements": ["Cedar", "deterministic"],
        "source_doc": "aws_agentcore.txt",
    },
]


def llm_judge(bedrock, question: str, context: str, answer: str) -> dict:
    """Use Claude as LLM judge to score context relevance and faithfulness."""
    prompt = f"""You are evaluating a RAG system. Score the following on a 0.0-1.0 scale.

QUESTION: {question}

RETRIEVED CONTEXT:
{context[:2000]}

GENERATED ANSWER:
{answer}

Score these three dimensions (respond ONLY with valid JSON, no other text):
{{
  "context_relevance": <0.0-1.0, how relevant is the context to the question?>,
  "faithfulness": <0.0-1.0, is the answer grounded in the context without hallucinating?>,
  "answer_relevance": <0.0-1.0, does the answer actually address the question?>,
  "context_relevance_reason": "<one sentence>",
  "faithfulness_reason": "<one sentence>",
  "answer_relevance_reason": "<one sentence>"
}}"""

    response = bedrock.invoke_model(
        modelId=JUDGE_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    text = json.loads(response["body"].read())["content"][0]["text"]
    try:
        # Extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {
            "context_relevance": 0.5, "faithfulness": 0.5, "answer_relevance": 0.5,
            "context_relevance_reason": "parse error",
            "faithfulness_reason": "parse error",
            "answer_relevance_reason": "parse error",
        }


def check_expected_elements(answer: str, elements: list[str]) -> float:
    """Check what fraction of expected answer elements appear in the answer."""
    answer_lower = answer.lower()
    found = sum(1 for e in elements if any(w in answer_lower for w in e.lower().split("/")))
    return found / len(elements) if elements else 1.0


def run_faiss_rag(bedrock, index, chunks, question: str) -> tuple[str, str]:
    """Run Stage 1 FAISS RAG and return (context, answer)."""
    import numpy as np

    r = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": question, "dimensions": 1024, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    vec = np.array([json.loads(r["body"].read())["embedding"]], dtype="float32")
    scores, indices = index.search(vec, 3)
    retrieved = [(chunks[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    context = "\n\n---\n".join(f"[{r['doc']} | {s:.3f}]\n{r['text']}" for r, s in retrieved)
    prompt = f"Context:\n{context}\n\nAnswer concisely:\n{question}"
    resp = bedrock.invoke_model(
        modelId=GENERATION_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }),
        contentType="application/json",
        accept="application/json",
    )
    answer = json.loads(resp["body"].read())["content"][0]["text"]
    return context, answer


def run_bedrock_kb_rag(bedrock_agent_rt, kb_id: str, question: str) -> tuple[str, str]:
    """Run Stage 2 Bedrock KB RAG and return (context, answer)."""
    retrieve_resp = bedrock_agent_rt.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3, "overrideSearchType": "HYBRID"}},
    )
    results = retrieve_resp.get("retrievalResults", [])
    context = "\n\n---\n".join(
        f"[{r.get('location', {}).get('s3Location', {}).get('uri', '').split('/')[-1]} | {r.get('score', 0):.3f}]\n"
        f"{r.get('content', {}).get('text', '')}"
        for r in results
    )

    rag_resp = bedrock_agent_rt.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": generation_model_arn(),
                "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 3}},
            },
        },
    )
    answer = rag_resp.get("output", {}).get("text", "")
    return context, answer


def evaluate_pipeline(bedrock, bedrock_agent_rt, kb_id: str, pipeline: str) -> list[dict]:
    import faiss

    index, chunks = None, None
    if pipeline in ("faiss", "both"):
        try:
            index = faiss.read_index(str(FAISS_INDEX_DIR / "index.faiss"))
            chunks = json.loads((FAISS_INDEX_DIR / "chunks.json").read_text())
        except Exception:
            console.print("[yellow]FAISS index not available — skipping FAISS evaluation.[/yellow]")

    results = []
    for item in EVAL_SET:
        question = item["question"]
        console.print(f"  [dim]Evaluating: {question[:60]}...[/dim]")

        if pipeline == "faiss" and index:
            context, answer = run_faiss_rag(bedrock, index, chunks, question)
            label = "FAISS"
        elif pipeline == "bedrock-kb" and kb_id:
            context, answer = run_bedrock_kb_rag(bedrock_agent_rt, kb_id, question)
            label = "Bedrock KB"
        else:
            label = "FAISS" if index else "Bedrock KB"
            if index:
                context, answer = run_faiss_rag(bedrock, index, chunks, question)
            elif kb_id:
                context, answer = run_bedrock_kb_rag(bedrock_agent_rt, kb_id, question)
            else:
                continue

        scores = llm_judge(bedrock, question, context, answer)
        element_score = check_expected_elements(answer, item["expected_elements"])

        results.append({
            "question": question,
            "pipeline": label,
            "answer": answer,
            "context_relevance": scores["context_relevance"],
            "faithfulness": scores["faithfulness"],
            "answer_relevance": scores["answer_relevance"],
            "element_coverage": element_score,
            "cr_reason": scores["context_relevance_reason"],
            "faith_reason": scores["faithfulness_reason"],
            "ar_reason": scores["answer_relevance_reason"],
        })
        time.sleep(1)

    return results


def render_report(results: list[dict]):
    console.print()
    console.print(Rule("[bold cyan]Evaluation Report[/bold cyan]", style="cyan"))

    table = Table(
        "Question", "Pipeline", "Ctx Rel", "Faithful", "Ans Rel", "Coverage",
        show_header=True, header_style="bold magenta",
    )
    totals = {"context_relevance": 0, "faithfulness": 0, "answer_relevance": 0, "element_coverage": 0}

    for r in results:
        def score_str(v):
            color = "green" if v >= 0.8 else "yellow" if v >= 0.6 else "red"
            return f"[{color}]{v:.2f}[/{color}]"

        table.add_row(
            r["question"][:45] + "…",
            r["pipeline"],
            score_str(r["context_relevance"]),
            score_str(r["faithfulness"]),
            score_str(r["answer_relevance"]),
            score_str(r["element_coverage"]),
        )
        for k in totals:
            totals[k] += r[k]

    console.print(table)

    n = len(results)
    if n > 0:
        console.print()
        console.print(Panel(
            "[bold]Average Scores:[/bold]\n\n"
            f"  Context Relevance:  {totals['context_relevance']/n:.3f}  "
            f"{'[green]PASS[/green]' if totals['context_relevance']/n >= 0.7 else '[red]FAIL[/red]'}\n"
            f"  Faithfulness:       {totals['faithfulness']/n:.3f}  "
            f"{'[green]PASS[/green]' if totals['faithfulness']/n >= 0.8 else '[red]FAIL[/red]'}\n"
            f"  Answer Relevance:   {totals['answer_relevance']/n:.3f}  "
            f"{'[green]PASS[/green]' if totals['answer_relevance']/n >= 0.7 else '[red]FAIL[/red]'}\n"
            f"  Element Coverage:   {totals['element_coverage']/n:.3f}  "
            f"{'[green]PASS[/green]' if totals['element_coverage']/n >= 0.6 else '[red]FAIL[/red]'}\n\n"
            "[bold]Thresholds:[/bold] Faithfulness ≥ 0.80, Context/Answer Relevance ≥ 0.70",
            title="Summary",
            border_style="green" if all(
                totals[k] / n >= t for k, t in
                [("faithfulness", 0.8), ("context_relevance", 0.7), ("answer_relevance", 0.7)]
            ) else "yellow",
        ))

    # Show worst performer
    if results:
        worst = min(results, key=lambda r: r["faithfulness"])
        console.print()
        console.print(Panel(
            f"[bold]Lowest faithfulness score ({worst['faithfulness']:.2f}):[/bold]\n\n"
            f"Question: {worst['question']}\n\n"
            f"Answer: {worst['answer'][:300]}\n\n"
            f"Reason: {worst['faith_reason']}",
            title="Worst Result — Investigate This",
            border_style="yellow",
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["faiss", "bedrock-kb", "both"], default="faiss")
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    kb_id = os.getenv("KNOWLEDGE_BASE_ID", "")

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 4 — RAG Evaluation[/bold cyan]\n"
        "[dim]RAG Triad: Context Relevance + Faithfulness + Answer Relevance[/dim]",
        border_style="cyan",
    ))

    console.print()
    console.print(Panel(
        "[bold]The RAG Triad:[/bold]\n\n"
        "  [cyan]Context Relevance[/cyan]  → Did we retrieve the RIGHT chunks for this query?\n"
        "                      Low score = retrieval failure, bad chunking, or wrong embedding model\n\n"
        "  [cyan]Faithfulness[/cyan]       → Is the answer supported by the retrieved context?\n"
        "                      Low score = hallucination, model ignoring context\n\n"
        "  [cyan]Answer Relevance[/cyan]   → Does the answer address what was actually asked?\n"
        "                      Low score = off-topic answers, over-hedging\n\n"
        "  [cyan]Element Coverage[/cyan]   → Does the answer contain the expected key facts?\n"
        "                      Custom metric for ground-truth checking",
        border_style="blue",
    ))

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    bedrock_agent_rt = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

    console.print(f"\n[bold]Evaluating {len(EVAL_SET)} questions on pipeline: {args.pipeline}[/bold]")
    console.print("[dim]  Using Claude 3.5 Sonnet as LLM judge...[/dim]")

    results = evaluate_pipeline(bedrock, bedrock_agent_rt, kb_id, args.pipeline)
    render_report(results)

    console.print()
    console.print(Panel(
        "Next step:\n"
        "  [bold]uv run 04_gateway_tool.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
