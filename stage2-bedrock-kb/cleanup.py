#!/usr/bin/env python3
"""
Stage 2 — Cleanup

Deletes all Stage 2 resources to stop OpenSearch Serverless charges:
  - Bedrock Knowledge Base (and its OpenSearch Serverless collection)
  - S3 bucket (empties it first)
  - IAM role and inline policies

⚠️  This is irreversible. Re-run 01_create_infrastructure.py and
    02_create_knowledge_base.py to recreate resources.

Usage:
    python cleanup.py
    python cleanup.py --yes   (skip confirmation)
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel

ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

console = Console()


def load_config():
    load_dotenv(ENV_FILE)
    return {
        "kb_id": os.getenv("KNOWLEDGE_BASE_ID"),
        "ds_id": os.getenv("KB_DATA_SOURCE_ID"),
        "bucket": os.getenv("S3_BUCKET_NAME"),
        "role_arn": os.getenv("KB_IAM_ROLE_ARN"),
    }


def delete_kb(bedrock_agent, kb_id: str):
    if not kb_id:
        console.print("[dim]No KNOWLEDGE_BASE_ID in .env — skipping[/dim]")
        return
    try:
        bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)
        console.print(f"  [green]✓ Deleted Knowledge Base:[/green] {kb_id}")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceNotFoundException",):
            console.print(f"  [dim]KB already gone: {kb_id}[/dim]")
        else:
            console.print(f"  [yellow]KB delete error:[/yellow] {e}")


def empty_and_delete_bucket(s3, bucket: str):
    if not bucket:
        console.print("[dim]No S3_BUCKET_NAME in .env — skipping[/dim]")
        return
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objects:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
                console.print(f"  [dim]Deleted {len(objects)} objects[/dim]")
        s3.delete_bucket(Bucket=bucket)
        console.print(f"  [green]✓ Deleted S3 bucket:[/green] {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            console.print(f"  [dim]Bucket already gone: {bucket}[/dim]")
        else:
            console.print(f"  [yellow]S3 delete error:[/yellow] {e}")


def delete_iam_role(iam, role_arn: str):
    if not role_arn:
        console.print("[dim]No KB_IAM_ROLE_ARN in .env — skipping[/dim]")
        return
    role_name = role_arn.split("/")[-1]
    try:
        # Remove inline policies first
        for policy in iam.list_role_policies(RoleName=role_name).get("PolicyNames", []):
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy)
        iam.delete_role(RoleName=role_name)
        console.print(f"  [green]✓ Deleted IAM role:[/green] {role_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            console.print(f"  [dim]Role already gone: {role_name}[/dim]")
        else:
            console.print(f"  [yellow]IAM delete error:[/yellow] {e}")


def clear_env():
    env_path = str(ENV_FILE)
    for key in ["KNOWLEDGE_BASE_ID", "KNOWLEDGE_BASE_ARN", "KB_DATA_SOURCE_ID",
                 "S3_BUCKET_NAME", "KB_IAM_ROLE_ARN"]:
        set_key(env_path, key, "")
    console.print("  [green]✓ Cleared Stage 2 values from .env[/green]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    config = load_config()

    console.print()
    console.print(Panel(
        "[bold red]Stage 2 Cleanup[/bold red]\n\n"
        "The following resources will be PERMANENTLY DELETED:\n\n"
        f"  • Knowledge Base:  {config['kb_id'] or '(not set)'}\n"
        f"  • S3 Bucket:       {config['bucket'] or '(not set)'}\n"
        f"  • IAM Role:        {config['role_arn'] or '(not set)'}\n",
        border_style="red",
    ))

    if not args.yes:
        confirm = console.input("Type [bold]yes[/bold] to confirm deletion: ").strip().lower()
        if confirm != "yes":
            console.print("[dim]Aborted.[/dim]")
            return

    bedrock_agent = boto3.client("bedrock-agent", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)

    console.print("\n[bold]Deleting resources...[/bold]")
    delete_kb(bedrock_agent, config["kb_id"])
    empty_and_delete_bucket(s3, config["bucket"])
    delete_iam_role(iam, config["role_arn"])
    clear_env()

    console.print()
    console.print(Panel(
        "[green]Stage 2 resources deleted.[/green]\n\n"
        "OpenSearch Serverless charges stop once the KB is deleted.\n"
        "Check the AWS Cost Explorer in 24 hours to confirm.",
        title="Cleanup Complete",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
