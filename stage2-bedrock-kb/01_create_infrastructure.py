#!/usr/bin/env python3
"""
Stage 2, Script 1 — Create Infrastructure

Creates the AWS resources needed for a Bedrock Knowledge Base:
  1. S3 bucket — stores the source documents
  2. IAM role — grants Bedrock permission to read S3 and write to OpenSearch
  3. Uploads the 5 workshop documents to S3

Outputs resource IDs to .env (in the repo root) for subsequent scripts.

Usage:
    python 01_create_infrastructure.py
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import dotenv_values, set_key
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DATA_DIR = Path(__file__).parent.parent / "stage0-setup" / "data"
ENV_FILE = Path(__file__).parent.parent / ".env"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SUFFIX = uuid.uuid4().hex[:8]
BUCKET_NAME = f"rag-workshop-{SUFFIX}"

console = Console()


def get_account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def create_s3_bucket(s3, bucket_name: str, region: str) -> str:
    console.print(f"\n[bold]Creating S3 bucket:[/bold] {bucket_name}")
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        s3.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        console.print(f"  [green]✓ Created:[/green] s3://{bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
            console.print(f"  [yellow]Already exists:[/yellow] s3://{bucket_name}")
        else:
            raise
    return bucket_name


def upload_documents(s3, bucket_name: str) -> list[str]:
    console.print(f"\n[bold]Uploading documents to s3://{bucket_name}/docs/[/bold]")
    uploaded = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        key = f"docs/{path.name}"
        s3.upload_file(str(path), bucket_name, key)
        size = path.stat().st_size
        console.print(f"  [green]✓[/green] {key} ({size:,} bytes)")
        uploaded.append(key)
    return uploaded


def create_kb_iam_role(iam, account_id: str, bucket_name: str, region: str) -> str:
    role_name = f"BedrockKBRole-{SUFFIX}"
    console.print(f"\n[bold]Creating IAM role:[/bold] {role_name}")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account_id},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/*"
                    },
                },
            }
        ],
    }

    inline_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ReadAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket"],
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*",
                ],
            },
            {
                "Sid": "BedrockEmbeddingAccess",
                "Effect": "Allow",
                "Action": ["bedrock:InvokeModel"],
                "Resource": [
                    f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-embed-text-v2:0"
                ],
            },
            {
                "Sid": "OpenSearchServerlessAccess",
                "Effect": "Allow",
                "Action": ["aoss:APIAccessAll"],
                "Resource": [f"arn:aws:aoss:{region}:{account_id}:collection/*"],
            },
        ],
    }

    try:
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Bedrock Knowledge Base service role for RAG workshop",
        )
        role_arn = role["Role"]["Arn"]
        console.print(f"  [green]✓ Created role:[/green] {role_arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
            console.print(f"  [yellow]Already exists:[/yellow] {role_arn}")
        else:
            raise

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="BedrockKBInlinePolicy",
        PolicyDocument=json.dumps(inline_policy),
    )
    console.print("  [green]✓ Inline policy attached[/green]")

    # IAM propagation takes a few seconds
    console.print("  [dim]Waiting 10s for IAM propagation...[/dim]")
    time.sleep(10)

    return role_arn


def save_env(bucket_name: str, role_arn: str) -> None:
    env_path = str(ENV_FILE)
    if not ENV_FILE.exists():
        ENV_FILE.write_text("")
    set_key(env_path, "S3_BUCKET_NAME", bucket_name)
    set_key(env_path, "KB_IAM_ROLE_ARN", role_arn)
    set_key(env_path, "AWS_REGION", AWS_REGION)
    console.print(f"\n  [green]✓ Saved to .env:[/green] S3_BUCKET_NAME, KB_IAM_ROLE_ARN")


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Stage 2 — Create Infrastructure[/bold cyan]\n"
        "[dim]S3 bucket + IAM role for Bedrock Knowledge Base[/dim]",
        border_style="cyan",
    ))

    sts = boto3.client("sts", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)

    account_id = get_account_id(sts)
    console.print(f"\n  AWS Account: [bold]{account_id}[/bold]")
    console.print(f"  Region:      [bold]{AWS_REGION}[/bold]")
    console.print(f"  Suffix:      [bold]{SUFFIX}[/bold]  (makes resource names unique)")

    bucket_name = create_s3_bucket(s3, BUCKET_NAME, AWS_REGION)
    uploaded = upload_documents(s3, bucket_name)
    role_arn = create_kb_iam_role(iam, account_id, bucket_name, AWS_REGION)
    save_env(bucket_name, role_arn)

    # Summary
    table = Table("Resource", "Value", show_header=True, header_style="bold magenta")
    table.add_row("S3 Bucket", bucket_name)
    table.add_row("Documents uploaded", str(len(uploaded)))
    table.add_row("IAM Role ARN", role_arn[:60] + "…")

    console.print()
    console.print(table)
    console.print()
    console.print(Panel(
        "[green]Infrastructure ready![/green]\n\n"
        "Next step:\n"
        "  [bold]python 02_create_knowledge_base.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
