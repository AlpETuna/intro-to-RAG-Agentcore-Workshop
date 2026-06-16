#!/usr/bin/env python3
"""
Stage 2, Script 1 — Create Infrastructure

Creates the AWS resources needed for a Bedrock Knowledge Base:
  1. S3 bucket — stores the source documents
  2. IAM role — grants Bedrock permission to read S3 and write to OpenSearch
  3. Uploads the 5 workshop documents to S3
  4. OpenSearch Serverless collection + vector index — the vector store the KB
     writes embeddings into (Bedrock requires this to exist before the KB is
     created; it is NOT auto-created by the CreateKnowledgeBase API)

Outputs resource IDs to .env (in the repo root) for subsequent scripts.

Usage:
    uv run 01_create_infrastructure.py
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
COLLECTION_NAME = f"rag-kb-{SUFFIX}"          # 3–32 chars, lowercase
INDEX_NAME = "rag-workshop-index"

console = Console()


def get_caller(sts) -> tuple[str, str]:
    """Return (account_id, principal_arn) usable in an aoss data-access policy.

    aoss principals must be an IAM user/role ARN, not an STS assumed-role
    session ARN, so normalize assumed-role ARNs back to the role ARN.
    """
    ident = sts.get_caller_identity()
    account_id = ident["Account"]
    arn = ident["Arn"]
    if ":assumed-role/" in arn:
        role_name = arn.split(":assumed-role/")[1].split("/")[0]
        arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    return account_id, arn


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


def create_oss_collection(aoss, account_id: str, kb_role_arn: str, caller_arn: str) -> tuple[str, str]:
    """Create an OpenSearch Serverless VECTORSEARCH collection + its policies.

    A collection needs three policies before it can be created:
      • encryption  (how data at rest is encrypted — we use the AWS-owned key)
      • network     (public access to the collection + dashboards endpoint)
      • data access (which principals may read/write the collection + indexes)
    """
    console.print(f"\n[bold]Creating OpenSearch Serverless collection:[/bold] {COLLECTION_NAME}")

    # Principals that need data access: the KB service role (to write embeddings)
    # and whoever runs this script (to create the vector index below).
    principals = sorted({kb_role_arn, caller_arn})

    policies = [
        ("encryption", f"rag-enc-{SUFFIX}", json.dumps({
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]}],
            "AWSOwnedKey": True,
        })),
        ("network", f"rag-net-{SUFFIX}", json.dumps([{
            "Rules": [
                {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"]},
                {"ResourceType": "dashboard", "Resource": [f"collection/{COLLECTION_NAME}"]},
            ],
            "AllowFromPublic": True,
        }])),
        ("data", f"rag-acc-{SUFFIX}", json.dumps([{
            "Rules": [
                {"ResourceType": "index", "Resource": [f"index/{COLLECTION_NAME}/*"],
                 "Permission": ["aoss:*"]},
                {"ResourceType": "collection", "Resource": [f"collection/{COLLECTION_NAME}"],
                 "Permission": ["aoss:*"]},
            ],
            "Principal": principals,
        }])),
    ]

    for ptype, name, policy in policies:
        try:
            if ptype == "data":
                aoss.create_access_policy(name=name, type="data", policy=policy)
            else:
                aoss.create_security_policy(name=name, type=ptype, policy=policy)
            console.print(f"  [green]✓ {ptype} policy:[/green] {name}")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("ConflictException",):
                console.print(f"  [yellow]{ptype} policy already exists:[/yellow] {name}")
            else:
                raise

    try:
        aoss.create_collection(name=COLLECTION_NAME, type="VECTORSEARCH",
                               description="RAG workshop vector store")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ConflictException",):
            raise
        console.print(f"  [yellow]Collection already exists:[/yellow] {COLLECTION_NAME}")

    # Wait for ACTIVE and grab the ARN + endpoint
    console.print("  [dim]Waiting for collection to become ACTIVE (1–2 min)...[/dim]")
    for _ in range(60):
        details = aoss.batch_get_collection(names=[COLLECTION_NAME])["collectionDetails"]
        if details:
            status = details[0]["status"]
            if status == "ACTIVE":
                arn = details[0]["arn"]
                endpoint = details[0]["collectionEndpoint"]
                console.print(f"  [green]✓ Collection ACTIVE:[/green] {arn}")
                return arn, endpoint
            if status == "FAILED":
                console.print("  [red]Collection creation FAILED[/red]")
                raise SystemExit(1)
        time.sleep(5)
    raise TimeoutError("OpenSearch Serverless collection did not become ACTIVE in time")


def create_vector_index(endpoint: str) -> None:
    """Create the kNN vector index Bedrock will write embeddings into."""
    try:
        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
    except ImportError:
        console.print("[red]opensearch-py not installed — run `uv sync` in stage2-bedrock-kb.[/red]")
        raise SystemExit(1)

    host = endpoint.replace("https://", "")
    creds = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(creds, AWS_REGION, "aoss")
    client = OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )

    index_body = {
        "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 512}},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw", "engine": "faiss", "space_type": "l2",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
                "text": {"type": "text"},
                "metadata": {"type": "text"},
            }
        },
    }

    console.print(f"\n[bold]Creating vector index:[/bold] {INDEX_NAME}")
    # A freshly-created data-access policy can take 1–3 minutes to propagate,
    # during which the data plane returns 403 security_exception. Retry for
    # up to ~5 minutes before giving up.
    last_err = None
    warned = False
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            if not client.indices.exists(index=INDEX_NAME):
                client.indices.create(index=INDEX_NAME, body=index_body)
            console.print(f"  [green]✓ Index created:[/green] {INDEX_NAME}")
            # Index needs a moment to be queryable before the KB references it.
            console.print("  [dim]Waiting 30s for the index to settle...[/dim]")
            time.sleep(30)
            return
        except Exception as e:  # opensearch-py raises various auth/connection errors
            last_err = e
            if not warned and "403" in str(e):
                console.print("  [dim]Waiting for the data-access policy to propagate "
                              "(403s are expected for a minute or two)...[/dim]")
                warned = True
            time.sleep(10)
    console.print(f"  [red]Failed to create index after 5 min:[/red] {last_err}")
    raise SystemExit(1)


def save_env(bucket_name: str, role_arn: str, collection_arn: str) -> None:
    env_path = str(ENV_FILE)
    if not ENV_FILE.exists():
        ENV_FILE.write_text("")
    set_key(env_path, "S3_BUCKET_NAME", bucket_name)
    set_key(env_path, "KB_IAM_ROLE_ARN", role_arn)
    set_key(env_path, "AWS_REGION", AWS_REGION)
    set_key(env_path, "OPENSEARCH_COLLECTION_ARN", collection_arn)
    set_key(env_path, "OPENSEARCH_COLLECTION_NAME", COLLECTION_NAME)
    set_key(env_path, "OPENSEARCH_INDEX_NAME", INDEX_NAME)
    console.print(f"\n  [green]✓ Saved to .env:[/green] S3_BUCKET_NAME, KB_IAM_ROLE_ARN, OPENSEARCH_COLLECTION_ARN")


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
    aoss = boto3.client("opensearchserverless", region_name=AWS_REGION)

    account_id, caller_arn = get_caller(sts)
    console.print(f"\n  AWS Account: [bold]{account_id}[/bold]")
    console.print(f"  Region:      [bold]{AWS_REGION}[/bold]")
    console.print(f"  Suffix:      [bold]{SUFFIX}[/bold]  (makes resource names unique)")

    bucket_name = create_s3_bucket(s3, BUCKET_NAME, AWS_REGION)
    uploaded = upload_documents(s3, bucket_name)
    role_arn = create_kb_iam_role(iam, account_id, bucket_name, AWS_REGION)
    collection_arn, endpoint = create_oss_collection(aoss, account_id, role_arn, caller_arn)
    create_vector_index(endpoint)
    save_env(bucket_name, role_arn, collection_arn)

    # Summary
    table = Table("Resource", "Value", show_header=True, header_style="bold magenta")
    table.add_row("S3 Bucket", bucket_name)
    table.add_row("Documents uploaded", str(len(uploaded)))
    table.add_row("IAM Role ARN", role_arn[:60] + "…")
    table.add_row("OpenSearch Collection", collection_arn)
    table.add_row("Vector Index", INDEX_NAME)

    console.print()
    console.print(table)
    console.print()
    console.print(Panel(
        "[green]Infrastructure ready![/green]\n\n"
        "Next step:\n"
        "  [bold]uv run 02_create_knowledge_base.py[/bold]",
        title="Done",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
