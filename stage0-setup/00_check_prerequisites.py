#!/usr/bin/env python3
"""
Stage 0 — Prerequisites Check

Run this first. It verifies your Python version, AWS CLI installation,
AWS login status, Bedrock model access, and optional tools (Docker, agentcore-cli).

If you are not logged in, it tells you to run `aws login`.

Usage:
    uv run 00_check_prerequisites.py
"""

import sys
import json
import subprocess
import importlib.util
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

REQUIRED_MODELS = [
    ("amazon.titan-embed-text-v2:0", "Stage 1–4 embeddings"),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Stage 1 generation"),
    ("us.anthropic.claude-sonnet-4-6", "Stage 3–4 agent reasoning"),
]

REQUIRED_PACKAGES = [
    ("boto3", "AWS SDK"),
    ("rich", "Terminal output"),
    ("dotenv", "python-dotenv: config management"),
]

STAGE1_PACKAGES = [
    ("faiss", "FAISS vector store (faiss-cpu)"),
    ("numpy", "Numerical computing"),
]

STAGE3_PACKAGES = [
    ("strands", "Strands Agents framework"),
]


def check(label: str, ok: bool, detail: str = "") -> tuple[str, str, str]:
    status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    return label, status, detail


def check_python() -> list:
    version = sys.version_info
    ok = version >= (3, 11)
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    return [check("Python >= 3.11", ok, detail)]


def check_packages(packages: list) -> list:
    results = []
    for module, label in packages:
        spec = importlib.util.find_spec(module)
        ok = spec is not None
        detail = "installed" if ok else f"uv pip install {module.replace('dotenv', 'python-dotenv').replace('faiss', 'faiss-cpu')}"
        results.append(check(label, ok, detail))
    return results


def check_aws_cli() -> list:
    """Check that AWS CLI v2 is installed."""
    try:
        result = subprocess.run(
            ["aws", "--version"], capture_output=True, text=True, timeout=5
        )
        ok = result.returncode == 0
        raw = (result.stdout + result.stderr).strip()
        if ok:
            # raw is like "aws-cli/2.15.0 Python/3.12.0 ..."
            detail = raw.split(" ")[0]  # "aws-cli/2.15.0"
            is_v2 = "/2." in detail
            if not is_v2:
                return [check("AWS CLI v2", False,
                              f"Found {detail} — upgrade to v2: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")]
            return [check("AWS CLI v2", True, detail)]
        return [check("AWS CLI v2", False, "Not found — see install instructions below")]
    except FileNotFoundError:
        return [check("AWS CLI v2", False,
                      "Not installed — https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")]
    except Exception as e:
        return [check("AWS CLI v2", False, str(e))]


def check_aws_credentials() -> list:
    """Verify boto3 can make an authenticated AWS call."""
    try:
        import boto3
        # Give STS an explicit region so we never hit NoRegionError when no
        # default region is configured (aws login does not set one).
        region = boto3.session.Session().region_name or "us-east-1"
        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        return [check("AWS login", True, f"Account {account} | {arn.split('/')[-1]}")]
    except Exception as e:
        err = str(e)
        low = err.lower()
        if "missingdependency" in low or "botocore[crt]" in low or "crt" in low:
            hint = 'Run: uv pip install "botocore[crt]"  (needed for aws login)'
        elif "nocredentials" in low or "unable to locate credentials" in low \
                or "token" in low or "expired" in low or "sso" in low:
            hint = "Not logged in — run: aws login"
        else:
            # Surface the real error (with its type) so it can be diagnosed.
            hint = f"{type(e).__name__}: {err}"[:90]
        return [check("AWS login", False, hint)]


def check_aws_region() -> list:
    try:
        import boto3
        region = boto3.session.Session().region_name or "not set"
        ok = region != "not set"
        return [check("AWS region configured", ok, region or "Set AWS_DEFAULT_REGION=us-east-1")]
    except Exception as e:
        return [check("AWS region configured", False, str(e))]


def check_bedrock_models(region: str = "us-east-1") -> list:
    results = []
    try:
        import boto3
        bedrock = boto3.client("bedrock", region_name=region)
        response = bedrock.list_foundation_models()
        available_ids = {m["modelId"] for m in response.get("modelSummaries", [])}

        bedrock_rt = boto3.client("bedrock-runtime", region_name=region)
        for model_id, purpose in REQUIRED_MODELS:
            try:
                if "embed" in model_id:
                    body = json.dumps({"inputText": "test", "dimensions": 256, "normalize": True})
                else:
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Hi"}],
                    })
                bedrock_rt.invoke_model(
                    modelId=model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                results.append(check(f"Model: {model_id}", True, purpose))
            except bedrock_rt.exceptions.AccessDeniedException:
                if model_id.startswith("anthropic") or ".anthropic." in model_id:
                    hint = f"Submit the one-time Anthropic usage form in the Bedrock playground ({purpose})"
                else:
                    hint = f"Check your IAM bedrock:InvokeModel permission ({purpose})"
                results.append(check(f"Model: {model_id}", False, hint))
            except Exception as e:
                results.append(check(f"Model: {model_id}", False, str(e)[:80]))
    except Exception as e:
        results.append(check("Bedrock API access", False, str(e)))
    return results


def check_docker() -> list:
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        ok = result.returncode == 0
        detail = result.stdout.strip() if ok else "Optional — agentcore deploy uses CodeBuild"
        return [check("Docker (optional)", ok, detail)]
    except FileNotFoundError:
        return [check("Docker (optional)", False,
                      "Optional — only needed for `agentcore deploy --local-build`")]
    except Exception as e:
        return [check("Docker (optional)", False, str(e))]


def check_agentcore_cli() -> list:
    try:
        # The agentcore CLI has no --version flag; --help exits 0 when installed.
        result = subprocess.run(
            ["agentcore", "--help"], capture_output=True, text=True, timeout=5
        )
        ok = result.returncode == 0
        detail = "installed" if ok else "uv pip install bedrock-agentcore-starter-toolkit"
        return [check("agentcore-cli (Stage 3)", ok, detail)]
    except FileNotFoundError:
        return [check("agentcore-cli (Stage 3)", False,
                      "Stage 3 only — uv pip install bedrock-agentcore-starter-toolkit")]
    except Exception as e:
        return [check("agentcore-cli (Stage 3)", False, str(e))]


def check_data_files() -> list:
    data_dir = Path(__file__).parent / "data"
    expected = [
        "rag_fundamentals.txt",
        "bedrock_models.txt",
        "aws_agentcore.txt",
        "vector_databases.txt",
        "serverless_aws.txt",
    ]
    results = []
    for fname in expected:
        path = data_dir / fname
        ok = path.exists()
        size = f"{path.stat().st_size // 1024} KB" if ok else "missing"
        results.append(check(f"data/{fname}", ok, size))
    return results


def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Intro to RAG with AgentCore[/bold cyan]\n"
        "[dim]Stage 0 — Prerequisites Check[/dim]",
        border_style="cyan",
    ))
    console.print()

    table = Table(
        title="Environment Check",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Check", style="bold", min_width=42)
    table.add_column("Status", justify="center", min_width=8)
    table.add_column("Detail", style="dim")

    all_results = []

    # ── Section 1: Python + packages ──────────────────────────────────────────
    console.print("[cyan]1/5  Checking Python and packages...[/cyan]")
    results = check_python() + check_packages(REQUIRED_PACKAGES)
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    # ── Section 2: AWS CLI ────────────────────────────────────────────────────
    table.add_section()
    console.print("[cyan]2/5  Checking AWS CLI...[/cyan]")
    results = check_aws_cli()
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    # ── Section 3: AWS login + region ─────────────────────────────────────────
    table.add_section()
    console.print("[cyan]3/5  Checking AWS login and region...[/cyan]")
    cred_results = check_aws_credentials()
    region_results = check_aws_region()
    results = cred_results + region_results
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    # ── Section 4: Bedrock models ─────────────────────────────────────────────
    table.add_section()
    import boto3
    region = boto3.session.Session().region_name or "us-east-1"
    creds_ok = "PASS" in cred_results[0][1]
    if creds_ok:
        console.print(f"[cyan]4/5  Checking Bedrock model access in {region}...[/cyan]")
        results = check_bedrock_models(region)
    else:
        console.print("[dim]4/5  Skipping Bedrock check (not logged in)[/dim]")
        results = [check("Bedrock model access", False, "Login first, then re-run")]
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    # ── Section 5: Stage 1 packages ───────────────────────────────────────────
    table.add_section()
    console.print("[cyan]5/5  Checking Stage 1 + optional packages...[/cyan]")
    results = (
        check_packages(STAGE1_PACKAGES)
        + check_docker()
        + check_agentcore_cli()
        + check_packages(STAGE3_PACKAGES)
    )
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    # ── Section 6: Data files ─────────────────────────────────────────────────
    table.add_section()
    results = check_data_files()
    all_results.extend(results)
    for label, status, detail in results:
        table.add_row(label, status, detail)

    console.print()
    console.print(table)

    fails = [r for r in all_results if "FAIL" in r[1]]
    # Optional checks: Stage 3 tools (Docker, agentcore-cli, strands)
    critical_fails = [
        r for r in fails
        if not any(skip in r[0] for skip in ("Stage 3", "Docker", "agentcore-cli", "Strands"))
    ]

    # ── Surface login hint if credentials are the blocker ────────────────────
    login_blocked = any("AWS login" in r[0] for r in critical_fails)
    if login_blocked:
        login_hint = (
            "You are not logged in to AWS. Run:\n\n"
            "  [bold]aws login[/bold]\n\n"
            "and make sure your default region is us-east-1."
        )
        console.print()
        console.print(Panel(login_hint, title="How to Log In", border_style="cyan"))

    console.print()
    if not critical_fails:
        console.print(Panel(
            "[green]All critical checks passed![/green]\n\n"
            "Next step:\n"
            "  [bold]cd ../stage1-basic-rag && uv sync[/bold]\n"
            "  [bold]uv run 01_chunk_and_embed.py[/bold]",
            title="Ready to Start",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[red]Fix the following before continuing:[/red]\n\n"
            + "\n".join(f"  • {r[0]}: {r[2]}" for r in critical_fails),
            title="Action Required",
            border_style="red",
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
