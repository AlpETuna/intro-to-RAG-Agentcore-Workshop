#!/usr/bin/env python3
"""
Stage 0 — Prerequisites Check

Run this first. It verifies your Python version, AWS CLI installation,
AWS login status, Bedrock model access, and optional tools (Docker, agentcore-cli).

If you are not logged in, it will offer to run `aws sso login` for you.

Usage:
    python 00_check_prerequisites.py
    python 00_check_prerequisites.py --login   (trigger aws sso login immediately)
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
    ("anthropic.claude-3-haiku-20240307-v1:0", "Stage 1 generation"),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "Stage 3–4 agent reasoning"),
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
    ok = version >= (3, 10)
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    return [check("Python >= 3.10", ok, detail)]


def check_packages(packages: list) -> list:
    results = []
    for module, label in packages:
        spec = importlib.util.find_spec(module)
        ok = spec is not None
        detail = "installed" if ok else f"pip install {module.replace('dotenv', 'python-dotenv').replace('faiss', 'faiss-cpu')}"
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


def get_sso_profiles() -> list[str]:
    """Return SSO-configured profile names from ~/.aws/config."""
    config_path = Path.home() / ".aws" / "config"
    if not config_path.exists():
        return []
    profiles = []
    current_profile = None
    is_sso = False
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            if current_profile and is_sso:
                profiles.append(current_profile)
            name = line.strip("[]").replace("profile ", "")
            current_profile = name
            is_sso = False
        elif "sso_start_url" in line or "sso_account_id" in line:
            is_sso = True
    if current_profile and is_sso:
        profiles.append(current_profile)
    return profiles


def run_sso_login(profile: str = None) -> bool:
    """Run aws sso login [--profile <name>] interactively."""
    cmd = ["aws", "sso", "login"]
    if profile:
        cmd += ["--profile", profile]
    console.print(f"\n  [cyan]Running:[/cyan] {' '.join(cmd)}")
    console.print("  [dim]Your browser will open to complete the login.[/dim]\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def check_aws_credentials() -> list:
    """Verify boto3 can make an authenticated AWS call."""
    try:
        import boto3
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        auth_type = "SSO" if ":assumed-role/" in arn or "AWSReservedSSO" in arn else "IAM"
        return [check("AWS login", True, f"Account {account} ({auth_type}) | {arn.split('/')[-1]}")]
    except Exception as e:
        err = str(e)
        if "token" in err.lower() or "expired" in err.lower() or "sso" in err.lower():
            hint = "Token expired — run: aws sso login"
        elif "credential" in err.lower() or "NoCredentials" in err:
            hint = "Not logged in — run: aws sso login  (or aws configure)"
        else:
            hint = err[:80]
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
                results.append(check(
                    f"Model: {model_id}", False,
                    f"Enable in Bedrock Console → Model access ({purpose})"
                ))
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
        detail = result.stdout.strip() if ok else "Install Docker Desktop"
        return [check("Docker (Stage 3)", ok, detail)]
    except FileNotFoundError:
        return [check("Docker (Stage 3)", False, "Not found — needed for Stage 3 only")]
    except Exception as e:
        return [check("Docker (Stage 3)", False, str(e))]


def check_agentcore_cli() -> list:
    try:
        result = subprocess.run(
            ["agentcore", "--version"], capture_output=True, text=True, timeout=5
        )
        ok = result.returncode == 0
        detail = result.stdout.strip() if ok else "pip install agentcore-cli"
        return [check("agentcore-cli (Stage 3)", ok, detail)]
    except FileNotFoundError:
        return [check("agentcore-cli (Stage 3)", False, "Optional — pip install agentcore-cli")]
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
    import argparse
    parser = argparse.ArgumentParser(description="Stage 0 prerequisites check")
    parser.add_argument("--login", action="store_true",
                        help="Run aws sso login before checking credentials")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Intro to RAG with AgentCore[/bold cyan]\n"
        "[dim]Stage 0 — Prerequisites Check[/dim]",
        border_style="cyan",
    ))
    console.print()

    # ── Early: offer SSO login if --login flag passed ─────────────────────────
    if args.login:
        sso_profiles = get_sso_profiles()
        if sso_profiles:
            console.print(Panel(
                f"SSO profiles found: [bold]{', '.join(sso_profiles)}[/bold]\n\n"
                "Launching login for the first profile…",
                title="AWS SSO Login",
                border_style="cyan",
            ))
            ok = run_sso_login(sso_profiles[0])
        else:
            console.print("[dim]No SSO profiles found — running default aws sso login[/dim]")
            ok = run_sso_login()
        if not ok:
            console.print("[red]Login failed. Check your SSO configuration and try again.[/red]")
            sys.exit(1)
        console.print("[green]Login succeeded![/green] Re-running checks…\n")

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

    # ── Surface SSO login hint if credentials are the blocker ─────────────────
    login_blocked = any("AWS login" in r[0] for r in critical_fails)
    if login_blocked and not args.login:
        sso_profiles = get_sso_profiles()
        if sso_profiles:
            login_hint = (
                f"SSO profile(s) detected: [bold]{', '.join(sso_profiles)}[/bold]\n\n"
                f"  Run:  [bold]python 00_check_prerequisites.py --login[/bold]\n"
                f"  Or:   [bold]aws sso login --profile {sso_profiles[0]}[/bold]"
            )
        else:
            login_hint = (
                "No SSO profile found. Options:\n\n"
                "  [bold]aws configure sso[/bold]          — set up SSO (recommended)\n"
                "  [bold]aws configure[/bold]               — use static access keys\n"
                "  [bold]aws sso login[/bold]               — login to existing SSO config"
            )
        console.print()
        console.print(Panel(login_hint, title="How to Log In", border_style="cyan"))

    console.print()
    if not critical_fails:
        console.print(Panel(
            "[green]All critical checks passed![/green]\n\n"
            "Next step:\n"
            "  [bold]cd ../stage1-basic-rag && pip install -r requirements.txt[/bold]\n"
            "  [bold]python 01_chunk_and_embed.py[/bold]",
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
