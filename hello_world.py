#!/usr/bin/env python3
"""Smoke-test every known model with a simple Hello, World! request."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.access import spec_selector  # noqa: E402
from src.experiments.runner import API_KEY_ENV, ENDPOINT_ENV, infer_provider_name  # noqa: E402
from src.experiments.selection import known_model_specs, parse_model_selectors  # noqa: E402
from src.providers import (  # noqa: E402
    ProviderError,
    RateLimitProviderError,
    TemporaryProviderError,
    get_provider,
)

console = Console()

HELLO_WORLD_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a connectivity smoke test. Reply with exactly this text and nothing else: "
            "Hello, World!"
        ),
    },
    {
        "role": "user",
        "content": "Please reply with exactly: Hello, World!",
    },
]
HELLO_WORLD_MAX_TOKENS = 32
HELLO_WORLD_MAX_ATTEMPTS = 3
HELLO_WORLD_MAX_WAIT_SECONDS = 20.0


@dataclass(frozen=True)
class HelloWorldResult:
    """Outcome of a single Hello, World! smoke test."""

    selector: str
    provider: str
    model: str
    endpoint: str
    reachable: bool
    exact_match: bool
    status: str
    response_excerpt: str = ""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Send a minimal Hello, World! prompt to each known model and report whether "
            "the current credentials and endpoints can get a response."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Optional repeatable model selector filter.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Optional repeatable provider filter, for example --provider cerebras.",
    )
    return parser.parse_args()


def endpoint_for_provider(provider_name: str) -> str:
    """Return the configured endpoint for a provider, if any."""
    endpoint_env = ENDPOINT_ENV.get(provider_name, "")
    if not endpoint_env:
        return "(provider default)"
    return os.getenv(endpoint_env, "").strip() or f"(missing {endpoint_env})"


def missing_provider_requirements(provider_name: str) -> list[str]:
    """Return missing env requirements for a provider."""
    missing: list[str] = []
    key_env = API_KEY_ENV.get(provider_name, "")
    endpoint_env = ENDPOINT_ENV.get(provider_name, "")

    if key_env and not os.getenv(key_env):
        missing.append(key_env)
    if endpoint_env and not os.getenv(endpoint_env):
        missing.append(endpoint_env)
    return missing


def summarize_response(content: str, *, limit: int = 60) -> str:
    """Compress a response into a small single-line excerpt."""
    compact = " ".join(content.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def provider_error_indicates_reachability(message: str) -> bool:
    """Return True when a provider error still proves the endpoint answered."""
    normalized = message.casefold()
    markers = (
        "429",
        "rate limit",
        "rate-limit",
        "rate limited",
        "too many requests",
        "resource_exhausted",
    )
    return any(marker in normalized for marker in markers)


def is_exact_hello_world(content: str) -> bool:
    """Return True when the provider replied with the expected text."""
    normalized = content.strip().strip('"').strip("'")
    return normalized == "Hello, World!"


async def check_model_hello_world(selector: str) -> HelloWorldResult:
    """Send the Hello, World! probe to a selected model."""
    load_dotenv(ROOT / ".env", override=False)
    spec = parse_model_selectors([selector])[0]
    provider_name = spec.provider or infer_provider_name(spec.model)
    endpoint = endpoint_for_provider(provider_name)
    missing = missing_provider_requirements(provider_name)

    if missing:
        return HelloWorldResult(
            selector=spec_selector(spec),
            provider=provider_name,
            model=spec.model,
            endpoint=endpoint,
            reachable=False,
            exact_match=False,
            status="missing " + ", ".join(missing),
        )

    api_key = os.getenv(API_KEY_ENV.get(provider_name, "")) or None
    base_url = os.getenv(ENDPOINT_ENV.get(provider_name, "")) or None

    try:
        client = get_provider(
            provider_name,
            model=spec.model,
            api_key=api_key,
            base_url=base_url,
        )
    except ProviderError as exc:
        return HelloWorldResult(
            selector=spec_selector(spec),
            provider=provider_name,
            model=spec.model,
            endpoint=endpoint,
            reachable=False,
            exact_match=False,
            status=str(exc),
        )

    last_retryable_error: TemporaryProviderError | None = None

    try:
        for attempt in range(HELLO_WORLD_MAX_ATTEMPTS):
            try:
                response = await client.complete(
                    messages=HELLO_WORLD_MESSAGES,
                    temperature=0.0,
                    max_tokens=HELLO_WORLD_MAX_TOKENS,
                )
                content = response.content.strip()
                if not content:
                    return HelloWorldResult(
                        selector=spec_selector(spec),
                        provider=provider_name,
                        model=spec.model,
                        endpoint=endpoint,
                        reachable=False,
                        exact_match=False,
                        status="empty response",
                    )
                exact_match = is_exact_hello_world(content)
                return HelloWorldResult(
                    selector=spec_selector(spec),
                    provider=provider_name,
                    model=spec.model,
                    endpoint=endpoint,
                    reachable=True,
                    exact_match=exact_match,
                    status="exact hello world" if exact_match else "responded",
                    response_excerpt=summarize_response(content),
                )
            except RateLimitProviderError as exc:
                return HelloWorldResult(
                    selector=spec_selector(spec),
                    provider=provider_name,
                    model=spec.model,
                    endpoint=endpoint,
                    reachable=True,
                    exact_match=False,
                    status=f"rate limited ({exc})",
                )
            except TemporaryProviderError as exc:
                last_retryable_error = exc
                if attempt == HELLO_WORLD_MAX_ATTEMPTS - 1:
                    break
                suggested_wait = getattr(exc, "retry_after_seconds", None)
                wait_seconds = max(1.0, suggested_wait or (2**attempt))
                await asyncio.sleep(min(HELLO_WORLD_MAX_WAIT_SECONDS, wait_seconds))
            except ProviderError as exc:
                message = str(exc)
                if provider_error_indicates_reachability(message):
                    return HelloWorldResult(
                        selector=spec_selector(spec),
                        provider=provider_name,
                        model=spec.model,
                        endpoint=endpoint,
                        reachable=True,
                        exact_match=False,
                        status=f"reachable but rate limited ({message})",
                    )
                return HelloWorldResult(
                    selector=spec_selector(spec),
                    provider=provider_name,
                    model=spec.model,
                    endpoint=endpoint,
                    reachable=False,
                    exact_match=False,
                    status=message,
                )
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    return HelloWorldResult(
        selector=spec_selector(spec),
        provider=provider_name,
        model=spec.model,
        endpoint=endpoint,
        reachable=False,
        exact_match=False,
        status=str(last_retryable_error or "probe failed"),
    )


def select_model_selectors(args: argparse.Namespace) -> list[str]:
    """Return the selected provider:model selectors for this run."""
    if args.model:
        return [spec_selector(spec) for spec in parse_model_selectors(args.model)]

    providers = {provider.strip().lower() for provider in args.provider if provider.strip()}
    selectors: list[str] = []
    for spec in known_model_specs():
        selector = spec_selector(spec)
        if providers and (spec.provider or infer_provider_name(spec.model)) not in providers:
            continue
        selectors.append(selector)
    return selectors


async def gather_results(selectors: list[str]) -> list[HelloWorldResult]:
    """Run the Hello, World! checks sequentially to avoid unnecessary rate limits."""
    results: list[HelloWorldResult] = []
    for selector in selectors:
        results.append(await check_model_hello_world(selector))
    return results


def render_results(results: list[HelloWorldResult]) -> None:
    """Show a readable summary table."""
    exact = sum(1 for result in results if result.exact_match)
    reachable = sum(1 for result in results if result.reachable)
    failed = sum(1 for result in results if not result.reachable)

    summary = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    summary.add_column("Label", style="bold cyan", no_wrap=True)
    summary.add_column("Value", style="white")
    summary.add_row("Checked models", str(len(results)))
    summary.add_row("Exact Hello, World!", str(exact))
    summary.add_row("Reachable", str(reachable))
    summary.add_row("Failed", str(failed))
    console.print(Panel(summary, title="Hello, World! Smoke Test", border_style="cyan", box=box.ROUNDED))
    console.print()

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Provider", style="bold cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Endpoint", style="dim")
    table.add_column("Status", no_wrap=True)
    table.add_column("Sample Response", style="green")

    for result in results:
        if result.exact_match:
            status_style = "bold green"
        elif result.reachable:
            status_style = "yellow"
        else:
            status_style = "red"

        table.add_row(
            result.provider,
            result.model,
            result.endpoint,
            f"[{status_style}]{result.status}[/{status_style}]",
            result.response_excerpt or "-",
        )

    console.print(table)
    console.print()


def main() -> int:
    """Execute the CLI."""
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args()
    selectors = select_model_selectors(args)
    if not selectors:
        console.print("[red]No models matched the requested filters.[/red]")
        return 1

    with console.status("[bold cyan]Checking model responses...[/bold cyan]", spinner="dots"):
        results = asyncio.run(gather_results(selectors))

    render_results(results)
    return 0 if all(result.reachable for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
