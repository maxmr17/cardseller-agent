"""
cardseller-agent — eBay sports card listing generator
------------------------------------------------------
Usage:
  python main.py                    # Interactive mode (prompts for card info)
  python main.py --demo             # Run with a built-in demo card
  python main.py --output listing.txt   # Save final listing to a file
"""

import argparse
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from agents import (
    generate_listing,
    validate_and_refine,
    research_and_authenticate,
    fetch_market_sales,
    maxmr17_approve,
)
from models import CardInfo

load_dotenv()
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: OPENAI_API_KEY not set.[/red]")
        console.print("Copy [bold].env.example[/bold] to [bold].env[/bold] and add your key.")
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = console.input(f"[bold]{label}{suffix}:[/bold] ").strip()
    return value or default


def yn_prompt(label: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    value = console.input(f"[bold]{label} [{default_str}]:[/bold] ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


def collect_card_query() -> str:
    """Collect a free-text card description from the user."""
    console.print(Panel("[bold]Enter Card Details[/bold]", border_style="blue"))
    console.print("[dim]Describe the card — player, set, year, parallel, grade, etc. The AI will confirm every detail.[/dim]\n")
    query = console.input("[bold]Card description:[/bold] ").strip()
    while not query:
        console.print("[red]Card description is required.[/red]")
        query = console.input("[bold]Card description:[/bold] ").strip()
    return query


def demo_query() -> str:
    return "Patrick Mahomes 2017 Panini Prizm Silver Prizm RC #15"


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_auth_report(report) -> None:
    status = "[green]✅ Authenticated[/green]" if report.is_authentic else "[red]❌ Could Not Authenticate[/red]"
    console.print(Panel(
        f"[bold]Player:[/bold] {report.confirmed_player}\n"
        f"[bold]Set:[/bold] {report.confirmed_set}\n"
        f"[bold]Card #:[/bold] {report.confirmed_card_number}\n"
        f"[bold]Parallel:[/bold] {report.confirmed_parallel or 'Base'}\n"
        f"[bold]Rookie:[/bold] {'Yes' if report.is_rookie_card else 'No'}\n"
        f"[bold]Verdict:[/bold] {status}\n"
        + (f"\n[bold]Red flags:[/bold] {' · '.join(report.red_flags)}" if report.red_flags else ""),
        title="[bold blue]Card Authentication[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))


def display_listing(title: str, listing, border_color: str = "cyan") -> None:
    pricing = listing.pricing
    strategy_str = pricing.strategy.value.replace("_", " ").title()

    price_lines = [f"[bold]Strategy:[/bold] {strategy_str}"]
    if pricing.starting_price:
        price_lines.append(f"[bold]Starting Price:[/bold] ${pricing.starting_price:.2f}")
    if pricing.buy_it_now_price:
        price_lines.append(f"[bold]Buy It Now:[/bold] ${pricing.buy_it_now_price:.2f}")
    price_lines.append(f"[bold]Best Offer:[/bold] {'✓ Yes' if pricing.allow_offers else '✗ No'}")
    price_lines.append(f"[bold]Rationale:[/bold] {pricing.pricing_rationale}")

    title_len = len(listing.title)
    title_color = "green" if title_len <= 80 else "red"

    content = (
        f"[bold]TITLE[/bold] ([{title_color}]{title_len}/80 chars[/{title_color}]):\n"
        f"{listing.title}\n\n"
        f"[bold]SUBTITLE:[/bold]\n{listing.subtitle}\n\n"
        f"[bold]CONDITION:[/bold] {listing.condition_grade}\n{listing.condition_description}\n\n"
        f"[bold]PRICING:[/bold]\n{chr(10).join(price_lines)}\n\n"
        f"[bold]DESCRIPTION:[/bold]\n{listing.description}\n\n"
        f"[bold]KEYWORDS:[/bold] {', '.join(listing.search_keywords)}"
    )

    console.print(Panel(content, title=f"[bold]{title}[/bold]", border_style=border_color, padding=(1, 2)))

    if listing.item_specifics:
        table = Table(title="Item Specifics", box=box.SIMPLE, show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for s in listing.item_specifics:
            table.add_row(s.key, s.value)
        console.print(table)


def display_market_summary(report) -> None:
    console.print(Panel(
        f"[bold]Avg:[/bold] ${report.avg_price:.2f} | "
        f"[bold]Range:[/bold] ${report.low_price:.2f}–${report.high_price:.2f} | "
        f"[bold]Trend:[/bold] {report.price_trend}\n\n"
        f"{report.market_summary}\n\n"
        f"[bold]Pricing verdict:[/bold] {report.pricing_verdict}",
        title="[bold green]Market Data[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))


def display_validation_summary(validated) -> None:
    score = validated.confidence_score
    score_color = "green" if score >= 8 else "yellow" if score >= 6 else "red"

    console.print(Panel(
        f"[bold]Confidence Score:[/bold] [{score_color}]{score}/10[/{score_color}]\n\n"
        f"[bold]Market Insight:[/bold]\n{validated.market_insight}\n\n"
        f"[bold]Validation Notes:[/bold]\n{validated.validation_notes}",
        title="[bold yellow]Validator Report[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))


def save_listing(listing, validated, card: CardInfo, output_path: str) -> None:
    pricing = listing.pricing

    lines = [
        "=" * 70,
        "EBAY LISTING — Generated by cardseller-agent",
        "=" * 70,
        "",
        f"Card: {card.player_name} | {card.card_set} | #{card.card_number}",
        "",
        f"TITLE ({len(listing.title)}/80 chars):",
        listing.title,
        "",
        "SUBTITLE:",
        listing.subtitle,
        "",
        "CONDITION:",
        f"  Grade: {listing.condition_grade}",
        f"  Notes: {listing.condition_description}",
        "",
        "PRICING:",
        f"  Strategy: {pricing.strategy.value.replace('_', ' ').title()}",
    ]
    if pricing.starting_price:
        lines.append(f"  Starting Price: ${pricing.starting_price:.2f}")
    if pricing.buy_it_now_price:
        lines.append(f"  Buy It Now: ${pricing.buy_it_now_price:.2f}")
    lines += [
        f"  Best Offer: {'Yes' if pricing.allow_offers else 'No'}",
        f"  Rationale: {pricing.pricing_rationale}",
        "",
        "ITEM SPECIFICS:",
        *[f"  {s.key}: {s.value}" for s in listing.item_specifics],
        "",
        "DESCRIPTION:",
        listing.description,
        "",
        f"SEARCH KEYWORDS: {', '.join(listing.search_keywords)}",
        "",
        "-" * 70,
        f"Confidence Score: {validated.confidence_score}/10",
        "",
        "MARKET INSIGHT:",
        validated.market_insight,
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[green]✓ Listing saved to[/green] [bold]{output_path}[/bold]")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(query: str, client: openai.OpenAI, output_path: str | None = None) -> None:
    console.print(Rule("[bold blue]cardseller-agent[/bold blue]"))

    # Step 1: Research + authenticate (2 API calls)
    research_result = research_and_authenticate(query, client)
    card = research_result.card
    auth = research_result.auth

    console.print(f"\n[bold]Card:[/bold] {card.player_name} — {card.card_set} #{card.card_number}\n")
    display_auth_report(auth)

    if not auth.is_authentic:
        console.print("[yellow]⚠ Authentication concerns found — review red flags above before listing.[/yellow]")

    # Step 2: Market data
    console.print(Rule("[dim]Market Research[/dim]"))
    try:
        market_report = fetch_market_sales(card, client)
        display_market_summary(market_report)
        market_context = (
            f"Recent sales avg: ${market_report.avg_price:.2f} | "
            f"Range: ${market_report.low_price:.2f}–${market_report.high_price:.2f} | "
            f"Trend: {market_report.price_trend}\n"
            + "\n".join(
                f"- {sale.sale_date}: ${sale.price:.2f} ({sale.condition}) on {sale.platform}"
                for sale in market_report.sales[:5]
            )
        )
        price_verdict = market_report.pricing_verdict
    except Exception as e:
        console.print(f"[yellow]⚠ Market data unavailable: {e}[/yellow]")
        market_report = None
        market_context = ""
        price_verdict = ""

    # Step 3: Generate draft
    console.print(Rule("[dim]Draft Listing[/dim]"))
    draft = generate_listing(card, client, market_context=market_context)
    display_listing("Draft", draft, border_color="dim")

    # Step 4: Validate + refine
    validated = validate_and_refine(
        card, draft, client,
        market_context=market_context,
        price_verdict=price_verdict,
    )
    display_validation_summary(validated)

    # Step 5: maxmr17 style loop (up to 4 passes)
    console.print(Rule("[bold]maxmr17 Style Review[/bold]"))
    current_listing = validated.refined_listing
    style_approval = None
    for attempt in range(1, 5):
        console.print(f"[magenta]Pass {attempt}/4…[/magenta]")
        style_approval = maxmr17_approve(card, current_listing, client, market_context=market_context)
        if style_approval.style_score >= 10:
            break
        current_listing = style_approval.final_listing

    final_listing = style_approval.final_listing if style_approval else validated.refined_listing

    console.print(Rule("[bold]Final Listing[/bold]"))
    display_listing("maxmr17-Approved Listing ✓", final_listing, border_color="green")

    if output_path:
        save_listing(final_listing, validated, card, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eBay listings for sports cards")
    parser.add_argument("--demo", action="store_true", help="Run with a demo card (Patrick Mahomes RC)")
    parser.add_argument("--output", type=str, default=None, help="Save final listing to this file path")
    args = parser.parse_args()

    client = get_client()

    if args.demo:
        query = demo_query()
        console.print(Panel(f"[bold]Demo card:[/bold] {query}", border_style="dim"))
    else:
        query = collect_card_query()

    run(query, client, output_path=args.output)


if __name__ == "__main__":
    main()
