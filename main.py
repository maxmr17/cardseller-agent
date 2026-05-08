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

from agents import generate_listing, validate_and_refine
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


def collect_card_info() -> CardInfo:
    console.print(Panel("[bold]Enter Card Details[/bold]", border_style="blue"))

    player_name = prompt("Player name")
    while not player_name:
        console.print("[red]Player name is required.[/red]")
        player_name = prompt("Player name")

    card_set = prompt("Card set (e.g. 2023 Panini Prizm Football)")
    while not card_set:
        console.print("[red]Card set is required.[/red]")
        card_set = prompt("Card set")

    card_number = prompt("Card number (e.g. #/99, BASE, #123)")
    while not card_number:
        console.print("[red]Card number is required.[/red]")
        card_number = prompt("Card number")

    sport = prompt("Sport", default="Football")
    is_rookie = yn_prompt("Rookie card (RC)?", default=False)
    is_auto = yn_prompt("Autographed?", default=False)

    is_graded = yn_prompt("Professionally graded?", default=False)
    grade = None
    if is_graded:
        grade = prompt("Grade (e.g. PSA 10, BGS 9.5, SGC 10)")

    serial = prompt("Serial number (leave blank if not numbered)") or None
    parallel = prompt("Parallel/variant (e.g. Silver Prizm, Gold Refractor — leave blank if base)") or None
    condition = prompt("Condition notes (leave blank to skip)") or None
    extra_notes = prompt("Any other details (leave blank to skip)") or None

    return CardInfo(
        player_name=player_name,
        card_number=card_number,
        card_set=card_set,
        sport=sport,
        is_rookie_card=is_rookie,
        is_autographed=is_auto,
        is_graded=is_graded,
        grade=grade,
        serial_number=serial,
        parallel=parallel,
        condition=condition,
        extra_notes=extra_notes,
    )


def demo_card() -> CardInfo:
    """A realistic demo card for testing."""
    return CardInfo(
        player_name="Patrick Mahomes",
        card_number="15",
        card_set="2017 Panini Prizm Football",
        sport="Football",
        is_rookie_card=True,
        is_autographed=False,
        is_graded=False,
        serial_number=None,
        parallel="Silver Prizm",
        condition="Near mint. Sharp corners, no creases. Light surface wear visible under direct light.",
        extra_notes=None,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_listing(title: str, listing, border_color: str = "cyan") -> None:
    """Print a CardListing in a formatted panel."""
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

    content = f"""[bold]TITLE[/bold] ([{title_color}]{title_len}/80 chars[/{title_color}]):
{listing.title}

[bold]SUBTITLE:[/bold]
{listing.subtitle}

[bold]CONDITION:[/bold] {listing.condition_grade}
{listing.condition_description}

[bold]PRICING:[/bold]
{chr(10).join(price_lines)}

[bold]DESCRIPTION:[/bold]
{listing.description}

[bold]KEYWORDS:[/bold] {', '.join(listing.search_keywords)}"""

    console.print(Panel(content, title=f"[bold]{title}[/bold]", border_style=border_color, padding=(1, 2)))

    # Item specifics table
    if listing.item_specifics:
        table = Table(title="Item Specifics", box=box.SIMPLE, show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for s in listing.item_specifics:
            table.add_row(s.key, s.value)
        console.print(table)


def display_validation_summary(validated) -> None:
    """Display the validator's notes and confidence score."""
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


def save_listing(validated, card: CardInfo, output_path: str) -> None:
    """Save the final listing to a plain-text file."""
    listing = validated.refined_listing
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
        f"SUBTITLE:",
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
    ]
    for s in listing.item_specifics:
        lines.append(f"  {s.key}: {s.value}")
    lines += [
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

def run(card: CardInfo, client: openai.OpenAI, output_path: str | None = None) -> None:
    console.print(Rule("[bold blue]cardseller-agent[/bold blue]"))
    console.print(f"\n[bold]Card:[/bold] {card.player_name} — {card.card_set} #{card.card_number}\n")

    # Step 1: Generate draft listing
    draft_listing = generate_listing(card, client)

    console.print(Rule("[dim]Draft Listing (before validation)[/dim]"))
    display_listing("Draft", draft_listing, border_color="dim")

    # Step 2: Validate and refine
    validated = validate_and_refine(card, draft_listing, client)

    console.print(Rule("[bold]Final Listing[/bold]"))
    display_listing("Refined Listing ✓", validated.refined_listing, border_color="green")
    display_validation_summary(validated)

    if output_path:
        save_listing(validated, card, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eBay listings for sports cards")
    parser.add_argument("--demo", action="store_true", help="Run with a demo card (Patrick Mahomes RC)")
    parser.add_argument("--output", type=str, default=None, help="Save final listing to this file path")
    args = parser.parse_args()

    client = get_client()

    if args.demo:
        card = demo_card()
        console.print(Panel(
            f"[bold]Demo card:[/bold] {card.player_name} | {card.card_set} | {card.parallel}",
            border_style="dim",
        ))
    else:
        card = collect_card_info()

    run(card, client, output_path=args.output)


if __name__ == "__main__":
    main()
