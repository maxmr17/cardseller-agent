import time

import openai
from rich.console import Console

from models.card_listing import CardListing

console = Console()

MODEL_SMART = "gpt-4o"
MODEL_FAST = "gpt-4o-mini"


def listing_to_text(listing: CardListing) -> str:
    """Format a CardListing as plain text for agent review prompts."""
    pricing = listing.pricing
    strategy_str = pricing.strategy.value.replace("_", " ").title()

    price_parts = [f"Strategy: {strategy_str}"]
    if pricing.starting_price:
        price_parts.append(f"Starting Price: ${pricing.starting_price:.2f}")
    if pricing.buy_it_now_price:
        price_parts.append(f"Buy It Now: ${pricing.buy_it_now_price:.2f}")
    price_parts.append(f"Best Offer: {'Yes' if pricing.allow_offers else 'No'}")
    price_parts.append(f"Rationale: {pricing.pricing_rationale}")

    specifics_text = "\n".join(f"  {s.key}: {s.value}" for s in listing.item_specifics)

    return (
        f"TITLE ({len(listing.title)}/80 chars): {listing.title}\n"
        f"SUBTITLE: {listing.subtitle}\n"
        f"CONDITION GRADE: {listing.condition_grade}\n"
        f"CONDITION NOTES: {listing.condition_description}\n"
        f"\nPRICING:\n"
        + "\n".join(f"  {p}" for p in price_parts)
        + f"\n\nITEM SPECIFICS:\n{specifics_text}"
        f"\n\nSEARCH KEYWORDS: {', '.join(listing.search_keywords)}"
        f"\n\nDESCRIPTION:\n{listing.description}"
    )


def with_retry(fn, *args, max_retries: int = 3, initial_delay: float = 2.0, **kwargs):
    """Retry an OpenAI API call on transient errors with exponential backoff."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            console.print(f"[yellow]⚠ Rate limit — retrying in {delay:.0f}s…[/yellow]")
            time.sleep(delay)
            delay *= 2
        except openai.APIConnectionError:
            if attempt == max_retries - 1:
                raise
            console.print(f"[yellow]⚠ Connection error — retrying in {delay:.0f}s…[/yellow]")
            time.sleep(delay)
            delay *= 2
