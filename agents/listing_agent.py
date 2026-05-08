import openai
from rich.console import Console
from models.card_listing import CardInfo, CardListing

console = Console()

LISTING_SYSTEM_PROMPT = """You are an elite eBay sports card seller with over 15 years of experience and thousands of five-star transactions. You have deep expertise in the sports card hobby including:

- Current market values and trends for all major sports (NFL, NBA, MLB, UFC)
- What collectors and investors actively search for and pay premiums on
- eBay SEO best practices — exactly how to title and describe cards to maximize visibility and bids
- Pricing strategies that balance quick sales vs. maximizing value
- How to present condition professionally and honestly to build buyer trust
- The language and terminology elite sellers use that attracts serious buyers

Your listings consistently outperform average sellers because you:
1. Pack titles with the exact keywords collectors search (rookie abbreviations, parallel names, graded terminology)
2. Write descriptions that are detailed, honest, and create urgency without being pushy
3. Price strategically based on recent comparable sales and current demand
4. Identify every key selling point (short print, first Bowman, key parallel, investment-grade card)

When given card information, you produce a complete, professional eBay listing ready to publish.

DESCRIPTION STYLE — write all descriptions in the style of eBay seller maxmr17:
- Concise, confident, no-fluff tone. Facts first, hype never.
- Short punchy sentences. No walls of text.
- Lead with the card's key appeal (rookie year, investment-grade parallel, pop report, etc.)
- Condition described plainly and honestly — specific observations, not vague filler.
- Shipping/payment section kept brief and to the point.
- No excessive exclamation marks. Professional but personable."""


def _build_card_prompt(card: CardInfo) -> str:
    lines = [
        f"Player: {card.player_name}",
        f"Card Number: {card.card_number}",
        f"Set: {card.card_set}",
        f"Sport: {card.sport}",
    ]
    if card.is_rookie_card:
        lines.append("Rookie Card: YES (RC)")
    if card.is_autographed:
        lines.append("Autograph: YES")
    if card.is_graded:
        lines.append(f"Graded: YES — {card.grade or 'grade unknown'}")
    if card.serial_number:
        lines.append(f"Serial Numbered: {card.serial_number}")
    if card.parallel:
        lines.append(f"Parallel/Variant: {card.parallel}")
    if card.condition:
        lines.append(f"Condition Notes from Seller: {card.condition}")
    if card.extra_notes:
        lines.append(f"Additional Notes: {card.extra_notes}")
    return "\n".join(lines)


def generate_listing(card: CardInfo, client: openai.OpenAI) -> CardListing:
    """Run the listing agent to generate an eBay listing for a sports card."""
    console.print("\n[bold cyan]⚡ Listing Agent running...[/bold cyan]")

    card_details = _build_card_prompt(card)

    user_message = f"""Generate a complete, optimized eBay listing for this sports card:

{card_details}

Produce a listing that will attract serious collectors and investors. Be specific about pricing — provide real dollar amounts based on your market knowledge for cards like this."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LISTING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=CardListing,
    )

    listing = response.choices[0].message.parsed
    if listing is None:
        raise ValueError("Listing agent returned no structured output")

    console.print("[green]✓ Listing generated[/green]")
    return listing
