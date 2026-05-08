import openai
from rich.console import Console
from models.card_listing import CardInfo

console = Console()

RESEARCH_SYSTEM_PROMPT = """You are a sports card research specialist with deep knowledge of trading card databases including PSA, Beckett, COMC, and eBay sold listings.

Given a card description, search the web to confirm every detail:
- Full player name (correct spelling, no nicknames unless official)
- Exact set name with year (e.g. "2017 Panini Prizm Football" not just "Prizm")
- Exact card number as printed on the card
- Parallel/variant name (exact name as used by the manufacturer)
- Serial numbering if applicable
- Whether this is a true Rookie Card (RC) as defined by hobby standards
- Whether it carries a certified autograph

Cross-reference multiple sources. Report what you find factually and flag any uncertainty."""

PARSE_SYSTEM_PROMPT = """Extract structured CardInfo fields from the research summary provided. Be precise — use the exact confirmed values from the research, not assumptions. If a field was not confirmed in the research, use the most reasonable inference from context."""

IMAGE_SYSTEM_PROMPT = """You are a sports card image researcher. Search the web for a clear, high-quality image of the exact sports card described.

Return ONLY a single direct image URL (ending in .jpg, .png, .webp, or similar) that shows the card. Prefer images from official sources like Panini, Topps, PSA, Beckett, COMC, or eBay listings. If no direct image URL is found, return the plain text: NO_IMAGE_FOUND"""


def research_card(query: str, client: openai.OpenAI) -> tuple[CardInfo, str]:
    """
    Search the web to confirm card details.
    Returns (confirmed CardInfo, research summary text).
    """
    console.print("\n[bold blue]🔎 Research Agent searching the web...[/bold blue]")

    search_response = client.responses.create(
        model="gpt-4o",
        instructions=RESEARCH_SYSTEM_PROMPT,
        input=f"Research and confirm all details for this sports card: {query}",
        tools=[{"type": "web_search_preview"}],
    )
    research_text = search_response.output_text

    parse_response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research summary:\n\n{research_text}"},
        ],
        response_format=CardInfo,
    )

    card = parse_response.choices[0].message.parsed
    if card is None:
        raise ValueError("Research agent could not parse card details")

    console.print("[green]✓ Card research complete[/green]")
    return card, research_text


def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """Search the web for a card image. Returns a direct image URL or None."""
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")
    try:
        query = f"{card.card_set} {card.player_name} #{card.card_number}"
        if card.parallel:
            query += f" {card.parallel}"

        response = client.responses.create(
            model="gpt-4o",
            instructions=IMAGE_SYSTEM_PROMPT,
            input=f"Find a card image for: {query}",
            tools=[{"type": "web_search_preview"}],
        )
        url = response.output_text.strip()
        if url and url != "NO_IMAGE_FOUND" and url.startswith("http"):
            console.print("[green]✓ Card image found[/green]")
            return url
    except Exception:
        pass
    console.print("[yellow]⚠ No card image found[/yellow]")
    return None
