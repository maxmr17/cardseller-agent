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

IMAGE_SYSTEM_PROMPT = """You are a sports card image hunter. Your only job is to find a working, embeddable image URL of the exact card described.

Search strategy — try these in order until you find one:
1. Google Images: search "{player} {year} {set} {card number} {parallel} card"
2. eBay active/sold listings for this exact card — eBay listing images are direct CDN URLs
3. COMC.com product page for this card
4. SportscardsPro.com
5. Beckett.com card database
6. Any other sports card marketplace

Rules for a valid URL:
- Must be a direct image file URL (ends in .jpg, .jpeg, .png, .webp, or contains /image/ or /photo/ in the path)
- Must actually show THIS specific card (correct player, set, parallel)
- Must be publicly accessible (no login required)
- eBay image CDN URLs look like: https://i.ebayimg.com/images/g/...
- COMC image URLs look like: https://www.comc.com/...

Return ONLY the raw image URL on a single line — no explanation, no markdown, no quotes.
If after exhausting all sources you truly cannot find one, return: NO_IMAGE_FOUND"""


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


def _is_likely_image_url(url: str) -> bool:
    url_lower = url.lower()
    direct_ext = any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"))
    image_path = any(p in url_lower for p in ("/image/", "/images/", "/photo/", "/photos/", "ebayimg.com", "comc.com", "imgix.net", "cloudinary.com"))
    return direct_ext or image_path


def _attempt_image_search(query: str, client: openai.OpenAI) -> str | None:
    response = client.responses.create(
        model="gpt-4o",
        instructions=IMAGE_SYSTEM_PROMPT,
        input=query,
        tools=[{"type": "web_search_preview"}],
    )
    url = response.output_text.strip().split("\n")[0].strip()
    if url and url != "NO_IMAGE_FOUND" and url.startswith("http") and _is_likely_image_url(url):
        return url
    # If not a direct image URL but looks like a valid HTTP URL, still return it
    # — Streamlit can sometimes render page-embedded images
    if url and url != "NO_IMAGE_FOUND" and url.startswith("http"):
        return url
    return None


def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    Search multiple sources for a card image.
    Tries up to 3 progressively broader queries before giving up.
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel:
        base += f" {card.parallel}"

    queries = [
        f"Find a direct image URL of this sports card from eBay, Google Images, or COMC: {base}",
        f"Search Google Images and eBay listings for a photo of this card and return the image URL: {card.player_name} {card.card_set} {card.parallel or ''} card #{card.card_number}",
        f"Find any clear photo of this sports card — try Beckett, PSA, SportscardsPro, or any card marketplace: {card.player_name} {card.card_set}",
    ]

    for i, query in enumerate(queries, 1):
        try:
            console.print(f"[dim]  Image search attempt {i}/3...[/dim]")
            url = _attempt_image_search(query, client)
            if url:
                console.print(f"[green]✓ Card image found on attempt {i}[/green]")
                return url
        except Exception as e:
            console.print(f"[yellow]  Attempt {i} failed: {e}[/yellow]")

    console.print("[yellow]⚠ No card image found after 3 attempts[/yellow]")
    return None
