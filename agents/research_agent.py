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

IMAGE_SYSTEM_PROMPT = """You are a sports card image hunter. Find a real photo of the EXACT card described — not a generic card image, not a placeholder, not the wrong parallel or wrong player.

Step 1 — Search eBay for an active or recently sold listing of this specific card.
  - Use the exact player name, year, set, card number, and parallel in your search.
  - Open an actual eBay listing that matches.
  - The main listing photo will have a CDN URL like: https://i.ebayimg.com/images/g/XXXXX/s-l500.jpg
  - Return that URL.

Step 2 — If no eBay listing found, try COMC.com for this card's product page image.

Step 3 — If COMC fails, try SportscardsPro.com or 130point.com sold listings.

Step 4 — If all else fails, search Google Images for: "{player} {year} {set} {card number} {parallel}" and return the most specific result.

REJECT any URL that:
- Is a generic card back or placeholder image
- Shows a different player, set, or parallel
- Is a site logo, banner, or thumbnail icon
- Comes from a site that requires login to view the image

Return ONLY the raw image URL on a single line — no explanation, no markdown, no quotes.
If you cannot find a real photo of this specific card after trying all sources, return: NO_IMAGE_FOUND"""


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


def _attempt_image_search(card: CardInfo, query: str, client: openai.OpenAI) -> str | None:
    """Run one image search attempt and verify the result is the correct card."""
    response = client.responses.create(
        model="gpt-4o",
        instructions=IMAGE_SYSTEM_PROMPT,
        input=query,
        tools=[{"type": "web_search_preview"}],
    )
    url = response.output_text.strip().split("\n")[0].strip()

    if not url or url == "NO_IMAGE_FOUND" or not url.startswith("http"):
        return None

    # Verify the model believes this URL is the correct card
    verify = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    f"I am about to display this image URL to a user selling a sports card:\n"
                    f"URL: {url}\n\n"
                    f"The card is: {card.player_name} — {card.card_set} #{card.card_number}"
                    f"{f' ({card.parallel})' if card.parallel else ''}\n\n"
                    f"Based on the URL structure and domain alone, does this look like a real photo "
                    f"of a specific sports card listing (not a generic placeholder, logo, or wrong card)?\n"
                    f"Reply with only YES or NO."
                ),
            }
        ],
    )
    verdict = (verify.choices[0].message.content or "").strip().upper()
    if verdict.startswith("YES"):
        return url
    return None


def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    Search multiple sources for a card image with verification.
    Tries up to 3 progressively targeted queries before giving up.
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel:
        base += f" {card.parallel}"

    queries = [
        (
            f"Search eBay for an active or sold listing of this exact card and return the listing's main photo URL: {base}. "
            f"The URL should start with https://i.ebayimg.com/"
        ),
        (
            f"Search COMC.com or SportscardsPro.com for this card and return the product image URL: {base}"
        ),
        (
            f"Search Google Images for a clear front-facing photo of this sports card and return the image URL: {base}"
        ),
    ]

    for i, query in enumerate(queries, 1):
        try:
            console.print(f"[dim]  Image search attempt {i}/3...[/dim]")
            url = _attempt_image_search(card, query, client)
            if url:
                console.print(f"[green]✓ Card image found on attempt {i}[/green]")
                return url
        except Exception as e:
            console.print(f"[yellow]  Attempt {i} failed: {e}[/yellow]")

    console.print("[yellow]⚠ No verified card image found after 3 attempts[/yellow]")
    return None
