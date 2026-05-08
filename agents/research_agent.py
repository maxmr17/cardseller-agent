import re
import urllib.parse

import httpx
import openai
from rich.console import Console
from models.card_listing import CardInfo

console = Console()

# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

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


def research_card(query: str, client: openai.OpenAI) -> tuple[CardInfo, str]:
    """Search the web to confirm card details. Returns (confirmed CardInfo, research summary)."""
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


# ---------------------------------------------------------------------------
# Image search — direct HTTP scrapers (primary) + AI fallback (secondary)
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Matches eBay CDN image URLs in HTML source
_EBAY_IMG_RE = re.compile(
    r'https://i\.ebayimg\.com/images/g/[A-Za-z0-9~_\-]+/s-l\d+\.(?:jpg|webp|png)',
    re.IGNORECASE,
)

# Matches 130point image URLs
_130PT_IMG_RE = re.compile(
    r'https://[^"\'>\s]*130point[^"\'>\s]*\.(?:jpg|jpeg|png|webp)',
    re.IGNORECASE,
)


def _upgrade_ebay_url(url: str) -> str:
    """Swap eBay thumbnail size suffix for full 500px version."""
    return re.sub(r's-l\d+', 's-l500', url)


def _scrape_ebay(query: str) -> str | None:
    """Search eBay and return the first listing image URL found in the HTML."""
    try:
        encoded = urllib.parse.quote(query)
        # _sacat=212 = Sports Trading Cards category
        url = f"https://www.ebay.com/sch/i.html?_nkw={encoded}&_sacat=212&LH_Sold=0"
        r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        matches = _EBAY_IMG_RE.findall(r.text)
        # Skip the first match — it's often eBay's own logo/nav image
        for match in matches[1:]:
            upgraded = _upgrade_ebay_url(match)
            console.print(f"[dim]  eBay image found: {upgraded[:80]}[/dim]")
            return upgraded
    except Exception as e:
        console.print(f"[yellow]  eBay scrape failed: {e}[/yellow]")
    return None


def _scrape_ebay_sold(query: str) -> str | None:
    """Search eBay SOLD listings for an image."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://www.ebay.com/sch/i.html?_nkw={encoded}&_sacat=212&LH_Sold=1&LH_Complete=1"
        r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        matches = _EBAY_IMG_RE.findall(r.text)
        for match in matches[1:]:
            upgraded = _upgrade_ebay_url(match)
            console.print(f"[dim]  eBay sold image found: {upgraded[:80]}[/dim]")
            return upgraded
    except Exception as e:
        console.print(f"[yellow]  eBay sold scrape failed: {e}[/yellow]")
    return None


def _scrape_130point(query: str) -> str | None:
    """Search 130point.com (eBay sold aggregator) for a card image."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://130point.com/sales/?search={encoded}"
        r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        # 130point embeds eBay CDN images
        matches = _EBAY_IMG_RE.findall(r.text)
        for match in matches:
            upgraded = _upgrade_ebay_url(match)
            console.print(f"[dim]  130point image found: {upgraded[:80]}[/dim]")
            return upgraded
        # Also try their own image URLs
        pt_matches = _130PT_IMG_RE.findall(r.text)
        if pt_matches:
            return pt_matches[0]
    except Exception as e:
        console.print(f"[yellow]  130point scrape failed: {e}[/yellow]")
    return None


def _scrape_comc(query: str) -> str | None:
    """Search COMC for a card image."""
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://www.comc.com/Cards/Baseball,Football,Basketball/*,s/{encoded}"
        r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        # COMC uses imgix CDN: https://img.comc.com/...
        pattern = re.compile(r'https://img\.comc\.com/[^"\'>\s]+\.(?:jpg|jpeg|png|webp)', re.IGNORECASE)
        matches = pattern.findall(r.text)
        if matches:
            console.print(f"[dim]  COMC image found: {matches[0][:80]}[/dim]")
            return matches[0]
    except Exception as e:
        console.print(f"[yellow]  COMC scrape failed: {e}[/yellow]")
    return None


def _ai_image_fallback(card: CardInfo, client: openai.OpenAI) -> str | None:
    """Last resort: use OpenAI web search to find any direct image URL."""
    try:
        base = f"{card.player_name} {card.card_set} #{card.card_number}"
        if card.parallel:
            base += f" {card.parallel}"

        response = client.responses.create(
            model="gpt-4o",
            instructions=(
                "Search eBay, Google Images, COMC, and Beckett for a real photo of this exact sports card. "
                "Return ONLY a single direct image URL (must end in .jpg, .jpeg, .png, or .webp and must be "
                "from a CDN like i.ebayimg.com, img.comc.com, or similar). "
                "Do NOT return a page URL or a URL without an image file extension. "
                "If you cannot find a direct image URL, return exactly: NO_IMAGE_FOUND"
            ),
            input=f"Find a direct image URL for: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        url = response.output_text.strip().split("\n")[0].strip()
        # Strip markdown formatting if present
        url = re.sub(r'^[`\[\(]|[`\]\)]$', '', url).strip()

        if url and url != "NO_IMAGE_FOUND" and url.startswith("http"):
            # Accept only URLs that look like direct image files
            if re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', url, re.IGNORECASE):
                console.print(f"[dim]  AI fallback image: {url[:80]}[/dim]")
                return url
    except Exception as e:
        console.print(f"[yellow]  AI fallback failed: {e}[/yellow]")
    return None


def _build_queries(card: CardInfo) -> list[str]:
    """Build progressively simpler search queries for the card."""
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    queries = [base]
    if card.parallel:
        queries.insert(0, f"{base} {card.parallel}")
    if card.is_graded and card.grade:
        queries.insert(0, f"{base} {card.parallel or ''} {card.grade}".strip())
    # Simplest fallback: just player + set
    queries.append(f"{card.player_name} {card.card_set}")
    return queries


def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    Find a card image using a waterfall of direct HTTP scrapers, falling back to AI search.
    Order: eBay active → eBay sold → 130point → COMC → AI web search
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    queries = _build_queries(card)

    # Try each scraper against the most specific query first, then broader ones
    scrapers = [
        ("eBay active", _scrape_ebay),
        ("eBay sold",   _scrape_ebay_sold),
        ("130point",    _scrape_130point),
        ("COMC",        _scrape_comc),
    ]

    for scraper_name, scraper_fn in scrapers:
        for query in queries:
            console.print(f"[dim]  Trying {scraper_name}: {query[:60]}...[/dim]")
            url = scraper_fn(query)
            if url:
                console.print(f"[green]✓ Image found via {scraper_name}[/green]")
                return url

    # Final fallback: AI-assisted search
    console.print("[dim]  Trying AI web search fallback...[/dim]")
    url = _ai_image_fallback(card, client)
    if url:
        console.print("[green]✓ Image found via AI fallback[/green]")
        return url

    console.print("[yellow]⚠ No card image found[/yellow]")
    return None
