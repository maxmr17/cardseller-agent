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
# Image search
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


def _extract_og_image(html: str) -> str | None:
    """Pull the og:image meta tag URL — always present, always a direct image URL."""
    patterns = [
        re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', re.IGNORECASE),
        re.compile(r'og:image["\'][^>]*content=["\']([^"\']+)["\']', re.IGNORECASE),
    ]
    for pat in patterns:
        m = pat.search(html)
        if m:
            url = m.group(1).strip()
            if url.startswith("http"):
                return url
    return None


def _extract_data_src(html: str) -> str | None:
    """Find lazy-loaded eBay CDN images stored in data-src attributes."""
    pat = re.compile(
        r'data-src=["\']([^"\']*i\.ebayimg\.com[^"\']+)["\']',
        re.IGNORECASE,
    )
    for m in pat.finditer(html):
        url = m.group(1)
        return re.sub(r's-l\d+', 's-l500', url)
    return None


def _extract_json_images(html: str) -> str | None:
    """eBay embeds image URLs in JSON blobs inside <script> tags."""
    pat = re.compile(
        r'"(?:imageUrl|src|image|mainImage|originalImg)":\s*"(https://i\.ebayimg\.com/[^"]+)"',
        re.IGNORECASE,
    )
    for m in pat.finditer(html):
        url = m.group(1)
        return re.sub(r's-l\d+', 's-l500', url)
    return None


def _fetch_page(url: str) -> str | None:
    """Fetch a URL and return the HTML text, or None on failure."""
    try:
        r = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        console.print(f"[yellow]  Fetch failed ({url[:60]}...): {e}[/yellow]")
        return None


def _extract_any_image(html: str) -> str | None:
    """Try every extraction method on a page's HTML."""
    return (
        _extract_og_image(html)
        or _extract_data_src(html)
        or _extract_json_images(html)
    )


# ── Source 1: AI finds a specific eBay listing URL, we fetch + extract ──────

def _ai_find_ebay_listing_url(card: CardInfo, client: openai.OpenAI) -> str | None:
    """Ask AI to find a real eBay listing URL (not an image URL)."""
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        base += f" {card.parallel}"

    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=(
                "Search eBay for a listing of this exact sports card. "
                "Return ONLY the full eBay listing URL (https://www.ebay.com/itm/DIGITS). "
                "No explanation. No markdown. Just the URL."
            ),
            input=f"Find an eBay listing for: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        text = response.output_text.strip()
        m = re.search(r'https?://(?:www\.)?ebay\.com/itm/\d+', text)
        if m:
            return m.group(0)
    except Exception as e:
        console.print(f"[yellow]  AI eBay listing search failed: {e}[/yellow]")
    return None


def _via_ebay_listing(card: CardInfo, client: openai.OpenAI) -> str | None:
    """Find eBay listing URL via AI, then scrape the listing page for its og:image."""
    console.print("[dim]  Strategy 1: AI → eBay listing URL → og:image[/dim]")
    listing_url = _ai_find_ebay_listing_url(card, client)
    if not listing_url:
        return None
    console.print(f"[dim]  Fetching listing: {listing_url}[/dim]")
    html = _fetch_page(listing_url)
    if not html:
        return None
    return _extract_any_image(html)


# ── Source 2: eBay search results page ──────────────────────────────────────

def _via_ebay_search(query: str) -> str | None:
    """Scrape eBay search results page for any card image."""
    encoded = urllib.parse.quote(query)
    for suffix in ["&LH_Sold=0", "&LH_Sold=1&LH_Complete=1"]:
        url = f"https://www.ebay.com/sch/i.html?_nkw={encoded}&_sacat=212{suffix}"
        console.print(f"[dim]  Strategy 2: eBay search ({url[:70]}...)[/dim]")
        html = _fetch_page(url)
        if html:
            img = _extract_any_image(html)
            if img:
                return img
    return None


# ── Source 3: 130point.com ───────────────────────────────────────────────────

def _via_130point(query: str) -> str | None:
    encoded = urllib.parse.quote(query)
    url = f"https://130point.com/sales/?search={encoded}"
    console.print(f"[dim]  Strategy 3: 130point ({url[:70]}...)[/dim]")
    html = _fetch_page(url)
    if html:
        return _extract_any_image(html)
    return None


# ── Source 4: COMC ───────────────────────────────────────────────────────────

def _via_comc(query: str) -> str | None:
    encoded = urllib.parse.quote(query)
    url = f"https://www.comc.com/Cards/*,s/{encoded}"
    console.print(f"[dim]  Strategy 4: COMC ({url[:70]}...)[/dim]")
    html = _fetch_page(url)
    if not html:
        return None
    pat = re.compile(r'https://img\.comc\.com/[^"\'>\s]+\.(?:jpg|jpeg|png|webp)', re.IGNORECASE)
    m = pat.search(html)
    if m:
        return m.group(0)
    return _extract_og_image(html)


# ── Source 5: PSA card database ──────────────────────────────────────────────

def _via_psa(query: str) -> str | None:
    encoded = urllib.parse.quote(query)
    url = f"https://www.psacard.com/cardfacts/search?q={encoded}"
    console.print(f"[dim]  Strategy 5: PSA ({url[:70]}...)[/dim]")
    html = _fetch_page(url)
    if html:
        return _extract_og_image(html)
    return None


# ── Source 6: AI direct image URL search (last resort) ──────────────────────

def _via_ai_direct(card: CardInfo, client: openai.OpenAI) -> str | None:
    """Last resort: ask AI to find a direct image URL ending in .jpg/.png/.webp."""
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        base += f" {card.parallel}"

    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=(
                "Search eBay, Google Images, Heritage Auctions, COMC, and Beckett for a photo of this sports card. "
                "Return ONLY a direct image file URL that ends in .jpg, .jpeg, .png, or .webp. "
                "It must be from a public CDN — not a page URL. "
                "No explanation. No markdown. Just the raw URL."
            ),
            input=f"Find a direct card image URL for: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        text = response.output_text.strip().split("\n")[0]
        text = re.sub(r'[`\[\]()]', '', text).strip()
        if text.startswith("http") and re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', text, re.IGNORECASE):
            return text
    except Exception as e:
        console.print(f"[yellow]  AI direct image search failed: {e}[/yellow]")
    return None


# ── Main entry point ─────────────────────────────────────────────────────────

def _card_queries(card: CardInfo) -> list[str]:
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    queries = []
    if card.parallel and card.parallel.lower() not in ("base", ""):
        queries.append(f"{base} {card.parallel}")
    queries.append(base)
    queries.append(f"{card.player_name} {card.card_set}")
    return queries


def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    Waterfall image search across 6 strategies.
    Returns first valid image URL found, or None.
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    queries = _card_queries(card)

    # Strategy 1: AI finds eBay listing URL → fetch page → og:image
    url = _via_ebay_listing(card, client)
    if url:
        console.print("[green]✓ Image found via eBay listing[/green]")
        return url

    # Strategies 2–5: direct HTTP scrapes with multiple query variants
    scrapers = [
        ("eBay search",  _via_ebay_search),
        ("130point",     _via_130point),
        ("COMC",         _via_comc),
        ("PSA",          _via_psa),
    ]
    for name, fn in scrapers:
        for q in queries:
            url = fn(q)
            if url:
                console.print(f"[green]✓ Image found via {name}[/green]")
                return url

    # Strategy 6: AI direct image URL search
    url = _via_ai_direct(card, client)
    if url:
        console.print("[green]✓ Image found via AI direct search[/green]")
        return url

    console.print("[yellow]⚠ No card image found[/yellow]")
    return None
