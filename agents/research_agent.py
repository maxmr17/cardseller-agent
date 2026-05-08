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

# Mimic a real browser as closely as possible to avoid WAF blocks
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


def _fetch(url: str, referer: str = "") -> str | None:
    headers = dict(_HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        console.print(f"[dim]    HTTP {r.status_code} — {url[:80]}[/dim]")
        if r.status_code == 200:
            return r.text
    except Exception as e:
        console.print(f"[dim]    Fetch error: {e}[/dim]")
    return None


def _og_image(html: str) -> str | None:
    """og:image meta tag — most reliable, always in raw HTML."""
    for pat in [
        re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', re.I),
        re.compile(r'"og:image"[^>]*content=["\']([^"\']+)["\']', re.I),
    ]:
        m = pat.search(html)
        if m:
            url = m.group(1).strip()
            if url.startswith("http") and not _is_logo(url):
                return url
    return None


def _data_src(html: str) -> str | None:
    """Lazy-loaded eBay images stored in data-src."""
    m = re.search(r'data-src=["\']([^"\']*i\.ebayimg\.com/images/g/[^"\']+)["\']', html, re.I)
    if m:
        return re.sub(r's-l\d+', 's-l500', m.group(1))
    return None


def _json_image(html: str) -> str | None:
    """Image URLs embedded in eBay JSON-LD / script blobs."""
    m = re.search(
        r'"(?:imageUrl|src|image|mainImage|originalImg)":\s*"(https://i\.ebayimg\.com/[^"]+)"',
        html, re.I,
    )
    if m:
        return re.sub(r's-l\d+', 's-l500', m.group(1))
    return None


def _is_logo(url: str) -> bool:
    """Reject obvious site logos / generic placeholders."""
    bad = ("logo", "icon", "favicon", "banner", "placeholder", "default", "noimage", "spinner")
    return any(b in url.lower() for b in bad)


def _best_image(html: str) -> str | None:
    return _og_image(html) or _data_src(html) or _json_image(html)


# ── Strategy 1: AI finds specific page URL → we fetch og:image ───────────────
#    Ask AI for a URL on a card-database or auction site (not eBay search),
#    then fetch that specific page and pull the og:image.

_URL_SOURCES = [
    ("Heritage Auctions", "ha.com"),
    ("PSA Card Facts",    "psacard.com/cardfacts"),
    ("Goldin Auctions",   "goldin.co"),
    ("eBay listing",      "ebay.com/itm"),
    ("PWCC",              "pwccmarketplace.com"),
]

def _ai_find_page_url(card: CardInfo, client: openai.OpenAI, source_name: str, source_domain: str) -> str | None:
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        base += f" {card.parallel}"
    try:
        resp = client.responses.create(
            model="gpt-4o",
            instructions=(
                f"Search {source_name} for this exact sports card. "
                f"Return ONLY the full page URL from {source_domain}. "
                "One URL, no explanation, no markdown."
            ),
            input=f"Find this card on {source_name}: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        text = resp.output_text.strip()
        m = re.search(rf'https?://[^\s"\'<>]*{re.escape(source_domain.split("/")[0])}[^\s"\'<>]*', text)
        if m:
            url = m.group(0).rstrip(".,)")
            console.print(f"[dim]    AI found {source_name} URL: {url[:80]}[/dim]")
            return url
    except Exception as e:
        console.print(f"[dim]    AI {source_name} search failed: {e}[/dim]")
    return None


def _via_ai_page(card: CardInfo, client: openai.OpenAI) -> str | None:
    for source_name, source_domain in _URL_SOURCES:
        console.print(f"[dim]  → Trying {source_name}...[/dim]")
        page_url = _ai_find_page_url(card, client, source_name, source_domain)
        if not page_url:
            continue
        html = _fetch(page_url)
        if not html:
            continue
        img = _best_image(html)
        if img:
            console.print(f"[green]✓ Image found on {source_name}[/green]")
            return img
    return None


# ── Strategy 2: Wikipedia API (free, no blocks, great for famous cards) ───────

def _via_wikipedia(card: CardInfo, client: openai.OpenAI) -> str | None:
    console.print("[dim]  → Trying Wikipedia...[/dim]")
    try:
        resp = client.responses.create(
            model="gpt-4o",
            instructions=(
                "Search Wikipedia for an article about this sports card or its set. "
                "Return ONLY the Wikipedia article title (e.g. '1952_Topps'). No URL, no explanation."
            ),
            input=f"Wikipedia article title for: {card.player_name} {card.card_set}",
            tools=[{"type": "web_search_preview"}],
        )
        title = resp.output_text.strip().replace(" ", "_").strip("\"'")
        if not title:
            return None

        # Wikipedia REST API — returns JSON with thumbnail and originalimage
        for t in [title, urllib.parse.quote(title)]:
            api = f"https://en.wikipedia.org/api/rest_v1/page/summary/{t}"
            r = httpx.get(api, headers={"User-Agent": "cardseller-agent/1.0"}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                img = (
                    (data.get("originalimage") or {}).get("source")
                    or (data.get("thumbnail") or {}).get("source")
                )
                if img and not _is_logo(img):
                    console.print(f"[green]✓ Image found on Wikipedia[/green]")
                    return img
    except Exception as e:
        console.print(f"[dim]    Wikipedia failed: {e}[/dim]")
    return None


# ── Strategy 3: direct scrape of card-database search pages ──────────────────

def _via_direct_scrape(card: CardInfo) -> str | None:
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        base += f" {card.parallel}"
    enc = urllib.parse.quote(base)

    targets = [
        (f"https://130point.com/sales/?search={enc}", "130point"),
        (f"https://www.comc.com/Cards/*,s/{enc}", "COMC"),
        (f"https://www.sportscardspro.com/search-products?q={enc}", "SportscardsPro"),
    ]
    for url, name in targets:
        console.print(f"[dim]  → Scraping {name}...[/dim]")
        html = _fetch(url)
        if html:
            img = _best_image(html)
            if img:
                console.print(f"[green]✓ Image found on {name}[/green]")
                return img
    return None


# ── Strategy 4: AI returns a direct image URL as last resort ─────────────────

def _via_ai_direct_url(card: CardInfo, client: openai.OpenAI) -> str | None:
    console.print("[dim]  → AI direct image URL search...[/dim]")
    base = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        base += f" {card.parallel}"
    try:
        resp = client.responses.create(
            model="gpt-4o",
            instructions=(
                "Search Google Images, Heritage Auctions, PSA, and eBay for a photo of this sports card. "
                "Find the actual image file URL — it must end in .jpg, .jpeg, .png, or .webp and must "
                "be hosted on a public image CDN (like i.ebayimg.com, ha.com, psacard.com, etc.). "
                "Return ONLY that raw image URL. Nothing else."
            ),
            input=f"Direct image URL for: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        text = resp.output_text.strip().split("\n")[0]
        text = re.sub(r'[`\[\]()"\']', "", text).strip()
        if text.startswith("http") and re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', text, re.I):
            if not _is_logo(text):
                console.print(f"[green]✓ Image found via AI direct URL[/green]")
                return text
    except Exception as e:
        console.print(f"[dim]    AI direct URL search failed: {e}[/dim]")
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    4-strategy image search waterfall with full console diagnostics.
    Returns first valid image URL found, or None.
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    console.print("[dim] Strategy 1: AI → card database/auction page → og:image[/dim]")
    url = _via_ai_page(card, client)
    if url:
        return url

    console.print("[dim] Strategy 2: Wikipedia REST API[/dim]")
    url = _via_wikipedia(card, client)
    if url:
        return url

    console.print("[dim] Strategy 3: Direct HTTP scrape (130point / COMC / SportscardsPro)[/dim]")
    url = _via_direct_scrape(card)
    if url:
        return url

    console.print("[dim] Strategy 4: AI direct image URL[/dim]")
    url = _via_ai_direct_url(card, client)
    if url:
        return url

    console.print("[yellow]⚠ All image strategies exhausted — no image found[/yellow]")
    return None
