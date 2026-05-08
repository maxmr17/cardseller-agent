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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


def _fetch(url: str, extra_headers: dict | None = None) -> str | None:
    h = {**_HEADERS, **(extra_headers or {})}
    try:
        r = httpx.get(url, headers=h, timeout=15, follow_redirects=True)
        console.print(f"[dim]    HTTP {r.status_code} ← {url[:90]}[/dim]")
        return r.text if r.status_code == 200 else None
    except Exception as e:
        console.print(f"[dim]    Fetch error: {e}[/dim]")
        return None


def _is_logo(url: str) -> bool:
    bad = ("logo", "icon", "favicon", "banner", "placeholder", "default",
           "noimage", "spinner", "blank", "avatar", "profile")
    u = url.lower()
    return any(b in u for b in bad)


def _is_image_url(url: str) -> bool:
    return bool(re.search(r'\.(jpg|jpeg|png|webp)(\?|$)', url, re.I))


# ── Extraction helpers ────────────────────────────────────────────────────────

def _og_image(html: str) -> str | None:
    for pat in [
        re.compile(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', re.I),
        re.compile(r'og:image.*?content=["\']([^"\']+)["\']', re.I),
    ]:
        m = pat.search(html)
        if m:
            u = m.group(1).strip()
            if u.startswith("http") and not _is_logo(u):
                return u
    return None


def _data_src(html: str) -> str | None:
    m = re.search(r'data-src=["\']([^"\']*i\.ebayimg\.com/images/g/[^"\']+)["\']', html, re.I)
    return re.sub(r's-l\d+', 's-l500', m.group(1)) if m else None


def _json_blob(html: str) -> str | None:
    m = re.search(
        r'"(?:imageUrl|src|image|mainImage|originalImg)":\s*"(https://i\.ebayimg\.com/[^"]+)"',
        html, re.I,
    )
    return re.sub(r's-l\d+', 's-l500', m.group(1)) if m else None


def _any_image(html: str) -> str | None:
    return _og_image(html) or _data_src(html) or _json_blob(html)


# ── Strategy 1: Wikipedia direct player article ───────────────────────────────
#  Most reliable for famous players. Hits /api/rest_v1/page/summary/{Player_Name}
#  directly — no search needed, no JS, always returns JSON with image fields.

def _via_wikipedia_player(card: CardInfo) -> str | None:
    player = card.player_name.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(player)}"
    console.print(f"[dim]  → Wikipedia direct: {card.player_name}[/dim]")
    try:
        r = httpx.get(url, timeout=10, headers={"User-Agent": "cardseller-agent/1.0"})
        console.print(f"[dim]    HTTP {r.status_code} ← Wikipedia summary[/dim]")
        if r.status_code == 200:
            data = r.json()
            img = (
                (data.get("originalimage") or {}).get("source")
                or (data.get("thumbnail") or {}).get("source")
            )
            if img and not _is_logo(img):
                console.print(f"[green]✓ Wikipedia player image: {img[:80]}[/green]")
                return img
            else:
                console.print(f"[dim]    No image in summary (img={img})[/dim]")
        elif r.status_code == 404:
            # Try with disambiguation — e.g. "Mickey_Mantle_(baseball)" won't exist
            # but the base article will; the 404 just means exact title not found.
            console.print(f"[dim]    Wikipedia 404 for '{player}' — skipping[/dim]")
    except Exception as e:
        console.print(f"[dim]    Wikipedia direct failed: {e}[/dim]")
    return None


# ── Strategy 2: Bing Images — murl data embedded in raw HTML ─────────────────
#  Bing encodes original image URLs as  murl&quot;:&quot;URL&quot;  in its
#  search results page source. No JavaScript needed, no API key required.

def _via_bing_images(card: CardInfo) -> str | None:
    base = _card_query(card)
    query = f"{base} sports card"
    url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&form=HDRSC2&first=1"
    console.print(f"[dim]  → Bing Images: {query[:70]}[/dim]")
    html = _fetch(url, extra_headers={"Referer": "https://www.bing.com/"})
    if not html:
        return None

    # Pattern 1: HTML-entity encoded (most common)
    for raw in re.findall(r'murl&quot;:&quot;(https://[^&"<]+)&quot;', html):
        u = urllib.parse.unquote(raw)
        if _is_image_url(u) and not _is_logo(u):
            return u

    # Pattern 2: plain JSON
    for raw in re.findall(r'"murl"\s*:\s*"(https://[^"]+)"', html):
        if _is_image_url(raw) and not _is_logo(raw):
            return raw

    # Pattern 3: iurl (Bing thumbnail source)
    for raw in re.findall(r'iurl&quot;:&quot;(https://[^&"<]+)&quot;', html):
        u = urllib.parse.unquote(raw)
        if _is_image_url(u) and not _is_logo(u):
            return u

    console.print(f"[dim]    Bing: no murl patterns found in {len(html)} bytes[/dim]")
    return None


# ── Strategy 3: AI searches for a direct image URL ───────────────────────────
#  Ask the model specifically to find and return a .jpg/.png URL.
#  Much more constrained than the og:image approach.

def _via_ai_direct_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    base = _card_query(card)
    console.print(f"[dim]  → AI direct image URL search...[/dim]")
    try:
        resp = client.responses.create(
            model="gpt-4o",
            instructions=(
                "You are a sports card image finder. Your ONLY job is to return a single direct "
                "image URL ending in .jpg, .jpeg, .png, or .webp. "
                "Search eBay sold listings, 130point.com, PSA card facts, COMC, or any card site. "
                "Find an actual image of the card and return ONLY the raw image URL. "
                "No explanation, no markdown, no text — just the URL. "
                "Example output: https://i.ebayimg.com/images/g/abc123/s-l500.jpg"
            ),
            input=f"Find a direct image URL (.jpg/.png) for this sports card: {base}",
            tools=[{"type": "web_search_preview"}],
        )
        text = resp.output_text.strip().split("\n")[0]
        # Strip any markdown/quote wrapping
        text = re.sub(r'^[`"\'\[\(<]+|[`"\'\]\)>]+$', '', text).strip()
        console.print(f"[dim]    AI returned: {text[:100]}[/dim]")
        if text.startswith("http") and not _is_logo(text):
            if _is_image_url(text):
                return text
            # If it's a page URL, try fetching og:image from it
            html = _fetch(text)
            if html:
                img = _any_image(html)
                if img:
                    return img
    except Exception as e:
        console.print(f"[dim]    AI direct image failed: {e}[/dim]")
    return None


# ── Strategy 4: Wikipedia search API fallback ────────────────────────────────
#  Searches Wikipedia for the player name and card set, extracts image
#  from summary pages of matching articles.

def _via_wikipedia_search(card: CardInfo) -> str | None:
    console.print("[dim]  → Wikipedia search API...[/dim]")
    queries = [
        card.player_name,
        f"{card.player_name} {card.card_set[:4]}",
        f"{card.card_set}",
    ]
    for q in queries:
        try:
            search_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={urllib.parse.quote(q)}"
                f"&srlimit=5&format=json&srprop=snippet"
            )
            r = httpx.get(search_url, timeout=10, headers={"User-Agent": "cardseller-agent/1.0"})
            results = r.json().get("query", {}).get("search", [])
            for result in results:
                title = result["title"]
                encoded = urllib.parse.quote(title.replace(" ", "_"))
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
                sr = httpx.get(summary_url, timeout=10, headers={"User-Agent": "cardseller-agent/1.0"})
                if sr.status_code == 200:
                    data = sr.json()
                    img = (
                        (data.get("originalimage") or {}).get("source")
                        or (data.get("thumbnail") or {}).get("source")
                    )
                    if img and not _is_logo(img):
                        console.print(f"[green]✓ Wikipedia search image via '{title}'[/green]")
                        return img
        except Exception as e:
            console.print(f"[dim]    Wikipedia search query '{q}' failed: {e}[/dim]")
    return None


# ── Strategy 5: direct scrape of card marketplace pages ──────────────────────

def _via_direct_scrape(card: CardInfo) -> str | None:
    base = _card_query(card)
    enc = urllib.parse.quote(base)
    targets = [
        (f"https://130point.com/sales/?search={enc}",                   "130point"),
        (f"https://www.comc.com/Cards/*,s/{enc}",                       "COMC"),
        (f"https://www.sportscardspro.com/search-products?q={enc}",     "SportscardsPro"),
        (f"https://www.psacard.com/cardfacts/search?q={enc}",           "PSA"),
    ]
    for url, name in targets:
        console.print(f"[dim]  → {name}: {base[:60]}[/dim]")
        html = _fetch(url)
        if html:
            img = _any_image(html)
            if img:
                console.print(f"[green]✓ Image from {name}[/green]")
                return img
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card_query(card: CardInfo) -> str:
    q = f"{card.player_name} {card.card_set} #{card.card_number}"
    if card.parallel and card.parallel.lower() not in ("base", ""):
        q += f" {card.parallel}"
    return q


# ── Main entry point ──────────────────────────────────────────────────────────

def search_card_image(card: CardInfo, client: openai.OpenAI) -> str | None:
    """
    5-strategy image search waterfall.
    Strategy 1: Wikipedia direct player article (most reliable for famous players)
    Strategy 2: Bing Images murl scrape
    Strategy 3: AI web search for direct .jpg/.png URL
    Strategy 4: Wikipedia search API fallback
    Strategy 5: Direct scrape of 130point, COMC, SportscardsPro, PSA
    """
    console.print("[bold blue]🖼  Searching for card image...[/bold blue]")

    console.print("[dim] Strategy 1: Wikipedia direct player article[/dim]")
    url = _via_wikipedia_player(card)
    if url:
        return url

    console.print("[dim] Strategy 2: Bing Images[/dim]")
    url = _via_bing_images(card)
    if url:
        console.print(f"[green]✓ Image found via Bing Images: {url[:80]}[/green]")
        return url

    console.print("[dim] Strategy 3: AI direct image URL[/dim]")
    url = _via_ai_direct_image(card, client)
    if url:
        console.print(f"[green]✓ Image found via AI search: {url[:80]}[/green]")
        return url

    console.print("[dim] Strategy 4: Wikipedia search API[/dim]")
    url = _via_wikipedia_search(card)
    if url:
        return url

    console.print("[dim] Strategy 5: Direct scrape[/dim]")
    url = _via_direct_scrape(card)
    if url:
        return url

    console.print("[yellow]⚠ All image strategies exhausted[/yellow]")
    return None
