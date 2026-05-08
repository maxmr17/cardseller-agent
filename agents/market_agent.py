import openai
from rich.console import Console
from models.card_listing import CardInfo, CardListing, MarketSalesReport

console = Console()

MARKET_SYSTEM_PROMPT = """You are a sports card market data analyst specialising in recent comparable sales. You search eBay sold listings, COMC, Whatnot, Alt, StockX, SportscardsPro, and SportscardsInvestor to find the most recent actual sales of a specific card.

Your rules:
- Find real, completed sales only — not active listings or asking prices.
- Match the exact card: same player, set, year, card number, AND parallel/variant. A Silver Prizm sale does not count for a Gold Prizm.
- Prefer sales from the last 90 days. Include older comps only if recent ones are unavailable.
- Report up to 10 sales, most recent first.
- For each sale: date, final price, condition/grade, platform, and any relevant notes.
- Calculate average, low, and high from the sales you found.
- Assess the price trend honestly.
- Be brutally honest in the pricing verdict — if the proposed price is off, say so and by how much."""

PRICE_VALIDATOR_SYSTEM_PROMPT = """You are the most rigorous sports card pricing analyst in the hobby. You scrutinize proposed eBay listing prices against real market data with zero tolerance for inaccuracy.

Your job:
1. Compare the proposed listing price against every comp found.
2. Identify whether the price is justified, too high, or too low — and by what percentage.
3. Consider condition premium or discount vs. the comps.
4. Factor in current demand, pop report if known, and seasonal trends.
5. Deliver a verdict that is specific and actionable.

A "correct" price means a buyer would find it fair AND the seller is not leaving money on the table. You will not approve a price that is more than 15% above or below what the comps support."""


def fetch_market_sales(card: CardInfo, client: openai.OpenAI) -> MarketSalesReport:
    """Search the web for recent comparable sales of this card."""
    console.print("\n[bold green]📊 Market Agent fetching recent sales...[/bold green]")

    query = f"{card.card_set} {card.player_name} #{card.card_number}"
    if card.parallel:
        query += f" {card.parallel}"
    if card.serial_number:
        query += f" {card.serial_number}"
    if card.is_graded and card.grade:
        query += f" {card.grade}"

    search_response = client.responses.create(
        model="gpt-4o",
        instructions=MARKET_SYSTEM_PROMPT,
        input=(
            f"Find the 10 most recent completed sales for this exact sports card: {query}\n\n"
            f"Search eBay sold listings, COMC, Whatnot, SportscardsPro, SportscardsInvestor, and any other platforms. "
            f"Report real sale prices only — no active listings."
        ),
        tools=[{"type": "web_search_preview"}],
    )
    sales_text = search_response.output_text

    parse_response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PRICE_VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Here are the recent sales data found for: {query}\n\n"
                f"{sales_text}\n\n"
                f"Parse this into a structured MarketSalesReport. "
                f"The proposed listing has no price yet — leave pricing_verdict focused on what the comps suggest the correct price range should be."
            )},
        ],
        response_format=MarketSalesReport,
    )

    report = parse_response.choices[0].message.parsed
    if report is None:
        raise ValueError("Market agent returned no structured output")

    console.print("[green]✓ Market data fetched[/green]")
    return report


def validate_listing_price(
    card: CardInfo,
    listing: CardListing,
    market_report: MarketSalesReport,
    client: openai.OpenAI,
) -> str:
    """Compare the listing price against market comps and return a verdict string."""
    console.print("[bold green]💰 Validating listing price against comps...[/bold green]")

    pricing = listing.pricing
    price_info = f"Strategy: {pricing.strategy.value}\n"
    if pricing.starting_price:
        price_info += f"Starting Price: ${pricing.starting_price:.2f}\n"
    if pricing.buy_it_now_price:
        price_info += f"Buy It Now: ${pricing.buy_it_now_price:.2f}\n"

    comps_text = "\n".join(
        f"- {s.sale_date} | ${s.price:.2f} | {s.condition} | {s.platform} | {s.details}"
        for s in market_report.sales
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PRICE_VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Card: {card.player_name} — {card.card_set} #{card.card_number}"
                f"{f' ({card.parallel})' if card.parallel else ''}\n\n"
                f"PROPOSED LISTING PRICE:\n{price_info}\n"
                f"RECENT COMPS:\n{comps_text}\n\n"
                f"Market avg: ${market_report.avg_price:.2f} | Range: ${market_report.low_price:.2f}–${market_report.high_price:.2f}\n\n"
                f"Deliver your pricing verdict. Be specific: is the price accurate? By how much is it off if so? "
                f"What exact price or range do the comps support?"
            )},
        ],
    )

    verdict = response.choices[0].message.content or "No verdict available."
    console.print("[green]✓ Price validation complete[/green]")
    return verdict
