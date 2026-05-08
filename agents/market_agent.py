import openai

from agents.utils import MODEL_SMART, console, with_retry
from models.card_listing import CardInfo, MarketSalesReport

MARKET_SYSTEM_PROMPT = """You are a sports card market data analyst specialising in recent comparable sales. You search eBay sold listings, COMC, Whatnot, Alt, StockX, SportscardsPro, and SportscardsInvestor to find the most recent actual sales of a specific card.

Your rules:
- Find real, completed sales only — not active listings or asking prices.
- Match the exact card: same player, set, year, card number, AND parallel/variant. A Silver Prizm sale does not count for a Gold Prizm.
- Prefer sales from the last 90 days. Include older comps only if recent ones are unavailable.
- Report up to 10 sales, most recent first.
- For each sale: date, final price, condition/grade, platform, and any relevant notes.
- Calculate average, low, and high from the sales you found.
- Assess the price trend honestly.
- Be brutally honest in the pricing verdict — state the exact price range the comps support."""

PARSE_SYSTEM_PROMPT = """You are a sports card pricing analyst. Parse the raw sales data into a structured MarketSalesReport.
The pricing_verdict field should state clearly what price range the comps support for a new listing — be specific and actionable."""


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

    search_response = with_retry(
        client.responses.create,
        model=MODEL_SMART,
        instructions=MARKET_SYSTEM_PROMPT,
        input=(
            f"Find the 10 most recent completed sales for this exact sports card: {query}\n\n"
            f"Search eBay sold listings, COMC, Whatnot, SportscardsPro, SportscardsInvestor, and any other platforms. "
            f"Report real sale prices only — no active listings."
        ),
        tools=[{"type": "web_search_preview"}],
    )
    sales_text = search_response.output_text

    parse_response = with_retry(
        client.beta.chat.completions.parse,
        model=MODEL_SMART,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Recent sales data for: {query}\n\n"
                f"{sales_text}\n\n"
                f"Parse into a structured MarketSalesReport. "
                f"In pricing_verdict, state the exact price range the comps support for listing this card."
            )},
        ],
        response_format=MarketSalesReport,
    )

    report = parse_response.choices[0].message.parsed
    if report is None:
        raise ValueError("Market agent returned no structured output")

    console.print("[green]✓ Market data fetched[/green]")
    return report
