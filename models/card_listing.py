from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class SellingStrategy(str, Enum):
    AUCTION = "auction"
    BUY_IT_NOW = "buy_it_now"
    AUCTION_WITH_BIN = "auction_with_buy_it_now"


class PricingRecommendation(BaseModel):
    strategy: SellingStrategy = Field(
        description="Recommended selling strategy"
    )
    starting_price: Optional[float] = Field(
        default=None,
        description="Starting auction price in USD (required for auction strategies)"
    )
    buy_it_now_price: Optional[float] = Field(
        default=None,
        description="Buy It Now price in USD"
    )
    allow_offers: bool = Field(
        default=False,
        description="Whether to enable Best Offer on the listing"
    )
    pricing_rationale: str = Field(
        description="Explanation of the pricing strategy and how it was determined"
    )


class ItemSpecific(BaseModel):
    key: str = Field(description="The item specific field name (e.g. Sport, Player, Year, Set)")
    value: str = Field(description="The item specific value")


class CardListing(BaseModel):
    title: str = Field(
        description="eBay listing title. Maximum 80 characters. Must include player name, year, set, card number, and key attributes like RC, Auto, /serial."
    )
    subtitle: str = Field(
        description="Optional eBay subtitle (max 55 chars) to add detail not in the title"
    )
    description: str = Field(
        description="Full eBay listing description. Use plain text with clear sections. Should cover card details, condition, shipping policy, and seller notes."
    )
    item_specifics: list[ItemSpecific] = Field(
        description="eBay item specifics as key-value pairs (e.g. Sport, Player, Year, Set, Card Number, Autographed, Graded, Grade)"
    )
    pricing: PricingRecommendation
    search_keywords: list[str] = Field(
        description="Top keywords collectors search for this card (for description optimization)"
    )
    condition_grade: str = Field(
        description="Standard eBay condition grade (Near Mint or Better, Excellent, Very Good, Good, Acceptable)"
    )
    condition_description: str = Field(
        description="Detailed seller condition notes describing the card's physical state"
    )


class CardInfo(BaseModel):
    player_name: str = Field(description="Full player name")
    card_number: str = Field(description="Card number as printed on the card (e.g. #/99, #123, BASE)")
    card_set: str = Field(description="Full set name including year (e.g. 2023 Panini Prizm)")
    sport: str = Field(default="Football", description="Sport (Football, Basketball, Baseball, UFC, etc.)")
    condition: Optional[str] = Field(default=None, description="Known condition notes from seller")
    is_rookie_card: bool = Field(default=False, description="Whether this is a rookie card (RC)")
    is_autographed: bool = Field(default=False, description="Whether the card has an on-card or sticker autograph")
    is_graded: bool = Field(default=False, description="Whether the card has been professionally graded")
    grade: Optional[str] = Field(default=None, description="Professional grade (e.g. PSA 10, BGS 9.5, SGC 10)")
    serial_number: Optional[str] = Field(default=None, description="Serial numbering (e.g. /99, /10, 1/1)")
    parallel: Optional[str] = Field(default=None, description="Parallel variant name (e.g. Silver Prizm, Gold Refractor, Holo)")
    extra_notes: Optional[str] = Field(default=None, description="Any other relevant details the seller wants to include")


class SaleRecord(BaseModel):
    sale_date: str = Field(description="Date of sale (e.g. 'May 2, 2026' or 'Apr 2026')")
    price: float = Field(description="Sale price in USD")
    condition: str = Field(description="Card condition or grade at time of sale")
    platform: str = Field(description="Platform where sold (e.g. eBay, COMC, Whatnot)")
    details: str = Field(description="Any additional context — parallel match, serial, graded vs raw, etc.")


class MarketSalesReport(BaseModel):
    sales: list[SaleRecord] = Field(description="Recent comparable sales found, most recent first")
    avg_price: float = Field(description="Average sale price across all found comps")
    low_price: float = Field(description="Lowest recent sale price")
    high_price: float = Field(description="Highest recent sale price")
    price_trend: str = Field(description="Overall trend: 'Rising', 'Stable', or 'Falling'")
    market_summary: str = Field(description="2-3 sentence summary of market conditions for this card")
    pricing_verdict: str = Field(description="Specific critique of whether the proposed listing price is accurate, too high, or too low given the comps")
    data_quality: str = Field(description="'High' if 5+ confirmed comps found, 'Medium' if 2-4, 'Low' if fewer than 2")


class AuthenticationReport(BaseModel):
    confirmed_player: str = Field(description="Confirmed full player name")
    confirmed_set: str = Field(description="Confirmed full set name including year")
    confirmed_card_number: str = Field(description="Confirmed card number as printed on the card")
    confirmed_parallel: Optional[str] = Field(default=None, description="Confirmed parallel/variant name, if any")
    confirmed_serial: Optional[str] = Field(default=None, description="Confirmed serial numbering, if any")
    confirmed_sport: str = Field(description="Confirmed sport")
    is_rookie_card: bool = Field(description="Whether this is a confirmed rookie card")
    is_autographed: bool = Field(description="Whether the card has a confirmed autograph")
    is_authentic: bool = Field(description="Overall authentication verdict — does the card match known database records?")
    confidence_score: int = Field(description="Authentication confidence 1-10", ge=1, le=10)
    red_flags: list[str] = Field(description="Any inconsistencies, discrepancies, or concerns found during research")
    authentication_notes: str = Field(description="Detailed findings from the authentication process")
    sources_consulted: list[str] = Field(description="Sources used to authenticate (e.g. PSA database, Beckett, COMC, recent eBay sold listings)")


class ValidatedListing(BaseModel):
    original_listing: CardListing
    refined_listing: CardListing
    validation_notes: str = Field(
        description="Summary of changes made and why, plus any important selling tips for this specific card"
    )
    confidence_score: int = Field(
        description="Confidence score 1-10 on how well this listing will perform",
        ge=1,
        le=10
    )
    market_insight: str = Field(
        description="Brief market insight for this card - current demand, comparable sales, what buyers look for"
    )


class ResearchOutput(BaseModel):
    """Combined output from the research + authentication step (one parse call instead of two)."""
    card: CardInfo
    auth: AuthenticationReport


class StyleApproval(BaseModel):
    approved: bool = Field(
        description="True if the listing authentically matches maxmr17's style without major issues. False if significant rewrites were required."
    )
    style_score: int = Field(
        description="Style match score 1-10 against maxmr17's exact format. 10 = indistinguishable from a real maxmr17 listing.",
        ge=1, le=10
    )
    title_verdict: str = Field(
        description="Specific feedback on the title — emoji selection, keyword order, character efficiency"
    )
    description_verdict: str = Field(
        description="Specific feedback on the description — which structural sections were missing, wrong, or off-voice"
    )
    final_listing: CardListing = Field(
        description="The maxmr17-signed-off listing. Always a complete CardListing — revised to match the style perfectly if needed, or confirmed as-is if already perfect."
    )
    approval_summary: str = Field(
        description="One paragraph from maxmr17's POV explaining what was changed and why, or confirming why this listing is ready to post."
    )
