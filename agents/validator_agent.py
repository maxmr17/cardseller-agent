import openai
from rich.console import Console
from models.card_listing import CardInfo, CardListing, ValidatedListing

console = Console()

VALIDATOR_SYSTEM_PROMPT = """You are a senior eBay sports card listing specialist. Your job is to review and refine listings to ensure they perfectly match the maxmr17 style — the exact format and voice of that specific seller.

━━━ MAXMR17 FORMAT CHECKLIST ━━━

TITLE must:
□ Start with 1–2 sport/energy-matched emojis (🏈🏀⚾🥊🔥🧊💜✨🧨🌟⚡🏆)
□ Follow format: [emoji(s)] [Player] [Parallel/Insert] [#Num] [Year] [Brand] [Set] [RC/Grade]
□ Be ≤80 characters, keyword-dense, no trailing punctuation

DESCRIPTION must contain these 8 sections IN ORDER:

1. HOOK LINE — 1 punchy line (optional single emoji). Specific to this card's appeal. Not generic.
2. CARD IDENTITY PARA — "This **[Player Insert #num]** from **[Year Brand Set]** features the **[finish]** — [sell line]."
3. WHY IT MATTERS — 2–3 sentences. Set's hobby standing + player value + this card's collector/investor appeal. Use: "chase", "color match", "hobby crown jewel", "case-hit", "blue-chip", "forward-thinking investors", "long-term holds"
4. VISUAL SENTENCE — 1 sentence on finish, shine, pattern specifics
5. CONDITION LINE — EXACTLY: "Fresh pull, immediately sleeved and top-loaded — [comma observations]." (For graded: "Flawless — professionally graded perfection")
6. EMOJI FACT BLOCK — Every field, bold labels, correct emojis: 🏈/🏀/⚾/🥊 Player · [match] Parallel/Insert/Grade · 📦 Set · 🆔 Card # · ⭐ Designation (RC only) · 🏆 Grade (graded only) · ✨ Finish · 🧼 Condition
7. PERFECT FOR — Header "Perfect for:" then exactly 4–5 bullets: [Player] collectors · [position] investors · [parallel/set] [chasers/builders] · [aesthetic] [fans] · Modern [sport] hobbyists
8. SEARCH KEYWORDS — Pipe-separated, no header, 10–15 lowercase phrases, last line of description

━━━ STYLE RULES TO ENFORCE ━━━
• Bold: player name + card ID in Section 2, parallel/insert names, key selling phrases. Never entire sentences.
• Emojis: title + hook line (max 1 emoji) + fact block ONLY. Zero emojis in body paragraphs.
• Tone: confident, factual, hobby-literate. No "amazing deal!", "great card!", "don't miss out!"
• Value framing: always speaks to both collectors AND investors
• Condition line is MANDATORY and must use the exact "Fresh pull..." phrase for ungraded cards
• Pricing must reflect the market comps provided — fix it if it's off

Your output is the refined listing that is ready for the maxmr17 style validator to sign off on."""


def _listing_to_text(listing: CardListing) -> str:
    pricing = listing.pricing
    strategy_str = pricing.strategy.value.replace("_", " ").title()

    price_parts = [f"Strategy: {strategy_str}"]
    if pricing.starting_price:
        price_parts.append(f"Starting Price: ${pricing.starting_price:.2f}")
    if pricing.buy_it_now_price:
        price_parts.append(f"Buy It Now: ${pricing.buy_it_now_price:.2f}")
    price_parts.append(f"Best Offer: {'Yes' if pricing.allow_offers else 'No'}")
    price_parts.append(f"Rationale: {pricing.pricing_rationale}")

    specifics_text = "\n".join(f"  {s.key}: {s.value}" for s in listing.item_specifics)

    return f"""TITLE ({len(listing.title)}/80 chars): {listing.title}
SUBTITLE: {listing.subtitle}
CONDITION GRADE: {listing.condition_grade}
CONDITION NOTES: {listing.condition_description}

PRICING:
{chr(10).join(f'  {p}' for p in price_parts)}

ITEM SPECIFICS:
{specifics_text}

SEARCH KEYWORDS: {', '.join(listing.search_keywords)}

DESCRIPTION:
{listing.description}"""


def validate_and_refine(
    card: CardInfo,
    draft_listing: CardListing,
    client: openai.OpenAI,
    market_context: str = "",
    price_verdict: str = "",
) -> ValidatedListing:
    """Run the validator agent to critique and refine a draft listing."""
    console.print("\n[bold yellow]🔍 Validator Agent running...[/bold yellow]")

    listing_text = _listing_to_text(draft_listing)

    market_section = f"\n\nRECENT MARKET DATA:\n{market_context}" if market_context else ""
    price_section = f"\n\nPRICE VALIDATION VERDICT:\n{price_verdict}" if price_verdict else ""

    user_message = f"""Review and refine this listing against the maxmr17 format checklist. Fix every deviation.

CARD:
Player: {card.player_name} | Set: {card.card_set} | Card #: {card.card_number}
Sport: {card.sport} | Rookie: {card.is_rookie_card} | Auto: {card.is_autographed} | Graded: {card.is_graded}{f' ({card.grade})' if card.grade else ''}{f' | Serial: {card.serial_number}' if card.serial_number else ''}{market_section}{price_section}

DRAFT TO REVIEW:
{listing_text}

Check every section in order:
1. Does the title start with correct sport/energy emojis?
2. Is the hook line punchy and specific — not generic filler?
3. Is the card identity paragraph bolded correctly?
4. Does "Why it matters" use proper hobby vocabulary?
5. Is there a visual/display sentence?
6. Does the condition line use EXACTLY "Fresh pull, immediately sleeved and top-loaded — ..."?
7. Is the emoji fact block complete with bold labels?
8. Is "Perfect for:" present with 4–5 bullets?
9. Are pipe-separated search keywords at the very end?
10. Is pricing accurate to the market comps provided?

Rewrite anything that fails these checks. Only assign confidence_score 9–10 if every section is correct."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=ValidatedListing,
    )

    validated = response.choices[0].message.parsed
    if validated is None:
        raise ValueError("Validator agent returned no structured output")

    validated.original_listing = draft_listing

    console.print("[green]✓ Validation complete[/green]")
    return validated
