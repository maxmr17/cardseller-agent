import openai

from agents.utils import MODEL_SMART, console, listing_to_text, with_retry
from models.card_listing import CardInfo, CardListing, ValidatedListing

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
SEPARATOR — the line  ___________________________________________  must appear between Section 7 and Section 8, with a blank line on each side
8. SEARCH KEYWORDS — Pipe-separated, no header, 10–15 lowercase phrases, after the separator

━━━ STYLE RULES TO ENFORCE ━━━
• Bold: player name + card ID in Section 2, parallel/insert names, key selling phrases. Never entire sentences.
• Emojis: title + hook line (max 1 emoji) + fact block ONLY. Zero emojis in body paragraphs.
• Tone: confident, factual, hobby-literate. No "amazing deal!", "great card!", "don't miss out!"
• Value framing: always speaks to both collectors AND investors
• Condition line is MANDATORY and must use the exact "Fresh pull..." phrase for ungraded cards
• Pricing must reflect the market comps provided — fix it if it's off

Your output is the refined listing that is ready for the maxmr17 style validator to sign off on."""


def validate_and_refine(
    card: CardInfo,
    draft_listing: CardListing,
    client: openai.OpenAI,
    market_context: str = "",
    price_verdict: str = "",
) -> ValidatedListing:
    """Run the validator agent to critique and refine a draft listing."""
    console.print("\n[bold yellow]🔍 Validator Agent running...[/bold yellow]")

    listing_text = listing_to_text(draft_listing)

    market_section = f"\n\nRECENT MARKET DATA:\n{market_context}" if market_context else ""
    price_section = f"\n\nPRICE VALIDATION VERDICT:\n{price_verdict}" if price_verdict else ""

    response = with_retry(
        client.beta.chat.completions.parse,
        model=MODEL_SMART,
        messages=[
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Review and refine this listing against the maxmr17 format checklist. Fix every deviation.\n\n"
                f"CARD:\n"
                f"Player: {card.player_name} | Set: {card.card_set} | Card #: {card.card_number}\n"
                f"Sport: {card.sport} | Rookie: {card.is_rookie_card} | Auto: {card.is_autographed} | "
                f"Graded: {card.is_graded}{f' ({card.grade})' if card.grade else ''}"
                f"{f' | Serial: {card.serial_number}' if card.serial_number else ''}"
                f"{market_section}{price_section}\n\n"
                f"DRAFT TO REVIEW:\n{listing_text}\n\n"
                f"Check every section in order:\n"
                f"1. Does the title start with correct sport/energy emojis?\n"
                f"2. Is the hook line punchy and specific — not generic filler?\n"
                f"3. Is the card identity paragraph bolded correctly?\n"
                f"4. Does \"Why it matters\" use proper hobby vocabulary?\n"
                f"5. Is there a visual/display sentence?\n"
                f"6. Does the condition line use EXACTLY \"Fresh pull, immediately sleeved and top-loaded — ...\"?\n"
                f"7. Is the emoji fact block complete with bold labels?\n"
                f"8. Is \"Perfect for:\" present with 4–5 bullets?\n"
                f"9. Is the separator line  ___________________________________________  present between \"Perfect for:\" and the keywords?\n"
                f"10. Are pipe-separated search keywords immediately after the separator?\n"
                f"11. Is pricing accurate to the market comps provided?\n\n"
                f"Rewrite anything that fails these checks. Only assign confidence_score 9–10 if every section is correct."
            )},
        ],
        response_format=ValidatedListing,
    )

    validated = response.choices[0].message.parsed
    if validated is None:
        raise ValueError("Validator agent returned no structured output")

    validated.original_listing = draft_listing

    console.print("[green]✓ Validation complete[/green]")
    return validated
