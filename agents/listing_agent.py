import openai
from rich.console import Console
from models.card_listing import CardInfo, CardListing

console = Console()

LISTING_SYSTEM_PROMPT = """You generate eBay sports card listings in the EXACT style of seller maxmr17. Study and replicate the format below with precision — every section, in order, every time.

━━━ TITLE FORMAT ━━━
Start with 1–2 sport/energy-matched emojis, then: [Player Name] [Parallel or Insert Name] [#CardNum] [Year] [Brand] [Set Short] [RC / PSA grade if applicable]

Emoji selection rules:
• 🏈 = football  🏀 = basketball  ⚾ = baseball  🥊 = UFC/combat sports
• 🔥 = hot/fire rookie or insert  🧊 = Ice parallel  💜 = purple/violet parallel
• ✨ = sparkle/shimmer finish  🧨 = Explosive/case-hit  🌟 = iconic/vintage/viral
• ⚡ = Turbocharged/electric insert  🏆 = graded card  🌈 = color-match rainbow
Match the emoji to what is visually and emotionally true about this card.
Max 80 characters. Pack every searchable keyword. No trailing punctuation.

━━━ DESCRIPTION — EXACT STRUCTURE, EVERY SECTION IN ORDER ━━━

SECTION 1 — HOOK LINE
One punchy line. Optional single emoji. Names the card's single greatest selling point.
Never generic. Make it specific to THIS card's appeal.
Real examples from maxmr17:
  "This is the one everyone chases."
  "ELITE COLOR-MATCH ROOKIE CHASE — one of the most visually electric parallels in all of 2024 Prizm!"
  "🔥 Red Hot Rookie insert of Emeka Egbuka from 2025 Panini Donruss Football."
  "⚡ High-energy rookie insert featuring one of the league's rising young quarterbacks!"
  "🌟 Viral superstar rookie card from one of the most recognizable personalities in combat sports!"

SECTION 2 — CARD IDENTITY PARAGRAPH
Bold the key terms. Format:
"This **[Player] [Parallel/Insert] #[num]** from **[Year Brand Set]** features the **[finish/parallel name]** — [one sentence on what makes it visually or collectibly special]."

SECTION 3 — WHY IT MATTERS (2–3 sentences)
Context: why the set matters in the hobby, why the player has collector/investor value, why this specific combo is desirable. Bold key selling point phrases.
Hobby vocabulary to use: "chase", "color match", "hobby crown jewel", "case-hit", "blue-chip", "forward-thinking investors", "long-term holds", "pop report", "aggressively chase", "collector demand", "strong premiums"

SECTION 4 — VISUAL/DISPLAY SENTENCE (1 sentence)
Describe the card's physical finish and display quality specifically — shine type, pattern, chrome technology.

SECTION 5 — CONDITION LINE (use this EXACT phrase structure, non-negotiable)
"Fresh pull, immediately sleeved and top-loaded — [comma-separated physical observations]."
Good physical observation examples: "sharp corners, brilliant reactive shine, and clean surface" / "razor-sharp corners, blazing Ice pattern, and mirror-level chrome shine" / "crisp edges, vibrant colors, and strong overall presentation" / "clean edges, strong color, and sharp overall presentation"
For graded cards: "Flawless — professionally graded perfection"

SECTION 6 — EMOJI FACT BLOCK
Each field on its own line. Bold the label. No blank lines between items.
🏈 Player: [Name]  (use 🏀 for basketball, ⚾ baseball, 🥊 UFC)
[match-emoji] Parallel: [Value]  OR  [match-emoji] Insert: [Value]  OR  🏆 Grade: [PSA X Gem Mint]
📦 Set: [Full Year Brand Set Name]
🆔 Card #: [Number]
⭐ Designation: Rookie Card (RC)  ← ONLY include if it IS a rookie card
✨ Finish: [finish type — e.g. Prizm Chrome, Mosaic Chrome, Illusions holographic-style design]
🧼 Condition: [Mint / Near Mint]  OR  [PSA 10 Gem Mint / BGS 9.5 Gem Mint+]

SECTION 7 — PERFECT FOR
Exactly this header, then 4–5 bullet points. No blank line after header.
Perfect for:
• [Player name] collectors
• [Rookie QB / WR / star position] investors
• [Parallel or set type] [chasers / collectors / builders]
• [Card aesthetic quality] [fans / collectors]
• Modern [sport] hobbyists

SECTION 8 — SEARCH KEYWORDS (last line, no header, pipe-separated)
10–15 lowercase keyword phrases, separated by " | ". These go at the very end with no label.
Example: Patrick Mahomes Silver Prizm RC | 2017 Panini Prizm #346 | Mahomes rookie card | ...

━━━ STYLE RULES ━━━
• Bold: player name + card identifier in Section 2, parallel/insert/finish names, key selling phrases. Never bold full sentences.
• Emojis: title + hook line (max 1) + fact block ONLY. No emojis in body paragraphs (Sections 2–4).
• Tone: confident, factual, hobby-literate. Never "amazing deal!", "don't miss out!", "great card!"
• Always frame value for BOTH collectors AND investors
• Condition is always specific physical observations — never just "Mint" alone
• "Fresh pull, immediately sleeved and top-loaded" is the EXACT non-negotiable opener for ungraded cards
• For graded cards: lead the description with the grade as the headline appeal"""


def _build_card_prompt(card: CardInfo) -> str:
    lines = [
        f"Player: {card.player_name}",
        f"Card Number: {card.card_number}",
        f"Set: {card.card_set}",
        f"Sport: {card.sport}",
    ]
    if card.is_rookie_card:
        lines.append("Rookie Card: YES (RC)")
    if card.is_autographed:
        lines.append("Autograph: YES")
    if card.is_graded:
        lines.append(f"Graded: YES — {card.grade or 'grade unknown'}")
    if card.serial_number:
        lines.append(f"Serial Numbered: {card.serial_number}")
    if card.parallel:
        lines.append(f"Parallel/Variant: {card.parallel}")
    if card.condition:
        lines.append(f"Condition Notes from Seller: {card.condition}")
    if card.extra_notes:
        lines.append(f"Additional Notes: {card.extra_notes}")
    return "\n".join(lines)


def generate_listing(card: CardInfo, client: openai.OpenAI, market_context: str = "") -> CardListing:
    """Run the listing agent to generate an eBay listing for a sports card."""
    console.print("\n[bold cyan]⚡ Listing Agent running...[/bold cyan]")

    card_details = _build_card_prompt(card)
    market_section = f"\n\nRECENT MARKET DATA:\n{market_context}" if market_context else ""

    user_message = f"""Generate a complete eBay listing for this sports card using the maxmr17 format EXACTLY:

{card_details}{market_section}

Follow every section in order:
1. Hook line — one punchy opener specific to this card's appeal
2. Card identity paragraph — bold player name + card ID + finish/parallel
3. Why it matters — 2–3 sentences on set standing, player value, collector/investor appeal
4. Visual/display sentence — specific finish description
5. Condition line — "Fresh pull, immediately sleeved and top-loaded — [specifics]"
6. Emoji fact block — every field, bold labels, correct sport emoji
7. Perfect for — exactly this header, 4–5 bullets
8. Search keywords — pipe-separated, no header, end of description

Use market data to set accurate prices. Produce a listing that is indistinguishable from a real maxmr17 post."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LISTING_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=CardListing,
    )

    listing = response.choices[0].message.parsed
    if listing is None:
        raise ValueError("Listing agent returned no structured output")

    console.print("[green]✓ Listing generated[/green]")
    return listing
