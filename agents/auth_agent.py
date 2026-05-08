import openai
from rich.console import Console
from models.card_listing import CardInfo, AuthenticationReport

console = Console()

AUTH_SYSTEM_PROMPT = """You are a PSA-certified authentication expert with 20+ years in the hobby and thousands of authentication decisions. Your job is to verify whether a sports card's claimed identity is accurate and consistent with known card databases.

You approach every authentication with extreme scrutiny. You check:

1. PLAYER IDENTITY — Does the player name match exactly? Are there any alternative spellings, prefixes (Jr./Sr./II), or common errors?
2. SET ACCURACY — Does the set year, brand, product name, and sport match known releases? Did this product actually exist?
3. CARD NUMBER — Is this card number valid within the known checklist? Does the card number format match the set's conventions?
4. PARALLEL/VARIANT — Is this a real parallel for this set? Does the name match exactly as the manufacturer labeled it (e.g. "Silver Prizm" not "Silver")?
5. SERIAL NUMBER — If serial numbered, is the print run (/99, /10, etc.) consistent with what was actually produced for this parallel?
6. ROOKIE CARD STATUS — Does this meet the official RC designation criteria? Was the player actually a rookie in this product's release year?
7. AUTOGRAPH — If claimed, is an auto version of this card known to exist? On-card or sticker?
8. CROSS-REFERENCE — Do the research sources align? Are there any contradictions between sources?

Your verdict must be binary: authentic (all details check out) or not authentic (one or more details cannot be verified or are inconsistent).
Be explicit about every source you consulted and every check you performed. A high confidence score requires multiple independent sources confirming the same details."""


def authenticate_card(
    card: CardInfo,
    research_summary: str,
    client: openai.OpenAI,
) -> AuthenticationReport:
    """
    Run PSA-style authentication against the proposed card identity.
    Returns an AuthenticationReport with confidence score and findings.
    """
    console.print("\n[bold magenta]🔐 Authentication Agent verifying card identity...[/bold magenta]")

    user_message = f"""Authenticate this card identity using the research findings below.

PROPOSED CARD DETAILS:
- Player: {card.player_name}
- Set: {card.card_set}
- Card Number: {card.card_number}
- Sport: {card.sport}
- Parallel: {card.parallel or 'Base'}
- Serial: {card.serial_number or 'Not serial numbered'}
- Rookie Card: {'Yes' if card.is_rookie_card else 'No'}
- Autographed: {'Yes' if card.is_autographed else 'No'}
- Grade: {card.grade or 'Ungraded'}

RESEARCH FINDINGS:
{research_summary}

Perform your full authentication checklist. Report every source consulted. Flag any inconsistency, no matter how minor. Assign a confidence score based on corroboration across multiple independent sources."""

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": AUTH_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format=AuthenticationReport,
    )

    report = response.choices[0].message.parsed
    if report is None:
        raise ValueError("Auth agent returned no structured output")

    console.print("[green]✓ Authentication complete[/green]")
    return report
