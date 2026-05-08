import openai

from agents.utils import MODEL_FAST, with_retry
from models.card_listing import CardInfo

PARSE_SYSTEM_PROMPT = """You are a sports card identification expert. Given a free-text string describing a sports card, extract all structured information from it and return a CardInfo object.

The input may contain any combination of: player name, year, set name, card number, sport, parallel/variant, serial number, grade, whether it's a rookie card, autograph, etc.

Make reasonable inferences where possible (e.g. if the set is "2017 Panini Prizm Football", the sport is Football and the year is 2017). If information is genuinely absent and cannot be inferred, leave the field as its default value."""


def parse_card_text(text: str, client: openai.OpenAI) -> CardInfo:
    """Use the LLM to parse a free-text card description into a structured CardInfo."""
    response = with_retry(
        client.beta.chat.completions.parse,
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this card description into structured data:\n\n{text}"},
        ],
        response_format=CardInfo,
    )

    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Card parser returned no structured output")
    return result
