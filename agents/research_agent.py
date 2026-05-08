import openai

from agents.utils import MODEL_SMART, console, with_retry
from models.card_listing import ResearchOutput

RESEARCH_SYSTEM_PROMPT = """You are a sports card research and authentication specialist with deep knowledge of PSA, Beckett, COMC, and eBay sold listings.

Given a card description, search the web to confirm every detail:
- Full player name (correct spelling, no nicknames unless official)
- Exact set name with year (e.g. "2017 Panini Prizm Football" not just "Prizm")
- Exact card number as printed on the card
- Parallel/variant name (exact name as used by the manufacturer)
- Serial numbering if applicable
- Whether this is a true Rookie Card (RC) as defined by hobby standards
- Whether it carries a certified autograph

Then authenticate the card identity against those findings:
- Verify player identity, set accuracy, card number validity, parallel name, serial print run, RC designation
- Cross-reference multiple sources and flag any inconsistencies
- Assign an authentication confidence score based on corroboration across sources

Cross-reference multiple sources. Report what you find factually and flag any uncertainty."""

PARSE_SYSTEM_PROMPT = """Extract a ResearchOutput from the research summary provided. This combines:
1. CardInfo — structured card details (use exact confirmed values from research)
2. AuthenticationReport — authentication verdict based on the research findings

Be precise. Use exact confirmed values from the research. If a field was not confirmed, use the most reasonable inference from context. The authentication report should reflect whether the card details are verified and consistent with known records."""


def research_and_authenticate(query: str, client: openai.OpenAI) -> ResearchOutput:
    """
    Search the web to confirm card details, then parse into CardInfo + AuthenticationReport
    in a single structured output call. Returns ResearchOutput containing both.

    Cost: 2 API calls (1 web search + 1 structured parse) vs the prior 3-call approach.
    """
    console.print("\n[bold blue]🔎 Research Agent searching the web...[/bold blue]")

    search_response = with_retry(
        client.responses.create,
        model=MODEL_SMART,
        instructions=RESEARCH_SYSTEM_PROMPT,
        input=f"Research and authenticate all details for this sports card: {query}",
        tools=[{"type": "web_search_preview"}],
    )
    research_text = search_response.output_text

    parse_response = with_retry(
        client.beta.chat.completions.parse,
        model=MODEL_SMART,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Research summary:\n\n{research_text}"},
        ],
        response_format=ResearchOutput,
    )

    result = parse_response.choices[0].message.parsed
    if result is None:
        raise ValueError("Research agent could not parse card details")

    console.print("[green]✓ Research and authentication complete[/green]")
    return result
