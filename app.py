"""
cardseller-agent — Streamlit web UI
Run with:  streamlit run app.py
"""

import os
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv

from agents import generate_listing, validate_and_refine, parse_card_text, research_card, authenticate_card
from models import CardInfo

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Card Seller Agent",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stAlert { border-radius: 8px; }
    div[data-testid="metric-container"] { background: #f8f9fa; border-radius: 8px; padding: 12px; }
    .listing-box { background: #f8f9fa; border-radius: 8px; padding: 1.2rem; border-left: 4px solid #dee2e6; margin-bottom: 1rem; }
    .listing-box-refined { border-left: 4px solid #28a745; }
    .listing-box-draft { border-left: 4px solid #6c757d; }
    .keyword-chip { display: inline-block; background: #e9ecef; border-radius: 20px; padding: 3px 10px; margin: 2px; font-size: 0.82rem; }
    .auth-card { border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
    h3 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client() -> openai.OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


def demo_card() -> CardInfo:
    return CardInfo(
        player_name="Patrick Mahomes",
        card_number="15",
        card_set="2017 Panini Prizm Football",
        sport="Football",
        is_rookie_card=True,
        is_autographed=False,
        is_graded=False,
        serial_number=None,
        parallel="Silver Prizm",
        condition="Near mint. Sharp corners, no creases. Light surface wear visible under direct light.",
        extra_notes=None,
    )


def render_auth_panel(card: CardInfo, report) -> None:
    """Render the card identification and authentication panel."""
    score = report.confidence_score
    is_auth = report.is_authentic

    verdict_color = "#28a745" if (is_auth and score >= 8) else "#ffc107" if score >= 6 else "#dc3545"
    verdict_icon = "✅" if (is_auth and score >= 8) else "⚠️" if score >= 6 else "❌"
    verdict_text = "Authenticated" if is_auth else "Could Not Authenticate"

    st.markdown("### 🔐 Card Identity & Authentication")

    col_id, col_auth = st.columns([3, 2])

    with col_id:
        st.markdown("**Confirmed Card Details**")
        details = {
            "Player": report.confirmed_player,
            "Set": report.confirmed_set,
            "Card Number": report.confirmed_card_number,
            "Sport": report.confirmed_sport,
            "Parallel": report.confirmed_parallel or "Base",
            "Serial": report.confirmed_serial or "Not numbered",
            "Rookie Card": "✓ Yes" if report.is_rookie_card else "No",
            "Autographed": "✓ Yes" if report.is_autographed else "No",
        }
        rows = "".join(
            f"<tr><td style='padding:3px 12px 3px 0;font-weight:600;color:#555;white-space:nowrap'>{k}</td>"
            f"<td style='padding:3px 0'>{v}</td></tr>"
            for k, v in details.items()
        )
        st.markdown(f"<table style='font-size:0.9rem'>{rows}</table>", unsafe_allow_html=True)

    with col_auth:
        st.markdown(
            f"<div style='text-align:center;padding:1rem;background:#f8f9fa;border-radius:10px;"
            f"border:2px solid {verdict_color}'>"
            f"<div style='font-size:2rem'>{verdict_icon}</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{verdict_color}'>{verdict_text}</div>"
            f"<div style='font-size:1.8rem;font-weight:800;color:{verdict_color}'>{score}/10</div>"
            f"<div style='font-size:0.75rem;color:#666'>confidence</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if report.red_flags:
        st.warning("**Red flags:** " + " · ".join(report.red_flags))

    with st.expander("Authentication details", expanded=False):
        st.write(report.authentication_notes)
        if report.sources_consulted:
            st.caption("Sources: " + ", ".join(report.sources_consulted))

    st.divider()


def render_listing(listing, label: str, is_refined: bool = False) -> None:
    box_class = "listing-box-refined" if is_refined else "listing-box-draft"
    pricing = listing.pricing
    strategy = pricing.strategy.value.replace("_", " ").title()

    title_len = len(listing.title)
    title_color = "#28a745" if title_len <= 80 else "#dc3545"
    title_badge = f'<span style="color:{title_color};font-size:0.8rem;font-weight:600;">{title_len}/80</span>'

    price_parts = [f"**Strategy:** {strategy}"]
    if pricing.starting_price:
        price_parts.append(f"**Starting bid:** ${pricing.starting_price:,.2f}")
    if pricing.buy_it_now_price:
        price_parts.append(f"**Buy It Now:** ${pricing.buy_it_now_price:,.2f}")
    price_parts.append(f"**Best Offer:** {'✅ Yes' if pricing.allow_offers else '❌ No'}")

    keywords_html = " ".join(
        f'<span class="keyword-chip">{kw}</span>' for kw in listing.search_keywords
    )

    specifics_rows = "".join(
        f"<tr><td style='padding:3px 10px 3px 0;font-weight:600;white-space:nowrap'>{s.key}</td>"
        f"<td style='padding:3px 0'>{s.value}</td></tr>"
        for s in listing.item_specifics
    )

    st.markdown(f"#### {label}")
    st.markdown(f"<div class='listing-box {box_class}'>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Title** {title_badge}", unsafe_allow_html=True)
        st.markdown(f"> {listing.title}")
        st.markdown(f"**Subtitle:** {listing.subtitle}")

    with c2:
        st.markdown("**Pricing**")
        for p in price_parts:
            st.markdown(f"- {p}")
        st.caption(f"*{pricing.pricing_rationale}*")

    st.divider()

    col_l, col_r = st.columns([2, 1])
    with col_l:
        with st.expander("📄 Description", expanded=True):
            st.text(listing.description)

    with col_r:
        st.markdown("**Condition**")
        st.markdown(f"Grade: `{listing.condition_grade}`")
        st.caption(listing.condition_description)
        st.markdown("**Item Specifics**")
        st.markdown(
            f"<table style='font-size:0.85rem'>{specifics_rows}</table>",
            unsafe_allow_html=True,
        )

    st.markdown("**Search Keywords**")
    st.markdown(keywords_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def build_text_output(validated, card: CardInfo) -> str:
    listing = validated.refined_listing
    pricing = listing.pricing
    lines = [
        "=" * 70,
        "EBAY LISTING — Generated by cardseller-agent",
        "=" * 70,
        "",
        f"Card: {card.player_name} | {card.card_set} | #{card.card_number}",
        "",
        f"TITLE ({len(listing.title)}/80 chars):",
        listing.title,
        "",
        "SUBTITLE:",
        listing.subtitle,
        "",
        "CONDITION:",
        f"  Grade: {listing.condition_grade}",
        f"  Notes: {listing.condition_description}",
        "",
        "PRICING:",
        f"  Strategy: {pricing.strategy.value.replace('_', ' ').title()}",
    ]
    if pricing.starting_price:
        lines.append(f"  Starting Price: ${pricing.starting_price:.2f}")
    if pricing.buy_it_now_price:
        lines.append(f"  Buy It Now: ${pricing.buy_it_now_price:.2f}")
    lines += [
        f"  Best Offer: {'Yes' if pricing.allow_offers else 'No'}",
        f"  Rationale: {pricing.pricing_rationale}",
        "",
        "ITEM SPECIFICS:",
        *[f"  {s.key}: {s.value}" for s in listing.item_specifics],
        "",
        "DESCRIPTION:",
        listing.description,
        "",
        f"SEARCH KEYWORDS: {', '.join(listing.search_keywords)}",
        "",
        "-" * 70,
        f"Confidence Score: {validated.confidence_score}/10",
        "",
        "MARKET INSIGHT:",
        validated.market_insight,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar — card input form
# ---------------------------------------------------------------------------

def render_sidebar() -> CardInfo | str | None:
    st.sidebar.title("🃏 Card Details")

    with st.sidebar.expander("🎯 Load demo card", expanded=False):
        if st.button("Patrick Mahomes 2017 Prizm RC", use_container_width=True):
            card = demo_card()
            for k, v in card.model_dump().items():
                st.session_state[f"field_{k}"] = v
            st.session_state["input_mode"] = "Manual entry"
            st.rerun()

    st.sidebar.divider()

    mode = st.sidebar.radio(
        "Input mode",
        ["Quick entry", "Manual entry"],
        index=0 if st.session_state.get("input_mode", "Quick entry") == "Quick entry" else 1,
        horizontal=True,
    )
    st.session_state["input_mode"] = mode

    # ------------------------------------------------------------------
    # Quick entry — single free-text field, web-researched by the AI
    # ------------------------------------------------------------------
    if mode == "Quick entry":
        quick_text = st.sidebar.text_area(
            "Card description",
            value=st.session_state.get("quick_text", ""),
            placeholder="e.g. Patrick Mahomes 2017 Panini Prizm Silver Prizm RC #15",
            height=100,
        )
        st.sidebar.caption("The AI will search the web to confirm every detail before generating the listing.")
        st.sidebar.divider()
        generate = st.sidebar.button(
            "⚡ Generate Listing",
            use_container_width=True,
            type="primary",
            disabled=not quick_text.strip(),
        )
        if not generate:
            return None
        st.session_state["quick_text"] = quick_text
        return quick_text.strip()

    # ------------------------------------------------------------------
    # Manual entry — full structured form
    # ------------------------------------------------------------------
    def _get(key, default=""):
        return st.session_state.get(f"field_{key}", default)

    player_name = st.sidebar.text_input("Player name *", value=_get("player_name"), placeholder="e.g. Patrick Mahomes")
    card_set = st.sidebar.text_input("Card set *", value=_get("card_set"), placeholder="e.g. 2023 Panini Prizm Football")
    card_number = st.sidebar.text_input("Card number *", value=_get("card_number"), placeholder="e.g. #/99, BASE, #123")
    sport = st.sidebar.selectbox(
        "Sport",
        ["Football", "Basketball", "Baseball", "UFC", "Soccer", "Hockey"],
        index=["Football", "Basketball", "Baseball", "UFC", "Soccer", "Hockey"].index(_get("sport") or "Football"),
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Card attributes**")
    is_rookie = st.sidebar.checkbox("Rookie card (RC)", value=bool(_get("is_rookie_card")))
    is_auto = st.sidebar.checkbox("Autographed", value=bool(_get("is_autographed")))
    is_graded = st.sidebar.checkbox("Professionally graded", value=bool(_get("is_graded")))

    grade = None
    if is_graded:
        grade = st.sidebar.text_input("Grade", value=_get("grade") or "", placeholder="e.g. PSA 10, BGS 9.5")

    st.sidebar.divider()
    st.sidebar.markdown("**Optional details**")
    serial = st.sidebar.text_input("Serial number", value=_get("serial_number") or "", placeholder="e.g. /99, /10, 1/1") or None
    parallel = st.sidebar.text_input("Parallel / variant", value=_get("parallel") or "", placeholder="e.g. Silver Prizm, Gold Refractor") or None
    condition = st.sidebar.text_area("Condition notes", value=_get("condition") or "", placeholder="e.g. Near mint, sharp corners, no creases", height=80) or None
    extra_notes = st.sidebar.text_area("Extra notes", value=_get("extra_notes") or "", placeholder="Anything else buyers should know", height=60) or None

    st.sidebar.divider()
    generate = st.sidebar.button(
        "⚡ Generate Listing",
        use_container_width=True,
        type="primary",
        disabled=not (player_name and card_set and card_number),
    )

    if not generate:
        return None

    if not player_name or not card_set or not card_number:
        st.sidebar.error("Player name, card set, and card number are required.")
        return None

    return CardInfo(
        player_name=player_name,
        card_number=card_number,
        card_set=card_set,
        sport=sport,
        is_rookie_card=is_rookie,
        is_autographed=is_auto,
        is_graded=is_graded,
        grade=grade if is_graded else None,
        serial_number=serial,
        parallel=parallel,
        condition=condition,
        extra_notes=extra_notes,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🃏 Card Seller Agent")
    st.caption("AI-powered eBay listing generator for sports cards")

    client = get_client()
    if client is None:
        st.error("**OPENAI_API_KEY not set.** Copy `.env.example` to `.env` and add your key, then restart.")
        st.stop()

    raw = render_sidebar()

    # Show landing state
    if "result" not in st.session_state and "auth_report" not in st.session_state and raw is None:
        st.markdown("### How it works")
        col1, col2, col3, col4 = st.columns(4)
        col1.info("**1. Describe your card**\nQuick free-text or detailed manual form.")
        col2.info("**2. Web research**\nAI searches the web to confirm every card detail.")
        col3.info("**3. PSA authentication**\nExpert-level identity verification with confidence score.")
        col4.info("**4. Listing generated**\nDraft written, then refined by a validator agent.")
        return

    # Run pipeline when submitted
    if raw is not None:
        # Clear previous results
        for key in ("result", "result_card", "auth_report"):
            st.session_state.pop(key, None)

        # Placeholders — auth panel renders first, listing panel renders after
        auth_placeholder = st.empty()
        listing_placeholder = st.empty()

        try:
            # ── Phase 1: Research + Authentication ──────────────────────────
            with auth_placeholder.status("🔎 Researching card details…", expanded=True) as auth_status:
                if isinstance(raw, str):
                    st.write("Searching the web to confirm card identity…")
                    card, research_summary = research_card(raw, client)
                else:
                    card = raw
                    query = f"{card.player_name} {card.card_set} #{card.card_number}"
                    st.write("Searching the web to confirm card identity…")
                    card, research_summary = research_card(query, client)

                auth_status.update(label="🔐 Authenticating card identity…", state="running")
                st.write("Running PSA-style authentication checks…")
                auth_report = authenticate_card(card, research_summary, client)
                auth_status.update(label="✅ Card identified and authenticated", state="complete")

            # Render auth panel immediately — before listing is generated
            with auth_placeholder.container():
                render_auth_panel(card, auth_report)
            st.session_state["auth_report"] = auth_report
            st.session_state["result_card"] = card

            # ── Phase 2: Listing Generation + Validation ─────────────────────
            with listing_placeholder.status("⚡ Generating listing…", expanded=True) as listing_status:
                st.write("Analyzing market conditions and crafting draft…")
                draft = generate_listing(card, client)
                listing_status.update(label="🔍 Validator refining listing…", state="running")
                st.write("Critiquing title, pricing, description, and item specifics…")
                validated = validate_and_refine(card, draft, client)
                listing_status.update(label="✅ Listing ready", state="complete")

            listing_placeholder.empty()
            st.session_state["result"] = validated

        except Exception as exc:
            auth_placeholder.empty()
            listing_placeholder.empty()
            st.error(f"Something went wrong: {exc}")
            return

        st.rerun()

    # ── Display persisted results ─────────────────────────────────────────────
    if "auth_report" in st.session_state:
        render_auth_panel(st.session_state["result_card"], st.session_state["auth_report"])

    if "result" in st.session_state:
        validated = st.session_state["result"]
        card = st.session_state["result_card"]

        score = validated.confidence_score
        m1, m2, m3 = st.columns(3)
        m1.metric("Listing Confidence", f"{score}/10")
        m2.metric("Title Length", f"{len(validated.refined_listing.title)}/80 chars")
        strategy = validated.refined_listing.pricing.strategy.value.replace("_", " ").title()
        m3.metric("Selling Strategy", strategy)

        st.divider()

        tab_final, tab_draft, tab_insights = st.tabs(["✅ Final Listing", "📝 Draft", "📊 Market Insights"])

        with tab_final:
            render_listing(validated.refined_listing, "Refined Listing", is_refined=True)
            text_output = build_text_output(validated, card)
            filename = f"{card.player_name.lower().replace(' ', '_')}_{card.card_set[:4]}_listing.txt"
            st.download_button(
                label="⬇️ Download listing as .txt",
                data=text_output.encode("utf-8"),
                file_name=filename,
                mime="text/plain",
            )

        with tab_draft:
            render_listing(validated.original_listing, "Original Draft")

        with tab_insights:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### 📈 Market Insight")
                st.info(validated.market_insight)
            with col_b:
                st.markdown("#### 🔍 Validator Notes")
                st.warning(validated.validation_notes)


if __name__ == "__main__":
    main()
