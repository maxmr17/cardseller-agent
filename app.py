"""
cardseller-agent — Streamlit web UI
Run with:  streamlit run app.py
"""

import os

import openai
import streamlit as st
from dotenv import load_dotenv

from agents import (
    generate_listing,
    validate_and_refine,
    parse_card_text,
    research_card,
    search_card_image,
    authenticate_card,
    fetch_market_sales,
    validate_listing_price,
)
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

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .keyword-chip {
        display: inline-block; background: #e9ecef; border-radius: 20px;
        padding: 3px 10px; margin: 2px; font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Client
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

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_auth_panel(card: CardInfo, report, image_url: str | None) -> None:
    score = report.confidence_score
    is_auth = report.is_authentic
    verdict_color = "#28a745" if (is_auth and score >= 8) else "#ffc107" if score >= 6 else "#dc3545"
    verdict_icon = "✅" if (is_auth and score >= 8) else "⚠️" if score >= 6 else "❌"
    verdict_text = "Authenticated" if is_auth else "Could Not Authenticate"

    st.markdown("### 🔐 Card Identity & Authentication")

    img_col, id_col, auth_col = st.columns([1, 2, 1])

    with img_col:
        if image_url:
            try:
                st.image(image_url, caption=f"{card.player_name} — {card.card_set}", use_container_width=True)
            except Exception:
                # Fallback: show as a clickable link if Streamlit can't render it directly
                st.markdown(f"[View card image]({image_url})", unsafe_allow_html=False)
        else:
            st.markdown(
                "<div style='height:160px;background:#f0f0f0;border-radius:8px;"
                "display:flex;align-items:center;justify-content:center;"
                "color:#aaa;font-size:0.85rem'>No image found</div>",
                unsafe_allow_html=True,
            )

    with id_col:
        st.markdown("**Confirmed Card Details**")
        details = {
            "Player": report.confirmed_player,
            "Set": report.confirmed_set,
            "Card #": report.confirmed_card_number,
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

    with auth_col:
        st.markdown(
            f"<div style='text-align:center;padding:1rem;background:#f8f9fa;border-radius:10px;"
            f"border:2px solid {verdict_color}'>"
            f"<div style='font-size:2rem'>{verdict_icon}</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{verdict_color}'>{verdict_text}</div>"
            f"<div style='font-size:1.8rem;font-weight:800;color:{verdict_color}'>{score}/10</div>"
            f"<div style='font-size:0.75rem;color:#666'>confidence</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if report.red_flags:
        st.warning("**Red flags:** " + " · ".join(report.red_flags))

    with st.expander("Authentication details & sources", expanded=False):
        st.write(report.authentication_notes)
        if report.sources_consulted:
            st.caption("Sources: " + ", ".join(report.sources_consulted))


def render_listing(listing, label: str, is_refined: bool = False) -> None:
    pricing = listing.pricing
    strategy = pricing.strategy.value.replace("_", " ").title()

    title_len = len(listing.title)
    title_color = "green" if title_len <= 80 else "red"

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

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Title** :{title_color}[{title_len}/80]")
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
        st.markdown(f"<table style='font-size:0.85rem'>{specifics_rows}</table>", unsafe_allow_html=True)

    st.markdown("**Search Keywords**")
    st.markdown(keywords_html, unsafe_allow_html=True)


def render_market_tab(market_report, price_verdict: str) -> None:
    if market_report is None:
        st.info("Market data not available for this card.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Sale Price", f"${market_report.avg_price:,.2f}")
    col2.metric("Low", f"${market_report.low_price:,.2f}")
    col3.metric("High", f"${market_report.high_price:,.2f}")
    trend_icon = "📈" if market_report.price_trend == "Rising" else "📉" if market_report.price_trend == "Falling" else "➡️"
    col4.metric("Trend", f"{trend_icon} {market_report.price_trend}")

    st.markdown(f"**Data quality:** `{market_report.data_quality}`")
    st.caption(market_report.market_summary)

    st.divider()
    st.markdown("#### Recent Comparable Sales")

    if market_report.sales:
        import pandas as pd
        rows = [
            {
                "Date": s.sale_date,
                "Price": f"${s.price:,.2f}",
                "Condition": s.condition,
                "Platform": s.platform,
                "Notes": s.details,
            }
            for s in market_report.sales
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No comparable sales found.")

    if price_verdict:
        st.divider()
        st.markdown("#### 💰 Price Validation Verdict")
        st.warning(price_verdict)


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
# Sidebar
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
            "⚡ Generate Listing", use_container_width=True, type="primary",
            disabled=not quick_text.strip(),
        )
        if not generate:
            return None
        st.session_state["quick_text"] = quick_text
        return quick_text.strip()

    def _get(key, default=""):
        return st.session_state.get(f"field_{key}", default)

    player_name = st.sidebar.text_input("Player name *", value=_get("player_name"), placeholder="e.g. Patrick Mahomes")
    card_set = st.sidebar.text_input("Card set *", value=_get("card_set"), placeholder="e.g. 2023 Panini Prizm Football")
    card_number = st.sidebar.text_input("Card number *", value=_get("card_number"), placeholder="e.g. #/99, BASE, #123")
    sport = st.sidebar.selectbox(
        "Sport", ["Football", "Basketball", "Baseball", "UFC", "Soccer", "Hockey"],
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
    condition = st.sidebar.text_area("Condition notes", value=_get("condition") or "", placeholder="e.g. Near mint, sharp corners", height=80) or None
    extra_notes = st.sidebar.text_area("Extra notes", value=_get("extra_notes") or "", placeholder="Anything else buyers should know", height=60) or None
    st.sidebar.divider()
    generate = st.sidebar.button(
        "⚡ Generate Listing", use_container_width=True, type="primary",
        disabled=not (player_name and card_set and card_number),
    )
    if not generate:
        return None
    if not player_name or not card_set or not card_number:
        st.sidebar.error("Player name, card set, and card number are required.")
        return None
    return CardInfo(
        player_name=player_name, card_number=card_number, card_set=card_set,
        sport=sport, is_rookie_card=is_rookie, is_autographed=is_auto,
        is_graded=is_graded, grade=grade if is_graded else None,
        serial_number=serial, parallel=parallel, condition=condition, extra_notes=extra_notes,
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

    # Landing state
    if not any(k in st.session_state for k in ("auth_report", "result")) and raw is None:
        st.markdown("### How it works")
        col1, col2, col3, col4 = st.columns(4)
        col1.info("**1. Describe your card**\nQuick free-text or detailed manual form.")
        col2.info("**2. Web research + auth**\nAI confirms every detail and authenticates card identity.")
        col3.info("**3. You confirm the card**\nReview the identified card before any listing is generated.")
        col4.info("**4. Listing generated**\nMarket-data-backed draft, then refined to perfection.")
        return

    # ── Phase 1: Research + Auth (triggered by form submit) ──────────────────
    if raw is not None:
        for key in ("auth_report", "result_card", "result", "card_confirmed",
                    "market_report", "price_verdict", "card_image_url"):
            st.session_state.pop(key, None)

        with st.status("🔎 Researching card details…", expanded=True) as s:
            query = raw if isinstance(raw, str) else (
                f"{raw.player_name} {raw.card_set} #{raw.card_number}"
                + (f" {raw.parallel}" if raw.parallel else "")
            )
            st.write("Searching the web to confirm card identity…")
            card, research_summary = research_card(query, client)

            st.write("Searching for card image…")
            image_url = search_card_image(card, client)

            s.update(label="🔐 Authenticating card identity…", state="running")
            st.write("Running PSA-style authentication checks…")
            auth_report = authenticate_card(card, research_summary, client)
            s.update(label="✅ Card identified — please confirm below", state="complete")

        st.session_state["auth_report"] = auth_report
        st.session_state["result_card"] = card
        st.session_state["card_image_url"] = image_url
        st.rerun()

    # ── Show auth panel + confirmation gate ──────────────────────────────────
    if "auth_report" in st.session_state and "result" not in st.session_state:
        render_auth_panel(
            st.session_state["result_card"],
            st.session_state["auth_report"],
            st.session_state.get("card_image_url"),
        )

        if not st.session_state.get("card_confirmed"):
            st.markdown("#### Is this the card you're listing?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("✅ Yes — generate listing", use_container_width=True, type="primary"):
                st.session_state["card_confirmed"] = True
                st.rerun()
            if col_no.button("❌ No — start over", use_container_width=True):
                for key in ("auth_report", "result_card", "card_confirmed",
                            "market_report", "price_verdict", "card_image_url"):
                    st.session_state.pop(key, None)
                st.rerun()
            st.stop()

    # ── Phase 2: Market data + listing (after user confirms) ─────────────────
    if st.session_state.get("card_confirmed") and "result" not in st.session_state:
        card = st.session_state["result_card"]

        with st.status("📊 Fetching market data…", expanded=True) as s:
            st.write("Searching eBay sold, SportscardsPro, SportscardsInvestor…")
            try:
                market_report = fetch_market_sales(card, client)
                market_context = (
                    f"Recent sales avg: ${market_report.avg_price:.2f} | "
                    f"Range: ${market_report.low_price:.2f}–${market_report.high_price:.2f} | "
                    f"Trend: {market_report.price_trend}\n"
                    + "\n".join(
                        f"- {s.sale_date}: ${s.price:.2f} ({s.condition}) on {s.platform}"
                        for s in market_report.sales[:5]
                    )
                )
            except Exception:
                market_report = None
                market_context = ""

            s.update(label="⚡ Generating listing…", state="running")
            st.write("Crafting optimised draft with market data…")
            draft = generate_listing(card, client, market_context=market_context)

            if market_report:
                price_verdict = validate_listing_price(card, draft, market_report, client)
            else:
                price_verdict = ""

            s.update(label="🔍 Validator refining to perfection…", state="running")
            st.write("Scrutinising every element — aiming for 10/10…")
            validated = validate_and_refine(
                card, draft, client,
                market_context=market_context,
                price_verdict=price_verdict,
            )
            s.update(label="✅ Listing ready", state="complete")

        st.session_state["result"] = validated
        st.session_state["market_report"] = market_report
        st.session_state["price_verdict"] = price_verdict
        st.rerun()

    # ── Display persisted results ─────────────────────────────────────────────
    if "auth_report" in st.session_state:
        render_auth_panel(
            st.session_state["result_card"],
            st.session_state["auth_report"],
            st.session_state.get("card_image_url"),
        )
        st.divider()

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

        tab_final, tab_draft, tab_market, tab_insights = st.tabs([
            "✅ Final Listing", "📝 Draft", "📊 Market Data", "💡 Insights"
        ])

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

        with tab_market:
            render_market_tab(
                st.session_state.get("market_report"),
                st.session_state.get("price_verdict", ""),
            )

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
