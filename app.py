"""
cardseller-agent — Streamlit web UI
Run with:  streamlit run app.py
"""

import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv

from agents import (
    generate_listing,
    validate_and_refine,
    parse_card_text,
    research_and_authenticate,
    fetch_market_sales,
    maxmr17_approve,
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
# Progress tracker
# ---------------------------------------------------------------------------

class PipelineProgress:
    """Renders a live progress bar + ETA that updates between pipeline steps."""

    def __init__(self, steps: list[tuple[str, int]]):
        # steps = [(label, estimated_seconds), ...]
        self.steps = steps
        self.n = len(steps)
        self.total_est = sum(s[1] for s in steps)
        self._bar = st.progress(0.0)
        self._text = st.empty()
        self._start = time.time()
        self._step_idx = 0
        self._done_est = 0
        self._render()

    def advance(self):
        """Mark the current step done and move to the next."""
        if self._step_idx < self.n:
            self._done_est += self.steps[self._step_idx][1]
            self._step_idx += 1
        self._render()

    def complete(self):
        self._bar.progress(1.0)
        elapsed = int(time.time() - self._start)
        self._text.markdown(
            f"<div style='font-size:0.85rem;color:#28a745'>"
            f"✅ &nbsp;<strong>Complete</strong> — finished in {elapsed}s"
            f"</div>",
            unsafe_allow_html=True,
        )

    def _render(self):
        elapsed = int(time.time() - self._start)
        pct = min(self._done_est / self.total_est, 0.97) if self.total_est else 0
        remaining = max(0, self.total_est - self._done_est)

        self._bar.progress(pct)

        if self._step_idx < self.n:
            label = self.steps[self._step_idx][0]
            mins, secs = divmod(remaining, 60)
            eta = f"{mins}m {secs}s" if mins else f"{secs}s"
            done_pct = int(pct * 100)
            self._text.markdown(
                f"<div style='font-size:0.85rem;color:#555;margin-top:2px'>"
                f"<strong>{label}</strong>"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"Step {self._step_idx + 1} of {self.n}"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{done_pct}% complete"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"~{eta} remaining"
                f"&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"{elapsed}s elapsed"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client() -> openai.OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_auth_panel(card: CardInfo, report) -> None:
    is_auth = report.is_authentic
    verdict_color = "#28a745" if is_auth else "#dc3545"
    verdict_icon = "✅" if is_auth else "❌"
    verdict_text = "Authenticated" if is_auth else "Could Not Authenticate"

    st.markdown("### 🔐 Card Identity & Authentication")

    id_col, auth_col = st.columns([3, 1])

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
            f"<div style='font-size:2.5rem'>{verdict_icon}</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{verdict_color}'>{verdict_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if report.red_flags:
        st.warning("**Red flags:** " + " · ".join(report.red_flags))

    with st.expander("Authentication details & sources", expanded=False):
        notes = report.authentication_notes.strip()
        bullets = [line.strip("•- ").strip() for line in notes.splitlines() if line.strip()]
        if len(bullets) <= 1:
            bullets = [s.strip() for s in notes.replace(". ", ".\n").splitlines() if s.strip()]
        for b in bullets:
            if b:
                st.markdown(f"- {b}")
        if report.sources_consulted:
            st.markdown("**Sources consulted:**")
            for src in report.sources_consulted:
                st.markdown(f"- {src}")


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
        st.markdown("**📄 Description** — copy and paste directly into eBay")
        st.code(listing.description, language=None)
    with col_r:
        st.markdown("**Condition**")
        st.markdown(f"Grade: `{listing.condition_grade}`")
        st.caption(listing.condition_description)
        st.markdown("**Item Specifics**")
        st.markdown(f"<table style='font-size:0.85rem'>{specifics_rows}</table>", unsafe_allow_html=True)

    st.markdown("**Search Keywords**")
    st.markdown(keywords_html, unsafe_allow_html=True)


def render_maxmr17_panel(approval) -> None:
    approved = approval.approved
    score = approval.style_score
    badge_color = "#28a745" if approved else "#e67e22"
    badge_icon = "✅" if approved else "✏️"
    badge_text = "Style Approved" if approved else "Style Revised"

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("### 👤 maxmr17 Style Review")
        st.caption(approval.approval_summary)
    with c2:
        st.markdown(
            f"<div style='text-align:center;padding:0.8rem;background:#f8f9fa;"
            f"border-radius:10px;border:2px solid {badge_color}'>"
            f"<div style='font-size:1.6rem'>{badge_icon}</div>"
            f"<div style='font-size:0.9rem;font-weight:700;color:{badge_color}'>{badge_text}</div>"
            f"<div style='font-size:1.6rem;font-weight:800;color:{badge_color}'>{score}/10</div>"
            f"<div style='font-size:0.7rem;color:#666'>style score</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    col_t, col_d = st.columns(2)
    with col_t:
        st.markdown("**Title feedback**")
        st.caption(approval.title_verdict)
    with col_d:
        st.markdown("**Description feedback**")
        st.caption(approval.description_verdict)


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
                "Date": sale.sale_date,
                "Price": f"${sale.price:,.2f}",
                "Condition": sale.condition,
                "Platform": sale.platform,
                "Notes": sale.details,
            }
            for sale in market_report.sales
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No comparable sales found.")

    if price_verdict:
        st.divider()
        st.markdown("#### 💰 Price Validation Verdict")
        st.warning(price_verdict)


def build_text_output_with_listing(listing, validated, card: CardInfo) -> str:
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
            placeholder="e.g. Patrick Mahomes, 2017 Panini Prizm, Silver Prizm RC, #15",
            height=100,
        )
        st.sidebar.caption("Comma-separated or free text — the AI confirms every detail before generating.")
        st.sidebar.divider()
        generate = st.sidebar.button(
            "⚡ Generate Listing", use_container_width=True, type="primary",
            disabled=not quick_text.strip(),
        )
        if not generate:
            return None
        st.session_state["quick_text"] = quick_text
        # Normalise comma-separated input into a space-joined query
        normalised = " ".join(p.strip() for p in quick_text.split(",") if p.strip())
        return normalised

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
                    "market_report", "style_approval"):
            st.session_state.pop(key, None)

        p1 = PipelineProgress([
            ("Researching & authenticating card identity", 30),
        ])

        with st.status("🔎 Researching card details…", expanded=True) as status:
            query = raw if isinstance(raw, str) else (
                f"{raw.player_name} {raw.card_set} #{raw.card_number}"
                + (f" {raw.parallel}" if raw.parallel else "")
            )
            st.write("Searching the web and running authentication checks…")
            result = research_and_authenticate(query, client)
            p1.complete()
            status.update(label="✅ Card identified — please confirm below", state="complete")

        st.session_state["auth_report"] = result.auth
        st.session_state["result_card"] = result.card
        st.rerun()

    # ── Show auth panel + confirmation gate ──────────────────────────────────
    if "auth_report" in st.session_state and "result" not in st.session_state:
        render_auth_panel(
            st.session_state["result_card"],
            st.session_state["auth_report"],
        )

        if not st.session_state.get("card_confirmed"):
            st.markdown("#### Is this the card you're listing?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("✅ Yes — generate listing", use_container_width=True, type="primary"):
                st.session_state["card_confirmed"] = True
                st.rerun()
            if col_no.button("❌ No — start over", use_container_width=True):
                for key in ("auth_report", "result_card", "card_confirmed",
                            "market_report", "style_approval"):
                    st.session_state.pop(key, None)
                st.rerun()
            st.stop()

    # ── Phase 2: Market data + listing (after user confirms) ─────────────────
    if st.session_state.get("card_confirmed") and "result" not in st.session_state:
        card = st.session_state["result_card"]

        p2 = PipelineProgress([
            ("Fetching market data",            20),
            ("Generating listing",              15),
            ("Refining listing",               20),
            ("maxmr17 style review — pass 1",  15),
            ("maxmr17 style review — pass 2",  15),
            ("maxmr17 style review — pass 3",  15),
            ("maxmr17 style review — pass 4",  15),
        ])

        with st.status("📊 Fetching market data…", expanded=True) as status:
            st.write("Searching eBay sold, SportscardsPro, SportscardsInvestor…")
            try:
                market_report = fetch_market_sales(card, client)
                market_context = (
                    f"Recent sales avg: ${market_report.avg_price:.2f} | "
                    f"Range: ${market_report.low_price:.2f}–${market_report.high_price:.2f} | "
                    f"Trend: {market_report.price_trend}\n"
                    + "\n".join(
                        f"- {sale.sale_date}: ${sale.price:.2f} ({sale.condition}) on {sale.platform}"
                        for sale in market_report.sales[:5]
                    )
                )
                # pricing_verdict comes directly from the market report — no extra LLM call needed
                price_verdict = market_report.pricing_verdict
            except Exception:
                market_report = None
                market_context = ""
                price_verdict = ""
            p2.advance()

            status.update(label="⚡ Generating listing…", state="running")
            st.write("Crafting optimised draft with market data…")
            draft = generate_listing(card, client, market_context=market_context)
            p2.advance()

            status.update(label="🔍 Validator refining to perfection…", state="running")
            st.write("Scrutinising every element — aiming for 10/10…")
            validated = validate_and_refine(
                card, draft, client,
                market_context=market_context,
                price_verdict=price_verdict,
            )
            p2.advance()

            status.update(label="👤 maxmr17 reviewing style…", state="running")
            current_listing = validated.refined_listing
            style_approval = None
            for attempt in range(1, 5):  # up to 4 passes
                st.write(f"maxmr17 style pass {attempt}/4…")
                style_approval = maxmr17_approve(
                    card, current_listing, client,
                    market_context=market_context,
                )
                p2.advance()
                if style_approval.style_score >= 10:
                    break
                current_listing = style_approval.final_listing

            p2.complete()
            status.update(
                label=f"✅ Listing ready — maxmr17 style {style_approval.style_score}/10",
                state="complete",
            )

        st.session_state["result"] = validated
        st.session_state["style_approval"] = style_approval
        st.session_state["market_report"] = market_report
        st.session_state["price_verdict"] = price_verdict
        st.rerun()

    # ── Display persisted results ─────────────────────────────────────────────
    if "auth_report" in st.session_state:
        render_auth_panel(
            st.session_state["result_card"],
            st.session_state["auth_report"],
        )
        st.divider()

    if "result" in st.session_state:
        validated = st.session_state["result"]
        style_approval = st.session_state.get("style_approval")
        card = st.session_state["result_card"]

        final_listing = style_approval.final_listing if style_approval else validated.refined_listing

        score = validated.confidence_score
        style_score = style_approval.style_score if style_approval else None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Listing Confidence", f"{score}/10")
        m2.metric("Title Length", f"{len(final_listing.title)}/80 chars")
        strategy = final_listing.pricing.strategy.value.replace("_", " ").title()
        m3.metric("Selling Strategy", strategy)
        if style_score is not None:
            approved_label = "✅ Approved" if style_approval.approved else "✏️ Revised"
            m4.metric("maxmr17 Style", f"{style_score}/10 {approved_label}")

        st.divider()

        tab_final, tab_style, tab_draft, tab_market, tab_insights = st.tabs([
            "✅ Final Listing", "👤 maxmr17 Review", "📝 Draft", "📊 Market Data", "💡 Insights"
        ])

        with tab_final:
            render_listing(final_listing, "maxmr17-Approved Listing", is_refined=True)
            text_output = build_text_output_with_listing(final_listing, validated, card)
            filename = f"{card.player_name.lower().replace(' ', '_')}_{card.card_set[:4]}_listing.txt"
            st.download_button(
                label="⬇️ Download listing as .txt",
                data=text_output.encode("utf-8"),
                file_name=filename,
                mime="text/plain",
            )

        with tab_style:
            if style_approval:
                render_maxmr17_panel(style_approval)
                if not style_approval.approved:
                    st.info("maxmr17 revised the listing above. The **Final Listing** tab shows the corrected version.")
            else:
                st.info("maxmr17 style review not available for this listing.")

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
