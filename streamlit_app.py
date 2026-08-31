"""
Mobile-friendly web UI for calscreener.py, built with Streamlit.

Run locally with:
    streamlit run streamlit_app.py

Or deploy for free on Streamlit Community Cloud (share.streamlit.io) pointed
at this file to get a URL you can open from your phone. See README.md.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from calscreener import (
    BACK_TARGET_DTE,
    DTE_TOLERANCE,
    FF_THRESHOLD,
    FRONT_TARGET_DTE,
    RISK_FREE_RATE,
    TARGET_DELTA,
    TOP_N_DEFAULT,
    get_sp500_tickers,
    rank_by_liquidity,
    rank_results,
    screen_ticker,
)

st.set_page_config(page_title="CalScreener", page_icon="📈", layout="wide")

st.title("📈 CalScreener")
st.caption(
    "Screens liquid S&P 500 names for options calendar spreads sitting in "
    "IV term-structure backwardation. Quotes are Yahoo Finance's last "
    "published values, not a live feed."
)

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("How many liquid names to scan", 5, 100, TOP_N_DEFAULT, step=5)
    front_dte = st.number_input("Front target DTE", 7, 365, FRONT_TARGET_DTE)
    back_dte = st.number_input("Back target DTE", 7, 365, BACK_TARGET_DTE)
    tolerance = st.number_input("DTE tolerance (+/-)", 1, 60, DTE_TOLERANCE)
    ff_threshold_pct = st.slider("Forward factor priority cutoff (%)", 0, 100,
                                  int(FF_THRESHOLD * 100))
    delta_target = st.slider("Double calendar target |delta|", 0.05, 0.50,
                              TARGET_DELTA, step=0.05)
    rate = st.number_input("Risk-free rate", 0.0, 0.20, RISK_FREE_RATE, step=0.005,
                            format="%.3f")
    custom_tickers = st.text_input(
        "Override tickers (comma-separated, optional)",
        placeholder="e.g. AAPL, MSFT, NVDA",
        help="Skips the liquidity ranking step and scans just these names.",
    )
    max_rows = st.number_input("Max rows to show", 5, 500, 50)
    run_button = st.button("Run screen", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_button:
    if custom_tickers.strip():
        universe = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        with st.spinner("Fetching S&P 500 constituent list..."):
            tickers = get_sp500_tickers()
        with st.spinner(f"Ranking {len(tickers)} names by liquidity..."):
            universe = rank_by_liquidity(tickers, top_n)

    st.write(f"Scanning **{len(universe)}** tickers: {', '.join(universe)}")

    progress = st.progress(0.0)
    status = st.empty()
    all_rows: list[dict] = []
    for i, ticker in enumerate(universe, 1):
        status.text(f"({i}/{len(universe)}) {ticker}...")
        rows = screen_ticker(ticker, front_dte, back_dte, tolerance, rate, delta_target)
        all_rows.extend(rows)
        progress.progress(i / len(universe))
        time.sleep(0.2)  # politeness delay for Yahoo
    status.empty()
    progress.empty()

    if not all_rows:
        st.warning("No candidates found.")
        st.session_state["results"] = None
    else:
        df = pd.DataFrame(all_rows)
        ranked = rank_results(df, front_dte, back_dte, ff_threshold_pct / 100)
        if ranked.empty:
            st.warning("No candidates with usable IV data.")
            st.session_state["results"] = None
        else:
            st.session_state["results"] = ranked

ranked = st.session_state["results"]
if ranked is not None:
    display_cols = [
        "ticker", "strategy", "spot", "forward_factor", "meets_ff_threshold",
        "negative_forward_variance", "forward_iv",
        "front_expiry", "front_dte", "front_strike", "front_iv", "front_oi",
        "back_expiry", "back_dte", "back_strike", "back_iv", "back_oi",
        "total_oi", "stale_quote",
    ]
    out = ranked[display_cols].head(max_rows)
    st.success(f"{len(ranked)} candidate(s) found, showing top {len(out)}.")
    if out["negative_forward_variance"].any():
        st.warning(
            "Rows flagged `negative_forward_variance` imply a forward vol so "
            "extreme the term-structure equation has no real solution -- the "
            "most severe form of backwardation. They're shown but excluded "
            "from the forward-factor ranking; review them manually."
        )
    st.dataframe(
        out.style.format({
            "forward_factor": "{:.1%}",
            "forward_iv": "{:.1%}",
            "front_iv": "{:.1%}",
            "back_iv": "{:.1%}",
            "spot": "{:.2f}",
        }, na_rep="n/a"),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download full results as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="calscreener_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("Set your parameters in the sidebar and tap **Run screen** to start. "
            "A full 50-name scan can take a few minutes.")
