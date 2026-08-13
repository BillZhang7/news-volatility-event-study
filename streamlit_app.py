"""
News Volatility Event Study — Streamlit Demo
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"

HYPOTHESIS_ID = "H-A3"
HYPOTHESIS_TEXT = (
    "Extreme negative news shocks increase post-event stock volatility "
    "more than normal days, without requiring a predictable return direction."
)
SENTIMENT_LOOKBACK_DAYS = 60
SENTIMENT_STD_THRESHOLD = 1.5
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10


def news_source_label() -> str:
    meta = ROOT / "data" / "fetch_meta.json"
    if meta.exists():
        data = json.loads(meta.read_text(encoding="utf-8"))
        return "Demo headlines" if data.get("news_source") == "demo" else "yfinance live"
    return "Pre-computed results (see GitHub pipeline)"


@st.cache_data
def load_outputs() -> dict:
    curve = pd.read_csv(OUTPUT / "vol_event_curve.csv")
    if "day" not in curve.columns:
        curve = curve.rename(columns={curve.columns[0]: "day"})
    curve["day"] = pd.to_numeric(curve["day"])

    events = pd.read_csv(OUTPUT / "events.csv", parse_dates=["event_date"])
    summary = json.loads((OUTPUT / "event_study_summary.json").read_text(encoding="utf-8"))
    return {"curve": curve, "events": events, "summary": summary}


try:
    st.set_page_config(page_title="News Vol Event Study", layout="wide")
    st.title("News Sentiment → Volatility Event Study")
    st.caption(f"Data: {news_source_label()} | Plan B events | H-A3 hypothesis")

    st.markdown(f"**Hypothesis ({HYPOTHESIS_ID}):** {HYPOTHESIS_TEXT}")
    st.markdown(
        f"**Event rule (Plan B):** daily sentiment below "
        f"{SENTIMENT_STD_THRESHOLD}σ of the stock's own {SENTIMENT_LOOKBACK_DAYS}-day history."
    )

    missing = [p.name for p in [
        OUTPUT / "vol_event_curve.csv",
        OUTPUT / "events.csv",
        OUTPUT / "event_study_summary.json",
    ] if not p.exists()]

    if missing:
        st.error(f"Missing files in outputs/: {', '.join(missing)}")
        st.stop()

    data = load_outputs()
    summary = data["summary"]["full_sample"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", summary["event_count"])
    c2.metric("Pre vol", f"{summary['pre_vol_avg']:.4f}")
    c3.metric("Post vol (+5d)", f"{summary['post_vol_5d_avg']:.4f}")
    ratio = summary.get("post_to_pre_ratio_5d")
    c4.metric("Post/Pre ratio", f"{ratio:.2f}x" if ratio else "N/A")

    st.subheader(f"Volatility around events [{-EVENT_WINDOW_PRE}, +{EVENT_WINDOW_POST}]")
    chart = data["curve"].set_index("day")
    st.line_chart(chart)

    png = OUTPUT / "vol_event_curve.png"
    if png.exists():
        st.image(str(png), caption="Average volatility path around events")

    st.subheader("Detected events")
    st.dataframe(
        data["events"].sort_values("event_date", ascending=False),
        width="stretch",
    )

    with st.expander("Research notes & limitations"):
        st.markdown(
            """
            - **Data:** yfinance headlines (limited history, no API key)
            - **Sentiment:** VADER / optional FinBERT locally
            - **Events:** stock-specific negative sentiment shocks (Plan B)
            - **H-A3:** tests volatility, not return direction
            - Past patterns do not guarantee future results
            """
        )

    st.markdown("---")
    st.markdown("[GitHub Repository](https://github.com/BillZhang7/news-volatility-event-study)")

except Exception as err:  # noqa: BLE001 - show full error in Streamlit Cloud logs/UI
    st.error("App crashed while loading. Details below:")
    st.exception(err)
