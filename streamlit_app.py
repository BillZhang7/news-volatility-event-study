"""
News Volatility Event Study — Streamlit Demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    EVENT_WINDOW_POST,
    EVENT_WINDOW_PRE,
    HYPOTHESIS_TEXT,
    HYPOTHESIS_ID,
    SENTIMENT_LOOKBACK_DAYS,
    SENTIMENT_STD_THRESHOLD,
    UNIVERSE,
)

OUTPUT = ROOT / "outputs"
DATA = ROOT / "data"
META = DATA / "fetch_meta.json"


def news_source_label() -> str:
    if META.exists():
        import json

        meta = json.loads(META.read_text())
        return "Demo headlines" if meta.get("news_source") == "demo" else "yfinance live"
    return "Unknown"


@st.cache_data
def load_outputs() -> dict:
    curve = pd.read_csv(OUTPUT / "vol_event_curve.csv", index_col=0)
    events = pd.read_csv(OUTPUT / "events.csv", parse_dates=["event_date"])
    summary = json.loads((OUTPUT / "event_study_summary.json").read_text())
    daily = pd.read_csv(DATA / "daily_sentiment.csv", parse_dates=["date"])
    return {"curve": curve, "events": events, "summary": summary, "daily": daily}


def main() -> None:
    st.set_page_config(page_title="News Vol Event Study", layout="wide")
    st.title("News Sentiment → Volatility Event Study")
    st.caption(f"Data: {news_source_label()} | Plan B events | H-A3 hypothesis")

    st.markdown(f"**Hypothesis ({HYPOTHESIS_ID}):** {HYPOTHESIS_TEXT}")
    st.markdown(
        f"**Event rule (Plan B):** daily sentiment below "
        f"{SENTIMENT_STD_THRESHOLD}σ of the stock's own {SENTIMENT_LOOKBACK_DAYS}-day history."
    )

    if not (OUTPUT / "vol_event_curve.csv").exists():
        st.error("Run the pipeline first: fetch_data → sentiment → events → event_study")
        st.code(
            "python3 src/fetch_data.py\n"
            "python3 src/sentiment.py\n"
            "python3 src/events.py\n"
            "python3 src/event_study.py"
        )
        return

    data = load_outputs()
    summary = data["summary"]["full_sample"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", summary["event_count"])
    c2.metric("Pre vol", f"{summary['pre_vol_avg']:.4f}")
    c3.metric("Post vol (+5d)", f"{summary['post_vol_5d_avg']:.4f}")
    ratio = summary.get("post_to_pre_ratio_5d")
    c4.metric("Post/Pre ratio", f"{ratio:.2f}x" if ratio else "N/A")

    st.subheader(f"Volatility around events [{-EVENT_WINDOW_PRE}, +{EVENT_WINDOW_POST}]")
    st.line_chart(data["curve"])

    st.subheader("Detected events")
    st.dataframe(
        data["events"].sort_values("event_date", ascending=False),
        use_container_width=True,
    )

    with st.expander("Research notes & limitations"):
        st.markdown(
            """
            - **Data:** yfinance headlines (limited history, no API key)
            - **Sentiment:** FinBERT with VADER fallback
            - **Value proxy events:** stock-specific negative sentiment shocks
            - **H-A3:** tests volatility, not return direction
            - Past patterns do not guarantee future results
            """
        )


if __name__ == "__main__":
    main()
