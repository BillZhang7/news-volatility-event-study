"""
Project configuration — documents YOUR research design decisions.
"""

from __future__ import annotations

# --- Research design (Plan B + H-A3) ---

# H-A3: extreme news raises volatility (not necessarily predictable returns)
HYPOTHESIS_ID = "H-A3"
HYPOTHESIS_TEXT = (
    "Extreme negative news shocks increase post-event stock volatility "
    "more than normal days, without requiring a predictable return direction."
)

# Focused professional universe (liquid US large caps, news-rich)
UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "GS",
    "BAC",
    "V",
    "MA",
    "UNH",
    "XOM",
    "WMT",
]

BENCHMARK = "SPY"
PRICE_START = "2021-01-01"
PRICE_END = "2024-12-31"

# Plan B: stock-specific negative sentiment surprise
SENTIMENT_LOOKBACK_DAYS = 60
SENTIMENT_STD_THRESHOLD = 1.5
MIN_NEWS_DAYS_IN_LOOKBACK = 5

# Event study windows (trading days relative to event)
EVENT_WINDOW_PRE = 5
EVENT_WINDOW_POST = 10

# Out-of-sample split for robustness
IN_SAMPLE_END = "2022-12-31"
OUT_OF_SAMPLE_START = "2023-01-01"

# Realized vol = rolling std of daily returns over N days
REALIZED_VOL_WINDOW = 5

FINBERT_MODEL = "ProsusAI/finbert"
