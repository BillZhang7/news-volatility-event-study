# News Volatility Event Study — Research Brief

## Research question (your decision)

**H-A3:** Do extreme negative news shocks increase post-event stock volatility?

We test **volatility**, not return direction — a modest, defensible hypothesis.

## Event definition (Plan B — your decision)

A (ticker, date) is an event when:

1. Daily average headline sentiment is **below** the stock's own 60-day mean by **1.5 standard deviations**
2. At least **5 news-days** exist in the lookback window

This captures **stock-specific sentiment shocks**, not just generic bad headlines.

## Event window (your decision)

Trading days **[-5, +10]** relative to the event date.

Primary read: compare realized volatility **before** (days -5..-1) vs **after** (+1..+5, +1..+10).

## Universe

15 liquid US large caps (tech + finance + consumer).

## Data limitations (state in interviews)

- yfinance news is free but **limited in history and coverage**
- FinBERT scores headlines only (not full article body)
- Not point-in-time institutional news feed
- Results are illustrative research, not tradable production alpha

## How to interpret results

- **Post/Pre vol ratio > 1** → supports H-A3 in this sample
- **Ratio ≈ 1** → no clear vol spike; still a valid finding
- Compare in-sample vs out-of-sample in `event_study_summary.json`
