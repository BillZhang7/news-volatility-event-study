"""
Step 3: Volatility event study (H-A3).

Tests whether post-event realized volatility rises vs pre-event baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BENCHMARK,
    EVENT_WINDOW_POST,
    EVENT_WINDOW_PRE,
    IN_SAMPLE_END,
    OUT_OF_SAMPLE_START,
    REALIZED_VOL_WINDOW,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
    return prices


def align_event_to_trading_day(event_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Use same day if listed, else next available trading day."""
    event_date = pd.Timestamp(event_date).normalize()
    if event_date in trading_index:
        return event_date
    future = trading_index[trading_index >= event_date]
    if len(future) == 0:
        return None
    return future[0]


def realized_vol_series(returns: pd.Series, window: int = REALIZED_VOL_WINDOW) -> pd.Series:
    return returns.rolling(window).std()


def extract_window_series(
    series: pd.Series,
    center_date: pd.Timestamp,
    pre: int,
    post: int,
    index: pd.DatetimeIndex,
) -> pd.Series | None:
    if center_date not in index:
        return None
    loc = index.get_loc(center_date)
    start = loc - pre
    end = loc + post
    if start < 0 or end >= len(index):
        return None
    window_index = index[start : end + 1]
    rel_days = range(-pre, post + 1)
    values = series.loc[window_index].values
    return pd.Series(values, index=rel_days)


def run_event_study(events: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bench_col = BENCHMARK if BENCHMARK in prices.columns else None
    results: list[pd.Series] = []
    bench_results: list[pd.Series] = []

    for _, ev in events.iterrows():
        ticker = ev["ticker"]
        if ticker not in prices.columns:
            continue
        trading_index = prices.index
        center = align_event_to_trading_day(ev["event_date"], trading_index)
        if center is None:
            continue

        ret = prices[ticker].pct_change()
        vol = realized_vol_series(ret)

        w = extract_window_series(
            vol, center, EVENT_WINDOW_PRE, EVENT_WINDOW_POST, trading_index
        )
        if w is None or w.isna().all():
            continue
        w.name = f"{ticker}_{center.date()}"
        results.append(w)

        if bench_col:
            bret = prices[bench_col].pct_change()
            bvol = realized_vol_series(bret)
            bw = extract_window_series(
                bvol, center, EVENT_WINDOW_PRE, EVENT_WINDOW_POST, trading_index
            )
            if bw is not None:
                bench_results.append(bw)

    if not results:
        raise ValueError("No valid event windows — check data overlap.")

    panel = pd.DataFrame(results).T
    avg_vol = panel.mean(axis=1)
    avg_bench = pd.DataFrame(bench_results).T.mean(axis=1) if bench_results else None

    pre = avg_vol.loc[-EVENT_WINDOW_PRE : -1].mean()
    post5 = avg_vol.loc[1:5].mean()
    post10 = avg_vol.loc[1:EVENT_WINDOW_POST].mean()

    summary = {
        "event_count": int(panel.shape[1]),
        "pre_vol_avg": float(pre),
        "post_vol_5d_avg": float(post5),
        "post_vol_10d_avg": float(post10),
        "post_to_pre_ratio_5d": float(post5 / pre) if pre > 0 else None,
        "post_to_pre_ratio_10d": float(post10 / pre) if pre > 0 else None,
        "supports_H_A3": bool(post5 > pre),
    }

    curve = pd.DataFrame({"stock_vol": avg_vol})
    if avg_bench is not None:
        curve["benchmark_vol"] = avg_bench
        curve["abnormal_vol"] = avg_vol - avg_bench

    return curve, summary


def split_summary(events: pd.DataFrame, summary_fn) -> dict:
    if events.empty:
        return {}
    ins = events[events["event_date"] <= pd.Timestamp(IN_SAMPLE_END)]
    oos = events[events["event_date"] >= pd.Timestamp(OUT_OF_SAMPLE_START)]
    out: dict = {}
    if len(ins) >= 3:
        _, s = summary_fn(ins)
        out["in_sample"] = s
    if len(oos) >= 3:
        _, s = summary_fn(oos)
        out["out_of_sample"] = s
    return out


def plot_vol_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve.index, curve["stock_vol"], label="Avg stock realized vol", linewidth=2)
    if "benchmark_vol" in curve.columns:
        ax.plot(curve.index, curve["benchmark_vol"], label=f"{BENCHMARK} vol", linewidth=2)
    if "abnormal_vol" in curve.columns:
        ax.plot(curve.index, curve["abnormal_vol"], label="Stock - SPY vol", linestyle="--")
    ax.axvline(0, color="red", linestyle=":", alpha=0.7, label="Event day")
    ax.set_xlabel("Trading days relative to event")
    ax.set_ylabel(f"Realized vol ({REALIZED_VOL_WINDOW}d rolling std)")
    ax.set_title("Volatility Event Study (H-A3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 50)
    print("Step 3: Volatility event study")
    print("=" * 50)

    events = pd.read_csv(OUTPUT_DIR / "events.csv", parse_dates=["event_date"])
    prices = load_prices()

    print(f"Events: {len(events)}")
    curve, summary = run_event_study(events, prices)

    def _run(sub: pd.DataFrame):
        return run_event_study(sub, prices)

    period_summaries = split_summary(events, _run)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUTPUT_DIR / "vol_event_curve.csv")
    plot_vol_curve(curve, OUTPUT_DIR / "vol_event_curve.png")

    payload = {"full_sample": summary, "by_period": period_summaries}
    (OUTPUT_DIR / "event_study_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("\nFull sample:")
    print(f"  Events used     : {summary['event_count']}")
    print(f"  Pre-event vol   : {summary['pre_vol_avg']:.4f}")
    print(f"  Post (+1..+5)   : {summary['post_vol_5d_avg']:.4f}")
    print(f"  Post (+1..+10)  : {summary['post_vol_10d_avg']:.4f}")
    print(f"  Post/Pre (5d)   : {summary['post_to_pre_ratio_5d']:.2f}x")
    print(f"  Supports H-A3?  : {summary['supports_H_A3']}")

    for name, s in period_summaries.items():
        print(f"\n{name}:")
        print(f"  Events: {s['event_count']}, Post/Pre 5d: {s['post_to_pre_ratio_5d']:.2f}x")

    print(f"\nSaved: {OUTPUT_DIR / 'vol_event_curve.png'}")
    print("Step 3 complete.")


if __name__ == "__main__":
    main()
