from __future__ import annotations

from datetime import datetime, date
import math
import time
import random
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Put 年化收益率查询器", layout="wide")

# =========================
# Math
# =========================
def ann_by_strike(bid: float, strike: float, dte: int) -> float:
    """Cash-secured annualized return: (bid/strike) * 365/dte"""
    if dte <= 0 or strike <= 0 or not math.isfinite(bid) or bid <= 0:
        return float("nan")
    return (bid / strike) * (365.0 / dte)

def ann_by_margin(bid: float, strike: float, dte: int, margin_ratio: float) -> float:
    """
    Margin annualized return using capital = strike * margin_ratio
    annualized = (bid / (strike*margin_ratio)) * 365/dte
    """
    if margin_ratio <= 0:
        return float("nan")
    denom = strike * margin_ratio
    if dte <= 0 or denom <= 0 or not math.isfinite(bid) or bid <= 0:
        return float("nan")
    return (bid / denom) * (365.0 / dte)

def exp_to_dte(exp: str, today: date) -> int:
    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    return (exp_date - today).days


# =========================
# Robust yfinance helpers (anti rate-limit)
# =========================
def is_rate_limit_error(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e).lower()
    # yfinance has YFRateLimitError; sometimes wrapped
    return ("ratelimit" in name.lower()) or ("rate limit" in msg) or ("too many requests" in msg)

def with_retry(fn, *, retries: int = 3, base_sleep: float = 1.0):
    """
    Retry wrapper with exponential backoff + jitter.
    Only retries for rate-limit-like errors; for other errors, it still retries once or twice
    because yfinance can be flaky, but stops early if needed.
    """
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # backoff: 1,2,4 + jitter
            sleep_s = base_sleep * (2 ** i) + random.uniform(0, 0.3)
            time.sleep(sleep_s)
    raise last_err

@st.cache_data(ttl=1800)  # 30 minutes cache to reduce Yahoo hits on Streamlit Cloud
def fetch_expirations(ticker: str) -> List[str]:
    def _do():
        t = yf.Ticker(ticker)
        return t.options or []
    return with_retry(_do, retries=3, base_sleep=1.0)

@st.cache_data(ttl=600)  # 10 minutes cache for spot
def fetch_spot(ticker: str) -> Optional[float]:
    """
    Multi-fallback spot fetch:
    1) fast_info last_price
    2) info regularMarketPrice/currentPrice
    3) history close
    """
    def _do():
        t = yf.Ticker(ticker)

        spot = None
        # 1) fast_info
        try:
            if hasattr(t, "fast_info") and t.fast_info:
                spot = t.fast_info.get("last_price") or t.fast_info.get("lastPrice")
        except Exception:
            pass

        # 2) info
        if spot is None:
            try:
                info = t.info  # slower, but sometimes more stable
                spot = info.get("regularMarketPrice") or info.get("currentPrice")
            except Exception:
                pass

        # 3) history
        if spot is None:
            try:
                h = t.history(period="5d", interval="1d")
                if h is not None and not h.empty:
                    spot = float(h["Close"].dropna().iloc[-1])
            except Exception:
                pass

        # normalize
        try:
            spot = float(spot) if spot is not None else None
            if spot is not None and (not math.isfinite(spot) or spot <= 0):
                spot = None
        except Exception:
            spot = None

        return spot

    return with_retry(_do, retries=3, base_sleep=1.0)

@st.cache_data(ttl=1800)  # 30 minutes cache per (ticker, exp)
def fetch_put_chain(ticker: str, exp: str) -> pd.DataFrame:
    def _do():
        t = yf.Ticker(ticker)
        chain = t.option_chain(exp)
        puts = chain.puts.copy()
        puts["exp"] = exp
        return puts
    return with_retry(_do, retries=3, base_sleep=1.0)


# =========================
# UI
# =========================
st.title("PUT 期权年化收益率查询器（bid / K）")

with st.sidebar:
    st.subheader("查询条件")

    # Clear cache button
    if st.button("强制刷新数据（清缓存）"):
        st.cache_data.clear()
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_meta", None)
        st.rerun()

    # IMPORTANT: only collect inputs here, do NOT fetch yfinance here
    with st.form("filters"):
        ticker = st.text_input("股票代码 (Ticker)", value="QQQ").strip().upper()

        dte_min, dte_max = st.slider(
            "到期区间（DTE 天）",
            min_value=1,
            max_value=365,
            value=(7, 30),
            help="例如 7–30 表示只看 7~30 天内到期的合约"
        )

        margin_ratio = st.number_input(
            "margin_ratio（默认 1）",
            min_value=0.05,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="保证金占用比例：1=全额现金担保；0.3≈占用30%资金（回报放大但风险也更高）"
        )

        only_otm = st.checkbox("只看 OTM puts（推荐）", value=True)
        min_oi = st.number_input("最小 Open Interest（可选）", min_value=0, value=0, step=50)

        st.divider()
        strike_mode = st.radio(
            "行权价筛选方式",
            options=["按K区间筛选", "按OTM%区间筛选（基于 strike/spot）"],
            index=1
        )

        strike_min = st.number_input("K 下限（可选，0 表示不限制）", min_value=0.0, value=0.0, step=1.0)
        strike_max = st.number_input("K 上限（可选，0 表示不限制）", min_value=0.0, value=0.0, step=1.0)

        otm_low, otm_high = st.slider(
            "OTM% 区间（PUT 通常为负数）",
            min_value=-80.0,
            max_value=20.0,
            value=(-20.0, -10.0),
            step=0.5,
            help="moneyness_pct=(K/spot-1)*100。-20%~-10% 表示 K 低于现价 10%~20%。"
        )

        max_exps_to_scan = st.number_input(
            "最多扫描到期日数量（防限流，建议 3~10）",
            min_value=1,
            max_value=30,
            value=8,
            step=1
        )

        top_n = st.number_input("显示 Top N（按 margin 年化排序）", min_value=10, max_value=500, value=100, step=10)

        submitted = st.form_submit_button("查询/刷新")

if not submitted and "last_result" not in st.session_state:
    st.info("在左侧设置条件后，点“查询/刷新”。")
    st.stop()

# =========================
# Run query (only on submit)
# =========================
if submitted:
    today = date.today()

    try:
        # fetch expirations (cached)
        exps = fetch_expirations(ticker)
    except Exception as e:
        if is_rate_limit_error(e):
            st.error("数据源被限流（yfinance/Yahoo）。请等待 1~5 分钟后再试，或点左侧“强制刷新数据（清缓存）”。")
        else:
            st.error(f"获取到期日失败：{type(e).__name__}")
        st.stop()

    if not exps:
        st.warning("未获取到到期日列表（可能数据源暂时不可用）。稍后重试。")
        st.stop()

    # compute DTE and filter by range
    exp_candidates: List[Tuple[str, int]] = []
    for exp in exps:
        dte = exp_to_dte(exp, today)
        if dte_min <= dte <= dte_max:
            exp_candidates.append((exp, dte))

    if not exp_candidates:
        st.warning("没有找到落在该 DTE 区间内的到期日。请放宽 DTE 范围。")
        st.stop()

    # limit number of expirations scanned to reduce rate-limit risk
    exp_candidates.sort(key=lambda x: x[1])  # nearest first
    exp_candidates = exp_candidates[: int(max_exps_to_scan)]

    # fetch spot (cached + fallback)
    spot = None
    try:
        spot = fetch_spot(ticker)
    except Exception as e:
        # spot failure should not kill the whole app unless user insists on OTM%
        spot = None

    # if user chose OTM% mode but spot missing -> soft fail with guidance
    if strike_mode.startswith("按OTM") and not spot:
        st.warning("当前无法获取 spot（可能限流/休市/数据源波动）。已建议：切换为“按K区间筛选”，或稍后重试。")

    # fetch option chains
    frames = []
    errors = 0
    with st.spinner("抓取期权链并计算中（云端可能需要几秒）..."):
        for exp, dte in exp_candidates:
            try:
                puts = fetch_put_chain(ticker, exp)
            except Exception as e:
                errors += 1
                # If rate limited mid-way, stop early to avoid hammering
                if is_rate_limit_error(e):
                    break
                continue

            if puts is None or puts.empty:
                continue

            df = puts.copy()
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
            df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
            df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
            df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
            df["inTheMoney"] = df["inTheMoney"].astype(bool)
            df["dte"] = dte
            df["spot"] = spot

            # moneyness
            if spot and math.isfinite(float(spot)):
                df["moneyness_pct"] = (df["strike"] / float(spot) - 1.0) * 100
            else:
                df["moneyness_pct"] = float("nan")

            # base filters
            df = df[df["bid"].fillna(0) > 0]
            if only_otm:
                df = df[df["inTheMoney"] == False]
            if min_oi > 0:
                df = df[df["openInterest"].fillna(0) >= min_oi]

            # strike filters
            if strike_mode == "按K区间筛选":
                if strike_min > 0:
                    df = df[df["strike"] >= strike_min]
                if strike_max > 0:
                    df = df[df["strike"] <= strike_max]
            else:
                # OTM% requires spot; if missing, skip this mode’s filter (but keep data so user sees something)
                if spot and math.isfinite(float(spot)):
                    df = df[df["moneyness_pct"].notna()]
                    df = df[(df["moneyness_pct"] >= otm_low) & (df["moneyness_pct"] <= otm_high)]

            if df.empty:
                continue

            # annualized
            df["ann_return_pct"] = df.apply(
                lambda r: ann_by_strike(float(r["bid"]), float(r["strike"]), int(r["dte"])) * 100, axis=1
            )
            df["ann_return_margin_pct"] = df.apply(
                lambda r: ann_by_margin(float(r["bid"]), float(r["strike"]), int(r["dte"]), float(margin_ratio)) * 100, axis=1
            )

            frames.append(df)

    if not frames:
        if errors > 0:
            st.error("期权链抓取失败（可能触发限流）。请等待 1~5 分钟后重试，或减少扫描到期日数量。")
        else:
            st.warning("没有符合筛选条件的合约。请放宽 DTE / OTM / K / OI 条件。")
        st.stop()

    result = pd.concat(frames, ignore_index=True)

    # sort
    result = result.sort_values(
        ["ann_return_margin_pct", "ann_return_pct", "dte"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # store result in session_state so the page remains viewable without re-fetching
    st.session_state["last_result"] = result
    st.session_state["last_meta"] = {
        "ticker": ticker,
        "spot": spot,
        "dte_min": dte_min,
        "dte_max": dte_max,
        "margin_ratio": float(margin_ratio),
        "strike_mode": strike_mode,
        "otm_low": float(otm_low),
        "otm_high": float(otm_high),
        "strike_min": float(strike_min),
        "strike_max": float(strike_max),
        "scanned_exps": [x[0] for x in exp_candidates],
    }

# =========================
# Display (use cached session result)
# =========================
result = st.session_state.get("last_result")
meta = st.session_state.get("last_meta", {})

if result is None or result.empty:
    st.warning("暂无结果。请在左侧点“查询/刷新”。")
    st.stop()

# Summary cards
spot_show = meta.get("spot", None)
dte_min = meta.get("dte_min")
dte_max = meta.get("dte_max")
margin_ratio = meta.get("margin_ratio", 1.0)

top_margin = float(result["ann_return_margin_pct"].iloc[0]) if len(result) else float("nan")
top_cash = float(result["ann_return_pct"].iloc[0]) if len(result) else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot_show:.2f}" if (spot_show and math.isfinite(float(spot_show))) else "N/A")
c2.metric("DTE 区间", f"{dte_min}–{dte_max} 天")
c3.metric("合约数量", f"{len(result)}")
c4.metric("Top 年化（Margin）", f"{top_margin:.2f}%")

st.caption(f"Top 年化（Cash-secured, bid/K）：{top_cash:.2f}% ｜ margin_ratio={margin_ratio:.2f}")

# Show filters used
with st.expander("本次查询参数（便于复查）", expanded=False):
    st.write(meta)

# Table
st.subheader(f"{meta.get('ticker','')} PUT 年化收益率结果（按 Margin 年化排序）")

show_cols = [
    "contractSymbol", "exp", "dte",
    "strike", "bid", "ask",
    "ann_return_pct", "ann_return_margin_pct",
    "moneyness_pct", "openInterest", "inTheMoney", "spot"
]

top_n = int(meta.get("top_n", 100)) if "top_n" in meta else 100
view = result[show_cols].head(top_n)

st.dataframe(view, use_container_width=True, height=580)

csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "下载 CSV（当前显示结果）",
    data=csv_bytes,
    file_name=f"{meta.get('ticker','TICKER')}_puts_DTE{dte_min}-{dte_max}_annualized.csv",
    mime="text/csv"
)
