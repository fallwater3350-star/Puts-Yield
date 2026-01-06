from __future__ import annotations

from datetime import datetime, date
import math
import time
import random
import re
from typing import List, Tuple, Optional, Dict

import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Put 年化收益率查询器", layout="wide")


# =========================
# Math
# =========================
def ann_by_strike(bid: float, strike: float, dte: int) -> float:
    """Cash-secured annualized: (bid/strike)*365/dte"""
    if dte <= 0 or strike <= 0 or not math.isfinite(bid) or bid <= 0:
        return float("nan")
    return (bid / strike) * (365.0 / dte)

def calc_capital(strike: float, margin_ratio: float, ask: float, spot: float) -> float:
    """
    Your estimated capital formula:
      capital = K*margin_ratio + ask - 0.75*(spot - K)
    """
    if strike <= 0 or margin_ratio <= 0:
        return float("nan")
    if not (math.isfinite(spot) and spot > 0):
        return float("nan")
    if not (math.isfinite(ask) and ask >= 0):
        return float("nan")

    capital = (strike * margin_ratio) + ask - 0.75 * (spot - strike)
    if (not math.isfinite(capital)) or capital <= 0:
        return float("nan")
    return capital

def ann_by_margin(bid: float, dte: int, capital: float) -> float:
    """Margin annualized: (bid/capital)*365/dte"""
    if dte <= 0 or not math.isfinite(bid) or bid <= 0:
        return float("nan")
    if not math.isfinite(capital) or capital <= 0:
        return float("nan")
    return (bid / capital) * (365.0 / dte)

def exp_to_dte(exp: str, today: date) -> int:
    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    return (exp_date - today).days


# =========================
# OCC contractSymbol -> strike parser (fixes yfinance strike weirdness)
# Example: NVDA260116P01820000 -> 182.0
#          TQQQ260116P00155000 -> 155.0
# =========================
_OCC_RE = re.compile(r"^[A-Z\.]{1,10}\d{6}[CP](\d{8})$")

def parse_strike_from_contract_symbol(sym: str) -> float:
    if not isinstance(sym, str):
        return float("nan")
    m = _OCC_RE.match(sym.strip().upper())
    if not m:
        return float("nan")
    strike_int = int(m.group(1))
    # OCC uses 3 decimals fixed-point (x1000) in most feeds
    return strike_int / 1000.0


# =========================
# Robust yfinance helpers (anti rate-limit / flaky)
# =========================
def is_rate_limit_error(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return ("ratelimit" in name) or ("rate limit" in msg) or ("too many requests" in msg)

def with_retry(fn, *, retries: int = 3, base_sleep: float = 1.0):
    last_err = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** i) + random.uniform(0, 0.35))
    raise last_err

@st.cache_data(ttl=3600)  # 1 hour
def fetch_expirations(ticker: str) -> List[str]:
    def _do():
        t = yf.Ticker(ticker)
        return t.options or []
    return with_retry(_do, retries=3, base_sleep=1.0)

@st.cache_data(ttl=1200)  # 20 min
def fetch_spot(ticker: str) -> Optional[float]:
    """
    Multi-fallback spot:
    1) fast_info last_price
    2) info regularMarketPrice/currentPrice
    3) history close
    """
    def _do():
        t = yf.Ticker(ticker)

        spot = None
        try:
            if hasattr(t, "fast_info") and t.fast_info:
                spot = t.fast_info.get("last_price") or t.fast_info.get("lastPrice")
        except Exception:
            pass

        if spot is None:
            try:
                info = t.info
                spot = info.get("regularMarketPrice") or info.get("currentPrice")
            except Exception:
                pass

        if spot is None:
            try:
                h = t.history(period="5d", interval="1d")
                if h is not None and not h.empty:
                    spot = float(h["Close"].dropna().iloc[-1])
            except Exception:
                pass

        try:
            spot = float(spot) if spot is not None else None
            if spot is not None and (not math.isfinite(spot) or spot <= 0):
                spot = None
        except Exception:
            spot = None

        return spot

    return with_retry(_do, retries=3, base_sleep=1.0)

@st.cache_data(ttl=3600)  # 1 hour per (ticker, exp)
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

    if st.button("强制刷新数据（清缓存）"):
        st.cache_data.clear()
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_meta", None)
        st.session_state.pop("last_diag", None)
        st.rerun()

    with st.form("filters"):
        ticker = st.text_input("股票代码 (Ticker)", value="NVDA").strip().upper()

        dte_min, dte_max = st.slider(
            "到期区间（DTE 天）",
            min_value=1,
            max_value=365,
            value=(7, 30)
        )

        margin_ratio = st.number_input(
            "margin_ratio（默认 1）",
            min_value=0.05,
            max_value=1.0,
            value=0.30,
            step=0.05
        )

        only_otm = st.checkbox("只看 OTM puts（推荐）", value=True)
        min_oi = st.number_input("最小 Open Interest（可选）", min_value=0, value=0, step=50)

        st.divider()
        strike_mode = st.radio(
            "行权价筛选方式",
            options=["按K区间筛选", "按OTM%区间筛选（基于 strike/spot）"],
            index=0
        )
        strike_min = st.number_input("K 下限（可选，0 表示不限制）", min_value=0.0, value=0.0, step=1.0)
        strike_max = st.number_input("K 上限（可选，0 表示不限制）", min_value=0.0, value=0.0, step=1.0)

        otm_low, otm_high = st.slider(
            "OTM% 区间（PUT 通常为负数）",
            min_value=-80.0,
            max_value=20.0,
            value=(-25.0, -5.0),
            step=0.5
        )

        # anti-limit knobs
        max_exps_to_scan = st.number_input(
            "最多扫描到期日数量（防限流，建议 3~10）",
            min_value=1, max_value=30, value=10, step=1
        )

        # NEW: data-quality guard
        min_rows_per_exp = st.number_input(
            "每个到期日最少返回行数（少于则视为残缺链并跳过）",
            min_value=10, max_value=500, value=80, step=10
        )

        top_n = st.number_input("显示 Top N（按 Margin 年化排序）", min_value=10, max_value=500, value=120, step=10)

        submitted = st.form_submit_button("查询/刷新")

if not submitted and "last_result" not in st.session_state:
    st.info("在左侧设置条件后，点“查询/刷新”。")
    st.stop()


# =========================
# Run query (only on submit)
# =========================
if submitted:
    today = date.today()

    # 1) expirations
    try:
        exps = fetch_expirations(ticker)
    except Exception as e:
        if is_rate_limit_error(e):
            st.error("数据源被限流（yfinance/Yahoo）。请等待 1~5 分钟后再试，或点左侧“强制刷新数据（清缓存）”。")
        else:
            st.error(f"获取到期日失败：{type(e).__name__}：{e}")
        st.stop()

    if not exps:
        st.warning("未获取到到期日列表（可能数据源暂时不可用）。稍后重试。")
        st.stop()

    # 2) DTE filter
    exp_candidates: List[Tuple[str, int]] = []
    for exp in exps:
        dte = exp_to_dte(exp, today)
        if dte_min <= dte <= dte_max:
            exp_candidates.append((exp, dte))

    if not exp_candidates:
        st.warning("没有找到落在该 DTE 区间内的到期日。请放宽 DTE 范围。")
        st.stop()

    # scan nearest first; limit for rate-limit protection
    exp_candidates.sort(key=lambda x: x[1])
    exp_candidates = exp_candidates[: int(max_exps_to_scan)]

    # 3) spot
    try:
        spot = fetch_spot(ticker)
    except Exception:
        spot = None

    if strike_mode.startswith("按OTM") and not spot:
        st.warning("当前无法获取 spot（可能限流/休市/数据源波动）。OTM% 筛选将无法生效，建议切换为“按K区间筛选”。")

    # 4) fetch chains with data-quality checks
    frames = []
    diag_rows: List[Dict] = []
    errors = 0
    skipped_incomplete = 0
    rate_limited_midway = False

    with st.spinner("抓取期权链并计算中（云端可能需要几秒）..."):
        for exp, dte in exp_candidates:
            try:
                puts = fetch_put_chain(ticker, exp)
            except Exception as e:
                errors += 1
                note = "fetch_error"
                if is_rate_limit_error(e):
                    note = "rate_limited_stop"
                    rate_limited_midway = True
                    diag_rows.append({"exp": exp, "dte": dte, "rows": 0, "minK": None, "maxK": None, "note": note})
                    break
                diag_rows.append({"exp": exp, "dte": dte, "rows": 0, "minK": None, "maxK": None, "note": note})
                continue

            if puts is None or puts.empty:
                diag_rows.append({"exp": exp, "dte": dte, "rows": 0, "minK": None, "maxK": None, "note": "empty"})
                continue

            rows = int(len(puts))

            # If too few rows, treat as incomplete chain (common yfinance cloud issue)
            if rows < int(min_rows_per_exp):
                skipped_incomplete += 1
                # still record diagnostic min/max from raw if possible
                try:
                    minK_raw = float(pd.to_numeric(puts["strike"], errors="coerce").min())
                    maxK_raw = float(pd.to_numeric(puts["strike"], errors="coerce").max())
                except Exception:
                    minK_raw, maxK_raw = None, None
                diag_rows.append({"exp": exp, "dte": dte, "rows": rows, "minK": minK_raw, "maxK": maxK_raw, "note": "too_few_rows_skip"})
                continue

            df = puts.copy()

            # normalize columns
            df["strike_raw"] = pd.to_numeric(df.get("strike"), errors="coerce")
            df["bid"] = pd.to_numeric(df.get("bid"), errors="coerce")
            df["ask"] = pd.to_numeric(df.get("ask"), errors="coerce")
            df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce")
            df["inTheMoney"] = df.get("inTheMoney").astype(bool)
            df["dte"] = dte
            df["spot"] = spot

            # ====== STRIKE FIX: parse from contractSymbol and override ======
            df["strike_parsed"] = df["contractSymbol"].apply(parse_strike_from_contract_symbol)
            mask = df["strike_parsed"].notna() & (df["strike_parsed"] > 0)
            df.loc[mask, "strike"] = df.loc[mask, "strike_parsed"]
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce")

            # ask fallback (your capital uses ask)
            df["ask_filled"] = df["ask"]
            df.loc[df["ask_filled"].isna(), "ask_filled"] = df.loc[df["ask_filled"].isna(), "bid"]

            # moneyness
            if spot and math.isfinite(float(spot)):
                df["moneyness_pct"] = (df["strike"] / float(spot) - 1.0) * 100
            else:
                df["moneyness_pct"] = float("nan")

            # Diagnostic for this exp (after strike fix)
            try:
                minK = float(df["strike"].min())
                maxK = float(df["strike"].max())
            except Exception:
                minK, maxK = None, None
            diag_rows.append({"exp": exp, "dte": dte, "rows": rows, "minK": minK, "maxK": maxK, "note": "ok"})

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
                if spot and math.isfinite(float(spot)):
                    df = df[df["moneyness_pct"].notna()]
                    df = df[(df["moneyness_pct"] >= otm_low) & (df["moneyness_pct"] <= otm_high)]
                # if no spot, skip OTM filter to avoid empty results

            if df.empty:
                continue

            # annualized
            df["ann_return_pct"] = df.apply(
                lambda r: ann_by_strike(float(r["bid"]), float(r["strike"]), int(r["dte"])) * 100,
                axis=1
            )

            df["capital"] = df.apply(
                lambda r: calc_capital(
                    strike=float(r["strike"]),
                    margin_ratio=float(margin_ratio),
                    ask=float(r["ask_filled"]) if pd.notna(r["ask_filled"]) else float("nan"),
                    spot=float(r["spot"]) if pd.notna(r["spot"]) else float("nan"),
                ),
                axis=1
            )

            df["ann_return_margin_pct"] = df.apply(
                lambda r: ann_by_margin(
                    bid=float(r["bid"]),
                    dte=int(r["dte"]),
                    capital=float(r["capital"]) if pd.notna(r["capital"]) else float("nan"),
                ) * 100,
                axis=1
            )

            frames.append(df)

    diag_df = pd.DataFrame(diag_rows)

    if not frames:
        # If we skipped most exps due to incomplete data, tell user explicitly
        if skipped_incomplete > 0:
            st.error(
                "本次未得到可用结果：多个到期日返回数据行数过少（疑似 yfinance/Yahoo 返回残缺链）。"
                "请点“强制刷新数据（清缓存）”，或稍后重试；也可降低每个到期日最少返回行数阈值。"
            )
        elif rate_limited_midway or errors > 0:
            st.error("期权链抓取失败（可能触发限流/数据源异常）。请等待 1~5 分钟后重试，或减少扫描到期日数量。")
        else:
            st.warning("没有符合筛选条件的合约。请放宽 DTE / OTM / K / OI 条件。")

        st.session_state["last_diag"] = diag_df
        st.stop()

    result = pd.concat(frames, ignore_index=True)

    # sort
    result = result.sort_values(
        ["ann_return_margin_pct", "ann_return_pct", "dte"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    # persist
    st.session_state["last_result"] = result
    st.session_state["last_diag"] = diag_df
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
        "max_exps_to_scan": int(max_exps_to_scan),
        "min_rows_per_exp": int(min_rows_per_exp),
        "only_otm": bool(only_otm),
        "min_oi": int(min_oi),
        "top_n": int(top_n),
        "rate_limited_midway": bool(rate_limited_midway),
        "skipped_incomplete": int(skipped_incomplete),
        "errors": int(errors),
        "scanned_exps": [x[0] for x in exp_candidates],
    }


# =========================
# Display (from session)
# =========================
result = st.session_state.get("last_result")
meta = st.session_state.get("last_meta", {})
diag_df = st.session_state.get("last_diag")

if result is None or result.empty:
    st.warning("暂无结果。请在左侧点“查询/刷新”。")
    if diag_df is not None:
        with st.expander("诊断信息（数据源是否残缺/限流）", expanded=True):
            st.dataframe(diag_df, use_container_width=True)
    st.stop()

spot_show = meta.get("spot", None)
dte_min = meta.get("dte_min", 0)
dte_max = meta.get("dte_max", 0)
margin_ratio = meta.get("margin_ratio", 1.0)

top_margin = float(result["ann_return_margin_pct"].iloc[0]) if len(result) else float("nan")
top_cash = float(result["ann_return_pct"].iloc[0]) if len(result) else float("nan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{spot_show:.2f}" if (spot_show and math.isfinite(float(spot_show))) else "N/A")
c2.metric("DTE 区间", f"{dte_min}–{dte_max} 天")
c3.metric("合约数量", f"{len(result)}")
c4.metric("Top 年化（Margin）", f"{top_margin:.2f}%")
st.caption(f"Top 年化（Cash-secured, bid/K）：{top_cash:.2f}% ｜ margin_ratio={margin_ratio:.2f}")

with st.expander("诊断信息（数据源是否残缺/限流）", expanded=False):
    st.write(meta)
    if diag_df is not None:
        st.dataframe(diag_df, use_container_width=True)

st.subheader(f"{meta.get('ticker','')} PUT 年化收益率结果（按 Margin 年化排序）")

show_cols = [
    "contractSymbol", "exp", "dte",
    "spot",
    "strike_raw", "strike_parsed", "strike",
    "bid", "ask", "ask_filled",
    "moneyness_pct", "openInterest", "inTheMoney",
    "capital",
    "ann_return_pct", "ann_return_margin_pct",
]

top_n = int(meta.get("top_n", 120))
view = result[show_cols].head(top_n)

st.dataframe(view, use_container_width=True, height=620)

csv_bytes = view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "下载 CSV（当前显示结果）",
    data=csv_bytes,
    file_name=f"{meta.get('ticker','TICKER')}_puts_DTE{dte_min}-{dte_max}_annualized.csv",
    mime="text/csv"
)
