from datetime import datetime, date
import math
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Put 年化收益率查询器", layout="wide")

# ========= Core math =========
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

# ========= Data fetching =========
@st.cache_data(ttl=180)
def get_meta(ticker: str):
    t = yf.Ticker(ticker)
    exps = t.options or []

    spot = None

    # 1) fast_info（快，但云端偶发为空）
    try:
        if hasattr(t, "fast_info") and t.fast_info:
            spot = t.fast_info.get("last_price") or t.fast_info.get("lastPrice")
    except Exception:
        pass

    # 2) info（慢一些，但有时更稳）
    if spot is None:
        try:
            info = t.info  # 可能较慢
            spot = info.get("regularMarketPrice") or info.get("currentPrice")
        except Exception:
            pass

    # 3) history close（兜底最稳）
    if spot is None:
        try:
            h = t.history(period="5d", interval="1d")
            if h is not None and not h.empty:
                spot = float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass

    # 清理/标准化
    try:
        spot = float(spot) if spot is not None else None
        if spot is not None and (not math.isfinite(spot) or spot <= 0):
            spot = None
    except Exception:
        spot = None

    return exps, spot

@st.cache_data(ttl=180)
def get_puts_for_exp(ticker: str, exp: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    chain = t.option_chain(exp)
    puts = chain.puts.copy()
    puts["exp"] = exp
    return puts

def exp_to_dte(exp: str, today: date) -> int:
    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    return (exp_date - today).days

# ========= UI =========
st.title("PUT 期权年化收益率查询器（bid / K）")

with st.sidebar:
    st.subheader("查询条件")

    # 一键清缓存（解决 yfinance 偶发不稳定/spot取不到）
    if st.button("强制刷新数据（清缓存）"):
        st.cache_data.clear()
        st.rerun()

    with st.form("filters"):
        ticker = st.text_input("1) 股票代码 (Ticker)", value="QQQ").strip().upper()

        exps, spot = get_meta(ticker)
        if not exps:
            st.error("未获取到到期日列表（该标的可能无期权或数据源暂时不可用）。")
            st.stop()

        # 1) 到期日区间：用 DTE 区间选择
        dte_min, dte_max = st.slider(
            "2) 到期区间（DTE 天）",
            min_value=1,
            max_value=365,
            value=(7, 30),
            help="例如 7–30 表示只看 7~30 天内到期的合约"
        )

        # 2) margin_ratio 替换杠杆参数
        margin_ratio = st.number_input(
            "3) margin_ratio（默认 1）",
            min_value=0.05,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="保证金占用比例：1=全额现金担保；0.3≈资金占用30%（回报会放大，但风险也更高）"
        )

        only_otm = st.checkbox("只看 OTM puts（推荐）", value=True)
        min_oi = st.number_input("最小 Open Interest（可选）", min_value=0, value=0, step=50)

        st.divider()
        st.caption("4) 行权价筛选方式（二选一）")

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
            help="moneyness_pct=(K/spot-1)*100。比如 -20% 到 -10% 代表行权价低于现价 10%~20%。"
        )

        top_n = st.number_input("显示 Top N（按 margin 年化排序）", min_value=10, max_value=500, value=80, step=10)

        submitted = st.form_submit_button("查询/刷新")

if not submitted:
    st.info("在左侧设置条件后，点“查询/刷新”。")
    st.stop()

# ========= Run query =========
today = date.today()

# 选出 DTE 区间内的 expirations
exp_candidates = []
for exp in exps:
    dte = exp_to_dte(exp, today)
    if dte_min <= dte <= dte_max:
        exp_candidates.append((exp, dte))

if not exp_candidates:
    st.warning("没有找到落在该 DTE 区间内的到期日。请放宽 DTE 范围。")
    st.stop()

# 抓取多个到期日 puts
rows = []
with st.spinner("抓取期权链并计算中（可能需要几秒）..."):
    for exp, dte in exp_candidates:
        puts = get_puts_for_exp(ticker, exp)
        if puts.empty:
            continue

        df = puts.copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
        df["inTheMoney"] = df["inTheMoney"].astype(bool)
        df["dte"] = dte

        # 只用 bid>0
        df = df[df["bid"].fillna(0) > 0]

        # OTM 过滤
        if only_otm:
            df = df[df["inTheMoney"] == False]

        # OI 过滤
        if min_oi > 0:
            df = df[df["openInterest"].fillna(0) >= min_oi]

        # spot / moneyness
        df["spot"] = spot
        if spot and math.isfinite(float(spot)):
            df["moneyness_pct"] = (df["strike"] / float(spot) - 1.0) * 100
        else:
            df["moneyness_pct"] = float("nan")

        # 行权价筛选
        if strike_mode == "按K区间筛选":
            if strike_min > 0:
                df = df[df["strike"] >= strike_min]
            if strike_max > 0:
                df = df[df["strike"] <= strike_max]
        else:
            # 按 OTM% 区间筛选，需要 spot
            if not (spot and math.isfinite(float(spot))):
                st.error("spot 未获取到，无法按 OTM% 区间筛选。请稍后刷新或改用按K区间筛选。")
                st.stop()
            df = df[df["moneyness_pct"].notna()]
            df = df[(df["moneyness_pct"] >= otm_low) & (df["moneyness_pct"] <= otm_high)]

        if df.empty:
            continue

        # 年化计算
        df["ann_return_pct"] = df.apply(
            lambda r: ann_by_strike(float(r["bid"]), float(r["strike"]), int(r["dte"])) * 100, axis=1
        )
        df["ann_return_margin_pct"] = df.apply(
            lambda r: ann_by_margin(float(r["bid"]), float(r["strike"]), int(r["dte"]), float(margin_ratio)) * 100, axis=1
        )

        rows.append(df)

if not rows:
    st.warning("没有符合筛选条件的合约。请放宽 DTE / OTM / K / OI 条件。")
    st.stop()

result = pd.concat(rows, ignore_index=True)

# 排序：先按 margin 年化
result = result.sort_values(
    ["ann_return_margin_pct", "ann_return_pct", "dte"],
    ascending=[False, False, True]
).reset_index(drop=True)

# ========= Summary cards =========
filtered_count = len(result)
spot_show = float(spot) if (spot and math.isfinite(float(spot))) else None

top_margin = float(result["ann_return_margin_pct"].iloc[0]) if filtered_count else float("nan")
top_cash = float(result["ann_return_pct"].iloc[0]) if filtered_count else float("nan")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Spot", f"{spot_show:.2f}" if spot_show else "N/A")
col2.metric("DTE 区间", f"{dte_min}–{dte_max} 天")
col3.metric("合约数量", f"{filtered_count}")
col4.metric("Top 年化（Margin）", f"{top_margin:.2f}%")

st.caption(f"Top 年化（Cash-secured, bid/K）：{top_cash:.2f}% ｜ margin_ratio={margin_ratio:.2f}")

# ========= Show table =========
st.subheader(f"{ticker} PUT 年化收益率结果（按 Margin 年化排序）")

show_cols = [
    "contractSymbol", "exp", "dte",
    "strike", "bid", "ask",
    "ann_return_pct", "ann_return_margin_pct",
    "moneyness_pct", "openInterest", "inTheMoney", "spot"
]
# 截取 top_n
result_view = result[show_cols].head(int(top_n))
st.dataframe(result_view, use_container_width=True, height=560)

# 下载 CSV
csv_bytes = result_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    "下载 CSV（当前显示结果）",
    data=csv_bytes,
    file_name=f"{ticker}_puts_DTE{dte_min}-{dte_max}_annualized.csv",
    mime="text/csv"
)
