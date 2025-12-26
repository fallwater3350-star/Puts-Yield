from datetime import datetime, date
import math
import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Put 年化收益率查询器", layout="wide")

def ann_by_strike(bid: float, strike: float, dte: int) -> float:
    if dte <= 0 or strike <= 0 or not math.isfinite(bid) or bid <= 0:
        return float("nan")
    return (bid / strike) * (365.0 / dte)

@st.cache_data(ttl=120)
def get_meta(ticker: str):
    t = yf.Ticker(ticker)
    exps = t.options
    spot = None
    try:
        spot = t.fast_info.get("last_price") if hasattr(t, "fast_info") else None
    except Exception:
        spot = None
    return exps, spot

@st.cache_data(ttl=120)
def get_puts_for_exp(ticker: str, exp: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    chain = t.option_chain(exp)
    puts = chain.puts.copy()
    puts["exp"] = exp
    return puts

st.title("PUT 期权年化收益率查询器（bid / K）")

with st.sidebar:
    st.subheader("查询条件")

    ticker = st.text_input("1) 股票代码 (Ticker)", value="QQQ").strip().upper()

    exps, spot = get_meta(ticker)
    if not exps:
        st.error("未获取到到期日列表（该标的可能无期权或数据源暂时不可用）。")
        st.stop()

    exp = st.selectbox("2) 选择到期日 (Expiration)", options=exps, index=0)

    leverage = st.number_input("6) 杠杆参数 Leverage（默认 1）", min_value=0.1, value=1.0, step=0.1)

    mode = st.radio(
        "3/4) 行权价筛选方式",
        options=["按行权价K筛选", "按OTM%自动换算K"],
        index=1
    )

    # 用于“按K筛选”
    strike_input = st.number_input("3) 行权价 K（仅在按K筛选时生效）", min_value=0.0, value=0.0, step=1.0)
    strike_tolerance = st.number_input("K 匹配容差（±$，默认 0 表示精确匹配）", min_value=0.0, value=0.0, step=0.5)

    # 用于“按OTM%”
    otm_pct = st.slider(
        "4) OTM 比例（PUT 通常为负数）",
        min_value=-80.0,
        max_value=20.0,
        value=-10.0,
        step=0.5,
        help="OTM% 定义：K = spot * (1 + OTM%/100)。例如 -10% 表示行权价为现价的 90%。"
    )

    only_otm = st.checkbox("只看 OTM puts（推荐）", value=True)
    min_oi = st.number_input("最小 Open Interest（可选）", min_value=0, value=0, step=50)

    premium_source = st.selectbox("权利金取值", options=["bid"], index=0)  # 你要求用 bid

    run = st.button("查询/刷新")

if run:
    with st.spinner("抓取期权链并计算中..."):
        puts = get_puts_for_exp(ticker, exp)

    if puts.empty:
        st.error("该到期日未返回期权链数据。请稍后再试或换一个到期日。")
        st.stop()

    today = date.today()
    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
    dte = (exp_date - today).days
    if dte <= 0:
        st.warning("你选择的到期日已到期或为今天。请换一个未来到期日。")
        st.stop()

    # spot（如果 fast_info 取不到，就用近似：期权链里无法直接拿 spot，所以只能提示）
    if not spot or not math.isfinite(float(spot)):
        st.warning("当前 spot 价格未获取到，OTM% 换算会受影响。建议稍后刷新或改用按K筛选。")
        spot = None

    df = puts.copy()

    # 基础字段
    df["strike"] = df["strike"].astype(float)
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["openInterest"] = pd.to_numeric(df["openInterest"], errors="coerce")
    df["inTheMoney"] = df["inTheMoney"].astype(bool)

    # 过滤：只用 bid>0
    df = df[df["bid"].fillna(0) > 0]

    # 过滤：OTM
    if only_otm:
        df = df[df["inTheMoney"] == False]

    # 过滤：OI
    if min_oi > 0:
        df = df[df["openInterest"].fillna(0) >= min_oi]

    # 计算 OTM%（若 spot 可用）
    df["spot"] = spot
    if spot:
        df["moneyness_pct"] = (df["strike"] / spot - 1.0) * 100
    else:
        df["moneyness_pct"] = float("nan")

    # 根据模式生成目标K，并筛选
    target_k = None
    if mode == "按OTM%自动换算K":
        if not spot:
            st.error("spot 未获取到，无法按 OTM% 换算 K。请刷新或切换到按K筛选。")
            st.stop()
        target_k = float(spot) * (1.0 + otm_pct / 100.0)
        # 选“最接近”的行权价（美股行权价是离散刻度）
        df["abs_diff_to_targetK"] = (df["strike"] - target_k).abs()
        df = df.sort_values("abs_diff_to_targetK").head(20)  # 给你最近的几个K，避免只剩1条
    else:
        if strike_input > 0:
            if strike_tolerance > 0:
                df = df[(df["strike"] >= strike_input - strike_tolerance) & (df["strike"] <= strike_input + strike_tolerance)]
            else:
                df = df[df["strike"] == strike_input]

    # 计算年化（按 bid/K）
    df["dte"] = dte
    df["ann_return_pct"] = df.apply(lambda r: ann_by_strike(float(r["bid"]), float(r["strike"]), dte) * 100, axis=1)
    df["ann_return_margin_pct"] = df["ann_return_pct"] * float(leverage)

    # 排序
    df = df.sort_values(["ann_return_margin_pct", "ann_return_pct"], ascending=[False, False]).reset_index(drop=True)

    # 展示
    st.subheader(f"{ticker} PUT 年化收益率（到期日 {exp}，DTE={dte} 天）")
    if target_k is not None:
        st.caption(f"按 OTM% 换算目标K：spot={spot:.2f}，OTM={otm_pct:.1f}% → targetK≈{target_k:.2f}（已列出最接近的若干行权价）")

    show_cols = [
        "contractSymbol", "spot", "exp", "dte",
        "strike", "bid", "ask",
        "ann_return_pct", "ann_return_margin_pct",
        "moneyness_pct", "openInterest", "inTheMoney"
    ]
    st.dataframe(df[show_cols], use_container_width=True, height=520)

    # 下载
    csv_bytes = df[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下载 CSV",
        data=csv_bytes,
        file_name=f"{ticker}_puts_{exp}_annualized.csv",
        mime="text/csv"
    )
else:
    st.info("左侧设置条件后，点“查询/刷新”。")