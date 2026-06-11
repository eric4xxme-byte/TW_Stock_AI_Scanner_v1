import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="台股 AI Scanner v1",
    page_icon="📈",
    layout="wide",
)

API_URL = "https://api.finmindtrade.com/api/v4/data"

try:
    FINMIND_TOKEN = st.secrets.get("FINMIND_TOKEN", "")
except Exception:
    FINMIND_TOKEN = ""


# =========================
# Basic helpers
# =========================
def to_number(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_int(value):
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return str(value)


# =========================
# Data fetchers
# =========================
def finmind_get(dataset, data_id=None, start_date=None, end_date=None):
    """FinMind API with retry and light backoff."""
    params = {"dataset": dataset}

    if data_id:
        params["data_id"] = str(data_id)
    if start_date:
        params["start_date"] = str(start_date)
    if end_date:
        params["end_date"] = str(end_date)

    headers = {"User-Agent": "Mozilla/5.0"}

    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN
        headers["Authorization"] = f"Bearer {FINMIND_TOKEN}"

    for attempt in range(3):
        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=30)

            if response.status_code != 200:
                time.sleep(1 + attempt * 0.5)
                continue

            payload = response.json()
            df = pd.DataFrame(payload.get("data", []))

            if not df.empty:
                return df

            time.sleep(1 + attempt * 0.5)

        except Exception:
            time.sleep(1.5 + attempt * 0.5)

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_info():
    info = finmind_get(dataset="TaiwanStockInfo")

    if info.empty:
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])

    keep_cols = [c for c in ["stock_id", "stock_name", "industry_category"] if c in info.columns]
    if not keep_cols:
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])

    info = info[keep_cols].drop_duplicates(subset=["stock_id"])

    for col in ["stock_id", "stock_name", "industry_category"]:
        if col not in info.columns:
            info[col] = "未知"

    info["stock_id"] = info["stock_id"].astype(str)
    return info[["stock_id", "stock_name", "industry_category"]]


@st.cache_data(ttl=3600)
def get_hot_stocks_by_turnover(limit=30):
    """
    自動抓上市市場最近交易日成交金額前 N 名。
    主要來源：證交所 MI_INDEX。
    若抓不到，回傳備援清單，避免畫面空白。
    """
    fallback_list = [
        "2330", "2317", "2382", "3231", "3441",
        "6285", "2313", "2409", "2344", "2618",
        "2303", "2454", "2603", "2609", "2615",
        "3706", "3661", "3017", "3037", "2881",
        "2882", "2883", "2884", "2891", "2892",
        "2356", "2379", "2345", "4938", "3711",
    ]

    today = datetime.today().date()
    headers = {"User-Agent": "Mozilla/5.0"}

    for i in range(0, 20):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y%m%d")

        url = (
            "https://www.twse.com.tw/exchangeReport/MI_INDEX"
            f"?response=json&date={date_str}&type=ALLBUT0999"
        )

        try:
            response = requests.get(url, headers=headers, timeout=20)
            data = response.json()
        except Exception:
            continue

        rows = data.get("data9", [])
        fields = data.get("fields9", [])

        if not rows or not fields:
            continue

        try:
            col_len = min(len(fields), len(rows[0]))
            df = pd.DataFrame([row[:col_len] for row in rows], columns=fields[:col_len])
        except Exception:
            continue

        stock_col = None
        money_col = None

        for col in df.columns:
            if "證券代號" in col:
                stock_col = col
            if "成交金額" in col:
                money_col = col

        if stock_col is None or money_col is None:
            continue

        df[stock_col] = df[stock_col].astype(str).str.strip()

        # 只保留四碼股票，排除 ETF / 權證 / 特殊商品
        df = df[df[stock_col].str.match(r"^\d{4}$", na=False)].copy()

        df["成交金額數字"] = (
            df[money_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("--", "0", regex=False)
        )
        df["成交金額數字"] = pd.to_numeric(df["成交金額數字"], errors="coerce").fillna(0)
        df = df.sort_values("成交金額數字", ascending=False)

        stock_list = df[stock_col].astype(str).head(limit).tolist()

        if stock_list:
            return stock_list[:limit]

    return fallback_list[:limit]


# =========================
# Indicators and scores
# =========================
def add_indicators(df):
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = ["open", "max", "min", "close", "Trading_Volume"]
    for col in numeric_cols:
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "max", "min", "close", "Trading_Volume"])
    df = df.sort_values("date")

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    df["vol_ma5"] = df["Trading_Volume"].rolling(5).mean()
    df["vol_ma20"] = df["Trading_Volume"].rolling(20).mean()

    df["high20_prev"] = df["max"].rolling(20).max().shift(1)
    df["volume_ratio"] = df["Trading_Volume"] / df["vol_ma5"]

    df["bias5"] = (df["close"] - df["ma5"]) / df["ma5"] * 100
    df["bias20"] = (df["close"] - df["ma20"]) / df["ma20"] * 100

    df["daily_range"] = df["max"] - df["min"]
    df["upper_shadow"] = df["max"] - df[["open", "close"]].max(axis=1)

    df["upper_shadow_ratio"] = np.where(
        df["daily_range"] > 0,
        df["upper_shadow"] / df["daily_range"],
        0,
    )

    df["near_high_ratio"] = np.where(
        df["daily_range"] > 0,
        (df["close"] - df["min"]) / df["daily_range"],
        0,
    )

    df["breakout_20d"] = df["close"] > df["high20_prev"]
    df["price_change_pct"] = df["close"].pct_change() * 100

    return df


def score_stock(df):
    if df.empty or len(df) < 60:
        return None

    df = add_indicators(df)
    if df.empty or len(df) < 60:
        return None

    latest = df.iloc[-1]

    required = [
        "date", "stock_id", "close", "min", "ma5", "ma10", "ma20", "ma60",
        "volume_ratio", "bias5", "bias20", "upper_shadow_ratio",
        "near_high_ratio", "breakout_20d", "price_change_pct", "high20_prev",
    ]
    for col in required:
        if col not in latest.index or pd.isna(latest[col]):
            return None

    technical_score = 0
    risk_score = 0
    reasons = []
    risks = []

    close = latest["close"]
    ma5 = latest["ma5"]
    ma10 = latest["ma10"]
    ma20 = latest["ma20"]
    ma60 = latest["ma60"]
    volume_ratio = latest["volume_ratio"]
    bias5 = latest["bias5"]
    bias20 = latest["bias20"]
    upper_shadow_ratio = latest["upper_shadow_ratio"]
    near_high_ratio = latest["near_high_ratio"]
    breakout_20d = latest["breakout_20d"]

    if close > ma5:
        technical_score += 15
        reasons.append("站上5日線")
    if close > ma10:
        technical_score += 15
        reasons.append("站上10日線")
    if close > ma20:
        technical_score += 15
        reasons.append("站上20日線")
    if close > ma60:
        technical_score += 10
        reasons.append("站上60日線")
    if breakout_20d:
        technical_score += 20
        reasons.append("突破近20日高點")
    if volume_ratio >= 1.5:
        technical_score += 15
        reasons.append("成交量放大")
    if near_high_ratio >= 0.7:
        technical_score += 10
        reasons.append("收盤接近當日高點")
    if latest["price_change_pct"] > 0:
        technical_score += 5
        reasons.append("今日收漲")

    if upper_shadow_ratio >= 0.45 and volume_ratio >= 1.5:
        risk_score += 30
        risks.append("爆量長上影，追高風險")
    if bias5 >= 10:
        risk_score += 20
        risks.append("短線乖離過大")
    if bias20 >= 20:
        risk_score += 20
        risks.append("波段乖離過大")
    if volume_ratio >= 3 and latest["price_change_pct"] < 1:
        risk_score += 20
        risks.append("爆量但漲不動")
    if close < ma5:
        risk_score += 20
        risks.append("跌破5日線")

    technical_score = min(technical_score, 100)
    risk_score = min(risk_score, 100)

    if technical_score >= 75 and risk_score <= 40:
        if breakout_20d:
            entry_note = "強勢突破型：不要早盤追高，等回測支撐或尾盤確認"
        else:
            entry_note = "偏多型：可觀察回測5日線或10日線"
    elif technical_score >= 60:
        entry_note = "觀察型：等待更明確突破或量價確認"
    else:
        entry_note = "暫不進場：分數不足或風險偏高"

    stop_loss = min(ma5, latest["min"])
    pressure_1 = latest["high20_prev"]

    return {
        "日期": latest["date"].date(),
        "代號": str(latest["stock_id"]),
        "收盤價": round(float(close), 2),
        "技術分": round(float(technical_score), 1),
        "風險分": round(float(risk_score), 1),
        "量比": round(float(volume_ratio), 2),
        "5日乖離率": round(float(bias5), 2),
        "20日乖離率": round(float(bias20), 2),
        "突破20日高點": bool(breakout_20d),
        "AI進場判斷": entry_note,
        "停損參考": round(float(stop_loss), 2),
        "壓力參考": round(float(pressure_1), 2) if pd.notna(pressure_1) else None,
        "技術面原因": "、".join(reasons) if reasons else "技術面無明顯加分",
        "技術風險": "、".join(risks) if risks else "暫無明顯高風險訊號",
    }


def get_institutional_summary(stock_id, start_date, end_date):
    df = finmind_get(
        dataset="TaiwanStockInstitutionalInvestorsBuySell",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty or "buy" not in df.columns or "sell" not in df.columns:
        return {"inst_net_1d": 0, "inst_net_3d": 0}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0)
    df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["net"] = df["buy"] - df["sell"]

    daily = df.groupby("date")["net"].sum().reset_index().sort_values("date")

    if daily.empty:
        return {"inst_net_1d": 0, "inst_net_3d": 0}

    return {
        "inst_net_1d": int(daily.iloc[-1]["net"]),
        "inst_net_3d": int(daily.tail(3)["net"].sum()),
    }


def get_margin_summary(stock_id, start_date, end_date):
    df = finmind_get(
        dataset="TaiwanStockMarginPurchaseShortSale",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return {"margin_change": 0, "short_change": 0}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest = df.iloc[-1]

    required_cols = [
        "MarginPurchaseTodayBalance",
        "MarginPurchaseYesterdayBalance",
        "ShortSaleTodayBalance",
        "ShortSaleYesterdayBalance",
    ]

    if not all(col in df.columns for col in required_cols):
        return {"margin_change": 0, "short_change": 0}

    margin_today = pd.to_numeric(latest["MarginPurchaseTodayBalance"], errors="coerce")
    margin_yesterday = pd.to_numeric(latest["MarginPurchaseYesterdayBalance"], errors="coerce")
    short_today = pd.to_numeric(latest["ShortSaleTodayBalance"], errors="coerce")
    short_yesterday = pd.to_numeric(latest["ShortSaleYesterdayBalance"], errors="coerce")

    if pd.isna(margin_today) or pd.isna(margin_yesterday):
        margin_change = 0
    else:
        margin_change = int(margin_today - margin_yesterday)

    if pd.isna(short_today) or pd.isna(short_yesterday):
        short_change = 0
    else:
        short_change = int(short_today - short_yesterday)

    return {"margin_change": margin_change, "short_change": short_change}


def calculate_chip_score(inst_net_1d, inst_net_3d, margin_change, short_change):
    chip_score = 50
    chip_reasons = []
    chip_risks = []

    if inst_net_1d > 0:
        chip_score += 15
        chip_reasons.append("法人單日買超")
    elif inst_net_1d < 0:
        chip_score -= 15
        chip_risks.append("法人單日賣超")

    if inst_net_3d > 0:
        chip_score += 15
        chip_reasons.append("法人近3日合計買超")
    elif inst_net_3d < 0:
        chip_score -= 10
        chip_risks.append("法人近3日合計賣超")

    if margin_change > 1000:
        chip_score -= 20
        chip_risks.append("融資大增，散戶追高風險")
    elif margin_change > 300:
        chip_score -= 10
        chip_risks.append("融資增加，籌碼略偏雜")
    elif margin_change < -300:
        chip_score += 10
        chip_reasons.append("融資減少，籌碼較乾淨")

    if short_change > 300:
        chip_score += 5
        chip_reasons.append("融券增加，可能有軋空動能")
    elif short_change < -300:
        chip_score -= 5
        chip_risks.append("融券回補，短線軋空力道可能減弱")

    chip_score = max(0, min(chip_score, 100))

    return (
        round(float(chip_score), 1),
        "、".join(chip_reasons) if chip_reasons else "籌碼無明顯加分",
        "、".join(chip_risks) if chip_risks else "籌碼無明顯風險",
    )


# =========================
# Main analyzer
# =========================
@st.cache_data(ttl=3600)
def analyze_stocks(watchlist):
    today = datetime.today().date()
    start_date = today - timedelta(days=220)

    stock_info = get_stock_info()
    results = []

    for stock_id in watchlist:
        stock_id = str(stock_id).strip()
        if not stock_id:
            continue

        df = finmind_get(
            dataset="TaiwanStockPrice",
            data_id=stock_id,
            start_date=str(start_date),
            end_date=str(today),
        )

        base = score_stock(df)

        if base is None:
            time.sleep(0.35)
            continue

        inst = get_institutional_summary(stock_id, str(start_date), str(today))
        margin = get_margin_summary(stock_id, str(start_date), str(today))

        chip_score, chip_reasons, chip_risks = calculate_chip_score(
            inst["inst_net_1d"],
            inst["inst_net_3d"],
            margin["margin_change"],
            margin["short_change"],
        )

        base["法人單日買賣超"] = inst["inst_net_1d"]
        base["法人近3日買賣超"] = inst["inst_net_3d"]
        base["融資變化"] = margin["margin_change"]
        base["融券變化"] = margin["short_change"]
        base["籌碼分"] = chip_score
        base["籌碼面原因"] = chip_reasons
        base["籌碼風險"] = chip_risks

        base["AI總分"] = round(
            base["技術分"] * 0.55 +
            base["籌碼分"] * 0.35 -
            base["風險分"] * 0.10,
            1,
        )

        results.append(base)
        time.sleep(0.35)

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    result_df["代號"] = result_df["代號"].astype(str)

    if not stock_info.empty:
        stock_info = stock_info.copy()
        stock_info["stock_id"] = stock_info["stock_id"].astype(str)

        result_df = result_df.merge(
            stock_info,
            left_on="代號",
            right_on="stock_id",
            how="left",
        )

        result_df["名稱"] = result_df["stock_name"].fillna(result_df["代號"])
        result_df["產業"] = result_df["industry_category"].fillna("未知")
    else:
        result_df["名稱"] = result_df["代號"]
        result_df["產業"] = "未知"

    result_df = result_df.sort_values("AI總分", ascending=False).reset_index(drop=True)
    return result_df


# =========================
# Chart and judgement
# =========================
def make_chart(stock_id):
    today = datetime.today().date()
    start_date = today - timedelta(days=220)

    df = finmind_get(
        dataset="TaiwanStockPrice",
        data_id=stock_id,
        start_date=str(start_date),
        end_date=str(today),
    )

    if df.empty:
        return None

    df = add_indicators(df)
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("股價 + 均線", "成交量"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["max"],
            low=df["min"],
            close=df["close"],
            name="K線",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=df["date"], y=df["ma5"], mode="lines", name="MA5"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma10"], mode="lines", name="MA10"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], mode="lines", name="MA20"), row=1, col=1)
    fig.add_trace(go.Bar(x=df["date"], y=df["Trading_Volume"], name="成交量"), row=2, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    return fig


def make_judgement(row):
    ai_score = to_number(row.get("AI總分", 0))
    risk_score = to_number(row.get("風險分", 0))
    chip_score = to_number(row.get("籌碼分", 0))

    if ai_score >= 80 and risk_score <= 35 and chip_score >= 60:
        return "高關注：技術與籌碼同步偏強，可列入隔日重點觀察，但仍不建議早盤直接追高。"
    if ai_score >= 70 and risk_score <= 45:
        return "偏多觀察：條件不差，適合等回測支撐或尾盤確認。"
    if ai_score >= 60:
        return "中性偏多：有部分條件轉強，但訊號還不夠完整。"
    return "暫不進場：綜合分數不足，或風險、籌碼、技術條件不佳。"


# =========================
# Streamlit UI
# =========================
st.title("📈 台股 AI Scanner v1")
st.caption("技術面 + 籌碼面 + 風險控管的盤後選股系統")

st.sidebar.header("設定")

scan_mode = st.sidebar.radio(
    "掃描模式",
    ["自選清單", "自動掃描熱門股"],
    index=1,
)

if scan_mode == "自選清單":
    default_watchlist = "2330,2317,2382,3231,3441,6285,2313,2409,2344,2618"

    watchlist_text = st.sidebar.text_area(
        "股票清單，用逗號分隔",
        default_watchlist,
        height=120,
    )

    watchlist = [x.strip() for x in watchlist_text.replace("，", ",").split(",") if x.strip()]

else:
    hot_limit = st.sidebar.slider(
        "自動掃描熱門股數量",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
    )

    with st.spinner("正在抓取市場熱門股..."):
        watchlist = get_hot_stocks_by_turnover(limit=hot_limit)

    st.sidebar.write("本次自動掃描候選股數：", len(watchlist))
    st.sidebar.caption(",".join(watchlist[:30]))

if st.sidebar.button("重新分析"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("分析中，請稍候..."):
    df = analyze_stocks(watchlist)

st.sidebar.write("送出候選股數：", len(watchlist))
st.sidebar.write("成功分析股票數：", len(df))

if len(df) < len(watchlist):
    st.sidebar.warning(f"{len(watchlist) - len(df)} 檔因資料不足、API限制或抓取失敗被略過")

if df.empty:
    st.warning("目前沒有分析結果，請確認股票代號或稍後再試。")
    st.stop()

# Ensure column order exists
show_cols = [
    "日期", "代號", "名稱", "產業", "收盤價", "AI總分", "技術分", "籌碼分", "風險分", "量比",
    "法人單日買賣超", "法人近3日買賣超", "融資變化", "融券變化", "AI進場判斷", "停損參考", "壓力參考",
]
show_cols = [c for c in show_cols if c in df.columns]

top_df = df.head(5)

c1, c2, c3, c4 = st.columns(4)
c1.metric("分析股票數", len(df))
c2.metric("最高 AI 分數", df["AI總分"].max())
c3.metric("平均風險分", round(df["風險分"].mean(), 1))
c4.metric("高關注股票數", len(df[df["AI總分"] >= 75]))

st.divider()

st.subheader("今日 AI 前 5 名")

top_cols = st.columns(min(5, len(top_df)))

for col, (_, row) in zip(top_cols, top_df.iterrows()):
    with col:
        st.markdown(f"### {row['名稱']} {row['代號']}")
        st.caption(row["產業"])
        st.metric("AI總分", row["AI總分"])
        st.write(f"技術分：{row['技術分']}")
        st.write(f"籌碼分：{row['籌碼分']}")
        st.write(f"風險分：{row['風險分']}")
        st.info(row["AI進場判斷"])

st.divider()

st.subheader("完整 AI 排名表")
st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

st.divider()

st.subheader("單檔詳細分析")

options = (df["代號"].astype(str) + " " + df["名稱"].astype(str)).tolist()
selected = st.selectbox("選擇股票", options)
selected_id = selected.split(" ")[0]

row = df[df["代號"].astype(str) == selected_id].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("收盤價", row["收盤價"])
k2.metric("AI總分", row["AI總分"])
k3.metric("籌碼分", row["籌碼分"])
k4.metric("風險分", row["風險分"])

st.markdown(f"## {row['名稱']} {row['代號']}")
st.write(f"**產業：** {row['產業']}")

fig = make_chart(selected_id)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("此股票目前無法產生 K 線圖。")

st.markdown("### AI 總結")
st.info(make_judgement(row))

st.markdown("### 進場策略")
st.success(row["AI進場判斷"])

st.markdown("### 出場策略")
st.warning(
    f"停損參考：{row['停損參考']}。壓力參考：{row['壓力參考']}。"
    " 若出現爆量長上影、跌破5日線、法人轉賣或融資暴增，應考慮減碼或出場。"
)

st.markdown("### 技術面原因")
st.success(row["技術面原因"])

st.markdown("### 籌碼面原因")
st.success(row["籌碼面原因"])

st.markdown("### 風險提醒")
st.error(f"技術風險：{row['技術風險']}｜籌碼風險：{row['籌碼風險']}")
