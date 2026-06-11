
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    page_title="台股 AI Scanner v2.0 Fast + Risk",
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
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def fmt_num(value):
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return value


def finmind_get(dataset, data_id=None, start_date=None, end_date=None, timeout=8, retries=1):
    """
    輕量版 FinMind 抓取器：
    - timeout 縮短，避免卡死
    - 失敗只重試少量次數
    - token 同時放 query 與 header，相容不同情境
    """
    params = {"dataset": dataset}

    if data_id:
        params["data_id"] = str(data_id)
    if start_date:
        params["start_date"] = str(start_date)
    if end_date:
        params["end_date"] = str(end_date)
    if FINMIND_TOKEN:
        params["token"] = FINMIND_TOKEN

    headers = {"User-Agent": "Mozilla/5.0"}
    if FINMIND_TOKEN:
        headers["Authorization"] = f"Bearer {FINMIND_TOKEN}"

    for attempt in range(retries + 1):
        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=timeout)
            if response.status_code != 200:
                time.sleep(0.2)
                continue

            payload = response.json()
            data = payload.get("data", [])
            if not data:
                return pd.DataFrame()

            return pd.DataFrame(data)

        except Exception:
            time.sleep(0.2)

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_stock_info():
    info = finmind_get(dataset="TaiwanStockInfo", timeout=10, retries=1)

    if info.empty:
        return pd.DataFrame(columns=["stock_id", "stock_name", "industry_category"])

    keep_cols = [c for c in ["stock_id", "stock_name", "industry_category", "type", "market"] if c in info.columns]
    info = info[keep_cols].drop_duplicates(subset=["stock_id"])
    info["stock_id"] = info["stock_id"].astype(str)

    return info


@st.cache_data(ttl=1800)
def get_hot_stocks_by_turnover(limit=30):
    """
    v2.0.2 只修候選股來源，不動後面的籌碼、風險、評分邏輯。
    來源順序：
    1) 證交所 OpenAPI：STOCK_DAY_ALL
    2) 證交所 open_data：STOCK_DAY_ALL
    3) 證交所 RWD MI_INDEX
    4) 原本 MI_INDEX
    5) 備援熱門股清單
    """
    fallback_list = [
        "2330", "2317", "2382", "3231", "3441", "6285", "2313", "2409", "2344", "2618",
        "2303", "2454", "2603", "2609", "2615", "3706", "3661", "3017", "3037", "2881",
        "2882", "2883", "2884", "2891", "2892", "2356", "2379", "2345", "4938", "3711",
        "2327", "2308", "2368", "3034", "2357", "2605", "2885", "1101", "1216", "2002"
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json,text/csv,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/",
    }

    def clean_money_series(series):
        return pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("--", "0", regex=False)
            .str.replace("-", "0", regex=False)
            .str.replace("nan", "0", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(0)

    def pick_columns(df):
        stock_col = None
        name_col = None
        money_col = None

        for col in df.columns:
            c = str(col).strip().replace("\ufeff", "")

            if stock_col is None and (
                c in ["證券代號", "Code", "STOCK_SYMBOL", "代號", "stock_id"]
                or "證券代號" in c
            ):
                stock_col = col

            if name_col is None and (
                c in ["證券名稱", "Name", "NAME", "名稱", "stock_name"]
                or "證券名稱" in c
            ):
                name_col = col

            if money_col is None and (
                c in ["成交金額", "TradeValue", "TRADE_VALUE", "Trading_money"]
                or "成交金額" in c
                or "TradeValue" in c
            ):
                money_col = col

        return stock_col, name_col, money_col

    def build_stock_list(df, source_text):
        if df is None or df.empty:
            return None

        df = df.copy()
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

        stock_col, name_col, money_col = pick_columns(df)
        if stock_col is None or money_col is None:
            return None

        df[stock_col] = df[stock_col].astype(str).str.strip()
        df = df[df[stock_col].str.match(r"^\d{4}$", na=False)].copy()
        if df.empty:
            return None

        df["成交金額數字"] = clean_money_series(df[money_col])
        df = df[df["成交金額數字"] > 0].copy()
        if df.empty:
            return None

        df = df.sort_values("成交金額數字", ascending=False)
        top_df = df.head(limit).copy()
        stock_list = top_df[stock_col].astype(str).tolist()

        if not stock_list:
            return None

        name_map = {}
        if name_col is not None and name_col in top_df.columns:
            name_map = {
                str(code).strip(): str(name).strip()
                for code, name in zip(top_df[stock_col], top_df[name_col])
                if str(code).strip() and str(name).strip() and str(name).strip().lower() != "nan"
            }

        return {
            "stocks": stock_list,
            "source": source_text,
            "used_fallback": False,
            "name_map": name_map,
        }

    # 1) 證交所 OpenAPI：上市個股日成交資訊
    # 這個端點是最新上市全市場日成交資料，通常最適合做成交金額熱門股排序。
    try:
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code == 200 and response.text.strip():
            data = response.json()
            market_df = pd.DataFrame(data if isinstance(data, list) else data.get("data", []))
            result = build_stock_list(market_df, "證交所上市成交金額排行（OpenAPI STOCK_DAY_ALL）")
            if result:
                return result
    except Exception:
        pass

    # 2) 證交所 open_data：上市個股日成交資訊 CSV
    try:
        url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data"
        response = requests.get(url, headers=headers, timeout=12)
        if response.status_code == 200 and response.text.strip():
            market_df = pd.read_csv(io.StringIO(response.text))
            result = build_stock_list(market_df, "證交所上市成交金額排行（STOCK_DAY_ALL open_data）")
            if result:
                return result
    except Exception:
        pass

    # 3) RWD MI_INDEX：近 15 日內找最近有資料的交易日
    today = datetime.today().date()
    for i in range(0, 15):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y%m%d")

        url = (
            "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
            f"?date={date_str}&type=ALLBUT0999&response=json"
        )

        try:
            response = requests.get(url, headers=headers, timeout=12)
            if response.status_code != 200:
                continue

            data = response.json()
            rows = data.get("data9", [])
            fields = data.get("fields9", [])
            if not rows or not fields:
                continue

            col_len = min(len(fields), len(rows[0]))
            market_df = pd.DataFrame([row[:col_len] for row in rows], columns=fields[:col_len])
            result = build_stock_list(market_df, f"證交所上市成交金額排行（RWD MI_INDEX）{target_date}")
            if result:
                return result
        except Exception:
            continue

    # 4) 原本 MI_INDEX：保留舊版可用來源
    for i in range(0, 15):
        target_date = today - timedelta(days=i)
        date_str = target_date.strftime("%Y%m%d")

        url = (
            "https://www.twse.com.tw/exchangeReport/MI_INDEX"
            f"?response=json&date={date_str}&type=ALLBUT0999"
        )

        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                continue

            data = response.json()
            rows = data.get("data9", [])
            fields = data.get("fields9", [])
            if not rows or not fields:
                continue

            col_len = min(len(fields), len(rows[0]))
            market_df = pd.DataFrame([row[:col_len] for row in rows], columns=fields[:col_len])
            result = build_stock_list(market_df, f"證交所上市成交金額排行（舊版 MI_INDEX）{target_date}")
            if result:
                return result
        except Exception:
            continue

    # 5) 最後才走備援，讓系統不會空白
    fallback_name_map = {
        "2330": "台積電", "2317": "鴻海", "2382": "廣達", "3231": "緯創", "3441": "聯一光",
        "6285": "啟碁", "2313": "華通", "2409": "友達", "2344": "華邦電", "2618": "長榮航",
        "2303": "聯電", "2454": "聯發科", "2603": "長榮", "2609": "陽明", "2615": "萬海",
        "3706": "神達", "3661": "世芯-KY", "3017": "奇鋐", "3037": "欣興", "2881": "富邦金",
        "2882": "國泰金", "2883": "凱基金", "2884": "玉山金", "2891": "中信金", "2892": "第一金",
        "2356": "英業達", "2379": "瑞昱", "2345": "智邦", "4938": "和碩", "3711": "日月光投控",
        "2327": "國巨", "2308": "台達電", "2368": "金像電", "3034": "聯詠", "2357": "華碩",
        "2605": "新興", "2885": "元大金", "1101": "台泥", "1216": "統一", "2002": "中鋼"
    }
    return {
        "stocks": fallback_list[:limit],
        "source": "備援熱門股清單",
        "used_fallback": True,
        "name_map": fallback_name_map,
    }


# =========================
# Indicator and scoring
# =========================
def add_indicators(df):
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = ["open", "max", "min", "close", "Trading_Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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


def score_stock_price_only(stock_id, start_date, end_date):
    df = finmind_get(
        dataset="TaiwanStockPrice",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        timeout=8,
        retries=1,
    )

    if df.empty or len(df) < 60:
        return None

    df = add_indicators(df)
    if df.empty:
        return None

    latest = df.iloc[-1]

    technical_score = 0
    risk_score = 0
    reasons = []
    risks = []

    close = safe_float(latest.get("close"))
    ma5 = safe_float(latest.get("ma5"))
    ma10 = safe_float(latest.get("ma10"))
    ma20 = safe_float(latest.get("ma20"))
    ma60 = safe_float(latest.get("ma60"))

    volume_ratio = safe_float(latest.get("volume_ratio"))
    bias5 = safe_float(latest.get("bias5"))
    bias20 = safe_float(latest.get("bias20"))
    upper_shadow_ratio = safe_float(latest.get("upper_shadow_ratio"))
    near_high_ratio = safe_float(latest.get("near_high_ratio"))
    breakout_20d = bool(latest.get("breakout_20d"))

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
    if safe_float(latest.get("price_change_pct")) > 0:
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
    if volume_ratio >= 3 and safe_float(latest.get("price_change_pct")) < 1:
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

    stop_loss = min(ma5, safe_float(latest.get("min")))
    pressure_1 = latest.get("high20_prev")

    return {
        "日期": latest["date"].date(),
        "代號": str(latest["stock_id"]),
        "收盤價": round(close, 2),
        "技術分": round(technical_score, 1),
        "風險分": round(risk_score, 1),
        "量比": round(volume_ratio, 2),
        "5日乖離率": round(bias5, 2),
        "20日乖離率": round(bias20, 2),
        "突破20日高點": breakout_20d,
        "AI進場判斷": entry_note,
        "停損參考": round(stop_loss, 2),
        "壓力參考": round(safe_float(pressure_1), 2) if pd.notna(pressure_1) else None,
        "技術面原因": "、".join(reasons) if reasons else "技術面無明顯加分",
        "技術風險": "、".join(risks) if risks else "暫無明顯高風險訊號",
        "籌碼分": 50,
        "法人單日買賣超": 0,
        "法人近3日買賣超": 0,
        "融資變化": 0,
        "融券變化": 0,
        "籌碼面原因": "快速模式：尚未抓取籌碼細節",
        "籌碼風險": "快速模式：尚未抓取籌碼細節",
        "籌碼狀態": "未抓",
        "初步分數": round(technical_score - risk_score * 0.25, 1),
    }


def get_institutional_summary(stock_id, start_date, end_date):
    df = finmind_get(
        dataset="TaiwanStockInstitutionalInvestorsBuySell",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
        timeout=6,
        retries=0,
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
        timeout=6,
        retries=0,
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

    margin_today = safe_int(latest["MarginPurchaseTodayBalance"])
    margin_yesterday = safe_int(latest["MarginPurchaseYesterdayBalance"])
    short_today = safe_int(latest["ShortSaleTodayBalance"])
    short_yesterday = safe_int(latest["ShortSaleYesterdayBalance"])

    return {
        "margin_change": margin_today - margin_yesterday,
        "short_change": short_today - short_yesterday,
    }


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
        chip_score,
        "、".join(chip_reasons) if chip_reasons else "籌碼無明顯加分",
        "、".join(chip_risks) if chip_risks else "籌碼無明顯風險",
    )


def enrich_chip_for_row(row, start_date, end_date):
    row = dict(row)
    stock_id = str(row["代號"])

    inst = get_institutional_summary(stock_id, start_date, end_date)
    margin = get_margin_summary(stock_id, start_date, end_date)

    chip_score, chip_reasons, chip_risks = calculate_chip_score(
        inst["inst_net_1d"],
        inst["inst_net_3d"],
        margin["margin_change"],
        margin["short_change"],
    )

    row["法人單日買賣超"] = inst["inst_net_1d"]
    row["法人近3日買賣超"] = inst["inst_net_3d"]
    row["融資變化"] = margin["margin_change"]
    row["融券變化"] = margin["short_change"]
    row["籌碼分"] = chip_score
    row["籌碼面原因"] = chip_reasons
    row["籌碼風險"] = chip_risks
    row["籌碼狀態"] = "已抓"

    return row


@st.cache_data(ttl=1800)
def analyze_stocks(watchlist, chip_detail_count=10, source_name_map=None):
    """
    加速版：
    1. 先平行抓股價，算技術分與風險分。
    2. 依初步分數排序。
    3. 只針對前 N 檔補抓法人與融資融券。
    這樣不會 30 檔就打 90+ 次 API 卡很久。
    """
    today = datetime.today().date()
    start_date = today - timedelta(days=220)
    start_date_str = str(start_date)
    today_str = str(today)

    cleaned_watchlist = []
    for x in watchlist:
        code = str(x).strip()
        if code and code not in cleaned_watchlist:
            cleaned_watchlist.append(code)

    watchlist = cleaned_watchlist

    stock_info = get_stock_info()
    source_name_map = source_name_map or {}
    results = []

    max_workers = min(10, max(1, len(watchlist)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(score_stock_price_only, stock_id, start_date_str, today_str): stock_id
            for stock_id in watchlist
        }

        for future in as_completed(future_map):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception:
                pass

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    result_df["代號"] = result_df["代號"].astype(str)

    source_names = result_df["代號"].map(source_name_map)

    if not stock_info.empty:
        result_df = result_df.merge(
            stock_info,
            left_on="代號",
            right_on="stock_id",
            how="left",
        )
        source_names = result_df["代號"].map(source_name_map)
        result_df["名稱"] = result_df["stock_name"].fillna(source_names).fillna(result_df["代號"])
        result_df["產業"] = result_df["industry_category"].fillna("未知")
    else:
        result_df["名稱"] = source_names.fillna(result_df["代號"])
        result_df["產業"] = "未知"

    result_df = result_df.sort_values("初步分數", ascending=False).reset_index(drop=True)

    chip_detail_count = int(min(max(chip_detail_count, 0), len(result_df)))
    enriched_rows = []

    if chip_detail_count > 0:
        top_part = result_df.head(chip_detail_count)
        rest_part = result_df.iloc[chip_detail_count:].copy()

        max_chip_workers = min(6, chip_detail_count)

        with ThreadPoolExecutor(max_workers=max_chip_workers) as executor:
            futures = [
                executor.submit(enrich_chip_for_row, row, start_date_str, today_str)
                for _, row in top_part.iterrows()
            ]

            for future in as_completed(futures):
                try:
                    enriched_rows.append(future.result())
                except Exception:
                    pass

        enriched_df = pd.DataFrame(enriched_rows)

        if not enriched_df.empty:
            result_df = pd.concat([enriched_df, rest_part], ignore_index=True)
        else:
            result_df = pd.concat([top_part, rest_part], ignore_index=True)

    # 用中性籌碼 50 補足尚未抓籌碼的股票
    if "籌碼分" not in result_df.columns:
        result_df["籌碼分"] = 50

    result_df["籌碼分"] = pd.to_numeric(result_df["籌碼分"], errors="coerce").fillna(50)
    result_df["技術分"] = pd.to_numeric(result_df["技術分"], errors="coerce").fillna(0)
    result_df["風險分"] = pd.to_numeric(result_df["風險分"], errors="coerce").fillna(0)

    result_df["AI總分"] = (
        result_df["技術分"] * 0.60 +
        result_df["籌碼分"] * 0.30 -
        result_df["風險分"] * 0.10
    ).round(1)

    fill_cols = {
        "法人單日買賣超": 0,
        "法人近3日買賣超": 0,
        "融資變化": 0,
        "融券變化": 0,
        "籌碼面原因": "快速模式：尚未抓取籌碼細節",
        "籌碼風險": "快速模式：尚未抓取籌碼細節",
        "籌碼狀態": "未抓",
    }

    for col, val in fill_cols.items():
        if col not in result_df.columns:
            result_df[col] = val
        else:
            result_df[col] = result_df[col].fillna(val)

    result_df = result_df.sort_values("AI總分", ascending=False).reset_index(drop=True)

    return result_df


# =========================
# Chart and text
# =========================
@st.cache_data(ttl=1800)
def make_chart(stock_id):
    today = datetime.today().date()
    start_date = today - timedelta(days=220)

    df = finmind_get(
        dataset="TaiwanStockPrice",
        data_id=stock_id,
        start_date=str(start_date),
        end_date=str(today),
        timeout=8,
        retries=1,
    )

    if df.empty:
        return None

    df = add_indicators(df)
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
    )

    return fig


def make_judgement(row):
    ai_score = safe_float(row.get("AI總分"))
    risk_score = safe_float(row.get("風險分"))
    chip_score = safe_float(row.get("籌碼分"))

    if ai_score >= 80 and risk_score <= 35 and chip_score >= 60:
        return "高關注：技術與籌碼同步偏強，可列入隔日重點觀察，但仍不建議早盤直接追高。"
    elif ai_score >= 70 and risk_score <= 45:
        return "偏多觀察：條件不差，適合等回測支撐或尾盤確認。"
    elif ai_score >= 60:
        return "中性偏多：有部分條件轉強，但訊號還不夠完整。"
    else:
        return "暫不進場：綜合分數不足，或風險、籌碼、技術條件不佳。"


# =========================
# Streamlit UI
# =========================
st.title("📈 台股 AI Scanner v2.0 Fast + Risk")
st.caption("技術面 + 籌碼面 + 風險控管的盤後選股系統")

st.sidebar.header("設定")

scan_mode = st.sidebar.radio(
    "掃描模式",
    ["自選清單", "自動掃描熱門股"],
    index=0,
)

if scan_mode == "自選清單":
    default_watchlist = "2330,2317,2382,3231,3441,6285,2313,2409,2344,2618"

    watchlist_text = st.sidebar.text_area(
        "股票清單，用逗號分隔",
        default_watchlist,
        height=120,
    )

    watchlist = [
        x.strip()
        for x in watchlist_text.replace("，", ",").split(",")
        if x.strip()
    ]

    source_text = "自選清單"
    source_name_map = {}

else:
    hot_limit = st.sidebar.slider(
        "自動掃描熱門股數量",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
    )

    hot_result = get_hot_stocks_by_turnover(limit=hot_limit)
    watchlist = hot_result["stocks"]
    source_text = hot_result["source"]
    source_name_map = hot_result.get("name_map", {})

    st.sidebar.write("候選股來源：", source_text)
    st.sidebar.write("本次自動掃描候選股數：", len(watchlist))
    st.sidebar.write("候選股名稱對應數：", len(source_name_map))
    st.sidebar.caption(",".join(watchlist[:50]))

chip_detail_count = st.sidebar.slider(
    "抓取籌碼細節檔數",
    min_value=0,
    max_value=30,
    value=10,
    step=5,
    help="數字越高越慢。建議先用 10，等需要更精準再提高。",
)

if st.sidebar.button("重新分析"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("分析中，請稍候..."):
    df = analyze_stocks(
        watchlist,
        chip_detail_count=chip_detail_count,
        source_name_map=source_name_map,
    )

if df.empty:
    st.warning("目前沒有分析結果，請確認股票代號或稍後再試。")
    st.stop()

st.sidebar.write("送出候選股數：", len(watchlist))
st.sidebar.write("成功分析股票數：", len(df))

if len(df) < len(watchlist):
    st.sidebar.warning(f"{len(watchlist) - len(df)} 檔因資料不足、API限制或抓取失敗被略過")

if "籌碼狀態" in df.columns:
    st.sidebar.write("已抓籌碼檔數：", int((df["籌碼狀態"] == "已抓").sum()))

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

show_cols = [
    "日期",
    "代號",
    "名稱",
    "產業",
    "收盤價",
    "AI總分",
    "技術分",
    "籌碼分",
    "風險分",
    "量比",
    "法人單日買賣超",
    "法人近3日買賣超",
    "融資變化",
    "融券變化",
    "籌碼狀態",
    "AI進場判斷",
    "停損參考",
    "壓力參考",
]

available_cols = [col for col in show_cols if col in df.columns]
st.dataframe(df[available_cols], use_container_width=True, hide_index=True)

st.divider()

st.subheader("高風險不要追")
st.caption("這裡不是做空建議，而是提醒：分數看起來不錯也可能有追高、倒貨或籌碼轉弱風險。")

risk_df = df.copy()

risk_score_series = pd.to_numeric(risk_df.get("風險分", 0), errors="coerce").fillna(0)
technical_risk_series = risk_df.get("技術風險", "").astype(str)
chip_risk_series = risk_df.get("籌碼風險", "").astype(str)
margin_series = pd.to_numeric(risk_df.get("融資變化", 0), errors="coerce").fillna(0)
foreign_1d_series = pd.to_numeric(risk_df.get("法人單日買賣超", 0), errors="coerce").fillna(0)
foreign_3d_series = pd.to_numeric(risk_df.get("法人近3日買賣超", 0), errors="coerce").fillna(0)

risk_mask = (
    (risk_score_series >= 40)
    | (technical_risk_series != "暫無明顯高風險訊號")
    | chip_risk_series.str.contains("法人單日賣超|法人近3日合計賣超|融資大增|融資增加|融券回補", na=False)
    | (margin_series > 1000)
    | (foreign_1d_series < 0)
    | (foreign_3d_series < 0)
)

risk_view = risk_df[risk_mask].copy()

if risk_view.empty:
    st.success("目前掃描清單內沒有明顯高風險不要追標的。")
else:
    risk_view["風險排序"] = pd.to_numeric(risk_view["風險分"], errors="coerce").fillna(0)
    risk_view = risk_view.sort_values(["風險排序", "AI總分"], ascending=[False, False]).head(15)

    risk_cols = [
        "日期",
        "代號",
        "名稱",
        "產業",
        "收盤價",
        "AI總分",
        "技術分",
        "籌碼分",
        "風險分",
        "量比",
        "法人單日買賣超",
        "法人近3日買賣超",
        "融資變化",
        "籌碼狀態",
        "技術風險",
        "籌碼風險",
        "AI進場判斷",
    ]
    risk_cols = [col for col in risk_cols if col in risk_view.columns]

    st.dataframe(risk_view[risk_cols], use_container_width=True, hide_index=True)


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

with st.spinner("載入 K 線圖..."):
    fig = make_chart(selected_id)

if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("這檔股票暫時無法載入 K 線圖。")

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
