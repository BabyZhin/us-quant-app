import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Basic Configuration
st.set_page_config(page_title="US Quant Strategy Screener", layout="wide")

# Multilingual Dictionary Configuration
TEXTS = {
    "zh": {
        "title": "📊 美股量化策略智能扫描系统",
        "sidebar_config": "⚙️ 扫描控制台",
        "mode_select": "选择扫描模式",
        "mode_auto": "自动扫描美股（市值>40亿美金 & 股本>4亿股）",
        "mode_custom": "单独输入一组自选股扫描",
        "timeframe": "选择时间周期",
        "weekly": "周线大波段 (Weekly)",
        "daily": "日线短线 (Daily)",
        "custom_input": "输入股票代码（逗号分隔，例如: AAPL, MSFT, NVDA）",
        "run_btn": "🚀 开始执行量化扫描",
        "scanning": "正在全市场扫描、获取财务数据与计算指标，请稍候...",
        "success": "扫描完成！",
        "no_result": "本期扫描未发现符合条件的标的。",
        "col_ticker": "股票代码",
        "col_close": "最新收盘价",
        "col_score": "KEIAS评分",
        "col_signal": "触发信号类型",
        "buy_title": "🟢 KEIAS 买入信号列表（机构建仓/低位地量）",
        "sell_title": "🔴 卖出预警列表（KXKX出货 & KEIAS动能衰竭）"
    },
    "en": {
        "title": "📊 US Quantitative Strategy Screening System",
        "sidebar_config": "⚙️ Control Panel",
        "mode_select": "Select Scan Mode",
        "mode_auto": "Auto Scan (Market Cap > $40B & Shares > 400M)",
        "mode_custom": "Custom Watchlist Scan",
        "timeframe": "Select Timeframe",
        "weekly": "Weekly Strategy",
        "daily": "Daily Strategy",
        "custom_input": "Enter Tickers (comma-separated, e.g., AAPL, MSFT, NVDA)",
        "run_btn": "🚀 Run Quant Screener",
        "scanning": "Scanning market, fetching fundamentals & calculating indicators...",
        "success": "Scan complete!",
        "no_result": "No stocks matched the criteria in this run.",
        "col_ticker": "Ticker",
        "col_close": "Latest Close",
        "col_score": "KEIAS Score",
        "col_signal": "Signal Type",
        "buy_title": "🟢 KEIAS Buy Signals (Accumulation/Low Volume)",
        "sell_title": "🔴 Sell Signals (KXKX Distribution & KEIAS Exhaustion)"
    }
}

# Sidebar Language Selection
lang_choice = st.sidebar.selectbox("🌐 语言 / Language", ["English", "中文 (Chinese)"])
lang = "zh" if "Chinese" in lang_choice else "en"
t = TEXTS[lang]

st.title(t["title"])

# Sidebar Control Panel
st.sidebar.header(t["sidebar_config"])
scan_mode = st.sidebar.radio(t["mode_select"], [t["mode_auto"], t["mode_custom"]])

timeframe_option = st.sidebar.selectbox(t["timeframe"], [t["weekly"], t["daily"]])
tf = "weekly" if "Weekly" in timeframe_option or "周线" in timeframe_option else "daily"

custom_tickers = ""
if scan_mode == t["mode_custom"]:
    custom_tickers = st.sidebar.text_area(t["custom_input"], value="AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, RMBS, GM, FIG")

if st.sidebar.button(t["run_btn"]):
    if scan_mode == t["mode_custom"]:
        tickers = [tk.strip().upper() for tk in custom_tickers.split(",") if tk.strip()]
    else:
        tickers = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "TSM", 
            "LLY", "JPM", "WMT", "V", "XOM", "MA", "UNH", "COST", "JNJ", "HD", 
            "PG", "NFLX", "BAC", "AMD", "ABBV", "CRM", "MRK", "CVX", "KO", "ORCL",
            "PEP", "ACN", "TMO", "MCD", "CSCO", "ADBE", "ABT", "LIN", "WFC", "DHR",
            "QCOM", "IBM", "TXN", "AMGN", "DIS", "PM", "CAT", "GE", "INTU", "VZ",
            "UBER", "SHOP", "PDD", "SNOW", "PLTR", "COIN", "INTC", "PYPL", "SQ"
        ]
    
    with st.spinner(t["scanning"]):
        results_buy = []
        results_sell = []
        vma_window = 60
        
        for ticker in tickers:
            try:
                if scan_mode == t["mode_auto"]:
                    stock_info = yf.Ticker(ticker).info
                    market_cap = stock_info.get("marketCap", 0) / 1e9  
                    float_shares = stock_info.get("floatShares", 0)     
                    shares_out = stock_info.get("sharesOutstanding", 0) 
                    
                    if market_cap < 40 or (float_shares < 4e8 and shares_out < 4e8):
                        continue

                period_val = "1y" if tf == "weekly" else "6mo"
                interval_val = "1wk" if tf == "weekly" else "1d"
                df = yf.download(ticker, period=period_val, interval=interval_val, progress=False)
                
                if df.empty or len(df) < vma_window:
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                high, low, close, volume = df['High'], df['Low'], df['Close'], df['Volume']
                
                high_low_diff = np.where((high - low) == 0, 0.001, high - low)
                pos = ((close - low) - (high - close)) / high_low_diff
                inst_net = pos * volume
                
                vma = volume.rolling(window=vma_window).mean()
                dlxp = volume < (vma * 0.5)
                
                v_power = (close - close.shift(1)) / close.shift(1) * volume
                ma_power = v_power.ewm(span=10, adjust=False).mean()
                abs_power = v_power.abs().ewm(span=10, adjust=False).mean()
                kxias = (ma_power / abs_power) * 100
                
                recent_dlxp = dlxp.iloc[-5:].any() if tf == "weekly" else dlxp.iloc[-3:].any()
                oversold = kxias.iloc[-1] < -40
                vol_breakout = (volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5) and (inst_net.iloc[-1] > 0)
                
                if recent_dlxp and (oversold or vol_breakout):
                    results_buy.append({
                        t["col_ticker"]: ticker,
                        t["col_close"]: round(float(close.iloc[-1]), 2),
                        t["col_score"]: round(float(kxias.iloc[-1]), 2),
                        t["col_signal"]: "KEIAS Buy (Low Vol/Oversold)" if lang == "en" else "KEIAS 买入 (低位地量/超卖)"
                    })
                
                is_high_zone = close.iloc[-1] >= (high.rolling(window=30).max().iloc[-1] * 0.90)
                heavy_vol = volume.iloc[-1] > (volume.rolling(20).mean().iloc[-1] * 1.5)
                keke_sell = is_high_zone and heavy_vol and inst_net.iloc[-1] < 0
                
                high_hist = (kxias.iloc[-5:] > 40).any()
                turning_down = kxias.iloc[-1] < kxias.iloc[-2] and kxias.iloc[-1] < kxias.iloc[-3]
                keias_sell = is_high_zone and high_hist and turning_down
                
                if keke_sell or keias_sell:
                    reason = []
                    if keke_sell: reason.append("KXKX Distribution" if lang == "en" else "KXKX放量流出")
                    if keias_sell: reason.append("KEIAS Exhaustion" if lang == "en" else "KEIAS动能拐头")
                    
                    results_sell.append({
                        t["col_ticker"]: ticker,
                        t["col_close"]: round(float(close.iloc[-1]), 2),
                        t["col_score"]: round(float(kxias.iloc[-1]), 2),
                        t["col_signal"]: " + ".join(reason)
                    })
                    
            except Exception as e:
                pass
                
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t["buy_title"])
            if results_buy:
                df_buy = pd.DataFrame(results_buy)
                st.dataframe(df_buy, use_container_width=True)
                csv_buy = df_buy.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Buy CSV" if lang == "en" else "📥 下载买入清单 CSV", data=csv_buy, file_name="keias_buy_results.csv", mime="text/csv")
            else:
                st.info(t["no_result"])
                
        with col2:
            st.subheader(t["sell_title"])
            if results_sell:
                df_sell = pd.DataFrame(results_sell)
                st.dataframe(df_sell, use_container_width=True)
                csv_sell = df_sell.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Sell CSV" if lang == "en" else "📥 下载卖出清单 CSV", data=csv_sell, file_name="sell_results.csv", mime="text/csv")
            else:
                st.info(t["no_result"])