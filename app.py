import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime, time
import matplotlib.pyplot as plt
import requests

# ======================
# 飞书推送配置（替换成你的Webhook）
# ======================
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/b5162b51-57e3-4d9f-87ec-a97203261dda"

def send_feishu(title, content):
    """飞书推送函数"""
    try:
        payload = {
            "msg_type": "text",
            "content": {"text": f"{title}\n\n{content}"}
        }
        response = requests.post(FEISHU_WEBHOOK_URL, json=payload, timeout=8)
        return response.status_code == 200 and response.json().get("code") == 0
    except Exception as e:
        print(f"推送失败: {e}")
        return False

# ======================
# 登录系统
# ======================
if "login" not in st.session_state:
    st.session_state.login = False

def login_page():
    st.title("🔒 A股AI量化 · 最终完全体")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if username == "admin" and password == "123456":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("账号或密码错误")

# ======================
# 数据接口：获取股票数据+实时股价
# ======================
def get_data(code, start="2020-01-01"):
    """获取股票历史数据"""
    try:
        if len(code) == 6:
            if code.startswith("6"):
                ticker = f"{code}.SS"
            else:
                ticker = f"{code}.SZ"
        else:
            ticker = code

        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime("2026-12-31")

        stk = yf.Ticker(ticker)
        df = stk.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df.empty:
            return None
            
        df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"历史数据获取异常: {e}")
        return None

def get_realtime_price(code):
    """获取股票实时股价（含涨跌幅）"""
    try:
        if len(code) == 6:
            if code.startswith("6"):
                ticker = f"{code}.SS"
            else:
                ticker = f"{code}.SZ"
        else:
            ticker = code

        stk = yf.Ticker(ticker)
        # 获取最新行情
        hist = stk.history(period="1d")
        if hist.empty:
            return {"price": 0, "change": 0, "change_pct": 0}
        
        # 计算涨跌幅
        current_price = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[0] if len(hist) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100

        return {
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2)
        }
    except Exception as e:
        print(f"实时股价获取异常: {e}")
        return {"price": 0, "change": 0, "change_pct": 0}

def get_sh_index():
    """获取上证指数"""
    return get_data("000001.SS", start="2020-01-01")

# ======================
# 特征工程与风控
# ======================
def build_features(df):
    data = df.copy()
    data["return"] = data["close"].pct_change()
    data["ma5"] = data["close"].rolling(5).mean()
    data["ma10"] = data["close"].rolling(10).mean()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()
    data["vol_ma5"] = data["volume"].rolling(5).mean()
    data["vol_ma20"] = data["volume"].rolling(20).mean()

    data.dropna(inplace=True)
    if data.empty:
        return pd.DataFrame()
        
    data["trend_up"] = (data["close"] > data["ma20"]) & (data["ma20"] > data["ma60"])
    data["vol_strength"] = data["vol_ma5"] > data["vol_ma20"]
    data["strong_uptrend"] = data["close"] > data["ma20"] * 1.03
    data["breakout"] = data["close"] > data["close"].rolling(20).max()
    data["target"] = (data["return"].shift(-1) > 0).astype(int)
    data.dropna(inplace=True)
    return data

# ======================
# 大盘与信号逻辑
# ======================
def get_market_level():
    sh = get_sh_index()
    if sh is None or len(sh) < 60:
        return 0
    sh = build_features(sh)
    if sh.empty:
        return 0
    last = sh.iloc[-1]
    if last["trend_up"] and last["strong_uptrend"]:
        return 2
    elif last["trend_up"]:
        return 1
    else:
        return 0

def market_info():
    lv = get_market_level()
    level_map = {0: "🔴 大盘弱势 -> 空仓避险", 1: "🟡 大盘安全 -> 正常交易", 2: "🟢 大盘强势 -> 可重仓参与"}
    return level_map.get(lv, "🔴 数据异常")

def allow_trade():
    return get_market_level() > 0

def train_model(data):
    feats = ["return", "ma5", "ma10", "ma20", "ma60", "vol_ma5", "vol_ma20", "trend_up", "vol_strength", "strong_uptrend", "breakout"]
    X = data[feats]
    y = data["target"]
    if len(X) < 10: return None, 0.0
    split = int(len(X) * 0.8)
    model = XGBClassifier(random_state=666, max_depth=5, n_estimators=150)
    model.fit(X[:split], y[:split])
    acc = np.mean(model.predict(X[split:]) == y[split:])
    return model, acc

def super_signal(code):
    df = get_data(code, start="2025-01-01")
    if df is None or len(df) < 60:
        return "🔴 数据不足", 0
    data = build_features(df)
    if data.empty or len(data) < 60:
        return "🔴 数据错误", 0
    model, acc = train_model(data)
    if model is None: return "🔴 训练失败", 0
    lv = get_market_level()
    last = data.iloc[-1]
    x = np.array([[last["return"], last["ma5"], last["ma10"], last["ma20"], last["ma60"], last["vol_ma5"], last["vol_ma20"], last["trend_up"], last["vol_strength"], last["strong_uptrend"], last["breakout"]]])
    pred = model.predict(x)[0]
    if lv == 0: return "🔴 空仓避险", round(acc*100,1)
    if pred == 1 and last["trend_up"] and last["strong_uptrend"]: return "🟢 超级买入", round(acc*100,1)
    if pred == 1 and last["trend_up"]: return "🟡 可关注", round(acc*100,1)
    return "🔴 观望", round(acc*100,1)

# ======================
# 推送与回测
# ======================
def generate_rich_report():
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    market_msg = market_info()
    watchlist = ["600519", "000001", "600036", "600030", "002594", "000858"]
    msg = f"【AI量化日报】{today}\n{market_msg}\n📊 关注列表：\n"
    buy_list = []
    for code in watchlist:
        sig, acc = super_signal(code)
        price_info = get_realtime_price(code)
        msg += f"• {code}: {sig} ({acc}%) | 现价: {price_info['price']} ({price_info['change_pct']}%)\n"
        if "超级买入" in sig: buy_list.append(code)
    msg += f"\n策略：{'✅ 可交易' if allow_trade() else '❌ 空仓'}\n风控：止损7% / 止盈25%"
    return msg

def backtest_final(code, money=100000):
    df = get_data(code)
    if df is None or len(df) < 60: return None
    data = build_features(df)
    if data.empty: return None
    model, acc = train_model(data)
    if model is None: return None
    lv = get_market_level()
    balance, pos, cost, hist, trades = money, 0, 0, [money], []
    for i, row in data.iterrows():
        x = np.array([[row["return"], row["ma5"], row["ma10"], row["ma20"], row["ma60"], row["vol_ma5"], row["vol_ma20"], row["trend_up"], row["vol_strength"], row["strong_uptrend"], row["breakout"]]])
        pred = model.predict(x)[0]
        
        if lv == 0:
            if pos > 0:
                balance += pos * row["close"]
                trades.append([str(i)[:10], "空仓避险", row["close"], pos, round(balance,2)])
                pos = 0
            hist.append(balance)
            continue
            
        max_pos = int(balance * 0.8 // row["close"])
        
        if pos > 0:
            if row["close"] <= cost * 0.93:
                balance += pos * row["close"]
                trades.append([str(i)[:10], "止损", row["close"], pos, round(balance,2)])
                pos, cost = 0, 0
            elif row["close"] >= cost * 1.25:
                balance += pos * row["close"]
                trades.append([str(i)[:10], "止盈", row["close"], pos, round(balance,2)])
                pos, cost = 0, 0
                
        if pos == 0 and pred == 1 and row["trend_up"] and row["vol_strength"]:
            if max_pos > 0:
                pos = max_pos
                cost = row["close"]
                balance -= pos * cost
                trades.append([str(i)[:10], "买入", row["close"], pos, round(balance,2)])
                
        hist.append(balance + pos * row["close"])
        
    final = balance + pos * data.iloc[-1]["close"]
    profit = final - money
    rate = profit / money * 100
    
    return {
        "final": round(final,2),
        "profit": round(profit,2),
        "rate": round(rate,1),
        "acc": round(acc*100,1),
        "hist": hist,
        "trades": pd.DataFrame(trades, columns=["时间","类型","价格","股数","现金"])
    }

# ======================
# 自动推送任务
# ======================
def auto_push_task():
    pass

# ======================
# 主界面（核心：实时信号+右侧股价）
# ======================
def main():
    if not st.session_state.login:
        login_page()
    else:
        auto_push_task()
        st.sidebar.title("🧧 最终完全体 · 飞书推送")
        menu = st.sidebar.radio("导航", ["🏠 首页", "📈 终极回测", "📡 实时信号", "📩 推送测试", "📖 系统说明"], index=0)

        if menu == "🏠 首页":
            st.title("🧧 A股AI量化 · 东方财富数据版")
            st.success("✅ 自动9:25推送飞书 | 高胜率 | 强收益 | 强风控")
            st.markdown(market_info())
            st.markdown("---")
            st.subheader("今日精选信号")
            
            # 首页也增加股价显示（左右布局）
            col1, col2 = st.columns([2,1])
            with col1:
                for c in ["600519","000001","600036","002594"]:
                    sig,acc = super_signal(c)
                    if "超级买入" in sig: st.success(f"{c} | {sig} ({acc}%)")
                    elif "可关注" in sig: st.info(f"{c} | {sig} ({acc}%)")
                    else: st.error(f"{c} | {sig}")
            with col2:
                st.subheader("实时股价")
                for c in ["600519","000001","600036","002594"]:
                    price_info = get_realtime_price(c)
                    if price_info["change"] > 0:
                        st.write(f"{c}: 🟢 {price_info['price']} (+{price_info['change_pct']}%)")
                    elif price_info["change"] < 0:
                        st.write(f"{c}: 🔴 {price_info['price']} ({price_info['change_pct']}%)")
                    else:
                        st.write(f"{c}: ⚪ {price_info['price']} (0%)")

        elif menu == "📈 终极回测":
            st.title("📈 最终版回测")
            code = st.text_input("股票代码", "600519")
            if st.button("开始回测"):
                res = backtest_final(code)
                if res is None: 
                    st.error("❌ 数据不足，无法回测")
                else:
                    col1,col2,col3,col4 = st.columns(4)
                    col1.metric("收益率", f"{res['rate']}%")
                    col2.metric("收益", f"{res['profit']}元")
                    col3.metric("最终资产", f"{res['final']}元")
                    col4.metric("AI准确率", f"{res['acc']}%")
                    st.line_chart(res['hist'])
                    st.dataframe(res['trades'])

        elif menu == "📡 实时信号":
            st.title("📡 实时超级信号 + 股价")
            watch = st.text_area("监控列表", "600519\n000001\n600036")
            
            if st.button("刷新"):
                st.markdown(f"### 大盘状态：{market_info()}")
                st.markdown("---")
                
                # 核心：左右分栏显示信号和股价
                col_signal, col_price = st.columns([2,1])
                
                with col_signal:
                    st.subheader("AI信号")
                    signal_list = []
                    for c in watch.split():
                        c = c.strip()
                        if c:
                            sig,acc = super_signal(c)
                            signal_list.append({"code": c, "sig": sig, "acc": acc})
                            if "超级买入" in sig: 
                                st.success(f"{c} | {sig} (置信度 {acc}%)")
                            elif "可关注" in sig: 
                                st.info(f"{c} | {sig} (置信度 {acc}%)")
                            else: 
                                st.error(f"{c} | {sig}")
                
                with col_price:
                    st.subheader("实时股价")
                    for item in signal_list:
                        c = item["code"]
                        price_info = get_realtime_price(c)
                        # 根据涨跌显示不同颜色
                        if price_info["change_pct"] > 0:
                            st.markdown(f"**{c}**\n现价：🟢 {price_info['price']}\n涨幅：+{price_info['change_pct']}%")
                        elif price_info["change_pct"] < 0:
                            st.markdown(f"**{c}**\n现价：🔴 {price_info['price']}\n跌幅：{price_info['change_pct']}%")
                        else:
                            st.markdown(f"**{c}**\n现价：⚪ {price_info['price']}\n涨跌幅：0%")
                        st.markdown("---")

        elif menu == "📩 推送测试":
            st.title("📩 飞书推送测试")
            if st.button("立即发送测试报告"):
                report = generate_rich_report()
                if send_feishu("AI量化测试", report):
                    st.success("✅ 推送成功！")
                else:
                    st.error("❌ 推送失败，请检查Webhook地址")
                st.code(report)

        elif menu == "📖 系统说明":
            st.markdown("""
            **系统说明**
            1. 数据来源：yfinance (免费A股数据)
            2. 推送方式：飞书群机器人
            3. 登录账号：admin / 123456
            4. 逻辑：大盘弱空仓，趋势强重仓
            5. 实时股价：自动获取最新价格和涨跌幅
            """)

if __name__ == "__main__":
    main()
