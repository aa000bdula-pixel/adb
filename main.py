import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, time
import pytz
import yfinance as yf

# إعدادات التنبيهات
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
APP_PASSWORD = os.environ.get("APP_PASSWORD")


def is_within_trading_hours():
    saudi_tz = pytz.timezone("Asia/Riyadh")
    now_time = datetime.now(saudi_tz).time()
    start_time = time(2, 0)   # 2:00 AM
    end_time = time(0, 30)    # 12:30 AM (اليوم التالي)
    return now_time >= start_time or now_time <= end_time


def fetch_market_data():
    if not is_within_trading_hours():
        print("خارج أوقات التداول المحددة (02:00 ص - 12:30 ص بتوقيت السعودية).")
        return pd.DataFrame()

    # جلب بيانات مؤشر S&P 500 مجاناً عبر yfinance
    df = yf.download("^GSPC", period="5d", interval="5m", progress=False)

    if df is None or df.empty:
        raise RuntimeError("فشل جلب البيانات من Yahoo Finance: البيانات فارغة.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.rename(columns={
        "Datetime": "datetime",
        "Date": "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }, inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def calculate_obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def find_fvgs(df):
    fvgs = []
    for i in range(2, len(df)):
        if df["high"].iloc[i - 2] < df["low"].iloc[i]:
            fvgs.append(("bullish_fvg", df["high"].iloc[i - 2], df["low"].iloc[i]))
        elif df["low"].iloc[i - 2] > df["high"].iloc[i]:
            fvgs.append(("bearish_fvg", df["high"].iloc[i], df["low"].iloc[i - 2]))
    return fvgs


def detect_crt_and_fvg(df):
    if len(df) < 10:
        return None

    df = df.copy()
    df["obv"] = calculate_obv(df)

    c_prev = df.iloc[-2]
    c_curr = df.iloc[-1]

    candle_range = c_curr["high"] - c_curr["low"]
    if candle_range <= 0:
        return None

    upper_wick = c_curr["high"] - max(c_curr["open"], c_curr["close"])
    lower_wick = min(c_curr["open"], c_curr["close"]) - c_curr["low"]

    bearish_div = (c_curr["high"] > df["high"].iloc[-3]) and (
        df["obv"].iloc[-1] < df["obv"].iloc[-3]
    )
    bullish_div = (c_curr["low"] < df["low"].iloc[-3]) and (
        df["obv"].iloc[-1] > df["obv"].iloc[-3]
    )

    fvgs = find_fvgs(df)
    zone_label = "نطاق عادي"
    for ftype, z_low, z_high in fvgs[-5:]:
        if z_low <= c_curr["high"] and c_curr["low"] <= z_high:
            zone_label = f"FVG ({z_low:.2f} - {z_high:.2f})"
            break

    swept_high = (
        c_curr["high"] > c_prev["high"]
        and c_prev["low"] <= c_curr["close"] <= c_prev["high"]
        and c_curr["close"] < c_curr["open"]
    )
    swept_low = (
        c_curr["low"] < c_prev["low"]
        and c_prev["low"] <= c_curr["close"] <= c_prev["high"]
        and c_curr["close"] > c_curr["open"]
    )

    saudi_tz = pytz.timezone("Asia/Riyadh")
    dt_val = c_curr["datetime"]
    if dt_val.tzinfo is None:
        dt_val = pytz.utc.localize(dt_val).astimezone(saudi_tz)
    else:
        dt_val = dt_val.astimezone(saudi_tz)

    if swept_high:
        prob = int(min(95, max(40, round((upper_wick / candle_range) * 100))))
        return {
            "signal": "CRT سلبي 🔴",
            "prob": prob,
            "price": float(c_curr["close"]),
            "div": bearish_div,
            "zone": zone_label,
            "time": dt_val.strftime("%I:%M %p"),
        }

    if swept_low:
        prob = int(min(95, max(40, round((lower_wick / candle_range) * 100))))
        return {
            "signal": "CRT إيجابي 🟢",
            "prob": prob,
            "price": float(c_curr["close"]),
            "div": bullish_div,
            "zone": zone_label,
            "time": dt_val.strftime("%I:%M %p"),
        }

    return None


def generate_chart(df, setup):
    sub = df.tail(10).reset_index(drop=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, row in sub.iterrows():
        color = "#2b9980" if row["close"] >= row["open"] else "#d9383a"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1.2)
        body_low = min(row["open"], row["close"])
        body_h = max(abs(row["close"] - row["open"]), 1e-4)
        rect = Rectangle(
            (i - 0.3, body_low), 0.6, body_h, facecolor=color, edgecolor=color
        )
        ax.add_patch(rect)

    ax.set_title(
        f"S&P 500 | {setup['signal']} | Prob: {setup['prob']}%", color="white"
    )
    plt.tight_layout()
    plt.savefig("chart.png", dpi=150)
    plt.close()


def send_telegram_alert(setup):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("إعدادات تيليجرام غير مكتملة.")
        return

    div_text = "نعم ✅" if setup.get("div") else "لا ❌"
    caption = (
        f"🚨 {setup['signal']}\n"
        f"💰 السعر: {setup['price']:.2f}\n"
        f"📊 المنطقة: {setup['zone']}\n"
        f"📉 دايفرجنس OBV: {div_text}\n"
        f"🔥 الاحتمالية: {setup['prob']}%\n"
        f"🕒 الوقت: {setup['time']}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open("chart.png", "rb") as photo:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
            files={"photo": photo},
            timeout=20,
        )
        response.raise_for_status()


if __name__ == "__main__":
    df = fetch_market_data()
    if not df.empty:
        setup = detect_crt_and_fvg(df)
        if setup:
            generate_chart(df, setup)
            send_telegram_alert(setup)
            print("تم إرسال التنبيه بنجاح.")
        else:
            print("لا توجد إشارة مطابقة حالياً.")
