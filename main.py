import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime

# إعدادات المتغيرات السرية من GitHub Secrets
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# إعدادات المؤشر والفريم الزمني
SYMBOL = "SPX"       # رمز مؤشر S&P 500 في Twelve Data
INTERVAL = "1h"      # الفريم الزمني (1h للساعة، 15min لربع ساعة، 5min لخمس دقائق)
OUTPUT_SIZE = 50     # عدد الشموع المطلوب جلبها للتحليل


def fetch_market_data(symbol=SYMBOL, interval=INTERVAL, outputsize=OUTPUT_SIZE):
    """جلب البيانات اللحظية من Twelve Data API"""
    if not TWELVE_DATA_API_KEY:
        raise ValueError("مفتاح TWELVE_DATA_API_KEY غير موجود في إعدادات Secrets!")

    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"خطأ في جلب البيانات من Twelve Data: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # تحويل الأعمدة السعرية إلى أرقام عشرية
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    return df


def detect_crt_and_fvg(df):
    """
    تحليل آخر الشموع لاكتشاف:
    1. سحب السيولة (CRT - Candle Range Theory / Liquidity Sweep)
    2. فجوة القيمة العادلة (FVG - Fair Value Gap)
    3. حساب نسبة الاحتمالية بناءً على طول الذيل مقارنة بالجسم
    """
    if len(df) < 5:
        return None

    # الشموع الأخيرة (نستثني الشمعة الحالية غير المغلقة [-1] ونفحص آخر شمعة مغلقة [-2] وما قبلها)
    c_prev2 = df.iloc[-4]  # شمعة تكوين الفجوة الأولى
    c_prev1 = df.iloc[-3]  # شمعة المرجع / السيولة
    c_signal = df.iloc[-2] # شمعة التأكيد / سحب السيولة

    signal_type = None
    fvg_zone = None
    swept_level = None

    # --- سيناريو الشراء (Bullish CRT + Bullish FVG) ---
    # كسر قاع الشمعة السابقة ثم الإغلاق فوقه (سحب سيولة سفلي)
    bullish_sweep = (c_signal["low"] < c_prev1["low"]) and (c_signal["close"] > c_prev1["low"])
    # وجود فجوة صاعدة بين قمة الشمعة [-4] وقاع الشمعة [-2]
    bullish_fvg = c_signal["low"] > c_prev2["high"]

    # --- سيناريو البيع (Bearish CRT + Bearish FVG) ---
    # كسر قمة الشمعة السابقة ثم الإغلاق تحتها (سحب سيولة علوي)
    bearish_sweep = (c_signal["high"] > c_prev1["high"]) and (c_signal["close"] < c_prev1["high"])
    # وجود فجوة هابطة بين قاع الشمعة [-4] وقمة الشمعة [-2]
    bearish_fvg = c_signal["high"] < c_prev2["low"]

    # حساب قوة الشمعة (نسبة الذيل إلى الجسم)
    body = abs(c_signal["close"] - c_signal["open"])
    total_range = c_signal["high"] - c_signal["low"]
    if total_range == 0:
        return None

    if bullish_sweep:
        signal_type = "شراء (Bullish CRT)"
        swept_level = c_prev1["low"]
        lower_wick = min(c_signal["open"], c_signal["close"]) - c_signal["low"]
        wick_ratio = lower_wick / total_range
        if bullish_fvg:
            fvg_zone = (c_prev2["high"], c_signal["low"])

    elif bearish_sweep:
        signal_type = "بيع (Bearish CRT)"
        swept_level = c_prev1["high"]
        upper_wick = c_signal["high"] - max(c_signal["open"], c_signal["close"])
        wick_ratio = upper_wick / total_range
        if bearish_fvg:
            fvg_zone = (c_signal["high"], c_prev2["low"])
    else:
        return None

    # حساب الاحتمالية التقديرية
    probability = min(95, int(55 + (wick_ratio * 35) + (10 if fvg_zone else 0)))

    return {
        "signal": signal_type,
        "swept_level": swept_level,
        "fvg_zone": fvg_zone,
        "probability": probability,
        "price": c_signal["close"],
        "time": c_signal["datetime"].strftime("%Y-%m-%d %H:%M")
    }


def generate_chart(df, result, filename="spx_signal.png"):
    """رسم آخر 6 شموع مع توضيح خط سحب السيولة ومنطقة الفجوة FVG"""
    plot_df = df.tail(6).reset_index(drop=True)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, row in plot_df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        # رسم الذيل (الفتيل)
        ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1.8)
        # رسم جسم الشمعة
        lower = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), 0.15)
        rect = Rectangle((idx - 0.3, lower), 0.6, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # رسم خط مستوى السيولة المسحوبة (CRT)
    ax.axhline(
        result["swept_level"],
        color="#ffca28",
        linestyle="--",
        linewidth=1.5,
        label=f"CRT Liquidity Sweep: {result['swept_level']:.2f}"
    )

    # رسم منطقة الفجوة السعرية (FVG) إن وجدت
    if result["fvg_zone"]:
        low_fvg, high_fvg = result["fvg_zone"]
        ax.axhspan(
            low_fvg,
            high_fvg,
            color="#42a5f5",
            alpha=0.25,
            label=f"FVG Zone: {low_fvg:.2f} - {high_fvg:.2f}"
        )

    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["datetime"].dt.strftime("%H:%M"), fontsize=9)
    ax.set_title(
        f"S&P 500 ({INTERVAL}) | {result['signal']} | Prob: {result['probability']}%",
        fontsize=12,
        fontweight="bold",
        pad=12
    )
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def send_telegram_alert(image_path, result):
    """إرسال صورة الشارت مع تفاصيل الإشارة إلى تليجرام"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("بيانات تليجرام غير مكتملة، تم تخطي الإرسال.")
        return

    fvg_text = (
        f"{result['fvg_zone'][0]:.2f} - {result['fvg_zone'][1]:.2f}"
        if result["fvg_zone"]
        else "لا توجد فجوة مكتملة"
    )

    caption = (
        f"🚨 **تنبيه S&P 500 ({INTERVAL})**\n\n"
        f"📌 **الاتجاه:** {result['signal']}\n"
        f"💰 **سعر الإغلاق:** {result['price']:.2f}\n"
        f"🎯 **مستوى السيولة (CRT):** {result['swept_level']:.2f}\n"
        f"📊 **منطقة الفجوة (FVG):** {fvg_text}\n"
        f"🔥 **نسبة الاحتمالية:** {result['probability']}%\n"
        f"🕒 **الوقت:** {result['time']}"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        files = {"photo": photo}
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption,
            "parse_mode": "Markdown"
        }
        res = requests.post(url, data=data, files=files, timeout=15)
        print("حالة إرسال تليجرام:", res.status_code, res.text)


if __name__ == "__main__":
    print("جاري جلب البيانات اللحظية لـ S&P 500...")
    df = fetch_market_data()
    print(f"تم جلب {len(df)} شمعة بنجاح. آخر سعر إغلاق: {df.iloc[-1]['close']}")

    signal_result = detect_crt_and_fvg(df)
    if signal_result:
        print("تم رصد إشارة:", signal_result)
        chart_file = generate_chart(df, signal_result)
        send_telegram_alert(chart_file, signal_result)
    else:
        print("لا توجد إشارة CRT مكتملة في الشمعة الأخيرة حالياً.")
