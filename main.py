import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ==========================================
# 1. إعدادات المتغيرات والرموز
# ==========================================
SYMBOL = "^GSPC"  # رمز مؤشر S&P 500 على Yahoo Finance
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("SENDER_EMAIL")  # يتم الإرسال لنفس الإيميل

# الأطر الزمنية المطلوبة (ساعة، ساعتين، 3 ساعات، 4 ساعات، يوم)
TIMEFRAMES = {
    "1h": {"interval": "1h", "period": "15d", "resample": None},
    "2h": {"interval": "1h", "period": "30d", "resample": "2h"},
    "3h": {"interval": "1h", "period": "30d", "resample": "3h"},
    "4h": {"interval": "1h", "period": "60d", "resample": "4h"},
    "1d": {"interval": "1d", "period": "100d", "resample": None},
}


# ==========================================
# 2. التحقق من ساعات التداول (توقيت السعودية)
# ==========================================
def is_market_open():
    tz = pytz.timezone("Asia/Riyadh")
    now = datetime.now(tz)
    # إغلاق عطلة نهاية الأسبوع (السبت والأحد)
    if now.weekday() in [5, 6]:
        return False
    current_minutes = now.hour * 60 + now.minute
    # نافذة العمل: من 02:00 ص (120 دقيقة) حتى 12:30 ص (30 دقيقة)
    if current_minutes >= 120 or current_minutes <= 30:
        return True
    return False


# ==========================================
# 3. جلب البيانات وتجميع الفريمات بدقة لحظية
# ==========================================
def fetch_data(tf_key, config):
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(
        period=config["period"], interval=config["interval"], prepost=True
    )

    if df.empty:
        return pd.DataFrame()

    # توحيد أسماء الأعمدة وإزالة مستويات MultiIndex إن وجدت
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # تجميع الفريمات (2h, 3h, 4h) من شموع الساعة
    if config["resample"]:
        df = (
            df.resample(config["resample"])
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            })
            .dropna()
        )

    # جلب السعر اللحظي المباشر من إطار الدقيقة لتحديث آخر شمعة
    live_df = ticker.history(period="1d", interval="1m")
    if not live_df.empty:
        latest_price = float(live_df["Close"].iloc[-1])
        df.iloc[-1, df.columns.get_loc("Close")] = latest_price
        if latest_price > df["High"].iloc[-1]:
            df.iloc[-1, df.columns.get_loc("High")] = latest_price
        if latest_price < df["Low"].iloc[-1]:
            df.iloc[-1, df.columns.get_loc("Low")] = latest_price

    return df


# ==========================================
# 4. حساب مؤشر OBV والدايفرجنس
# ==========================================
def analyze_divergence(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    recent = df.tail(10)
    price_diff = recent["Close"].iloc[-1] - recent["Close"].iloc[0]
    obv_diff = recent["OBV"].iloc[-1] - recent["OBV"].iloc[0]

    if price_diff < 0 and obv_diff > 0:
        return "دايفر إيجابي"
    elif price_diff > 0 and obv_diff < 0:
        return "دايفر سلبي"
    elif obv_diff >= 0:
        return "دايفر إيجابي"
    else:
        return "دايفر سلبي"


# ==========================================
# 5. تحليل شمعة CRT وقوة الذيل
# ==========================================
def analyze_crt(df):
    if len(df) < 5:
        return None

    prev_candle = df.iloc[-2]  # الشمعة المرجعية
    curr_candle = df.iloc[-1]  # الشمعة الحالية / الأخيرة

    high_curr = curr_candle["High"]
    low_curr = curr_candle["Low"]
    open_curr = curr_candle["Open"]
    close_curr = curr_candle["Close"]

    candle_range = high_curr - low_curr
    if candle_range == 0:
        return None

    signal = None
    wick_ratio = 0.0

    # CRT إيجابي: كسر قاع الشمعة السابقة مع ذيل سفلي قوي وإغلاق أعلى القاع
    if low_curr < prev_candle["Low"] and close_curr > prev_candle["Low"]:
        lower_wick = min(open_curr, close_curr) - low_curr
        wick_ratio = round((lower_wick / candle_range) * 100, 1)
        if wick_ratio >= 30:  # فلتر قوة الذيل
            signal = "CRT إيجابي 🟢"

    # CRT سلبي: كسر قمة الشمعة السابقة مع ذيل علوي قوي وإغلاق أسفل القمة
    elif high_curr > prev_candle["High"] and close_curr < prev_candle["High"]:
        upper_wick = high_curr - max(open_curr, close_curr)
        wick_ratio = round((upper_wick / candle_range) * 100, 1)
        if wick_ratio >= 30:
            signal = "CRT سلبي 🔴"

    if not signal:
        return None

    div_status = analyze_divergence(df)

    return {
        "signal": signal,
        "price": round(close_curr, 2),
        "wick_ratio": wick_ratio,
        "divergence": div_status,
    }


# ==========================================
# 6. رسم الشارت للشموع الأخيرة
# ==========================================
def generate_chart(df, tf_key, signal_text, filename="chart.png"):
    plot_df = df.tail(12).reset_index(drop=True)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, row in plot_df.iterrows():
        color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
        ax.plot([i, i], [row["Low"], row["High"]], color=color, linewidth=1.5)
        bottom = min(row["Open"], row["Close"])
        height = max(abs(row["Close"] - row["Open"]), 0.2)
        ax.bar(i, height, bottom=bottom, width=0.55, color=color)

    ax.set_title(f"S&P 500 ({tf_key}) | {signal_text}", fontsize=11, pad=10)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ==========================================
# 7. إرسال التنبيه إلى تيليجرام والإيميل
# ==========================================
def send_telegram_alert(message, image_path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
            files={"photo": img},
            timeout=15,
        )


def send_email_alert(subject, message, image_path):
    if not SENDER_EMAIL or not APP_PASSWORD:
        return
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    msg.attach(MIMEText(message, "plain", "utf-8"))

    with open(image_path, "rb") as f:
        img = MIMEImage(f.read())
        img.add_header("Content-Disposition", "attachment", filename="chart.png")
        msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())


# ==========================================
# 8. التشغيل الرئيسي
# ==========================================
def main():
    # ملاحظة: عند التشغيل اليدوي عبر workflow_dispatch سيعمل دائماً
    if os.getenv("GITHUB_EVENT_NAME") == "schedule" and not is_market_open():
        print("خارج أوقات التداول المحددة.")
        return

    tz = pytz.timezone("Asia/Riyadh")
    now_str = datetime.now(tz).strftime("%I:%M %p")

    for tf_key, config in TIMEFRAMES.items():
        try:
            df = fetch_data(tf_key, config)
            if df.empty:
                continue

            result = analyze_crt(df)
            if result:
                chart_file = f"chart_{tf_key}.png"
                generate_chart(df, tf_key, result["signal"], chart_file)

                # تنسيق الرسالة المطلوب بدقة
                alert_text = (
                    f"{result['signal']}\n"
                    f"الفريم: {tf_key}\n"
                    f"السعر: {result['price']}\n"
                    f"قوة الذيل: {result['wick_ratio']}%\n"
                    f"{result['divergence']}\n"
                    f"الوقت: {now_str}"
                )

                send_telegram_alert(alert_text, chart_file)
                send_email_alert(
                    f"SPX500 {result['signal']} ({tf_key})",
                    alert_text,
                    chart_file,
                )
                print(f"تم إرسال التنبيه بنجاح لفريم {tf_key}")
            else:
                print(f"لا توجد إشارة CRT على فريم {tf_key}")

        except Exception as e:
            print(f"خطأ في فريم {tf_key}: {e}")


if __name__ == "__main__":
    main()
