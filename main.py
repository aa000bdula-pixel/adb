import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import requests
import yfinance as yf

# ==========================================
# 1. الإعدادات والمتغيرات
# ==========================================
SYMBOL = "^GSPC"  # مؤشر S&P 500 على Yahoo Finance
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL") or SENDER_EMAIL

# الأطر الزمنية المعتمدة (1h, 2h, 3h, 4h, 1d)
TIMEFRAMES = {
    "1h": {"interval": "1h", "period": "15d", "resample": None},
    "2h": {"interval": "1h", "period": "30d", "resample": "2h"},
    "3h": {"interval": "1h", "period": "30d", "resample": "3h"},
    "4h": {"interval": "1h", "period": "60d", "resample": "4h"},
    "1d": {"interval": "1d", "period": "100d", "resample": None},
}


# ==========================================
# 2. التحقق من وقت التشغيل (توقيت السعودية)
# ==========================================
def is_market_open():
  tz = pytz.timezone("Asia/Riyadh")
  now = datetime.now(tz)
  if now.weekday() in [5, 6]:
    return False
  current_minutes = now.hour * 60 + now.minute
  # من 02:00 ص (120 دقيقة) حتى 12:30 ص (30 دقيقة)
  if current_minutes >= 120 or current_minutes <= 30:
    return True
  return False


# ==========================================
# 3. جلب البيانات وتحديث السعر اللحظي
# ==========================================
def fetch_data(tf_key, config):
  ticker = yf.Ticker(SYMBOL)
  df = ticker.history(
      period=config["period"], interval=config["interval"], prepost=True
  )

  if df.empty:
    return pd.DataFrame()

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

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

  # تحديث آخر شمعة بالسعر اللحظي المباشر من فريم الدقيقة
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
# 4. حساب OBV والدايفرجنس
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
# 5. تحليل CRT وقوة الذيل
# ==========================================
def analyze_crt(df):
  if len(df) < 5:
    return None

  prev_candle = df.iloc[-2]
  curr_candle = df.iloc[-1]

  high_curr = curr_candle["High"]
  low_curr = curr_candle["Low"]
  open_curr = curr_candle["Open"]
  close_curr = curr_candle["Close"]

  candle_range = high_curr - low_curr
  if candle_range == 0:
    return None

  signal = None
  wick_ratio = 0.0

  if low_curr < prev_candle["Low"] and close_curr > prev_candle["Low"]:
    lower_wick = min(open_curr, close_curr) - low_curr
    wick_ratio = round((lower_wick / candle_range) * 100, 1)
    if wick_ratio >= 30:
      signal = "CRT إيجابي 🟢"

  elif high_curr > prev_candle["High"] and close_curr < prev_candle["High"]:
    upper_wick = high_curr - max(open_curr, close_curr)
    wick_ratio = round((upper_wick / candle_range) * 100, 1)
    if wick_ratio >= 30:
      signal = "CRT سلبي 🔴"

  if not signal:
    return None

  return {
      "signal": signal,
      "price": round(close_curr, 2),
      "wick_ratio": wick_ratio,
      "divergence": analyze_divergence(df),
  }


# ==========================================
# 6. رسم الشارت
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
# 7. إرسال التنبيهات (تيليجرام + إيميل)
# ==========================================
def send_telegram_alert(message, image_path):
  if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("تنبيه: بيانات تيليجرام غير مكتملة.")
    return
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
  try:
    with open(image_path, "rb") as img:
      resp = requests.post(
          url,
          data={"chat_id": TELEGRAM_CHAT_ID, "caption": message},
          files={"photo": img},
          timeout=20,
      )
    if resp.ok:
      print("تم إرسال تنبيه تيليجرام بنجاح.")
    else:
      print(f"فشل إرسال تيليجرام: {resp.text}")
  except Exception as e:
    print(f"خطأ في إرسال تيليجرام: {e}")


def send_email_alert(subject, message, image_path):
  if not SENDER_EMAIL or not APP_PASSWORD or not RECEIVER_EMAIL:
    print("تنبيه: بيانات الإيميل غير مكتملة.")
    return
  try:
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(message, "plain", "utf-8"))

    with open(image_path, "rb") as f:
      img = MIMEImage(f.read())
      img.add_header(
          "Content-Disposition",
          "attachment",
          filename=os.path.basename(image_path),
      )
      msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
      server.login(SENDER_EMAIL, APP_PASSWORD)
      server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("تم إرسال الإيميل بنجاح.")
  except Exception as e:
    print(f"خطأ في إرسال الإيميل: {e}")


# ==========================================
# 8. تشغيل المونيتور
# ==========================================
def run_monitor():
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
            f"SPX500 {result['signal']} ({tf_key})", alert_text, chart_file
        )
        print(f"تم إرسال التنبيهات بنجاح لفريم {tf_key}")
      else:
        print(f"لا توجد إشارة CRT على فريم {tf_key}")

    except Exception as e:
      print(f"خطأ في فريم {tf_key}: {e}")


if __name__ == "__main__":
  run_monitor()
