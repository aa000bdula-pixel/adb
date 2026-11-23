import io
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf

# ---------------- إعدادات الرمز والفواصل الزمنية ----------------
SYMBOL = "^GSPC"
MASSIVE_TICKER = "I:SPX"  # رمز مؤشر S&P 500 في Massive/Polygon عند توفر الخطة المناسبة
TIMEFRAMES = ["1m", "5m", "15m"]  # تجنب 3m مع yfinance لأنها غير مدعومة

TZ = pytz.timezone("Asia/Riyadh")
ENFORCE_TIME_WINDOW = False  # اجعلها True لتقييد العمل بنافذة زمنية محددة
WINDOW_START_HOUR = 17
WINDOW_START_MINUTE = 30
WINDOW_END_HOUR = 23
WINDOW_END_MINUTE = 59

# ---------------- إعدادات التنبيهات ----------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "").strip()


def is_within_execution_window(now: datetime | None = None) -> bool:
    if not ENFORCE_TIME_WINDOW:
        return True
    current = now.astimezone(TZ) if now else datetime.now(TZ)
    start = current.replace(
        hour=WINDOW_START_HOUR, minute=WINDOW_START_MINUTE, second=0, microsecond=0
    )
    end = current.replace(
        hour=WINDOW_END_HOUR, minute=WINDOW_END_MINUTE, second=59, microsecond=0
    )
    if end >= start:
        return start <= current <= end
    return current >= start or current <= end


def _interval_to_yf_period(interval: str) -> str:
    if interval == "1m":
        return "5d"
    if interval in {"2m", "5m", "15m", "30m"}:
        return "30d"
    return "60d"


def _fetch_from_massive(interval: str) -> pd.DataFrame:
    if not MASSIVE_API_KEY:
        return pd.DataFrame()

    multiplier_map = {
        "1m": (1, "minute"),
        "3m": (3, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "1h": (1, "hour"),
        "60m": (1, "hour"),
    }
    if interval not in multiplier_map:
        return pd.DataFrame()

    multiplier, timespan = multiplier_map[interval]
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=7)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{MASSIVE_TICKER}/range/"
        f"{multiplier}/{timespan}/{start_dt.strftime('%Y-%m-%d')}/{end_dt.strftime('%Y-%m-%d')}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": MASSIVE_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(TZ)
        df = df.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
            }
        )
        df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as exc:
        print(f"Massive fetch error ({MASSIVE_TICKER} {interval}): {exc}")
        return pd.DataFrame()


def _fetch_from_yfinance(symbol: str, interval: str) -> pd.DataFrame:
    if interval == "3m":
        print(f"Skipping unsupported interval '3m' for yfinance on {symbol}")
        return pd.DataFrame()

    try:
        df = yf.download(
            tickers=symbol,
            period=_interval_to_yf_period(interval),
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(TZ)
        else:
            df.index = df.index.tz_convert(TZ)
        return df
    except Exception as exc:
        print(f"yfinance fetch error ({symbol} {interval}): {exc}")
        return pd.DataFrame()


def fetch_ohlc(symbol: str, interval: str) -> pd.DataFrame:
    df = _fetch_from_massive(interval)
    if not df.empty:
        return df
    return _fetch_from_yfinance(symbol, interval)


def detect_crt_and_obv_divergence(df: pd.DataFrame) -> dict | None:
    """مثال أساسي لفحص شمعة داخل نطاق السابقة (CRT) مع انحراف OBV."""
    if df is None or len(df) < 10:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float).fillna(0)

    direction = np.where(close > close.shift(1), 1.0, np.where(close < close.shift(1), -1.0, 0.0))
    obv = (direction * volume).cumsum()

    prev_high, prev_low = high.iloc[-3], low.iloc[-3]
    curr_high, curr_low = high.iloc[-2], low.iloc[-2]
    curr_close = close.iloc[-2]
    curr_open = df["Open"].astype(float).iloc[-2]

    inside_bar = (curr_high <= prev_high) and (curr_low >= prev_low)
    bullish_crt = inside_bar and (curr_close > curr_open) and (obv.iloc[-2] > obv.iloc[-5])
    bearish_crt = inside_bar and (curr_close < curr_open) and (obv.iloc[-2] < obv.iloc[-5])

    ts_str = df.index[-2].strftime("%Y-%m-%d %H:%M %Z")
    if bullish_crt:
        return {
            "signal": "BULLISH CRT + OBV",
            "price": round(float(curr_close), 2),
            "timestamp": ts_str,
        }
    if bearish_crt:
        return {
            "signal": "BEARISH CRT + OBV",
            "price": round(float(curr_close), 2),
            "timestamp": ts_str,
        }
    return None


def build_chart_png(df: pd.DataFrame, interval: str, alert: dict) -> bytes | None:
    try:
        recent = df.tail(40)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(recent.index, recent["Close"], label="Close", linewidth=1.5)
        ax.set_title(f"{SYMBOL} ({interval}) - {alert['signal']}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        print(f"Chart build error: {exc}")
        return None


def send_telegram_alert(text: str, chart_png: bytes | None = None) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; skipping Telegram alert.")
        return

    try:
        if chart_png:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {"photo": ("chart.png", chart_png, "image/png")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
            resp = requests.post(url, data=data, files=files, timeout=20)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
            resp = requests.post(url, data=data, timeout=20)
        resp.raise_for_status()
        print("Telegram alert sent.")
    except Exception as exc:
        print(f"Telegram send error: {exc}")


def send_email_alert(subject: str, body: str, chart_png: bytes | None = None) -> None:
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email credentials missing; skipping Email alert.")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if chart_png:
            image_part = MIMEImage(chart_png, name="chart.png")
            image_part.add_header("Content-Disposition", "attachment", filename="chart.png")
            msg.attach(image_part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent.")
    except Exception as exc:
        print(f"Email send error: {exc}")


def run_scanner() -> None:
    now_local = datetime.now(TZ)
    if not is_within_execution_window(now_local):
        print(f"Outside execution window at {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return

    for interval in TIMEFRAMES:
        df = fetch_ohlc(SYMBOL, interval)
        if df.empty:
            print(f"No data returned for {SYMBOL} ({interval})")
            continue

        alert = detect_crt_and_obv_divergence(df)
        if not alert:
            print(f"No signal on {SYMBOL} ({interval})")
            continue

        chart_png = build_chart_png(df, interval, alert)
        text = (
            f"Alert: {alert['signal']}\n"
            f"Symbol: {SYMBOL}\n"
            f"Interval: {interval}\n"
            f"Price: {alert['price']}\n"
            f"Time: {alert['timestamp']}"
        )
        send_telegram_alert(text, chart_png)
        send_email_alert(f"{alert['signal']} - {SYMBOL} ({interval})", text, chart_png)
        print(f"Alert processed for {SYMBOL} ({interval}): {alert['signal']}")


if __name__ == "__main__":
    run_scanner()
