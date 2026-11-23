import os
import smtplib
from email.message import EmailMessage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

SYMBOL = "^GSPC"
TIMEFRAMES = ["15m", "30m", "1h", "2h", "3h", "4h", "1d"]

PERIOD_MAP = {
    "15m": "5d",
    "30m": "10d",
    "1h": "1mo",
    "2h": "2mo",
    "3h": "3mo",
    "4h": "6mo",
    "1d": "1y",
}

WICK_THRESHOLD = 0.30

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()


def fetch_ohlcv(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df = df[required_cols].dropna().copy()
    return df


def compute_obv(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    direction = np.sign(close.diff().fillna(0.0))
    obv = (direction * volume).cumsum()
    return obv


def detect_crt_and_obv_divergence(df: pd.DataFrame, wick_ratio_min: float = WICK_THRESHOLD):
    if len(df) < 6:
        return None

    obv = compute_obv(df)
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    o = float(curr["Open"])
    h = float(curr["High"])
    l = float(curr["Low"])
    c = float(curr["Close"])
    total_range = h - l

    if total_range <= 0:
        return None

    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    lower_wick_ratio = lower_wick / total_range
    upper_wick_ratio = upper_wick / total_range

    prev_high = float(prev["High"])
    prev_low = float(prev["Low"])

    # انحراف OBV على آخر نافذة قصيرة
    price_window = df["Close"].iloc[-5:]
    obv_window = obv.iloc[-5:]

    bullish_crt = (lower_wick_ratio >= wick_ratio_min) and (l <= prev_low or c > o)
    bearish_crt = (upper_wick_ratio >= wick_ratio_min) and (h >= prev_high or c < o)

    bullish_obv_div = (float(price_window.iloc[-1]) <= float(price_window.min())) and (
        float(obv_window.iloc[-1]) > float(obv_window.min())
    )
    bearish_obv_div = (float(price_window.iloc[-1]) >= float(price_window.max())) and (
        float(obv_window.iloc[-1]) < float(obv_window.max())
    )

    if bullish_crt and bullish_obv_div:
        return {
            "signal": "Bullish CRT + OBV Divergence",
            "wick_ratio": lower_wick_ratio,
            "close": c,
            "obv": float(obv.iloc[-1]),
        }

    if bearish_crt and bearish_obv_div:
        return {
            "signal": "Bearish CRT + OBV Divergence",
            "wick_ratio": upper_wick_ratio,
            "close": c,
            "obv": float(obv.iloc[-1]),
        }

    return None


def create_chart(df: pd.DataFrame, tf: str, signal_title: str, output_path: str) -> str:
    subset = df.tail(50).copy()
    obv = compute_obv(subset)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#111317")

    ax1.set_facecolor("#111317")
    ax1.plot(range(len(subset)), subset["Close"].values, color="#38bdf8", linewidth=1.8, label="Close")
    ax1.set_title(f"{SYMBOL} ({tf}) - {signal_title}", color="white", fontsize=11)
    ax1.tick_params(colors="white")
    ax1.grid(alpha=0.15)
    ax1.legend(loc="upper left")

    ax2.set_facecolor("#111317")
    ax2.plot(range(len(subset)), obv.values, color="#f59e0b", linewidth=1.5, label="OBV")
    ax2.tick_params(colors="white")
    ax2.grid(alpha=0.15)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def send_telegram_alert(text: str, chart_path: str | None = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("تنبيه: بيانات Telegram غير مكتملة، تم تخطي إرسال تيليجرام.")
        return

    try:
        if chart_path and os.path.exists(chart_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(chart_path, "rb") as photo:
                resp = requests.post(
                    url,
                    data={"chat_id": TELEGRAM_CHAT_ID, "caption": text},
                    files={"photo": photo},
                    timeout=30,
                )
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            resp = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
                timeout=20,
            )
        resp.raise_for_status()
        print("تم إرسال تنبيه Telegram بنجاح.")
    except Exception as e:
        print(f"خطأ أثناء إرسال تنبيه Telegram: {e}")


def send_email_alert(subject: str, body: str, chart_path: str | None = None):
    if not (EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECEIVER):
        print("تنبيه: بيانات البريد الإلكتروني غير مكتملة، تم تخطي إرسال الإيميل.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg.set_content(body)

        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as f:
                msg.add_attachment(
                    f.read(),
                    maintype="image",
                    subtype="png",
                    filename=os.path.basename(chart_path),
                )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print("تم إرسال تنبيه البريد الإلكتروني بنجاح.")
    except Exception as e:
        print(f"خطأ أثناء إرسال البريد الإلكتروني: {e}")


def run_monitor():
    print(f"بدء فحص {SYMBOL} على الفريمات: {', '.join(TIMEFRAMES)}")

    for tf in TIMEFRAMES:
        period = PERIOD_MAP.get(tf, "1mo")
        try:
            df = fetch_ohlcv(SYMBOL, tf, period)
            if df.empty:
                print(f"[{tf}] لا توجد بيانات متاحة حالياً.")
                continue

            result = detect_crt_and_obv_divergence(df, WICK_THRESHOLD)
            if result:
                alert_text = (
                    f"🚨 تنبيه {SYMBOL} | الفريم: {tf}\n"
                    f"الإشارة: {result['signal']}\n"
                    f"نسبة الذيل: {result['wick_ratio'] * 100:.1f}%\n"
                    f"سعر الإغلاق: {result['close']:.2f}\n"
                    f"قيمة OBV: {result['obv']:.0f}"
                )
                chart_file = f"chart_{tf}.png"
                create_chart(df, tf, result["signal"], chart_file)
                send_telegram_alert(alert_text, chart_file)
                send_email_alert(f"تنبيه {SYMBOL} ({tf}) - {result['signal']}", alert_text, chart_file)
            else:
                print(f"[{tf}] لا توجد إشارة مكتملة حالياً.")

        except Exception as e:
            print(f"[{tf}] خطأ أثناء الفحص: {e}")


if __name__ == "__main__":
    run_monitor()
