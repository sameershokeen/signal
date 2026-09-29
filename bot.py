"""
Heikin Ashi x HMA(50) Crossover -> Telegram signals only (3m + 5m)

Install:  pip install ccxt pandas numpy requests
Run:      python ha_hma_telegram_signals.py
"""

import time
import math
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt
import requests

# ============================ SETTINGS ============================
TG_BOT_TOKEN = "8392707199:AAHjWHGLoZ3Udm4rS5JlgSaPLez1qZbHMOo"     # from @BotFather
TG_CHAT_ID   = "1950462171"       # your chat / channel / group id

EXCHANGE_ID = "binance"
SYMBOL      = "SOL/USDT"
TIMEFRAMES  = ["3m", "5m"]
HMA_LEN     = 25
TP_MULT     = 5.0
CANDLES     = 300
POLL_SECS   = 10
MIN_TICK    = 0.01
# ==================================================================


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def wma(series: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    half = int(length / 2)
    root = int(round(math.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), root)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha["close"].iloc[i - 1]) / 2
    ha["open"] = ha_open
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def fetch_ohlc(exchange, tf: str) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=CANDLES)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("time", inplace=True)
    return df.iloc[:-1]  # closed candles only


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    ha = heikin_ashi(df)
    ha["hma"] = hma(ha["close"], HMA_LEN)
    ha["signal"] = (ha["close"] > ha["hma"]) & (ha["close"].shift(1) <= ha["hma"].shift(1))
    return ha


def build_message(tf: str, candle_time, row) -> str:
    entry = row["close"]
    sl = row["low"]
    risk = entry - sl
    if risk <= 0:
        risk = MIN_TICK * 10
        sl = entry - risk
    tp = entry + TP_MULT * risk
    return (
        f"🟢 <b>BUY SIGNAL</b>\n"
        f"<b>{SYMBOL}</b> | {tf} | Heikin Ashi\n"
        f"Time: {candle_time}\n\n"
        f"Entry: <code>{entry:.4f}</code>\n"
        f"SL: <code>{sl:.4f}</code>\n"
        f"TP ({TP_MULT}R): <code>{tp:.4f}</code>"
    )


def main():
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    last_seen = {tf: None for tf in TIMEFRAMES}

    while True:
        for tf in TIMEFRAMES:
            try:
                ha = analyse(fetch_ohlc(exchange, tf))
                candle_time = ha.index[-1]
                if last_seen[tf] == candle_time:
                    continue
                first_run = last_seen[tf] is None
                last_seen[tf] = candle_time
                # skip alert on very first load (old candle), only alert on new closes
                if first_run:
                    continue
                if bool(ha.iloc[-1]["signal"]):
                    send_telegram(build_message(tf, candle_time, ha.iloc[-1]))
            except Exception:
                pass
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
