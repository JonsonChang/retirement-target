# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "requests", "yfinance", "fugle-marketdata"]
# ///
"""抓取各標的近 N 年月報酬，計算年化波動率、歷史年化報酬與相關係數矩陣，另取 USD/TWD 匯率。
另抓近 12 個月配息殖利率（div_yield_ttm，台股用 yfinance .TW → .TWO），用於高登方程式、債券殖利率與股利預扣稅拖累 drag。

用法：
    uv run fetch_market.py 0050 00679B VOO VXUS --years 10

台股（純數字代號，可帶英文尾碼，如 00679B）：Fugle 還原權值月K → yfinance .TW → .TWO
其他市場：yfinance（auto_adjust）→ Yahoo chart API
Fugle 金鑰從環境變數 FUGLE_API_KEY 讀取，未設定時讀專案目錄下的 .env；都沒有就跳過 Fugle。
輸出 JSON 到 stdout，進度與錯誤訊息寫到 stderr。
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests


def eprint(*a):
    print(*a, file=sys.stderr)


def is_tw(symbol):
    return re.fullmatch(r"\d{4,6}[A-Z]?", symbol) is not None


def _retry(fn, tries=3, wait=1.5):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:  # 網路錯誤重試
            last = e
            time.sleep(wait)
    raise last


# 專案目錄 = scripts/ 往上四層（<專案>/.claude/skills/retirement-target/scripts）
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".env")


def load_env_file(path=ENV_FILE):
    """讀取專案目錄下的 .env（KEY=VALUE），不覆蓋已存在的環境變數。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def monthly_from_fugle(symbol, years):
    key = os.environ.get("FUGLE_API_KEY")
    if not key:
        raise RuntimeError("未設定 FUGLE_API_KEY")
    from fugle_marketdata import RestClient

    client = RestClient(api_key=key)
    today = dt.date.today()
    rows = []
    # Fugle 限制每次查詢區間少於一年，按年分段
    for k in range(years + 1):
        end = today - dt.timedelta(days=364 * k)
        start = end - dt.timedelta(days=363)
        try:
            resp = _retry(lambda: client.stock.historical.candles(
                symbol=symbol, timeframe="M", adjusted=True,
                **{"from": start.isoformat(), "to": end.isoformat()}))
        except Exception:
            if k > 0 and rows:
                break  # 上市前的區間 Fugle 回 404，視為沒有更早的資料
            raise
        data = (resp or {}).get("data", [])
        if not data and k > 0:
            break  # 上市前，沒有更早的資料
        rows.extend(data)
        time.sleep(0.3)
    if not rows:
        raise RuntimeError("Fugle 無資料")
    s = pd.Series({pd.Timestamp(r["date"]).to_period("M"): r["close"] for r in rows}).sort_index()
    return s


def monthly_from_yfinance(ticker, years):
    import yfinance as yf

    h = _retry(lambda: yf.Ticker(ticker).history(period=f"{years + 1}y", interval="1mo", auto_adjust=True))
    s = h["Close"].dropna()
    if len(s) < 13:
        raise RuntimeError(f"yfinance {ticker} 資料不足")
    s.index = s.index.tz_localize(None).to_period("M")
    return s[~s.index.duplicated(keep="last")]


def monthly_from_yahoo_chart(ticker, years):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={years + 1}y&interval=1mo&events=div,split")
    r = _retry(lambda: requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15))
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"].get("adjclose", [{}])[0].get("adjclose") or res["indicators"]["quote"][0]["close"]
    idx = pd.to_datetime(res["timestamp"], unit="s").to_period("M")
    s = pd.Series(closes, index=idx).dropna()
    if len(s) < 13:
        raise RuntimeError(f"Yahoo chart {ticker} 資料不足")
    return s[~s.index.duplicated(keep="last")]


def fetch_one(symbol, years):
    if is_tw(symbol):
        sources = [("fugle", lambda: monthly_from_fugle(symbol, years)),
                   ("yfinance", lambda: monthly_from_yfinance(f"{symbol}.TW", years)),
                   ("yfinance", lambda: monthly_from_yfinance(f"{symbol}.TWO", years))]
    else:
        sources = [("yfinance", lambda: monthly_from_yfinance(symbol, years)),
                   ("yahoo_chart", lambda: monthly_from_yahoo_chart(symbol, years))]
    errors = {}
    for name, fn in sources:
        try:
            s = fn()
            # 只保留最近 years 年，並去掉當月（尚未收盤）
            cur = pd.Timestamp.today().to_period("M")
            s = s[(s.index < cur) & (s.index >= cur - 12 * years)]
            return name, s, errors
        except Exception as e:
            errors[f"{name}#{len(errors)}"] = str(e)
            eprint(f"[{symbol}] {name} 失敗: {e}")
    return None, None, errors


def div_yield_yfinance(ticker):
    import yfinance as yf

    h = _retry(lambda: yf.Ticker(ticker).history(period="13mo", interval="1d", auto_adjust=False))
    d = h["Dividends"][h.index > h.index[-1] - pd.Timedelta(days=365)]
    return float(d.sum() / h["Close"].iloc[-1]), int((d > 0).sum())


def div_yield_yahoo_chart(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1y&interval=1d&events=div"
    r = _retry(lambda: requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15))
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    divs = [v["amount"] for v in res.get("events", {}).get("dividends", {}).values()]
    close = [c for c in res["indicators"]["quote"][0]["close"] if c is not None][-1]
    return float(sum(divs) / close), len(divs)


def div_yield(tickers):
    """近 12 個月配息 ÷ 最新收盤價（未還原）。依序嘗試各 ticker，回傳 (殖利率, 配息次數)，失敗回傳 (None, 0)。"""
    for t in tickers:
        for name, fn in (("yfinance", div_yield_yfinance), ("yahoo_chart", div_yield_yahoo_chart)):
            try:
                return fn(t)
            except Exception as e:
                eprint(f"[{t}] 殖利率 {name} 失敗: {e}")
    return None, 0


def usd_twd():
    try:
        d = requests.get("https://open.er-api.com/v6/latest/USD", timeout=15).json()
        return d["rates"]["TWD"] if d.get("result") == "success" else None
    except Exception as e:
        eprint(f"匯率取得失敗: {e}")
        return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    load_env_file()
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--years", type=int, default=10)
    args = ap.parse_args()

    out = {"as_of": dt.date.today().isoformat(), "years_requested": args.years,
           "usd_twd": usd_twd(), "assets": {}, "failed": {}}
    rets = {}
    for sym in args.symbols:
        eprint(f"抓取 {sym} ...")
        src, s, errors = fetch_one(sym, args.years)
        if s is None:
            out["failed"][sym] = errors
            continue
        r = np.log(s).diff().dropna()
        rets[sym] = r
        out["assets"][sym] = {
            "source": src,
            "months": int(len(r)),
            "from": str(r.index[0]),
            "to": str(r.index[-1]),
            # 對數月報酬年化
            "vol_annual": float(r.std(ddof=1) * np.sqrt(12)),
            "hist_return_annual": float(np.exp(r.mean() * 12) - 1),
        }
        y, n = div_yield([f"{sym}.TW", f"{sym}.TWO"] if is_tw(sym) else [sym])
        out["assets"][sym].update(div_yield_ttm=y, div_count_ttm=n)

    if rets:
        df = pd.DataFrame(rets)
        corr = df.corr(min_periods=24).fillna(0.0)
        m = corr.values.copy()
        np.fill_diagonal(m, 1.0)
        corr = pd.DataFrame(m, index=corr.index, columns=corr.columns)
        out["correlation"] = {"symbols": list(corr.columns), "matrix": corr.round(4).values.tolist()}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
