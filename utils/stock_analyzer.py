"""
Stock Analysis Utility Module

Fetches stock data from Finnhub (primary) → Alpha Vantage → Yahoo Finance
fallback chain, and computes technical indicators and analytics.
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Log to stderr so Render captures it in the deploy logs."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Finnhub API (primary — 60 calls/min free tier)
# ---------------------------------------------------------------------------

_FH_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()


def _has_fh() -> bool:
    """Whether Finnhub API key is configured."""
    return bool(_FH_API_KEY)


def _fh_request(path: str) -> dict | None:
    """Make a request to Finnhub and return parsed JSON."""
    url = f"https://finnhub.io/api/v1{path}"
    try:
        r = requests.get(url, params={"token": _FH_API_KEY}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("error"):
            _log(f"[FH] API error: {data['error']}")
            return None
        return data
    except Exception as e:
        _log(f"[FH] Request failed: {type(e).__name__}: {e}")
        return None


def _fh_get_quote(ticker: str) -> dict | None:
    """Fetch current quote from Finnhub."""
    if not _has_fh():
        return None
    _log(f"[FH] quote for {ticker}")
    data = _fh_request(f"/quote?symbol={ticker}")
    if not data:
        return None
    # Finnhub quote: {c, d, dp, h, l, o, pc, t}
    price = data.get("c")
    if price is None or price == 0:
        return None
    try:
        prev_close = data.get("pc", price)
        return {
            "price": float(price),
            "prev_close": float(prev_close) if prev_close else float(price),
            "open": float(data.get("o", price)),
            "day_high": float(data.get("h", price)),
            "day_low": float(data.get("l", price)),
            "volume": 0,  # Finnhub free quote doesn't include volume
        }
    except (ValueError, TypeError) as e:
        _log(f"[FH] Parse error for {ticker}: {e}")
        return None


def _fh_get_profile(ticker: str) -> dict:
    """Fetch company profile from Finnhub (limited metadata)."""
    if not _has_fh():
        return {}
    _log(f"[FH] profile2 for {ticker}")
    data = _fh_request(f"/stock/profile2?symbol={ticker}")
    if not data or "name" not in data:
        return {}
    # Convert market cap from millions to actual
    market_cap_m = data.get("marketCapitalization")
    market_cap = float(market_cap_m) * 1_000_000 if market_cap_m else None
    return {
        "name": data.get("name", ticker.upper()),
        "industry": data.get("finnhubIndustry", "N/A"),
        "market_cap": market_cap,
    }


def _fh_get_candles(ticker: str, months: int = 12) -> list[dict] | None:
    """Fetch daily candles from Finnhub."""
    if not _has_fh():
        return None
    to_ts = int(datetime.now().timestamp())
    from_ts = int((datetime.now() - timedelta(days=months * 32)).timestamp())
    _log(f"[FH] candle for {ticker} ({months}mo)")
    data = _fh_request(
        f"/stock/candle?symbol={ticker}&resolution=D&from={from_ts}&to={to_ts}"
    )
    if not data or data.get("s") != "ok":
        _log(f"[FH] No candle data for {ticker}: {data.get('s', 'no data')}")
        return None
    t_stamps = data.get("t", [])
    opens = data.get("o", [])
    highs = data.get("h", [])
    lows = data.get("l", [])
    closes = data.get("c", [])
    volumes = data.get("v", [])
    if not t_stamps or not closes:
        return None

    results = []
    for i, ts in enumerate(t_stamps):
        dt = datetime.utcfromtimestamp(ts)
        results.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(opens[i]), 2),
            "high": round(float(highs[i]), 2),
            "low": round(float(lows[i]), 2),
            "close": round(float(closes[i]), 2),
            "volume": int(volumes[i]) if i < len(volumes) else 0,
        })
    _log(f"[FH] Got {len(results)} candle rows for {ticker}")
    return results if results else None


# ---------------------------------------------------------------------------
# Alpha Vantage API (fallback — 25 calls/day free tier)
# ---------------------------------------------------------------------------

AV_BASE = "https://www.alphavantage.co/query"
_AV_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()


def _has_av() -> bool:
    """Whether Alpha Vantage API key is configured."""
    return bool(_AV_API_KEY)


def _av_request(params: dict) -> dict | None:
    """Make a request to Alpha Vantage and return parsed JSON."""
    params["apikey"] = _AV_API_KEY
    try:
        r = requests.get(AV_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        # Check for rate-limit / error messages
        if "Information" in data and "rate limit" in str(data.get("Information", "")).lower():
            _log("[AV] Rate limited by Alpha Vantage")
            return None
        if "Error Message" in data:
            _log(f"[AV] API error: {data['Error Message']}")
            return None
        if "Note" in data:
            _log(f"[AV] API note (likely rate limit): {data['Note']}")
            return None
        return data
    except Exception as e:
        _log(f"[AV] Request failed: {type(e).__name__}: {e}")
        return None


def _av_get_quote(ticker: str) -> dict | None:
    """Fetch current quote from Alpha Vantage GLOBAL_QUOTE."""
    if not _has_av():
        return None
    _log(f"[AV] GLOBAL_QUOTE for {ticker}")
    data = _av_request({"function": "GLOBAL_QUOTE", "symbol": ticker})
    if not data:
        return None
    quote = data.get("Global Quote", {})
    if not quote or not quote.get("05. price"):
        return None

    try:
        price = float(quote.get("05. price", 0))
        prev_close = float(quote.get("08. previous close", price))
        return {
            "price": price,
            "prev_close": prev_close,
            "open": float(quote.get("02. open", price)),
            "day_high": float(quote.get("03. high", price)),
            "day_low": float(quote.get("04. low", price)),
            "volume": int(quote.get("06. volume", 0)),
        }
    except (ValueError, TypeError) as e:
        _log(f"[AV] Parse error for {ticker}: {e}")
        return None


def _av_get_overview(ticker: str) -> dict:
    """Fetch company overview from Alpha Vantage OVERVIEW."""
    if not _has_av():
        return {}
    _log(f"[AV] OVERVIEW for {ticker}")
    data = _av_request({"function": "OVERVIEW", "symbol": ticker})
    if not data or "Symbol" not in data:
        return {}

    def _s(key, default=None):
        val = data.get(key, default)
        if isinstance(val, str) and val.strip().lower() in ("none", "", "n/a"):
            return default
        return val

    def _num(key):
        val = _s(key)
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _intnum(key):
        val = _s(key)
        if val is None:
            return 0
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    return {
        "name": _s("Name", ticker.upper()),
        "sector": _s("Sector", "N/A"),
        "industry": _s("Industry", "N/A"),
        "description": _s("Description", ""),
        "market_cap": _num("MarketCapitalization"),
        "pe_ratio": _num("PERatio"),
        "forward_pe": _num("ForwardPE"),
        "eps": _num("EPS"),
        "dividend_yield": _num("DividendYield"),
        "beta": _num("Beta"),
        "fifty_two_week_high": _num("52WeekHigh"),
        "fifty_two_week_low": _num("52WeekLow"),
        "fifty_day_ma": _num("50DayMovingAverage"),
        "two_hundred_day_ma": _num("200DayMovingAverage"),
        "avg_volume": _intnum("AverageVolume"),
    }


def _av_get_daily(ticker: str, months: int = 12) -> list[dict] | None:
    """Fetch daily historical data from Alpha Vantage TIME_SERIES_DAILY."""
    if not _has_av():
        return None
    outputsize = "full" if months > 3 else "compact"
    _log(f"[AV] TIME_SERIES_DAILY for {ticker} outputsize={outputsize}")
    data = _av_request({"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": outputsize})
    if not data:
        return None
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        _log(f"[AV] No daily data for {ticker}")
        return None

    cutoff = datetime.now() - timedelta(days=months * 31)
    results = []
    for date_str, values in sorted(ts.items()):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt < cutoff:
                continue
            results.append({
                "date": date_str,
                "open": round(float(values["1. open"]), 2),
                "high": round(float(values["2. high"]), 2),
                "low": round(float(values["3. low"]), 2),
                "close": round(float(values["4. close"]), 2),
                "volume": int(values["5. volume"]),
            })
        except (ValueError, KeyError, TypeError):
            continue

    _log(f"[AV] Got {len(results)} daily rows for {ticker}")
    return results if results else None


# ---------------------------------------------------------------------------
# Yahoo Finance fallback (kept for when no API keys are set)
# ---------------------------------------------------------------------------

_yf_session = None


def _get_yf_session():
    global _yf_session
    if _yf_session is None:
        _yf_session = requests.Session()
        _yf_session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
    return _yf_session


def _yf_get_price(ticker: str) -> dict | None:
    """Fallback: get price via yfinance Ticker.info."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker, session=_get_yf_session())
        info = stock.info
        if not info or not isinstance(info, dict):
            return None
        price = (info.get("currentPrice") or info.get("regularMarketPrice")
                 or info.get("regularMarketOpen") or 0)
        if not price:
            try:
                price = stock.fast_info.get("lastPrice", 0)
            except Exception:
                pass
        if not price:
            return None
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or price
        return {
            "price": float(price),
            "prev_close": float(prev_close),
            "open": float(info.get("regularMarketOpen", price)),
            "day_high": float(info.get("dayHigh", price)),
            "day_low": float(info.get("dayLow", price)),
            "volume": int(info.get("volume", 0)),
        }
    except Exception as e:
        _log(f"[YF] get_price failed for {ticker}: {type(e).__name__}: {e}")
        return None


def _yf_get_info(ticker: str) -> dict:
    """Fallback: get metadata via yfinance Ticker.info."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker, session=_get_yf_session())
        info = stock.info
        if not info or not isinstance(info, dict):
            return {}
        return {
            "name": info.get("longName") or info.get("shortName", ticker.upper()),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "fifty_day_ma": info.get("fiftyDayAverage"),
            "two_hundred_day_ma": info.get("twoHundredDayAverage"),
            "avg_volume": info.get("averageVolume", 0),
        }
    except Exception as e:
        _log(f"[YF] get_info failed for {ticker}: {type(e).__name__}: {e}")
        return {}


def _yf_get_historical(ticker: str, period: str = "1y") -> list[dict] | None:
    """Fallback: get historical data via yfinance."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False,
                         auto_adjust=True, session=_get_yf_session())
        if df is None or df.empty:
            return None
        results = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            results.append({
                "date": date_str,
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return results
    except Exception as e:
        _log(f"[YF] get_historical failed for {ticker}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# StockAnalyzer — unified interface
# ---------------------------------------------------------------------------

class StockAnalyzer:
    """Fetches stock data (Finnhub → Alpha Vantage → Yahoo Finance)
    and computes technical indicators."""

    @staticmethod
    def get_stock_info(ticker: str) -> dict | None:
        """Fetch basic stock info and key metrics.

        Fallback chain:
          Price:  Finnhub → Alpha Vantage → Yahoo Finance
          Meta:   Alpha Vantage → Yahoo Finance (+ Finnhub profile for name/industry)
        """
        t = ticker.upper()
        fh = "yes" if _has_fh() else "no"
        av = "yes" if _has_av() else "no"
        _log(f"[get_stock_info] {t} (FH={fh}, AV={av})")

        # --- Price data (Finnhub → AV → YF) ---
        price_data = None
        if _has_fh():
            price_data = _fh_get_quote(t)
        if not price_data and _has_av():
            price_data = _av_get_quote(t)
        if not price_data:
            _log(f"[get_stock_info] FH/AV quote failed, trying YF fallback")
            price_data = _yf_get_price(t)
        if not price_data:
            _log(f"[get_stock_info] FAIL: no price for {t}")
            return None

        # --- Metadata (AV → YF, + FH profile for name/industry) ---
        meta = {}
        fh_profile = _fh_get_profile(t) if _has_fh() else {}
        if _has_av():
            meta = _av_get_overview(t)
        if not meta.get("name"):
            yf_meta = _yf_get_info(t)
            if yf_meta:
                meta = {**yf_meta, **meta}  # AV takes priority where present
        # Fill in name from Finnhub if still missing
        if not meta.get("name") and fh_profile.get("name"):
            meta["name"] = fh_profile["name"]
        if not meta.get("industry") or meta.get("industry") == "N/A":
            if fh_profile.get("industry"):
                meta["industry"] = fh_profile["industry"]
        if meta.get("market_cap") is None and fh_profile.get("market_cap"):
            meta["market_cap"] = fh_profile["market_cap"]

        price = price_data["price"]
        prev_close = price_data["prev_close"]
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        _log(f"[get_stock_info] SUCCESS: {t} = ${price:.2f} ({meta.get('name', t)})")

        return {
            "ticker": t,
            "name": meta.get("name", t),
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "prev_close": round(prev_close, 2),
            "open": round(price_data["open"], 2),
            "day_high": round(price_data["day_high"], 2),
            "day_low": round(price_data["day_low"], 2),
            "volume": int(price_data["volume"]),
            "avg_volume": meta.get("avg_volume", 0),
            "market_cap": meta.get("market_cap"),
            "pe_ratio": meta.get("pe_ratio"),
            "forward_pe": meta.get("forward_pe"),
            "eps": meta.get("eps"),
            "dividend_yield": meta.get("dividend_yield"),
            "beta": meta.get("beta"),
            "fifty_two_week_high": meta.get("fifty_two_week_high"),
            "fifty_two_week_low": meta.get("fifty_two_week_low"),
            "fifty_day_ma": meta.get("fifty_day_ma"),
            "two_hundred_day_ma": meta.get("two_hundred_day_ma"),
            "sector": meta.get("sector", "N/A"),
            "industry": meta.get("industry", "N/A"),
            "description": meta.get("description", ""),
        }

    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y") -> list[dict] | None:
        """Fetch historical price data.

        Fallback chain: Finnhub → Alpha Vantage → Yahoo Finance
        """
        t = ticker.upper()
        period_months = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12,
                         "2y": 24, "5y": 60}.get(period, 12)

        # Try Finnhub first
        if _has_fh():
            data = _fh_get_candles(t, months=period_months)
            if data:
                return data
            _log(f"[get_historical_data] FH failed for {t}, trying AV")

        # Try Alpha Vantage
        if _has_av():
            data = _av_get_daily(t, months=period_months)
            if data:
                return data
            _log(f"[get_historical_data] AV failed for {t}, trying YF")

        # Fall back to Yahoo Finance
        return _yf_get_historical(t, period=period)

    # ---- Technical indicators (unchanged) ----

    @staticmethod
    def compute_sma(data: list[float], window: int) -> list[float | None]:
        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        return [None if pd.isna(v) else round(float(v), 2) for v in sma]

    @staticmethod
    def compute_ema(data: list[float], window: int) -> list[float | None]:
        series = pd.Series(data)
        ema = series.ewm(span=window, adjust=False).mean()
        return [None if pd.isna(v) else round(float(v), 2) for v in ema]

    @staticmethod
    def compute_rsi(data: list[float], window: int = 14) -> list[float | None]:
        series = pd.Series(data)
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return [None if pd.isna(v) else round(float(v), 2) for v in rsi]

    @staticmethod
    def compute_macd(data: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        series = pd.Series(data)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": [None if pd.isna(v) else round(float(v), 4) for v in macd_line],
            "signal": [None if pd.isna(v) else round(float(v), 4) for v in signal_line],
            "histogram": [None if pd.isna(v) else round(float(v), 4) for v in histogram],
        }

    @staticmethod
    def compute_bollinger_bands(data: list[float], window: int = 20, num_std: float = 2.0) -> dict:
        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        return {
            "sma": [None if pd.isna(v) else round(float(v), 2) for v in sma],
            "upper": [None if pd.isna(v) else round(float(v), 2) for v in upper],
            "lower": [None if pd.isna(v) else round(float(v), 2) for v in lower],
        }

    @staticmethod
    def get_full_analysis(ticker: str, period: str = "1y") -> dict | None:
        """Get complete analysis including info, historical data, and all indicators."""
        info = StockAnalyzer.get_stock_info(ticker)
        if not info:
            return None

        hist_data = StockAnalyzer.get_historical_data(ticker, period=period)
        if not hist_data:
            return {"info": info, "indicators": None}

        closes = [d["close"] for d in hist_data]

        return {
            "info": info,
            "historical": hist_data,
            "indicators": {
                "sma_20": StockAnalyzer.compute_sma(closes, 20),
                "sma_50": StockAnalyzer.compute_sma(closes, 50),
                "sma_200": StockAnalyzer.compute_sma(closes, 200),
                "ema_12": StockAnalyzer.compute_ema(closes, 12),
                "ema_26": StockAnalyzer.compute_ema(closes, 26),
                "rsi": StockAnalyzer.compute_rsi(closes),
                "macd": StockAnalyzer.compute_macd(closes),
                "bollinger": StockAnalyzer.compute_bollinger_bands(closes),
            },
            "dates": [d["date"] for d in hist_data],
        }

    @staticmethod
    def validate_ticker(query: str) -> dict | None:
        """Validate that a ticker symbol exists and return basic info."""
        info = StockAnalyzer.get_stock_info(query.strip())
        if info:
            return {"ticker": info["ticker"], "name": info["name"]}
        return None

    @staticmethod
    def compare_stocks(tickers: list[str]) -> list[dict]:
        """Compare key metrics across multiple stocks."""
        results = []
        for ticker in tickers:
            info = StockAnalyzer.get_stock_info(ticker.strip())
            if info:
                results.append({
                    "ticker": info["ticker"],
                    "name": info["name"],
                    "price": info["price"],
                    "change_pct": info["change_pct"],
                    "market_cap": info["market_cap"],
                    "pe_ratio": info["pe_ratio"],
                    "volume": info["volume"],
                    "beta": info["beta"],
                    "dividend_yield": info["dividend_yield"],
                })
        return results
