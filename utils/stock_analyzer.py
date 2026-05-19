"""
Stock Analysis Utility Module

Handles fetching stock data from Yahoo Finance and computing
technical indicators and analytics.
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta


def _log(msg: str) -> None:
    """Log to stderr so Render captures it in the deploy logs."""
    print(msg, file=sys.stderr, flush=True)


def _get_yf_session():
    """Create a requests session with browser-like headers so Yahoo Finance
    doesn't block us — especially important on cloud platforms like Render."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Cache-Control": "no-cache",
    })
    return session


def _ensure_session():
    """Ensure the shared session exists and return it."""
    if StockAnalyzer._session is None:
        StockAnalyzer._session = _get_yf_session()
    return StockAnalyzer._session


class StockAnalyzer:
    """Fetches and analyzes stock data using yfinance."""

    # Shared session so yfinance reuses the connection
    _session = None

    @classmethod
    def _get_ticker(cls, ticker: str) -> yf.Ticker:
        """Get a yfinance Ticker with our custom session."""
        return yf.Ticker(ticker, session=_ensure_session())

    @staticmethod
    def _get_price_from_download(ticker: str) -> dict | None:
        """Get latest price data from yf.download (uses different, more
        cloud-friendly Yahoo endpoint than Ticker.info)."""
        try:
            _log(f"[_get_price_from_download] Attempting yf.download for {ticker}")
            df = yf.download(
                ticker,
                period="5d",
                progress=False,
                auto_adjust=False,
                session=_ensure_session(),
            )
            if df is None or df.empty:
                _log(f"[_get_price_from_download] Empty DataFrame for {ticker}")
                return None

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest

            price = float(latest["Close"])
            prev_close = float(prev["Close"])
            open_price = float(latest["Open"])
            high = float(latest["High"])
            low = float(latest["Low"])
            volume = int(latest["Volume"])

            _log(f"[_get_price_from_download] {ticker}: price={price}, prev_close={prev_close}")
            return {
                "price": price,
                "prev_close": prev_close,
                "open": open_price,
                "day_high": high,
                "day_low": low,
                "volume": volume,
            }
        except Exception as e:
            _log(f"[_get_price_from_download] ERROR for {ticker}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def get_stock_info(ticker: str) -> dict | None:
        """Fetch basic stock info and key metrics.

        Uses yf.download for price data (more reliable on cloud) and
        Ticker.info for metadata (name, sector, etc.)."""
        _log(f"[get_stock_info] Starting for {ticker}")

        # --- Step 1: Get price data from yf.download (primary) ---
        price_data = StockAnalyzer._get_price_from_download(ticker)
        if not price_data:
            _log(f"[get_stock_info] yf.download failed for {ticker}, falling back to Ticker.info")

        # --- Step 2: Get metadata from Ticker.info ---
        info = {}
        try:
            stock = StockAnalyzer._get_ticker(ticker)
            info = stock.info
            _log(f"[get_stock_info] Ticker.info returned {len(info)} keys for {ticker}")
        except Exception as e:
            _log(f"[get_stock_info] Ticker.info failed for {ticker}: {type(e).__name__}: {e}")

        if not isinstance(info, dict):
            info = {}

        # --- Step 3: Determine price ---
        if price_data:
            price = price_data["price"]
            prev_close = price_data["prev_close"]
            open_price = price_data["open"]
            day_high = price_data["day_high"]
            day_low = price_data["day_low"]
            volume = price_data["volume"]
        else:
            # Fallback to info dict
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("regularMarketOpen")
                or 0
            )
            if price == 0:
                try:
                    price = stock.fast_info.get("lastPrice", 0) if stock else 0
                except Exception:
                    pass
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or price
            open_price = info.get("regularMarketOpen", 0)
            day_high = info.get("dayHigh", 0)
            day_low = info.get("dayLow", 0)
            volume = info.get("volume", 0)

        if not price or price == 0:
            _log(f"[get_stock_info] FAIL: No price for {ticker}. info keys: {list(info.keys())[:15]}")
            return None

        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        ticker_name = info.get("longName") or info.get("shortName") or ticker.upper()

        _log(f"[get_stock_info] SUCCESS: {ticker} = ${price} ({ticker_name})")

        return {
            "ticker": ticker.upper(),
            "name": ticker_name,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "prev_close": round(prev_close, 2),
            "open": round(open_price, 2),
            "day_high": round(day_high, 2),
            "day_low": round(day_low, 2),
            "volume": int(volume),
            "avg_volume": info.get("averageVolume", 0),
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
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "description": info.get("longBusinessSummary", ""),
        }

    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> list[dict] | None:
        """Fetch historical price data using yf.download (more reliable on cloud)."""
        _log(f"[get_historical_data] Fetching {ticker} period={period}")
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                session=_ensure_session(),
            )

            if df is None or df.empty:
                _log(f"[get_historical_data] Empty data for {ticker}")
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
            _log(f"[get_historical_data] Got {len(results)} rows for {ticker}")
            return results
        except Exception as e:
            _log(f"[get_historical_data] ERROR for {ticker}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def compute_sma(data: list[float], window: int) -> list[float | None]:
        """Compute Simple Moving Average."""
        series = pd.Series(data)
        sma = series.rolling(window=window).mean()
        return [None if pd.isna(v) else round(float(v), 2) for v in sma]

    @staticmethod
    def compute_ema(data: list[float], window: int) -> list[float | None]:
        """Compute Exponential Moving Average."""
        series = pd.Series(data)
        ema = series.ewm(span=window, adjust=False).mean()
        return [None if pd.isna(v) else round(float(v), 2) for v in ema]

    @staticmethod
    def compute_rsi(data: list[float], window: int = 14) -> list[float | None]:
        """Compute Relative Strength Index."""
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
        """Compute MACD (Moving Average Convergence Divergence)."""
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
        """Compute Bollinger Bands."""
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

        analysis = {
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

        return analysis

    @staticmethod
    def validate_ticker(query: str) -> dict | None:
        """Validate that a ticker symbol exists and return basic info."""
        try:
            stock = StockAnalyzer._get_ticker(query.strip().upper())
            info = stock.info
            if info and isinstance(info, dict) and ("longName" in info or "shortName" in info):
                return {
                    "ticker": query.strip().upper(),
                    "name": info.get("longName") or info.get("shortName", query.strip().upper()),
                }
            return None
        except Exception:
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

