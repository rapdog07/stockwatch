"""
Stock Analysis Utility Module

Handles fetching stock data from Yahoo Finance and computing
technical indicators and analytics.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class StockAnalyzer:
    """Fetches and analyzes stock data using yfinance."""

    @staticmethod
    def get_stock_info(ticker: str) -> dict | None:
        """Fetch basic stock info and key metrics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            if not info or ("regularMarketPrice" not in info and "currentPrice" not in info):
                return None

            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or price
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0

            return {
                "ticker": ticker.upper(),
                "name": info.get("longName") or info.get("shortName", ticker.upper()),
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(prev_close, 2),
                "open": round(info.get("regularMarketOpen", 0), 2),
                "day_high": round(info.get("dayHigh", 0), 2),
                "day_low": round(info.get("dayLow", 0), 2),
                "volume": info.get("volume", 0),
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
        except Exception as e:
            print(f"Error fetching stock info for {ticker}: {e}")
            return None

    @staticmethod
    def get_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> list[dict] | None:
        """Fetch historical price data."""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            if hist.empty:
                return None

            results = []
            for idx, row in hist.iterrows():
                results.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]),
                })
            return results
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
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
            ticker = yf.Ticker(query.strip().upper())
            info = ticker.info
            if info and ("longName" in info or "shortName" in info):
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

