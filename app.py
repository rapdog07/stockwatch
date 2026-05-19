"""
Stock Analysis Web Application

A Flask-based web app for looking up and analyzing stocks
using Yahoo Finance data via yfinance.
"""

import sys
from flask import Flask, render_template, request, jsonify
from utils.stock_analyzer import StockAnalyzer

app = Flask(__name__)


@app.route("/")
def index():
    """Home page with stock search."""
    return render_template("index.html")


@app.route("/stock/<ticker>")
def stock_detail(ticker: str):
    """Stock detail page with full analysis."""
    period = request.args.get("period", "1y")
    analysis = StockAnalyzer.get_full_analysis(ticker.upper(), period=period)
    if not analysis:
        return render_template(
            "index.html",
            error=(
                f"Could not find stock data for '{ticker.upper()}'. "
                f"This may be due to a temporary Yahoo Finance connectivity issue "
                f"on the server. Try refreshing in a moment, or visit "
                f"/debug/{ticker.upper()} to diagnose."
            ),
        )
    return render_template("stock.html", analysis=analysis, period=period)


@app.route("/debug/<ticker>")
def debug_ticker(ticker: str):
    """Diagnostic endpoint: shows raw yfinance responses for a ticker."""
    t = ticker.upper()
    result = {
        "ticker": t,
        "steps": [],
    }

    # Step 1: Test raw connectivity
    try:
        import requests
        r = requests.get("https://finance.yahoo.com", timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0.0.0"
        })
        result["steps"].append({
            "step": "connectivity_check",
            "ok": r.status_code == 200,
            "status": r.status_code,
        })
    except Exception as e:
        result["steps"].append({
            "step": "connectivity_check",
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })

    # Step 2: Try yf.download
    try:
        price_data = StockAnalyzer._get_price_from_download(t)
        result["steps"].append({
            "step": "yf_download_price",
            "ok": price_data is not None,
            "data": price_data,
        })
    except Exception as e:
        result["steps"].append({
            "step": "yf_download_price",
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })

    # Step 3: Try Ticker.info
    try:
        stock = StockAnalyzer._get_ticker(t)
        info = stock.info
        result["steps"].append({
            "step": "ticker_info",
            "ok": bool(info),
            "keys_found": list(info.keys())[:30] if info else [],
            "longName": info.get("longName") if info else None,
            "currentPrice": info.get("currentPrice") if info else None,
        })
    except Exception as e:
        result["steps"].append({
            "step": "ticker_info",
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })

    # Step 4: Try full analysis
    try:
        analysis = StockAnalyzer.get_full_analysis(t)
        result["steps"].append({
            "step": "full_analysis",
            "ok": analysis is not None,
        })
    except Exception as e:
        result["steps"].append({
            "step": "full_analysis",
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        })

    return jsonify(result)


@app.route("/api/stock/<ticker>")
def api_stock(ticker: str):
    """API endpoint for stock data."""
    period = request.args.get("period", "1y")
    analysis = StockAnalyzer.get_full_analysis(ticker.upper(), period=period)
    if not analysis:
        return jsonify({"error": "Stock not found"}), 404
    return jsonify(analysis)


@app.route("/api/search")
def api_search():
    """API endpoint for stock search."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    result = StockAnalyzer.validate_ticker(query)
    return jsonify([result] if result else [])


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """Compare multiple stocks side by side."""
    if request.method == "POST":
        tickers_str = request.form.get("tickers", "")
        tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
        if not tickers:
            return render_template("compare.html", error="Please enter at least one ticker symbol.")
        results = StockAnalyzer.compare_stocks(tickers)
        return render_template("compare.html", results=results, tickers_input=tickers_str)
    return render_template("compare.html")


@app.route("/api/compare")
def api_compare():
    """API endpoint for stock comparison."""
    tickers_str = request.args.get("tickers", "")
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400
    results = StockAnalyzer.compare_stocks(tickers)
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
