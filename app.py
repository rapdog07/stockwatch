"""
Stock Analysis Web Application

A Flask-based web app for looking up and analyzing stocks
using Yahoo Finance data via yfinance.
"""

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
        return render_template("index.html", error=f"Could not find stock data for '{ticker.upper()}'. Please check the ticker symbol and try again.")
    return render_template("stock.html", analysis=analysis, period=period)


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
