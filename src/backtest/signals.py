"""
src/backtest/signals.py — Market signal generation using yfinance.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.config import CONTRADICTION_THRESHOLD
from src.utils.helpers import new_uuid, now_iso


class SignalGenerator:
    """Compute CAR and classify market signals from contradictions."""

    @staticmethod
    def compute_car(
        ticker: str,
        event_date: str,
        window_before: int = 2,
        window_after: int = 3,
    ) -> dict[str, float]:
        """
        Compute Cumulative Abnormal Return over [-window_before, +window_after]
        around *event_date* using yfinance + SPY as benchmark.
        """
        try:
            import yfinance as yf

            dt = datetime.strptime(event_date, "%Y-%m-%d")
            start = dt - timedelta(days=window_before + 10)
            end = dt + timedelta(days=window_after + 10)

            stock = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                end=end.strftime("%Y-%m-%d"), progress=False)
            spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"), progress=False)

            if stock.empty or spy.empty:
                return {"car_1d": 0.0, "car_3d": 0.0, "car_5d": 0.0, "price": 0.0, "price_change_pct": 0.0}

            stock_ret = stock["Close"].pct_change().dropna()
            spy_ret = spy["Close"].pct_change().dropna()

            # Align indices
            common = stock_ret.index.intersection(spy_ret.index)
            abnormal = stock_ret.loc[common] - spy_ret.loc[common]

            # Find closest index to event_date
            event_idx = abnormal.index.searchsorted(dt)
            event_idx = min(event_idx, len(abnormal) - 1)

            def cum_ret(n):
                s = max(0, event_idx - 1)
                e = min(len(abnormal), event_idx + n)
                vals = abnormal.iloc[s:e]
                return float(vals.sum()) if len(vals) > 0 else 0.0

            price = float(stock["Close"].iloc[-1]) if not stock.empty else 0.0
            price_first = float(stock["Close"].iloc[0]) if not stock.empty else 1.0
            pct = ((price - price_first) / price_first * 100) if price_first else 0.0

            return {
                "car_1d": round(cum_ret(1), 4),
                "car_3d": round(cum_ret(3), 4),
                "car_5d": round(cum_ret(5), 4),
                "price": round(price, 2),
                "price_change_pct": round(pct, 2),
            }
        except Exception:
            return {"car_1d": 0.0, "car_3d": 0.0, "car_5d": 0.0, "price": 0.0, "price_change_pct": 0.0}

    @staticmethod
    def classify_signal(
        contradiction_score: float,
        sentiment_shift: float,
    ) -> str:
        """
        Classify a signal based on contradiction score and sentiment shift.
        """
        if contradiction_score > CONTRADICTION_THRESHOLD:
            if sentiment_shift < -0.2:
                return "BEARISH"
            elif sentiment_shift > 0.2:
                return "BULLISH"
            else:
                return "WATCH"
        elif contradiction_score >= 0.5:
            return "WATCH"
        else:
            return "NEUTRAL"

    def generate_signal(
        self,
        ticker: str,
        contradiction_score: float,
        sentiment_shift: float,
        event_date: str,
        contradiction_id: str = "",
        description: str = "",
    ) -> dict:
        """Generate a complete signal record for one contradiction."""
        car = self.compute_car(ticker, event_date)
        signal_type = self.classify_signal(contradiction_score, sentiment_shift)

        return {
            "id": new_uuid(),
            "ticker": ticker,
            "signal_type": signal_type,
            "confidence": round(contradiction_score, 3),
            "price": car["price"],
            "price_change_pct": car["price_change_pct"],
            "description": description or f"{signal_type} signal for {ticker}",
            "event_date": event_date,
            "car_1d": car["car_1d"],
            "car_3d": car["car_3d"],
            "car_5d": car["car_5d"],
            "contradiction_id": contradiction_id,
        }
