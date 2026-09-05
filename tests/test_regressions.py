import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt.analytics import performanceSummary
from quantbt.portfolio import Portfolio, sanitizeJSON


class UnfundedRebalanceTests(unittest.TestCase):
    def test_unfunded_long_leverage_raises_by_default(self):
        portfolio = Portfolio({}, 100.0)
        with self.assertRaises(ValueError) as ctx:
            portfolio.rebalance({"AAA": 2.0}, {"AAA": 100.0}, pd.Timestamp("2024-01-01"))
        self.assertIn("cash", str(ctx.exception).lower())

    def test_unfunded_leverage_permitted_when_opted_in(self):
        portfolio = Portfolio({}, 100.0)
        portfolio.setAllowNegativeCash(True)
        portfolio.rebalance({"AAA": 2.0}, {"AAA": 100.0}, pd.Timestamp("2024-01-01"))
        self.assertEqual(portfolio.getCash(), -100.0)

    def test_fully_invested_with_costs_is_allowed(self):
        # Full investment with 1bp costs ends slightly cash-negative; that is
        # explained by costs and must NOT raise.
        portfolio = Portfolio({}, 1_000.0)
        portfolio.setFixedTransactionCosts({"AAA": 0.0001, "BBB": 0.0001})
        portfolio.rebalance(
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 100.0, "BBB": 50.0},
            pd.Timestamp("2024-01-01"),
        )
        self.assertLess(portfolio.getCash(), 0.0)  # small cost-driven deficit
        self.assertGreater(portfolio.getCash(), -1.0)


class MissingPriceTests(unittest.TestCase):
    def test_rebalance_reports_missing_asset_price(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        with self.assertRaises(ValueError) as ctx:
            portfolio.rebalance({}, {"BBB": 100.0}, pd.Timestamp("2024-01-02"))
        self.assertIn("AAA", str(ctx.exception))

    def test_signoff_reports_missing_asset_price(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        with self.assertRaises(ValueError) as ctx:
            portfolio.signOff(pd.Timestamp("2024-01-02"), {"BBB": 100.0})
        self.assertIn("AAA", str(ctx.exception))

    def test_zero_position_asset_may_be_absent_from_prices(self):
        portfolio = Portfolio({"AAA": 0.0}, 100.0)
        portfolio.signOff(pd.Timestamp("2024-01-02"), {})  # must not raise
        self.assertEqual(portfolio.getHistoricalNAV().iloc[-1, 0], 100.0)


class SlippageConfigTests(unittest.TestCase):
    def test_slippage_without_impact_params_raises_actionable_error(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        portfolio.setSlippageModel("squarerootimpact")
        portfolio.FirstRebalanceDate = pd.Timestamp("2024-01-01")  # non-initial
        with self.assertRaises(ValueError) as ctx:
            portfolio.rebalance({"AAA": 0.5}, {"AAA": 100.0}, pd.Timestamp("2024-01-02"))
        self.assertIn("AAA", str(ctx.exception))
        self.assertIn("setImpactParams", str(ctx.exception))

    def test_slippage_with_incomplete_params_raises_actionable_error(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        portfolio.setSlippageModel("squarerootimpact")
        portfolio.setImpactParams({"AAA": {"BidAskSpread": 0.0}})  # incomplete
        portfolio.FirstRebalanceDate = pd.Timestamp("2024-01-01")
        with self.assertRaises(ValueError) as ctx:
            portfolio.rebalance({"AAA": 0.5}, {"AAA": 100.0}, pd.Timestamp("2024-01-02"))
        self.assertIn("Volatility", str(ctx.exception))


class AnalyticsConsistencyTests(unittest.TestCase):
    def test_volatility_uses_sample_std(self):
        dates = pd.date_range("2023-01-02", periods=3, freq="D")
        nav = dict(zip(dates, [100.0, 110.0, 132.0]))  # returns 10%, 20%
        stats = performanceSummary(nav)
        expected = np.array([0.10, 0.20]).std(ddof=1) * np.sqrt(260)
        self.assertAlmostEqual(stats["Annual Volatility"], expected)

    def test_annual_return_uses_trading_day_annualisation(self):
        # 5 business-day NAV observations span 4 elapsed weekdays = 4/260 years.
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        nav = dict(zip(dates, [100.0] * 4 + [100.0 * 1.1]))
        stats = performanceSummary(nav)
        self.assertAlmostEqual(stats["Annual Returns"], 1.1 ** (260 / 4) - 1)


class StatisticsCachingTests(unittest.TestCase):
    def test_historical_cost_rows_are_cumulative_to_date(self):
        portfolio = Portfolio({}, 1_000.0)
        d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
        portfolio.transactionCosts = 5.0
        portfolio.signOff(d1, {"AAA": 100.0})
        portfolio.transactionCosts = 9.0
        portfolio.signOff(d2, {"AAA": 100.0})

        stats = portfolio.getPerformanceStatistics(historical=True)
        self.assertEqual(stats.loc[d1, "Total Transaction Costs"], 5.0)
        self.assertEqual(stats.loc[d2, "Total Transaction Costs"], 9.0)

        latest = portfolio.getPerformanceStatistics(historical=False)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest.index[0], d2)


class JsonDumpTests(unittest.TestCase):
    def test_daily_dump_is_strict_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            portfolio = Portfolio({"AAA": 10.0}, 0.0, name="dump", datadump=True,
                                  backtestFolderName=tmp)
            portfolio.setFixedTransactionCosts({"AAA": 0.01})
            portfolio.rebalance({"AAA": 0.5}, {"AAA": 100.0}, pd.Timestamp("2024-01-01"))
            portfolio.signOff(pd.Timestamp("2024-01-01"), {"AAA": 100.0})

            files = list(Path(tmp).rglob("*.json"))
            self.assertEqual(len(files), 1)
            payload = json.loads(files[0].read_text())  # strict JSON, no NaN literal
            self.assertIn("Performance", payload[0])
            self.assertIn("NAV", payload[0])
            self.assertFalse("NaN" in files[0].read_text())

    def test_sanitize_json_removes_non_finite_floats(self):
        cleaned = sanitizeJSON({"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": -float("inf")}})
        self.assertIsNone(cleaned["a"])
        self.assertIsNone(cleaned["b"][0])
        self.assertEqual(cleaned["b"][1], 1.0)
        self.assertIsNone(cleaned["c"]["d"])


if __name__ == "__main__":
    unittest.main()
