import unittest

import numpy as np
import pandas as pd

from quantbt.analytics import performanceSummary


class PerformanceSummaryTests(unittest.TestCase):
    def test_single_observation_returns_nan_annual_metrics(self):
        date = pd.Timestamp("2024-01-01")
        stats = performanceSummary({date: 100.0}, historicalTCosts={date: 0.0})

        self.assertTrue(np.isnan(stats["Annual Returns"]))
        self.assertEqual(stats["Cumulative Return"], 0.0)

    def test_calmar_uses_absolute_drawdown(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        nav = dict(zip(dates, [100.0, 80.0, 90.0]))
        stats = performanceSummary(nav, historicalTCosts={dates[-1]: 2.5})

        self.assertAlmostEqual(stats["Maximum Drawdown"], -0.2)
        self.assertLess(stats["Calmar Ratio"], 0)
        self.assertEqual(stats["Total Transaction Costs"], 2.5)
