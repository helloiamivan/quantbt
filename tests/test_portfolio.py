import unittest

import pandas as pd

from quantbt.portfolio import Portfolio, flattenDictionary


class PortfolioTests(unittest.TestCase):
    def test_flatten_dictionary_does_not_mutate_history(self):
        history = {pd.Timestamp("2024-01-01"): {"AAA": 1.0}}

        records = flattenDictionary(history)

        self.assertEqual(records[0]["Dates"], pd.Timestamp("2024-01-01"))
        self.assertNotIn("Dates", next(iter(history.values())))

    def test_rebalance_updates_cash_and_accumulates_fixed_costs(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        portfolio.setFixedTransactionCosts({"AAA": 0.01})
        portfolio.rebalance({"AAA": 0.5}, {"AAA": 100.0}, pd.Timestamp("2024-01-01"))

        self.assertEqual(portfolio.getAssetPosition("AAA"), 5.0)
        self.assertEqual(portfolio.getCash(), 495.0)
        self.assertEqual(portfolio.getTransactionCosts(), 5.0)

    def test_unwind_charges_trade_cost_before_position_is_zeroed(self):
        portfolio = Portfolio({"AAA": 10.0}, 0.0)
        portfolio.setFixedTransactionCosts({"AAA": 0.01})
        portfolio.rebalance({}, {"AAA": 100.0}, pd.Timestamp("2024-01-01"))

        self.assertEqual(portfolio.getAssetPosition("AAA"), 0.0)
        self.assertEqual(portfolio.getTransactionCosts(), 10.0)
        self.assertEqual(portfolio.getCash(), 990.0)
