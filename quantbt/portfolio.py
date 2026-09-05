"""Portfolio state, rebalancing, and backtest history."""

import copy
import json
import math
import time
from numbers import Real
from pathlib import Path

import pandas as pd

from .analytics import getNAVPlot, getWeightsPlot, performanceSummary


def flattenDictionary(nestedDict):
    """Convert a date-keyed mapping of mappings into DataFrame-ready records."""
    return [{"Dates": date, **values} for date, values in nestedDict.items()]


def sanitizeJSON(value):
    """Recursively replace non-finite floats with ``None``.

    Keeps daily dumps strict JSON (``json.dumps(..., allow_nan=False)``) instead
    of writing the non-standard ``NaN`` literal that many parsers reject.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: sanitizeJSON(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitizeJSON(val) for val in value]
    return value


class Portfolio:
    def __init__(self, positions, cash, name="", datadump=False, backtestFolderName=None):
        self.name = name
        self.positions = dict(positions)
        self.cash = float(cash)
        self.datadump = datadump
        self.fixedTransactionCosts = {}
        self.borrowCosts = {}
        self.annualManagementFee = 0.0
        self.slippageModel = ""
        self.impactParams = {}
        self.unwindUndefinedAssetWeights = True
        # When False (default), rebalance() raises if cash would go negative for
        # reasons other than the trading costs incurred during that rebalance.
        # Set True to permit deliberate leverage / negative cash balances.
        self.allowNegativeCash = False
        self.transactionCosts = 0.0
        self.slippageCosts = 0.0
        self.LastRebalanceDate = None
        self.FirstRebalanceDate = None
        self.performanceStatistics = {}
        self.customData = {}
        self.historicalNAV = {}
        self.historicalPositions = {}
        self.historicalWeights = {}
        self.historicalTCosts = {}
        self.historicalSlippageCosts = {}
        self.historicalBorrowCosts = {}
        self.historicalCash = {}
        self.timestamp = str(time.time_ns())
        root = Path(backtestFolderName) if backtestFolderName else Path.cwd()
        self.backtestFolderName = root / "BackTestResults" / f"{self.timestamp}-{name}"

    # Read API retained for compatibility with existing notebooks.
    def getPortfolioName(self): return self.name
    def getBacktestFolderName(self): return str(self.backtestFolderName)
    def getPositions(self): return self.positions
    def getAssetPosition(self, asset): return self.positions[asset]
    def getCash(self): return self.cash
    def getTransactionCosts(self): return self.transactionCosts
    def getAssetsInPortfolio(self): return list(self.positions)
    def getFixedTransactionCosts(self, asset): return self.fixedTransactionCosts.get(asset, 0.0)
    def getAllFixedTransactionCosts(self): return self.fixedTransactionCosts
    def getBorrowCost(self, asset): return self.borrowCosts.get(asset, 0.0)
    def getAllBorrowCosts(self): return self.borrowCosts
    def getAnnualManagementFee(self): return self.annualManagementFee
    def getBacktestTimestamp(self): return self.timestamp
    def getFirstRebalanceDate(self): return self.FirstRebalanceDate
    def getLastRebalanceDate(self): return self.LastRebalanceDate
    def getSlippageModel(self): return self.slippageModel
    def getSlippageCosts(self): return self.slippageCosts
    def getImpactParams(self): return self.impactParams
    def getCustomData(self): return self.customData
    def getCustomDataByDate(self, date): return self.customData.get(date, {})

    def getNAV(self, lastPriceMap):
        return self.cash + sum(
            lastPriceMap[asset] * position
            for asset, position in self.positions.items()
            if position
        )

    def getWeights(self, lastPriceMap):
        nav = self.getNAV(lastPriceMap)
        if nav == 0:
            return {asset: math.nan for asset in self.positions if self.positions[asset]}
        return {
            asset: lastPriceMap[asset] * position / nav
            for asset, position in self.positions.items()
            if position
        }

    @staticmethod
    def _format_history(history, column_name, formatOut, nested=False):
        format_out = formatOut.lower()
        if format_out == "dictionary":
            return history
        if format_out != "dataframe":
            raise ValueError("formatOut must be 'dataframe' or 'dictionary'")
        if nested:
            return pd.DataFrame(flattenDictionary(history)).set_index("Dates") if history else pd.DataFrame()
        return pd.Series(history, name=column_name).to_frame()

    def getHistoricalWeights(self, formatOut="DataFrame"):
        return self._format_history(self.historicalWeights, None, formatOut, nested=True)
    def getHistoricalPositions(self, formatOut="DataFrame"):
        return self._format_history(self.historicalPositions, None, formatOut, nested=True)
    def getHistoricalNAV(self, formatOut="DataFrame"):
        return self._format_history(self.historicalNAV, "Historical NAV", formatOut)
    def getHistoricalTCosts(self, formatOut="DataFrame"):
        return self._format_history(self.historicalTCosts, "Cumulative Transaction Costs", formatOut)
    def getHistoricalSlippageCosts(self, formatOut="DataFrame"):
        return self._format_history(self.historicalSlippageCosts, "Cumulative Slippage Costs", formatOut)
    def getHistoricalBorrowCosts(self, formatOut="DataFrame"):
        return self._format_history(self.historicalBorrowCosts, "Daily Borrow Costs", formatOut)
    def getHistoricalCash(self, formatOut="DataFrame"):
        return self._format_history(self.historicalCash, "Historical Cash Account", formatOut)

    def getPerformanceStatistics(self, historical=False):
        stats = pd.DataFrame.from_dict(self._performance_rows(), orient="index")
        return stats if historical or stats.empty else stats.tail(1)

    def _performance_rows(self):
        """Return per-date statistics (cached; recomputed when new days appear)."""
        if len(self.performanceStatistics) == len(self.historicalNAV):
            return self.performanceStatistics
        items = list(self.historicalNAV.items())
        rows = {}
        for index, (date, _) in enumerate(items):
            rows[date] = self._performance_at_index(items, index)
        self.performanceStatistics = rows
        return rows

    def _performance_at(self, date):
        items = list(self.historicalNAV.items())
        index = next((i for i, (d, _) in enumerate(items) if d == date), None)
        if index is None:
            raise KeyError(f"No NAV recorded for date {date}")
        return self._performance_at_index(items, index)

    def _performance_at_index(self, items, index):
        """Summaries over the history prefix ending at ``items[index]``."""
        prefix = dict(items[: index + 1])
        if self.historicalTCosts:
            prefix_costs = {date: self.historicalTCosts[date] for date, _ in items[: index + 1]}
        else:
            prefix_costs = None
        return performanceSummary(prefix, historicalTCosts=prefix_costs)

    def setCash(self, cash):
        if not isinstance(cash, Real) or isinstance(cash, bool):
            raise TypeError("cash must be numeric")
        self.cash = float(cash)

    def _set_dict(self, attribute, value, description):
        if not isinstance(value, dict):
            raise TypeError(f"{description} must be a dictionary")
        setattr(self, attribute, value)

    def setFixedTransactionCosts(self, value): self._set_dict("fixedTransactionCosts", value, "Transaction costs")
    def setBorrowCosts(self, value): self._set_dict("borrowCosts", value, "Borrow costs")
    def setImpactParams(self, value): self._set_dict("impactParams", value, "Impact parameters")
    def setAnnualManagementFee(self, value): self.annualManagementFee = float(value)
    def setTransactionCosts(self, value): self.transactionCosts = float(value)
    def setSlippageCosts(self, value): self.slippageCosts = float(value)
    def setUnwindUndefinedAssetWeights(self, value): self.unwindUndefinedAssetWeights = bool(value)
    def setAllowNegativeCash(self, value): self.allowNegativeCash = bool(value)

    def setSlippageModel(self, model):
        if model not in {"", "squarerootimpact"}:
            raise ValueError("slippageModel must be 'squarerootimpact' or ''")
        self.slippageModel = model

    def _set_rebalance_date(self, attribute, date):
        if not isinstance(date, pd.Timestamp):
            raise TypeError("Rebalance date must be a pandas Timestamp")
        setattr(self, attribute, date)
    def setFirstRebalanceDate(self, date): self._set_rebalance_date("FirstRebalanceDate", date)
    def setLastRebalanceDate(self, date): self._set_rebalance_date("LastRebalanceDate", date)

    def setCustomData(self, date, data_dict):
        if not isinstance(date, pd.Timestamp) or not isinstance(data_dict, dict):
            raise TypeError("Custom data requires a pandas Timestamp and dictionary")
        self.customData[date] = data_dict

    def plotNAV(self): return getNAVPlot(self)
    def plotWeights(self): return getWeightsPlot(self)

    def buy(self, asset, quantity, lastPriceMap):
        if quantity < 0: raise ValueError("quantity must be non-negative")
        self.positions[asset] = self.positions.get(asset, 0.0) + quantity
        self.cash -= quantity * lastPriceMap[asset]

    def sell(self, asset, quantity, lastPriceMap):
        if quantity < 0: raise ValueError("quantity must be non-negative")
        self.positions[asset] = self.positions.get(asset, 0.0) - quantity
        self.cash += quantity * lastPriceMap[asset]

    def calcDailyBorrowCost(self, lastPriceMap):
        return sum(
            abs(position * lastPriceMap[asset]) * ((1 + self.getBorrowCost(asset)) ** (1 / 260) - 1)
            for asset, position in self.positions.items() if position < 0
        )

    def _fixed_cost(self, asset, units, price):
        return abs(units) * price * self.getFixedTransactionCosts(asset)

    def _slippage_cost(self, asset, units, price, is_initial_rebalance):
        if self.slippageModel != "squarerootimpact" or is_initial_rebalance or not units:
            return 0.0
        params = self.impactParams.get(asset)
        if params is None:
            raise ValueError(
                f"slippageModel 'squarerootimpact' requires setImpactParams(...) "
                f"for every traded asset; missing parameters for asset {asset!r}."
            )
        missing_keys = [key for key in ("BidAskSpread", "ScalingFactor", "Volatility", "ADV") if key not in params]
        if missing_keys:
            raise ValueError(
                f"Impact parameters for asset {asset!r} are missing keys: {missing_keys}. "
                f"Expected 'BidAskSpread', 'ScalingFactor', 'Volatility' (annualised) and 'ADV'."
            )
        impact = params["BidAskSpread"] + params["ScalingFactor"] * params["Volatility"] * math.sqrt(abs(units) / params["ADV"] / 252)
        return abs(units) * price * impact

    def _execute_trade(self, asset, units, lastPriceMap, is_initial_rebalance):
        if not units:
            return
        price = lastPriceMap[asset]
        (self.buy if units > 0 else self.sell)(asset, abs(units), lastPriceMap)
        fixed_cost = self._fixed_cost(asset, units, price)
        slippage_cost = self._slippage_cost(asset, units, price, is_initial_rebalance)
        self.cash -= fixed_cost + slippage_cost
        self.transactionCosts += fixed_cost
        self.slippageCosts += slippage_cost

    def _validate_prices(self, lastPriceMap, assets, date):
        """Fail fast with an actionable error when a needed price is missing.

        Without this, missing/NaN prices would either raise a bare ``KeyError``
        mid-backtest or silently poison the NAV/statistics with NaN values.
        """
        missing = []
        for asset in assets:
            price = lastPriceMap.get(asset)
            if price is None:
                missing.append(asset)
                continue
            try:
                finite = math.isfinite(float(price))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                missing.append(asset)
        if missing:
            raise ValueError(
                f"Missing or non-finite price(s) at {date.strftime('%Y-%m-%d')}: {missing}. "
                f"Provide a lastPriceMap covering all held/targeted assets "
                f"(e.g. forward-fill or last-known prices for delisted series)."
            )

    @staticmethod
    def _costs_since(costs_before, transaction_costs, slippage_costs):
        return (transaction_costs - costs_before[0]) + (slippage_costs - costs_before[1])

    def _enforce_cash_floor(self, costs_before, date):
        """Reject unfunded trades unless allowNegativeCash has been enabled."""
        if self.allowNegativeCash:
            return
        # Trading costs are the only legitimate reason a fully-invested book
        # ends a rebalance slightly cash-negative.
        costs_incurred = self._costs_since(
            costs_before, self.transactionCosts, self.slippageCosts
        )
        if self.cash < -(costs_incurred + 1e-6):
            raise ValueError(
                f"Rebalance at {date.strftime('%Y-%m-%d')} would leave cash at "
                f"{self.cash:.2f}, below what trading costs ({costs_incurred:.2f}) can "
                f"explain - target weights exceed available cash (implicit leverage). "
                f"Call setAllowNegativeCash(True) to permit this deliberately."
            )

    def rebalance(self, targetWeights, lastPriceMap, date):
        if not isinstance(date, pd.Timestamp):
            raise TypeError("Rebalance date must be a pandas Timestamp")
        self._validate_prices(
            lastPriceMap,
            {asset for asset, position in self.positions.items() if position} | set(targetWeights),
            date,
        )
        current_nav = self.getNAV(lastPriceMap)
        costs_before = (self.transactionCosts, self.slippageCosts)
        is_initial_rebalance = self.FirstRebalanceDate is None
        if is_initial_rebalance:
            self.FirstRebalanceDate = date
        self.LastRebalanceDate = date

        if self.unwindUndefinedAssetWeights:
            for asset, position in list(self.positions.items()):
                if asset not in targetWeights:
                    self._execute_trade(asset, -position, lastPriceMap, is_initial_rebalance)

        for asset, weight in targetWeights.items():
            target_units = weight * current_nav / lastPriceMap[asset]
            self._execute_trade(asset, target_units - self.positions.get(asset, 0.0), lastPriceMap, is_initial_rebalance)

        self._enforce_cash_floor(costs_before, date)

    def signOff(self, date, lastPriceMap):
        self._validate_prices(
            lastPriceMap,
            {asset for asset, position in self.positions.items() if position},
            date,
        )
        management_fee = self.getNAV(lastPriceMap) * ((1 + self.annualManagementFee) ** (1 / 260) - 1)
        borrow_costs = self.calcDailyBorrowCost(lastPriceMap)
        self.cash -= management_fee + borrow_costs
        self.historicalPositions[date] = copy.deepcopy(self.positions)
        self.historicalNAV[date] = float(self.getNAV(lastPriceMap))
        self.historicalWeights[date] = copy.deepcopy(self.getWeights(lastPriceMap))
        self.historicalTCosts[date] = self.transactionCosts
        self.historicalSlippageCosts[date] = self.slippageCosts
        self.historicalBorrowCosts[date] = borrow_costs
        self.historicalCash[date] = self.cash
        # Performance statistics are computed lazily (see _performance_at /
        # _performance_rows) so the daily loop stays linear.
        if self.datadump:
            self._write_daily_dump(date, lastPriceMap)

    def _write_daily_dump(self, date, lastPriceMap):
        self.backtestFolderName.mkdir(parents=True, exist_ok=True)
        daily_node = {"Date": date.strftime("%Y-%m-%d"), "NAV": self.historicalNAV[date], "Cash": self.cash,
                      "FirstRebalanceDate": self.FirstRebalanceDate, "LastRebalanceDate": self.LastRebalanceDate,
                      "TransactionCosts": self.transactionCosts, "FixedTransactionCosts": self.fixedTransactionCosts,
                      "DailyBorrowCost": self.historicalBorrowCosts[date], "SlippageModel": self.slippageModel,
                      "Positions": self.positions, "Weights": self.getWeights(lastPriceMap),
                      "Performance": self._performance_at(date), "CustomData": self.getCustomDataByDate(date)}
        path = self.backtestFolderName / f"{date:%Y-%m-%d}.json"
        path.write_text(json.dumps([sanitizeJSON(daily_node)], indent=2, default=str, allow_nan=False))
