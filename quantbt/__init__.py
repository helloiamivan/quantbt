"""Tools for researching and backtesting quantitative investment strategies."""

from .analytics import getNAVPlot, getWeightsPlot, performanceSummary
from .portfolio import Portfolio

# The data handlers are imported lazily (PEP 562) so that the optional
# yfinance dependency is only required when those adapters are actually used.
_DATA_HANDLERS = {"CsvDataHandler", "YahooFinanceDataHandler", "csvDataHandler", "yfDataHandler"}

__all__ = [
    "Portfolio",
    "CsvDataHandler",
    "YahooFinanceDataHandler",
    "csvDataHandler",
    "yfDataHandler",
    "performanceSummary",
    "getNAVPlot",
    "getWeightsPlot",
]


def __getattr__(name):
    if name in _DATA_HANDLERS:
        from . import data  # noqa: PLC0415

        return getattr(data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
